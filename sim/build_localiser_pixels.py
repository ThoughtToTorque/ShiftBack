#!/usr/bin/env python3
"""Run the fine-tuned OWLv2 point localiser on the dumped first frames and write the pixel table.

``live[suite/task/init]`` is the localiser's pixel for the stage-0 phrase on every first frame;
``mem[prompt]`` is its mean over the first frames of the demonstrated layout (suites without
``_swap`` in their name), i.e. the memorised scene as the localiser sees it, label-free.
Every prompt is queried with one phrase per grasp stage (:func:`shiftback.libero_phrases.stage_phrases`)
and one per placement (:func:`place_phrases`) on the same first frame (the stage-k object does not
move before stage k): ``mem_stages`` / ``live_stages`` hold the per-stage pixels, ``mem_place`` /
``live_place`` the release-anchor pixels. For "put both A and B" the grasp order is resolved by
matching the localiser's pixels on the demonstrated layout to the demonstrated stage pixels
(``--stages``, gripper pose only); for a duplicated phrase ("both moka pots") stage 1 takes the
second NMS peak of the same heat-map. All pixels are in the 256-px frame.

Inputs: ``--model`` (checkpoint dir from localiser/finetune_owlv2.py), ``--frames`` (npz files from
dump_first_frames.py), ``--stages`` (anchors_stages.json), ``--hold`` (hold_height.json, copied
into the table as ``mem_hold``). ``--sim-truth`` prints the stage-0 hit rate against the simulator
object pixel when the frames carry it (validation only; the method never uses it).
Output: ``--out`` json with the schema of ``data/libero/localiser_pixels_owlv2.json``.

Example
    python sim/build_localiser_pixels.py --model outputs/localiser/owlv2_ft/best \
        --frames outputs/first_frames/swap_frames.npz outputs/first_frames/base_frames.npz
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shiftback.libero_phrases import place_phrases, stage_phrases  # noqa: E402

GRID = 60
FRAMES_DIR = os.path.join(ROOT, "outputs", "first_frames")


def peaks(lg, n=2, nms_patches=6):
    """Top-n NMS peaks of a (GRID*GRID,) logit map -> list of (uv256, prob)."""
    g = lg.reshape(GRID, GRID).copy(); out = []
    for _ in range(n):
        j = int(g.argmax()); y, x = divmod(j, GRID)
        out.append(([float((x + 0.5) / GRID * 256), float((y + 0.5) / GRID * 256)], float(1 / (1 + np.exp(-g[y, x])))))
        g[max(0, y - nms_patches):y + nms_patches + 1, max(0, x - nms_patches):x + nms_patches + 1] = -1e9
    return out


def is_memorised(m):
    """Frames of the demonstrated layout contribute to ``mem``; relocated suites are live-only."""
    return "_swap" not in m["suite"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "libero", "localiser_pixels.json"))
    ap.add_argument("--frames", nargs="+", default=[os.path.join(FRAMES_DIR, "swap_frames.npz"), os.path.join(FRAMES_DIR, "base_frames.npz")])
    ap.add_argument("--hold", default=os.path.join(ROOT, "data", "libero", "hold_height.json"))
    ap.add_argument("--stages", default=os.path.join(ROOT, "data", "libero", "anchors_stages.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sim-truth", action="store_true", help="print the hit rate against the simulator object pixel")
    args = ap.parse_args()
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(args.model).to(args.device).eval()
    stage_tbl = json.load(open(args.stages)) if args.stages and os.path.exists(args.stages) else {}

    # ---- pass 1: per (frame, phrase) heat-map peaks ----
    frames, metas = [], []
    for f in args.frames:
        if not os.path.exists(f):
            print("missing", f); continue
        d = np.load(f); metas += json.loads(str(d["meta"])); frames.append(d["imgs"])
    imgs = np.concatenate(frames)
    queries = []  # (frame idx, stage k, phrase); release anchors use key k + 100
    for i, m in enumerate(metas):
        ph = stage_phrases(m["prompt"]) or [m["prompt"]]
        for k, p in enumerate(ph):
            queries.append((i, k, p))
        for k, p in enumerate(place_phrases(m["prompt"]) or []):
            if p is not None:
                queries.append((i, 100 + k, p))
    pk = {}  # (i, k) -> peaks
    for q in range(0, len(queries), 16):
        qs = queries[q:q + 16]
        inp = proc(text=[[f"a photo of a {p}"] for _, _, p in qs], images=[Image.fromarray(imgs[i]) for i, _, _ in qs],
                   return_tensors="pt").to(args.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**inp).logits[:, :, 0].float().cpu().numpy()
        for (i, k, _), l in zip(qs, lg):
            pk[(i, k)] = peaks(l)

    # ---- pass 2: grasp order of "put both" prompts from the demonstrated layout (label-side stage pixels) ----
    order = {}  # prompt -> permutation of text-order phrases
    for prompt in {m["prompt"] for m in metas}:
        ph = stage_phrases(prompt) or [prompt]
        if len(ph) < 2 or not prompt.startswith("put both") or prompt not in stage_tbl:
            order[prompt] = list(range(len(ph))); continue
        lab = [np.asarray(s["uv"], np.float64) for s in stage_tbl[prompt]["stages"]]
        base = [i for i, m in enumerate(metas) if m["prompt"] == prompt and is_memorised(m)]
        if ph[0] == ph[1]:  # duplicate phrase: stage 0 = the peak nearer label stage 0, stage 1 = the other peak
            order[prompt] = "second_peak"; continue
        det = [np.mean([pk[(i, k)][0][0] for i in base], 0) for k in range(2)]
        # the localiser has a small constant offset to the label frame; compare DIFFERENCES, which cancel it
        same = np.linalg.norm((det[0] - det[1]) - (lab[0] - lab[1])); flip = np.linalg.norm((det[1] - det[0]) - (lab[0] - lab[1]))
        order[prompt] = [0, 1] if same <= flip else [1, 0]
        print(f"[order] {prompt[:60]}: text {ph} -> stage order {order[prompt]} (same {same:.1f}px vs flipped {flip:.1f}px)")

    live, prob, memacc, live_st, memacc_st, diag = {}, {}, defaultdict(list), {}, defaultdict(list), []
    live_pl, memacc_pl = {}, defaultdict(list)  # release anchors, per stage (None = stage places nothing)
    for i, m in enumerate(metas):
        ph = stage_phrases(m["prompt"]) or [m["prompt"]]
        o = order[m["prompt"]]
        if o == "second_peak":
            p0, p1 = pk[(i, 0)][0], pk[(i, 0)][1]
            lab = [np.asarray(s["uv"], np.float64) for s in stage_tbl[m["prompt"]]["stages"]]
            d_lab = lab[0] - lab[1]  # the pairing whose difference vector matches the label difference
            if np.linalg.norm((np.array(p0[0]) - np.array(p1[0])) - d_lab) <= np.linalg.norm((np.array(p1[0]) - np.array(p0[0])) - d_lab):
                st = [p0, p1]
            else:
                st = [p1, p0]
        else:
            st = [pk[(i, k)][0] for k in o]
        key = f"{m['suite']}/{m['task']}/{m['init']}"
        live[key] = st[0][0]; prob[key] = st[0][1]
        live_st[key] = [s[0] for s in st]
        ks = range(len(ph)) if o == "second_peak" else o
        _pl = [pk[(i, 100 + k)][0][0] if (i, 100 + k) in pk else None for k in ks]
        live_pl[key] = _pl
        if is_memorised(m):
            memacc[m["prompt"]].append(st[0][0]); memacc_st[m["prompt"]].append([s[0] for s in st])
            memacc_pl[m["prompt"]].append(_pl)
        if args.sim_truth and m.get("true") is not None:
            P = np.asarray(m["w2p"]); pt = P @ np.append(np.asarray(m["true"]), 1.0); tb = pt[:2] / max(pt[2], 1e-6)
            diag.append((m["suite"], m["task"], float(np.hypot(st[0][0][0] - tb[0], st[0][0][1] - tb[1]))))
    mem = {k: np.mean(v, 0).tolist() for k, v in memacc.items()}
    sd = {k: np.std(v, 0).round(1).tolist() for k, v in memacc.items()}
    mem_st = {k: np.mean(np.asarray(v), 0).tolist() for k, v in memacc_st.items()}
    mem_st_sd = {k: np.std(np.asarray(v), 0).round(1).tolist() for k, v in memacc_st.items()}
    mem_pl = {k: [None if v[0][j] is None else np.mean([x[j] for x in v], 0).tolist() for j in range(len(v[0]))]
              for k, v in memacc_pl.items()}
    mem_pl_sd = {k: [None if v[0][j] is None else np.std([x[j] for x in v], 0).round(1).tolist() for j in range(len(v[0]))]
                 for k, v in memacc_pl.items()}
    table = dict(mem=mem, live=live, mem_std=sd, live_prob=prob, mem_stages=mem_st, live_stages=live_st,
                 mem_stages_std=mem_st_sd, mem_place=mem_pl, live_place=live_pl, mem_place_std=mem_pl_sd,
                 source=f"OWLv2 fine-tuned ({args.model}), NMS peaks, 256-px frame, per-stage phrases",
                 mem_hold=(json.load(open(args.hold)) if args.hold and os.path.exists(args.hold) else None))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=1)
    print(len(live), "live", len(mem), "mem ->", args.out)
    by = defaultdict(list)
    for s, t, e in diag:
        by[(s, t)].append(e)
    for s in sorted(set(k[0] for k in by)):
        tasks = sorted(t for (ss, t) in by if ss == s)
        print(s, "stage-0 hit<20px per task:", " ".join(str(int(10 * np.mean(np.array(by[(s, t)]) < 20))) for t in tasks))
    for k, v in mem_st.items():
        if len(v) > 1:
            print("[stages]", k[:60], np.round(v).tolist(), "std", mem_st_sd[k])
    for k, v in mem_pl.items():
        print("[place]", k[:60], [None if x is None else np.round(x).tolist() for x in v], "std", mem_pl_sd[k])


if __name__ == "__main__":
    main()
