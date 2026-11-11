#!/usr/bin/env python3
"""Training set for the OWLv2 point localiser from the LIBERO demonstrations (gripper labels only).

One row per sampled pre-grasp frame: the 256-px agentview image, the phrase the detector is asked
for, and the label pixel = the gripper position at the gripper event, projected by the fixed
per-task camera. No object pose is read. Frames are sampled every ``--stride`` steps from the start
of the demonstration up to the closure (the object is static in that window), at most
``--max-per-ep`` per window.

Default (first grasp only): phrase = :func:`shiftback.libero_phrases.first_grasp_phrase`, label =
gripper position at the first positive gripper command; "put both A and B" prompts are excluded.

``--stages`` (stage-aware superset): one phrase per grasp stage (:func:`stage_phrases`), closures
found with the debounced :func:`shiftback.anchors.gripper_events`, demonstrations whose closure
count differs from the prompt's modal count dropped. Stage-1 rows use the frames before the first
closure and the frames between the first opening and the second closure. For a duplicated phrase
("put both moka pots") each row carries a second target ``uv256_b`` (the other pot; after the first
placement, the projected release position of the gripper).

``--with-place`` adds release-anchor rows (``role`` = 1): phrase = :func:`place_phrases`, label =
gripper position at the k-th opening. ``--surface-release`` moves that label down to the surface
under the gripper, by h = min(z_close, z_open) - z_table, where z_table is the lowest first-grasp
height among the tasks sharing a camera minus 3 cm; the mean h per prompt and stage is written to
``--hold-out`` (the ``hold_height.json`` the evaluation reads).

Output npz: image (N,256,256,3) uint8, global_index, episode, task, phrase, uv256, uv256_b (NaN =
none), frame, stage, role, task_names.

Example
    python localiser/build_dataset.py --stages --with-place --surface-release \
        --out outputs/localiser/train_frames.npz
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shiftback.anchors import gripper_events  # noqa: E402
from shiftback.libero_phrases import first_grasp_phrase, place_phrases, stage_phrases  # noqa: E402

LOWDIM_COLUMNS = ["index", "episode_index", "frame_index", "task_index", "state", "actions"]
NAN2 = np.array([np.nan, np.nan], np.float32)


def open_dataset(repo_id):
    try:
        from lerobot.common.datasets import lerobot_dataset
    except ImportError:
        from lerobot.datasets import lerobot_dataset
    return lerobot_dataset.LeRobotDataset(repo_id)


def task_names(ds):
    tasks = ds.meta.tasks
    if hasattr(tasks, "iterrows"):
        d = {int(row["task_index"]): str(name) for name, row in tasks.iterrows()}
    else:
        d = {int(k): (str(v["task"]) if isinstance(v, dict) else str(v)) for k, v in tasks.items()}
    return [d[i] for i in range(len(d))]


def load_lowdim(ds, cache_npz):
    if cache_npz and os.path.exists(cache_npz):
        d = np.load(cache_npz, allow_pickle=True)
        return {k: d[k] for k in d.files}
    src = ds.hf_dataset.with_format("numpy", columns=LOWDIM_COLUMNS)
    cols = {k: np.asarray(src[k]) for k in LOWDIM_COLUMNS[:4]}
    cols["state"] = np.stack(src["state"]).astype(np.float32)
    cols["actions"] = np.stack(src["actions"]).astype(np.float32)
    cols["task_names"] = np.array(task_names(ds))
    if cache_npz:
        os.makedirs(os.path.dirname(os.path.abspath(cache_npz)), exist_ok=True)
        np.savez_compressed(cache_npz, **cols)
    return cols


def closes_and_opens(cmd, debounce=3):
    """Debounced closure / opening frames; padded so a flip in the final steps is accepted too."""
    cmd = np.asarray(cmd, np.float64)
    ev = gripper_events(np.concatenate([cmd, np.repeat(cmd[-1:], debounce - 1)]), debounce)
    return [t for t, s in ev if s > 0], [t for t, s in ev if s < 0]


def inside(uv):
    return 2 <= uv[0] <= 254 and 2 <= uv[1] <= 254


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-id", default="physical-intelligence/libero")
    ap.add_argument("--cache-npz", default=os.path.join(ROOT, "outputs", "libero", "libero_lowdim.npz"))
    ap.add_argument("--cameras", default=os.path.join(ROOT, "data", "libero", "camera_transforms.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "localiser", "train_frames.npz"))
    ap.add_argument("--stages", action="store_true", help="one phrase per grasp stage, debounced events")
    ap.add_argument("--with-place", action="store_true", help="add release-anchor rows (implies --stages)")
    ap.add_argument("--surface-release", action="store_true", help="release label on the surface under the gripper")
    ap.add_argument("--hold-out", default=os.path.join(ROOT, "outputs", "localiser", "hold_height.json"))
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-per-ep", type=int, default=12)
    ap.add_argument("--debounce", type=int, default=3)
    a = ap.parse_args()
    a.stages = a.stages or a.with_place or a.surface_release
    stride, cap = a.stride, a.max_per_ep

    ds = open_dataset(a.repo_id)
    d = load_lowdim(ds, a.cache_npz)
    w2p = {int(k): np.asarray(v["world_to_pixel"], float) for k, v in json.load(open(a.cameras))["tasks"].items()}
    ep, fr, ti, st, ac = d["episode_index"], d["frame_index"], d["task_index"], d["state"], d["actions"]
    names = d["task_names"]
    assert len(ds.hf_dataset) == len(ep), (len(ds.hf_dataset), len(ep))

    def proj(t, g):
        pix = w2p[t][:3] @ np.append(g, 1.0)
        return (pix[:2] / max(pix[2], 1e-6)).astype(np.float32)

    rows = []  # (global index, episode, task, phrase, uv256, uv256_b, frame, stage, role)
    if not a.stages:
        for e in np.unique(ep):
            idx = np.nonzero(ep == e)[0]
            idx = idx[np.argsort(fr[idx])]
            t = int(ti[idx[0]])
            phrase = first_grasp_phrase(str(names[t]))
            if phrase is None:
                continue
            close = np.nonzero(ac[idx, 6] > 0)[0]
            if len(close) == 0 or close[0] < stride:
                continue
            uv = proj(t, st[idx[close[0]], :3])
            if not inside(uv):
                continue
            for gi in idx[np.arange(0, close[0], stride)][:cap]:
                rows.append((int(gi), int(e), t, phrase, uv, NAN2, int(fr[gi]), 0, 0))
    else:
        per_task = {}
        for e in np.unique(ep):
            idx = np.nonzero(ep == e)[0]
            idx = idx[np.argsort(fr[idx])]
            t = int(ti[idx[0]])
            if stage_phrases(str(names[t])) is None:
                continue
            c, o = closes_and_opens(ac[idx, 6], a.debounce)
            per_task.setdefault(t, []).append((e, idx, c, o))
        # scene = tasks sharing a camera matrix; its table height from the lowest first-grasp height
        scene_of = {t: tuple(np.round(w2p[t], 3).ravel().tolist()) for t in per_task}
        z_table = {}
        for t, eps in per_task.items():
            zc = [st[idx[c[0]], 2] for _, idx, c, _ in eps if len(c)]
            if zc:
                z_table[scene_of[t]] = min(z_table.get(scene_of[t], 1e9), float(np.min(zc)) - 0.03)
        hold, pl_stats = {}, {}
        for t, eps in per_task.items():
            prompt = str(names[t])
            ph = stage_phrases(prompt)
            K = len(ph)
            modal = Counter(len(c) for _, _, c, _ in eps).most_common(1)[0][0]
            kept = [x for x in eps if len(x[2]) == modal and len(x[2]) >= K]
            pl = place_phrases(prompt) or [None] * K
            dup = K == 2 and ph[0] == ph[1]
            for e, idx, c, o in kept:
                if c[0] < stride:
                    continue
                uvs = [proj(t, st[idx[c[k]], :3]) for k in range(K)]
                if not all(inside(u) for u in uvs):
                    continue
                # release anchors: gripper at the k-th opening (which must follow the k-th closure)
                pl_uv = [None] * K
                for k in range(K):
                    if pl[k] is not None and len(o) > k and o[k] > c[k]:
                        g = np.array(st[idx[o[k]], :3], float)
                        if a.surface_release:
                            # whichever of the pick / place supports is the table gives the object's
                            # height above its support exactly (cabinet-to-plate placements otherwise
                            # fold the cabinet height in)
                            h = max(0.0, min(float(st[idx[c[k]], 2]), float(st[idx[o[k]], 2])) - z_table[scene_of[t]])
                            g[2] -= h
                            hold.setdefault((t, k), []).append(h)
                        u = proj(t, g)
                        if inside(u):
                            pl_uv[k] = u
                            pl_stats.setdefault((t, k), []).append(u)
                # frames before the first closure: stage-0 rows, stage-1 rows (the second object is
                # static before stage 1) and release rows
                for gi in idx[np.arange(0, c[0], stride)][:cap]:
                    rows.append((int(gi), int(e), t, ph[0], uvs[0], uvs[1] if dup else NAN2, int(fr[gi]), 0, 0))
                    for k in range(K):
                        if a.with_place and pl_uv[k] is not None:
                            rows.append((int(gi), int(e), t, pl[k], pl_uv[k], NAN2, int(fr[gi]), k, 1))
                    if K == 2 and not dup:
                        rows.append((int(gi), int(e), t, ph[1], uvs[1], NAN2, int(fr[gi]), 1, 0))
                # frames between the first opening and the second closure (first object placed)
                if K == 2 and len(o) >= 1 and c[1] - o[0] > stride:
                    rel = proj(t, st[idx[o[0]], :3])  # where the gripper let go = where object 1 now sits
                    for gi in idx[np.arange(o[0], c[1], stride)][:cap]:
                        rows.append((int(gi), int(e), t, ph[1], uvs[1], rel if dup else NAN2, int(fr[gi]), 1, 0))
                        if a.with_place and pl_uv[1] is not None:
                            rows.append((int(gi), int(e), t, pl[1], pl_uv[1], NAN2, int(fr[gi]), 1, 1))
            print(f"{prompt[:70]:70s} demos {len(eps)} kept {len(kept)} closures {modal}")
        for (t, k), v in sorted(pl_stats.items()):
            print(f"[release] {str(names[t])[:60]:60s} stage {k} n={len(v)} uv_mean={np.mean(v, 0).round(1).tolist()}")
        if a.surface_release:
            hj = {}
            for (t, k), v in hold.items():
                hj.setdefault(str(names[t]), {})[str(k)] = float(np.mean(v))
            os.makedirs(os.path.dirname(os.path.abspath(a.hold_out)), exist_ok=True)
            with open(a.hold_out, "w") as f:
                json.dump(hj, f, indent=1)
            print(f"[hold] {a.hold_out} ({len(hj)} prompts)")
    print(f"{len(rows)} rows from {len(set(r[1] for r in rows))} episodes; phrases {Counter(r[3] for r in rows).most_common()}", flush=True)

    imgs = np.zeros((len(rows), 256, 256, 3), np.uint8)
    cache = {}
    for k, r in enumerate(rows):
        if r[0] not in cache:
            im = np.asarray(ds.hf_dataset[r[0]]["image"])
            if im.ndim == 3 and im.shape[0] == 3:
                im = np.transpose(im, (1, 2, 0))
            if im.dtype != np.uint8:
                im = (255 * im).clip(0, 255).astype(np.uint8)
            assert im.shape == (256, 256, 3), im.shape
            cache[r[0]] = im
        imgs[k] = cache[r[0]]
        if k % 2000 == 0:
            print(k, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, image=imgs, global_index=np.array([r[0] for r in rows]), episode=np.array([r[1] for r in rows]),
        task=np.array([r[2] for r in rows]), phrase=np.array([r[3] for r in rows]),
        uv256=np.stack([r[4] for r in rows]), uv256_b=np.stack([r[5] for r in rows]),
        frame=np.array([r[6] for r in rows]), stage=np.array([r[7] for r in rows]),
        role=np.array([r[8] for r in rows]), task_names=names)
    print("->", a.out, imgs.shape)


if __name__ == "__main__":
    main()
