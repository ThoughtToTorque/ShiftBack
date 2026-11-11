#!/usr/bin/env python3
"""Grasp and release anchors of every LIBERO prompt, taken from the demonstrations' gripper alone.

For each prompt the demonstrations are segmented by the gripper command channel (``actions[:, 6]``,
+1 = close) with the three-step debounce of :func:`shiftback.anchors.gripper_events`. The k-th
closure of a demonstration gives the stage-k grasp point ``g`` (gripper position, robot base frame,
metres) and the k-th opening the release point ``release_g``; both are projected into the 224-px
policy image by the fixed per-task camera (``camera_transforms.json``). Nothing about the objects
is read: only the gripper position, the gripper command and the camera matrix enter the tables.

Demonstrations whose number of closures differs from the prompt's modal count (re-grasps) are
dropped from the per-stage averages. ``first_grasp_frac`` is the median fraction of the
demonstration elapsed before the first closure (prompts that spend most of the demo before any
grasp are left uncorrected by the evaluation, see ``eval_paired.py --skip-late-grasp``).

Inputs
  the LeRobot dataset ``physical-intelligence/libero`` (the slow column read is cached once in
  ``--cache-npz``), ``--cameras`` (per-task world -> 256-px pixel matrices) and optionally
  ``--exclude-episodes`` (episode indices to leave out, e.g. a validation split).
Outputs
  ``--grasp-out``   {prompt: {g, g_std, uv, uv_std, n}}  mean gripper position at the FIRST closure
                    over every demonstration of the prompt that has one, its pixel (224 px) and spread
  ``--stages-out``  the same keys plus K, n_episodes, n_kept, first_grasp_frac and ``stages[k]`` with
                    g, g_std, uv, uv_std, n, release_g, release_uv, release_g_std, release_n

Example
    python sim/build_anchors_libero.py --cache-npz outputs/libero/libero_lowdim.npz
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
from shiftback.geometry import project  # noqa: E402

MODEL_RES, LABEL_RES = 224.0, 256.0
LOWDIM_COLUMNS = ["index", "episode_index", "frame_index", "task_index", "state", "actions"]


def open_dataset(repo_id):
    try:
        from lerobot.common.datasets import lerobot_dataset
    except ImportError:  # newer lerobot moved the module
        from lerobot.datasets import lerobot_dataset
    return lerobot_dataset.LeRobotDataset(repo_id)


def task_names(ds):
    """Prompt text per task index, for the dict and the data-frame flavours of ``meta.tasks``."""
    tasks = ds.meta.tasks
    if hasattr(tasks, "iterrows"):
        d = {int(row["task_index"]): str(name) for name, row in tasks.iterrows()}
    else:
        d = {int(k): (str(v["task"]) if isinstance(v, dict) else str(v)) for k, v in tasks.items()}
    return [d[i] for i in range(len(d))]


def load_lowdim(cache_npz, repo_id):
    """Low-dimensional columns of the dataset as numpy arrays (read once, then cached)."""
    if cache_npz and os.path.exists(cache_npz):
        d = np.load(cache_npz, allow_pickle=True)
        return {k: d[k] for k in d.files}
    ds = open_dataset(repo_id)
    src = ds.hf_dataset.with_format("numpy", columns=LOWDIM_COLUMNS)
    cols = {k: np.asarray(src[k]) for k in LOWDIM_COLUMNS[:4]}
    cols["state"] = np.stack(src["state"]).astype(np.float32)
    cols["actions"] = np.stack(src["actions"]).astype(np.float32)
    cols["task_names"] = np.array(task_names(ds))
    if cache_npz:
        os.makedirs(os.path.dirname(os.path.abspath(cache_npz)), exist_ok=True)
        np.savez_compressed(cache_npz, **cols)
        print(f"[cache] {cache_npz}")
    return cols


def closes_and_opens(cmd, debounce=3):
    """Closure and opening frames of one demonstration.

    ``gripper_events`` commits a flip only after ``debounce`` steps; the demonstrations end with
    openings shorter than that, which count as events here (the command is padded with its last
    value so that a flip in the final steps is accepted).
    """
    cmd = np.asarray(cmd, np.float64)
    ev = gripper_events(np.concatenate([cmd, np.repeat(cmd[-1:], debounce - 1)]), debounce)
    return [t for t, s in ev if s > 0], [t for t, s in ev if s < 0]


def pix224(w2p, X):
    """World point(s) -> 224-px policy image (the camera matrices are in the 256-px render frame)."""
    return np.atleast_2d(project(w2p[:3], X)) * (MODEL_RES / LABEL_RES)


def read_episode_list(path):
    if not path:
        return set()
    txt = open(path).read()
    vals = json.loads(txt) if txt.lstrip().startswith("[") else txt.split()
    return set(int(x) for x in vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-id", default="physical-intelligence/libero")
    ap.add_argument("--cache-npz", default=os.path.join(ROOT, "outputs", "libero", "libero_lowdim.npz"))
    ap.add_argument("--cameras", default=os.path.join(ROOT, "data", "libero", "camera_transforms.json"))
    ap.add_argument("--exclude-episodes", default=None, help="json list or whitespace-separated episode indices to skip")
    ap.add_argument("--debounce", type=int, default=3)
    ap.add_argument("--min-episodes", type=int, default=5, help="release anchors need at least this many demos")
    ap.add_argument("--grasp-out", default=os.path.join(ROOT, "outputs", "libero", "anchors_grasp.json"))
    ap.add_argument("--stages-out", default=os.path.join(ROOT, "outputs", "libero", "anchors_stages.json"))
    a = ap.parse_args()

    L = load_lowdim(a.cache_npz, a.repo_id)
    cams = json.load(open(a.cameras))["tasks"]
    names = [str(x) for x in L["task_names"]]
    ep_idx, fr_idx, tk_idx = L["episode_index"], L["frame_index"], L["task_index"]
    state, actions = L["state"], L["actions"]
    excluded = read_episode_list(a.exclude_episodes)

    per_prompt = {p: [] for p in names}
    for e in sorted(set(int(x) for x in np.unique(ep_idx)) - excluded):
        r = np.flatnonzero(ep_idx == e)
        r = r[np.argsort(fr_idx[r])]
        closes, opens = closes_and_opens(actions[r, 6], a.debounce)
        per_prompt[names[int(tk_idx[r[0]])]].append((r, closes, opens))
    print(f"[data] {sum(len(v) for v in per_prompt.values())} demonstrations, {len(names)} prompts")

    grasp, stages_out = {}, {}
    for t, prompt in enumerate(names):
        eps = per_prompt[prompt]
        if not eps:
            continue
        w2p = np.asarray(cams[str(t)]["world_to_pixel"], np.float64)

        # first-closure anchor over every demonstration that has a closure; a prompt without any
        # closure (push / pull tasks) gets the final gripper position so that the keys exist, but the
        # evaluation never corrects such prompts (first_grasp_frac = 1)
        with_close = [(r, c) for r, c, _ in eps if c]
        if with_close:
            G = np.stack([state[r[c[0]], :3] for r, c in with_close])
        else:
            G = np.stack([state[r[-1], :3] for r, _, _ in eps])
        UV = pix224(w2p, G)
        grasp[prompt] = {"g": G.mean(0).tolist(), "g_std": G.std(0).tolist(),
                         "uv": UV.mean(0).tolist(), "uv_std": UV.std(0).tolist(), "n": int(len(G))}

        counts = Counter(len(c) for _, c, _ in eps)
        K = counts.most_common(1)[0][0]
        kept = [(r, c, o) for r, c, o in eps if len(c) == K]
        stages = []
        for k in range(K):
            Gk = np.stack([state[r[c[k]], :3] for r, c, _ in kept])
            rel = [state[r[o[k]], :3] for r, c, o in kept if len(o) > k]
            enough = len(rel) >= a.min_episodes
            R = np.stack(rel) if rel else None
            stages.append({
                "g": Gk.mean(0).tolist(), "g_std": Gk.std(0).tolist(),
                "uv": pix224(w2p, Gk.mean(0))[0].tolist(), "uv_std": pix224(w2p, Gk).std(0).tolist(),
                "n": int(len(Gk)),
                "release_g": R.mean(0).tolist() if enough else None,
                # stored, not re-projected on the fly: the place-side gate compares it with the
                # localiser's pixel as two independent quantities
                "release_uv": pix224(w2p, R.mean(0))[0].tolist() if enough else None,
                "release_g_std": R.std(0).tolist() if enough else None,
                "release_n": int(len(rel)),
            })
        fg = [(c[0] / len(r)) if c else 1.0 for r, c, _ in eps]
        stages_out[prompt] = dict(grasp[prompt])
        stages_out[prompt].update({"K": int(K), "n_episodes": len(eps), "n_kept": len(kept),
                                   "first_grasp_frac": float(np.median(fg)), "stages": stages})

    for path, table in ((a.grasp_out, grasp), (a.stages_out, stages_out)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(table, f, indent=1)
        print(f"[write] {path}  ({len(table)} prompts)")

    print("\nprompt (task_index)                                          K  kept/all  stage spreads")
    for t, prompt in enumerate(names):
        v = stages_out.get(prompt)
        if v is None:
            continue
        sp = "  ".join(f"s{k}: g_std={max(s['g_std'][:2]) * 100:.1f}cm uv_std={max(s['uv_std']):.1f}px n={s['n']}"
                       for k, s in enumerate(v["stages"]))
        if v["K"] >= 2:
            d = np.linalg.norm(np.array(v["stages"][1]["g"]) - np.array(v["stages"][0]["g"]))
            sp += f"  |s1-s0|={d * 100:.0f}cm"
        print(f"{prompt[:56]:56s} ({t:2d}) {v['K']}  {v['n_kept']:3d}/{v['n_episodes']:3d}  {sp}")


if __name__ == "__main__":
    main()
