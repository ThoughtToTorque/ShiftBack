"""Anchors taken from the gripper itself.

The grasp and release points of a task are the gripper's own positions at the moments its
command closes and opens in the demonstrations. Nothing about the object is measured; the only
inputs are the logged gripper position, the commanded gripper state and (for pixels) the fixed
camera's projection matrix.
"""
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import project


def gripper_events(cmd: Sequence[float], debounce: int = 3, start_sign: float = -1.0) -> List[Tuple[int, float]]:
    """Closing and opening events of a commanded gripper channel.

    ``cmd`` is the per-step gripper command, positive = close. An event is committed when the new
    sign has held for ``debounce`` consecutive steps, judged against the last *accepted* sign, so
    the returned list strictly alternates (+1 close, -1 open) and ``closes[k]`` is the k-th grasp.
    Returns ``[(t, sign), ...]`` with ``t`` the first step of the run.
    """
    sign, run, t0 = float(start_sign), 0, 0
    events: List[Tuple[int, float]] = []
    for t, c in enumerate(cmd):
        s = 1.0 if float(c) > 0.0 else -1.0
        if s != sign:
            if run == 0:
                t0 = t
            run += 1
            if run >= debounce:
                sign, run = s, 0
                events.append((t0, s))
        else:
            run = 0
    return events


def stage_events(cmd: Sequence[float], debounce: int = 3) -> List[Tuple[int, Optional[int]]]:
    """``[(close_t, open_t or None), ...]`` per close/open cycle of one demonstration."""
    ev = gripper_events(cmd, debounce)
    out: List[Tuple[int, Optional[int]]] = []
    for i, (t, s) in enumerate(ev):
        if s > 0:
            nxt = ev[i + 1][0] if i + 1 < len(ev) else None
            out.append((t, nxt))
    return out


def anchors_from_demos(demos: Sequence[Dict], P: Optional[np.ndarray] = None, debounce: int = 3,
                       min_episodes: int = 1) -> Dict:
    """Per-stage grasp and release anchors of one task from its demonstrations.

    ``demos`` is a list of dicts with ``"ee"`` (T,3 gripper position, robot base frame, metres) and
    ``"gripper_cmd"`` (T,) commanded gripper state, positive = close. Demonstrations whose number
    of close/open cycles differs from the modal count (re-grasps) are dropped from the averages.

    Returns a dict with the demonstrated grasp ``g`` and release ``r`` positions per stage, their
    spreads, their pixels under ``P`` (if given), the fraction of the demonstration elapsed before
    the first closure, and ``n`` the number of demonstrations used.
    """
    per_demo = []
    for d in demos:
        ee = np.asarray(d["ee"], np.float64)
        st = stage_events(d["gripper_cmd"], debounce)
        per_demo.append((ee, st))
    counts = Counter(len(st) for _, st in per_demo if st)
    if not counts:
        raise ValueError("no gripper closure found in any demonstration")
    K = counts.most_common(1)[0][0]
    kept = [(ee, st) for ee, st in per_demo if len(st) == K]
    if len(kept) < min_episodes:
        raise ValueError(f"only {len(kept)} demonstrations with {K} stages")
    stages = []
    for k in range(K):
        g = np.array([ee[st[k][0]] for ee, st in kept])
        r_list = [ee[st[k][1]] for ee, st in kept if st[k][1] is not None]
        r = np.array(r_list) if r_list else None
        entry = {"g": g.mean(0).tolist(), "g_std": g.std(0).tolist(), "n": int(len(g))}
        if P is not None:
            entry["uv"] = project(P, g.mean(0)).tolist()
            entry["uv_std"] = np.std(project(P, g), 0).tolist()
        if r is not None:
            entry["r"] = r.mean(0).tolist()
            entry["r_std"] = r.std(0).tolist()
            if P is not None:
                entry["uv_r"] = project(P, r.mean(0)).tolist()
        stages.append(entry)
    first_frac = float(np.median([st[0][0] / max(1, len(ee)) for ee, st in kept]))
    out = dict(stages[0])
    out.update(K=K, n_demos=len(per_demo), n_kept=len(kept), first_grasp_frac=first_frac, stages=stages)
    return out


def start_pose_from_demos(demos: Sequence[Dict]) -> np.ndarray:
    """Mean gripper position at the first step of the demonstrations (the home pose)."""
    return np.mean([np.asarray(d["ee"], np.float64)[0] for d in demos], 0)


__all__ = ["gripper_events", "stage_events", "anchors_from_demos", "start_pose_from_demos"]
