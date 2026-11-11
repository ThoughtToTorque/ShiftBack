#!/usr/bin/env python3
"""Paired statistics of a frozen-vs-ShiftBack LIBERO evaluation (the numbers the paper reports).

Every episode was run under identical initial state and policy noise in both modes, so the
comparison is paired: gained = frozen fails and ShiftBack succeeds, lost = the reverse. Reported:
success rates, gained / lost counts, the exact binomial (McNemar) p-value on the discordant pairs,
a 95 % bootstrap interval of the success-rate difference over episodes, and a 95 % task-clustered
bootstrap interval (tasks resampled with replacement, so that all episodes of a task move
together) of the baseline rate, the candidate rate and their difference.

Input: the ``rows.json`` (or ``metrics.json``, whose ``rows.json`` sibling is then read) written by
``eval_paired.py``. One file holding both modes, or two files (``--baseline`` / ``--candidate``).
Output: a json report (``--out``, default next to the input) and an optional per-task bar plot.

Examples
    python sim/paired_stats.py outputs/object_swap/rows.json
    python sim/paired_stats.py --baseline outputs/run_a/rows.json --candidate outputs/run_b/rows.json \
        --modes shiftback,shiftback --plot outputs/run_b/per_task.png
"""
import argparse
import json
import math
import os

import numpy as np


def load_rows(path):
    d = json.load(open(path))
    if isinstance(d, dict):
        if "rows" in d:
            return d["rows"]
        return json.load(open(os.path.join(os.path.dirname(path), "rows.json")))
    return d


def pick_mode(rows, wanted):
    present = sorted(set(r["mode"] for r in rows))
    if wanted in present:
        return [r for r in rows if r["mode"] == wanted]
    if len(present) == 1:
        return rows
    raise ValueError(f"mode {wanted!r} not in file (modes {present})")


def key_of(r):
    return (r.get("suite", ""), int(r["task_id"]), int(r["episode_index"]))


def mcnemar_exact(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    one_tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * one_tail))


def bootstrap_episodes(baseline, candidate, seed, samples=20000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    deltas = candidate[idx].mean(axis=1) - baseline[idx].mean(axis=1)
    return np.quantile(deltas, [0.025, 0.975]).astype(float).tolist()


def bootstrap_tasks(task_of, baseline, candidate, seed, reps=4000):
    """Resample tasks with replacement; -> 95 % intervals of baseline rate, candidate rate and delta."""
    tasks = sorted(set(task_of))
    arr = [np.stack([baseline[task_of == t], candidate[task_of == t]], 1) for t in tasks]
    rng = np.random.default_rng(seed)
    B, C, D = [], [], []
    for _ in range(reps):
        idx = rng.integers(0, len(tasks), len(tasks))
        s = np.concatenate([arr[i] for i in idx])
        B.append(s[:, 0].mean()); C.append(s[:, 1].mean()); D.append((s[:, 1] - s[:, 0]).mean())
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"baseline": ci(B), "candidate": ci(C), "delta": ci(D)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rows", nargs="?", help="rows.json / metrics.json holding both modes")
    ap.add_argument("--baseline", help="rows.json / metrics.json of the baseline (two-file form)")
    ap.add_argument("--candidate", help="rows.json / metrics.json of the candidate (two-file form)")
    ap.add_argument("--modes", default="frozen,shiftback", help="baseline,candidate mode names")
    ap.add_argument("--seed", type=int, default=20260722, help="episode bootstrap seed")
    ap.add_argument("--samples", type=int, default=20000, help="episode bootstrap replicates")
    ap.add_argument("--cluster-seed", type=int, default=0, help="task-clustered bootstrap seed")
    ap.add_argument("--cluster-reps", type=int, default=4000, help="task-clustered bootstrap replicates")
    ap.add_argument("--out", default=None, help="report json (default: paired_stats.json next to the input)")
    ap.add_argument("--plot", default=None, help="write a per-task successes bar plot (png)")
    a = ap.parse_args()
    m_base, m_cand = a.modes.split(",")
    if a.rows:
        rows = load_rows(a.rows)
        base_rows, cand_rows = pick_mode(rows, m_base), pick_mode(rows, m_cand)
        src = a.rows
    elif a.baseline and a.candidate:
        base_rows, cand_rows = pick_mode(load_rows(a.baseline), m_base), pick_mode(load_rows(a.candidate), m_cand)
        src = a.candidate
    else:
        ap.error("give one rows.json, or --baseline and --candidate")

    base = {key_of(r): r for r in base_rows}
    cand = {key_of(r): r for r in cand_rows}
    if set(base) != set(cand):
        raise ValueError(f"not paired: baseline-only={len(set(base) - set(cand))}, candidate-only={len(set(cand) - set(base))}")
    keys = sorted(base)
    b = np.asarray([float(base[k]["success"]) for k in keys])
    c = np.asarray([float(cand[k]["success"]) for k in keys])
    task_of = np.asarray([k[1] for k in keys])
    gained = int(np.sum((b == 0) & (c == 1)))
    lost = int(np.sum((b == 1) & (c == 0)))

    per_task = []
    for t in sorted(set(task_of.tolist())):
        m = task_of == t
        per_task.append({"task_id": int(t), "episodes": int(m.sum()), "baseline_successes": int(b[m].sum()),
                         "candidate_successes": int(c[m].sum()), "delta_successes": int(c[m].sum() - b[m].sum())})

    report = {
        "source": src, "baseline_mode": m_base, "candidate_mode": m_cand,
        "suites": sorted(set(k[0] for k in keys)),
        "paired_episodes": len(keys), "tasks": len(per_task),
        "baseline_successes": int(b.sum()), "candidate_successes": int(c.sum()),
        "baseline_success_rate": float(b.mean()), "candidate_success_rate": float(c.mean()),
        "success_rate_delta": float(c.mean() - b.mean()),
        "gained": gained, "lost": lost, "exact_binomial_p": mcnemar_exact(gained, lost),
        "delta_ci95_episodes": bootstrap_episodes(b, c, a.seed, a.samples),
        "ci95_task_clustered": bootstrap_tasks(task_of, b, c, a.cluster_seed, a.cluster_reps),
        "per_task": per_task,
    }
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(src)), "paired_stats.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    tc = report["ci95_task_clustered"]
    print(f"n={len(keys)} episodes, {len(per_task)} tasks")
    print(f"{m_base:>10s}: {100 * b.mean():.1f}%  [{100 * tc['baseline'][0]:.1f}, {100 * tc['baseline'][1]:.1f}] task-clustered")
    print(f"{m_cand:>10s}: {100 * c.mean():.1f}%  [{100 * tc['candidate'][0]:.1f}, {100 * tc['candidate'][1]:.1f}]")
    print(f"     delta: {100 * (c.mean() - b.mean()):+.1f} pp  episodes CI [{100 * report['delta_ci95_episodes'][0]:+.1f}, "
          f"{100 * report['delta_ci95_episodes'][1]:+.1f}]  task-clustered CI [{100 * tc['delta'][0]:+.1f}, {100 * tc['delta'][1]:+.1f}]")
    print(f"    gained {gained}  lost {lost}  exact binomial p = {report['exact_binomial_p']:.4f}")
    print(f"-> {out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(per_task)); w = 0.38
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(x - w / 2, [r["baseline_successes"] for r in per_task], w, label=m_base)
        ax.bar(x + w / 2, [r["candidate_successes"] for r in per_task], w, label=m_cand)
        ax.set_xticks(x); ax.set_xticklabels([str(r["task_id"]) for r in per_task])
        ax.set_xlabel("LIBERO task id"); ax.set_ylabel("successful paired trials")
        ax.set_ylim(0, max(r["episodes"] for r in per_task) + 0.5)
        ax.grid(axis="y", alpha=0.25); ax.legend(loc="best"); fig.tight_layout()
        fig.savefig(a.plot, dpi=180); plt.close(fig)
        print(f"-> {a.plot}")


if __name__ == "__main__":
    main()
