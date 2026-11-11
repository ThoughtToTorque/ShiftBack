#!/usr/bin/env python3
"""Demo 0 (no GPU, a few seconds): the whole ShiftBack computation on recorded robot frames.

Inputs, all shipped in ``data/real_sample/``: the fixed camera's projection matrix, the anchors
taken from the 40 demonstrations (demonstrated grasp/release pixels and heights), the first frame
of a recorded trial in which the tape or the tray had been moved, and the detections OWLv2 gave on
that frame (stored, so that no model is needed; pass ``--detect`` to run OWLv2 instead).

Outputs: the displacement Delta, the image offset, the carry at gripper closure, a figure showing
the demonstrated frame, the live frame and the frame the policy actually sees, and a check against
the numbers the robot used in that trial.

    python demos/00_shift_back_no_gpu.py                      # object moved 10 cm
    python demos/00_shift_back_no_gpu.py --scene tray_moved   # tray moved 9 cm
    python demos/00_shift_back_no_gpu.py --scene demo_layout  # nothing moved: inside the threshold, no shift
    python demos/00_shift_back_no_gpu.py --detect             # run OWLv2 zero-shot instead of the stored detections
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from shiftback import ShiftBackPlanner, kept_box, lift, load_projection, project  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "real_sample")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="object_moved", choices=["object_moved", "tray_moved", "demo_layout"])
    ap.add_argument("--detect", action="store_true", help="run OWLv2 zero-shot (needs torch + transformers)")
    ap.add_argument("--tau", type=float, default=0.04, help="minimum-shift threshold (m)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "outputs"))
    a = ap.parse_args()

    P = load_projection(os.path.join(DATA, "calib", "P.npy"))
    A = json.load(open(os.path.join(DATA, "calib", "anchors.json")))
    live = np.asarray(Image.open(os.path.join(DATA, "live", f"{a.scene}.jpg")).convert("RGB"))
    demo = np.asarray(Image.open(os.path.join(DATA, "demo_first_frames", "episode_000.jpg")).convert("RGB"))
    stored = json.load(open(os.path.join(DATA, "live", "detections.json")))[a.scene]

    if a.detect:
        from shiftback.localiser import OwlLocaliser
        loc = OwlLocaliser()
        phrases = A["tape_phrases"] + A["plate_phrases"]
        det = loc.locate(live, phrases, mode="box")
        best = lambda phs: max((d for p in phs if (d := det.get(p))), key=lambda d: d["score"], default=None)
        det_obj, det_place = best(A["tape_phrases"]), best(A["plate_phrases"])
        print("OWLv2 zero-shot:", {k: (None if v is None else np.round(v["uv"]).tolist()) for k, v in det.items()})
    else:
        det_obj, det_place = stored["pick"]["det"], stored["place"]["det"]

    planner = ShiftBackPlanner(P, grasp_uv=A["tape_uv_mem"], grasp_z=A["z_grasp"],
                               release_uv=A["plate_uv_mem"], release_z=A["z_release"], tau=a.tau, min_score=0.1)
    plan = planner.plan(det_obj, det_place, stored["pick"]["phrase"], stored["place"]["phrase"])

    print(f"scene: {a.scene}")
    for side in ("pick", "place"):
        s = getattr(plan, side)
        print(f"  {side:5s} detected at {np.round(s.det_uv or [np.nan, np.nan]).tolist()} px, taught at "
              f"{np.round(s.anchor_uv).tolist()} px, height {s.z:.3f} m")
        print(f"        displacement {np.round(s.delta_raw[:2], 3).tolist()} m (|Delta| = {100 * s.norm:.1f} cm) -> "
              f"{'ACT' if s.active else 'no action'} ({s.reason}); image offset {np.round(s.duv).astype(int).tolist()} px")
    print(f"  carry at gripper closure: {np.round(plan.carry[:2], 3).tolist()} m")

    rec = stored["recorded"]
    ok = (np.allclose(plan.pick.delta[:2], rec["pick_delta_m"], atol=1e-6) and
          np.allclose(np.round(plan.pick.duv), rec["pick_du"], atol=1) and
          np.allclose(plan.place.delta[:2], rec["place_delta_m"], atol=1e-6))
    print("  matches the values the robot used in this trial:", ok)

    # --- figure: demonstrated frame | live frame | what the policy sees ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    shifted = planner.shift_frame(live, "pick")
    H, W = live.shape[:2]
    du = plan.pick.duv
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    titles = ["demonstration (episode 0)", f"live first frame ({a.scene.replace('_', ' ')})",
              "what the policy sees: image moved back"]
    for ax, img, t in zip(axs, [demo, live, shifted], titles):
        ax.imshow(img); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    axs[0].add_patch(Circle(A["tape_uv_mem"], 10, fill=False, ec="tab:blue", lw=1.5))
    axs[0].text(A["tape_uv_mem"][0] + 14, A["tape_uv_mem"][1], "taught grasp pixel", color="tab:blue", fontsize=8)
    if plan.pick.det_uv:
        axs[1].add_patch(Circle(plan.pick.det_uv, 10, fill=False, ec="tab:red", lw=1.5))
        axs[1].add_patch(Circle(A["tape_uv_mem"], 10, fill=False, ec="tab:blue", lw=1.5, ls="--"))
        axs[1].text(plan.pick.det_uv[0] + 14, plan.pick.det_uv[1], f"detected, |Delta| = {100 * plan.pick.norm:.0f} cm",
                    color="tab:red", fontsize=8)
    if plan.pick.active:
        x0, y0, x1, y1 = kept_box(W, H, du[0], du[1])
        axs[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1, ls="--"))
        axs[2].add_patch(Rectangle((x0 + du[0], y0 + du[1]), x1 - x0, y1 - y0, fill=False, ec="white", lw=1, ls="--"))
        axs[2].add_patch(Circle(A["tape_uv_mem"], 10, fill=False, ec="tab:blue", lw=1.5))
        axs[2].text(A["tape_uv_mem"][0] + 14, A["tape_uv_mem"][1], "tape back at the taught pixel", color="tab:blue", fontsize=8)
        axs[2].text(W / 2, H - 8, "empty strip filled by repeating the edge pixels", color="white", fontsize=7,
                    ha="center", va="bottom", bbox=dict(fc="black", alpha=0.5, lw=0))
    else:
        axs[2].text(W / 2, H / 2, "inside the threshold: no shift", color="white", fontsize=10, ha="center",
                    bbox=dict(fc="black", alpha=0.5, lw=0))
    fig.suptitle(f"ShiftBack on the robot frames: arm start moved by {np.round(plan.pick.delta[:2], 3).tolist()} m, "
                 f"image moved by {np.round(du).astype(int).tolist()} px", fontsize=10)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"demo0_{a.scene}.png")
    fig.savefig(out, dpi=130)
    print("  figure:", out)


if __name__ == "__main__":
    main()
