#!/usr/bin/env python3
"""Anchors for a real task from the demonstrations alone: gripper events + first-frame detections.

    python real/build_anchors.py --demos data_root/T1 --calib calib_work/P.npy \
        --object-phrase "roll of tape" "a green roll of tape on the table" --place-phrase "black round tray" \
        --out anchors.json

Reads every demonstration .npz (base_rgb, ee_pose, gripper_cmd), finds the closing and opening events
with the three-step rule, averages the gripper position at these events (grasp / release points and
heights), runs OWLv2 zero-shot on each first frame for the object and placement phrases and stores the
mean detected pixels: the demonstrated pixels the live detection is compared with. Nothing about the
object is measured by hand.

Without a robot, the same script runs on the shipped sample (``--sample``): the 40 first frames and
the gripper logs in ``data/real_sample``; pass ``--detections`` to reuse stored detections instead of
running OWLv2.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shiftback import anchors_from_demos, lift, load_projection, project, start_pose_from_demos  # noqa: E402


def load_demos(a):
    """-> list of dict(ee, gripper_cmd, first_frame) from .npz demos or from the shipped sample."""
    from PIL import Image
    demos = []
    if a.sample:
        S = os.path.join(ROOT, "data", "real_sample")
        for e in json.load(open(os.path.join(S, "demo_gripper_log.json"))):
            img = np.asarray(Image.open(os.path.join(S, "demo_first_frames", e["episode"] + ".jpg")).convert("RGB"))
            demos.append(dict(name=e["episode"], ee=np.asarray(e["ee"]), gripper_cmd=np.asarray(e["gripper_cmd"]), first_frame=img))
    else:
        for f in sorted(x for x in os.listdir(a.demos) if x.endswith(".npz")):
            d = np.load(os.path.join(a.demos, f))
            demos.append(dict(name=f[:-4], ee=d["ee_pose"][:, :3, 3], gripper_cmd=d["gripper_cmd"], first_frame=d["base_rgb"][0]))
    return demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=None, help="directory of demonstration .npz files")
    ap.add_argument("--sample", action="store_true", help="use the shipped data/real_sample instead of --demos")
    ap.add_argument("--calib", default=None, help="P.npy (default: the sample's)")
    ap.add_argument("--object-phrase", nargs="+", default=["roll of tape", "a green roll of tape on the table"])
    ap.add_argument("--place-phrase", nargs="+", default=["black round tray"])
    ap.add_argument("--detections", default=None, help="json {episode: {phrase: {uv, score}}} to reuse instead of running OWLv2")
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.sample and not a.demos:
        ap.error("--demos or --sample")
    P = load_projection(a.calib or os.path.join(ROOT, "data", "real_sample", "calib", "P.npy"))
    demos = load_demos(a)

    # 1. gripper events -> grasp / release points and heights
    an = anchors_from_demos(demos, P)
    home = start_pose_from_demos(demos)
    print(f"{an['n_kept']}/{an['n_demos']} demonstrations with {an['K']} gripper cycle(s); grasp at "
          f"{np.round(an['g'], 3).tolist()} (std {np.round(100 * np.asarray(an['g_std']), 1).tolist()} cm), "
          f"release at {np.round(an['r'], 3).tolist()}")

    # 2. first-frame detections -> demonstrated pixels of the object and the placement surface
    phrases = list(a.object_phrase) + [p for p in a.place_phrase if p not in a.object_phrase]
    if a.detections:
        dets = json.load(open(a.detections))
    else:
        from shiftback.localiser import OwlLocaliser
        loc = OwlLocaliser()
        dets = {}
        for d in demos:
            dets[d["name"]] = loc.boxes(d["first_frame"], phrases)
            print(d["name"], {k: None if v is None else [round(x) for x in v["uv"]] for k, v in dets[d["name"]].items()}, flush=True)
        json.dump(dets, open(os.path.splitext(a.out)[0] + "_first_frame_detections.json", "w"), indent=1)

    def best(v, phs):
        c = [(v[p]["score"], v[p]["uv"]) for p in phs if v.get(p) and v[p]["score"] >= a.min_score]
        return max(c)[1] if c else None
    obj = np.array([u for u in (best(v, a.object_phrase) for v in dets.values()) if u is not None])
    plc = np.array([u for u in (best(v, a.place_phrase) for v in dets.values()) if u is not None])
    out = dict(tape_phrases=a.object_phrase, plate_phrases=a.place_phrase,
               tape_uv_mem=obj.mean(0).tolist(), tape_uv_std=obj.std(0).tolist(), n_tape=int(len(obj)),
               plate_uv_mem=plc.mean(0).tolist(), plate_uv_std=plc.std(0).tolist(), n_plate=int(len(plc)),
               z_grasp=float(an["g"][2]), z_release=float(an["r"][2]),
               grasp_tcp_mem=an["g"], grasp_tcp_std_cm=(100 * np.asarray(an["g_std"])).tolist(),
               release_tcp_mem=an["r"], home_tcp=home.tolist(),
               tape_lift_mem=lift(P, obj.mean(0), an["g"][2]).tolist(), plate_lift_mem=lift(P, plc.mean(0), an["r"][2]).tolist(),
               grasp_tcp_proj=project(P, an["g"]).tolist(), release_tcp_proj=project(P, an["r"]).tolist(),
               calib=a.calib)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"object pixel {np.round(out['tape_uv_mem'], 1).tolist()} (std {np.round(out['tape_uv_std'], 1).tolist()}, n={len(obj)}), "
          f"placement pixel {np.round(out['plate_uv_mem'], 1).tolist()} (n={len(plc)}); lifted object point vs logged grasp: "
          f"{np.round(100 * (np.asarray(out['tape_lift_mem'][:2]) - np.asarray(an['g'][:2])), 1).tolist()} cm -> {a.out}")


if __name__ == "__main__":
    main()
