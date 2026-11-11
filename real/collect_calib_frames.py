#!/usr/bin/env python3
"""Calibration frames WITHOUT clicking: hold the tape in the gripper and visit a grid of positions; at each one save the
base-camera frame and the EE pose.  The GPU side detects the held tape (OWLv2) and fits the camera from (EE, pixel) pairs.

    # 1. put the tape at its home cell (poses.json grasp_current), robot at home, run:
    python3 collect_calib_frames.py --out calib_frames            # grasps the tape itself, visits ~30 poses, puts it back

Grid (base frame, m): x 0.36..0.56 (5 values), y -0.45..-0.05 (3 values), z 0.09 / 0.16 / 0.24 (rotated through) -> 30 poses,
all inside the collection workspace box. Edit GRID below if the table / camera support is in the way.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from auto_collect import Abort, AutoCollector  # noqa: E402

GRID = dict(x=[0.36, 0.41, 0.46, 0.51, 0.56], y=[-0.45, -0.25, -0.05], z=[0.09, 0.16, 0.24])


class Calib(AutoCollector):
    def run_calib(self, out):
        import cv2
        os.makedirs(out, exist_ok=True); a = self.a
        self.gripper(False)
        st = self.robot.get_state(); p = st["ee_pos"].copy()
        if p[2] < self.transit_z:
            self.move_to(np.array([p[0], p[1], self.transit_z]), speed=a.speed_z)
        self.move_to(self.home, self.home_q)
        src = self.grasp_cur.copy()
        print(f"[calib] grasping the tape at {np.round(src, 3)}")
        self.move_to(np.array([src[0], src[1], self.transit_z]), self.grasp_q); self.move_to(src, self.grasp_q, speed=a.speed_z)
        self.gripper(True); self.move_to(np.array([src[0], src[1], self.transit_z]), speed=a.speed_z)
        poses = []; k = 0
        for xi, x in enumerate(GRID["x"]):
            ys = GRID["y"] if xi % 2 == 0 else GRID["y"][::-1]
            for yi, y in enumerate(ys):
                z = GRID["z"][(xi + yi) % len(GRID["z"])]
                tgt = np.clip(np.array([x, y, z]), self.wmin + 0.01, self.wmax - 0.01)
                self.move_to(np.array([tgt[0], tgt[1], max(tgt[2], self.transit_z)]), self.grasp_q)
                self.move_to(tgt, self.grasp_q, speed=a.speed_z); self.hold(0.6)
                st = self.robot.get_state(); fr = self.cams.get_all()
                b = fr["third_person"][0]
                if b is None: raise Abort("no base frame")
                cv2.imwrite(os.path.join(out, f"base_{k:03d}.png"), b)
                poses.append(dict(i=k, ee_pos=st["ee_pos"].tolist(), ee_quat=st["ee_quat"].tolist(), q=st["q"].tolist(), target=tgt.tolist(), t=time.time()))
                print(f"[calib] {k:2d} ee {np.round(st['ee_pos'], 3)}"); k += 1
                self.move_to(np.array([tgt[0], tgt[1], max(tgt[2], self.transit_z)]), speed=a.speed_z)
        json.dump(poses, open(os.path.join(out, "poses.json"), "w"), indent=1)
        print(f"[calib] putting the tape back at {np.round(src, 3)}")
        self.move_to(np.array([src[0], src[1], self.transit_z]), self.grasp_q); self.move_to(src, self.grasp_q, speed=a.speed_z)
        self.gripper(False); self.move_to(np.array([src[0], src[1], self.transit_z]), speed=a.speed_z); self.move_to(self.home, self.home_q)
        print(f"[calib] done: {k} frames in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml")); ap.add_argument("--poses", default=os.path.join(HERE, "poses.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "calib_frames")); ap.add_argument("--root", default=os.path.join(HERE, "data_root")); ap.add_argument("--task", default="T1")
    ap.add_argument("--instruction", default="put the tape on the plate"); ap.add_argument("--lift", type=float, default=0.05)
    ap.add_argument("--speed", type=float, default=0.05); ap.add_argument("--speed-z", type=float, default=0.03); ap.add_argument("--max-force", type=float, default=25.0)
    ap.add_argument("--min-grasp-width", type=float, default=0.0015); ap.add_argument("--settle-timeout", type=float, default=3.0); ap.add_argument("--push-max", type=float, default=0.012)
    ap.add_argument("--grip-wait", type=float, default=0.8); ap.add_argument("--reset-grasp-dz", type=float, default=0.02); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-frame-age", type=float, default=0.3); ap.add_argument("--preview", action="store_true"); ap.add_argument("--backend", default="cart", choices=["cart", "ros2"])
    ap.add_argument("--no-go-home-first", dest="go_home_first", action="store_false"); ap.add_argument("--joint-home", action="store_true"); ap.add_argument("--max-q-dev", type=float, default=0.35)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--mock-robot", action="store_true"); ap.add_argument("--jitter", type=float, default=0.0); ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--max-cam-retries", type=int, default=8); ap.add_argument("--reset-last", action="store_true"); ap.add_argument("--tape-on-plate", action="store_true"); ap.add_argument("--test-home", action="store_true")
    a = ap.parse_args()
    c = Calib(a, yaml.safe_load(open(a.config)))
    try:
        c.run_calib(a.out)
    except Abort as e:
        print(f"[calib] ABORT: {e}")
    except KeyboardInterrupt:
        print("[calib] Ctrl+C")
    finally:
        c.close()


if __name__ == "__main__":
    main()
