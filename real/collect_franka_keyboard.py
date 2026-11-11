#!/usr/bin/env python3
"""Keyboard teleoperation + recording for a single Franka arm with a wrist camera and a fixed third-person camera.

Records, at a fixed rate, everything ShiftBack needs later: both images, joint state, the robot's own end-effector
pose, the gripper width, and the COMMANDED end-effector delta action. See README_CN.md for the protocol.

    python collect_franka_keyboard.py --root data_root --task T1 --instruction "put the green tape in the tray"
    python collect_franka_keyboard.py --root data_root --calib          # calibration frames (press c at each TCP spot)
    python collect_franka_keyboard.py --dry-run                         # no robot: test cameras + keys + saving

Robot backend: `franky` by default (pip install franky-panda). If your machine uses another stack, implement the
four methods of RobotBackend (read_state, move_delta, set_gripper, go_home) for it and nothing else changes.
"""
import argparse, json, os, sys, threading, time
from pathlib import Path
import numpy as np

RATE_HZ = 15.0
STEP_M, STEP_RAD = 0.01, np.deg2rad(2.0)          # default per-cycle step while a key is held
HOME_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]   # <-- edit: your fixed start posture (rad)
IMG_W, IMG_H = 640, 480
GRIP_OPEN_W, GRIP_CLOSE_W = 0.08, 0.0


# --------------------------------------------------------------------------- cameras
class Camera:
    """OpenCV device or RealSense (serial). `source` = int / /dev/videoN  or  'rs:<serial>'."""
    def __init__(self, source, w=IMG_W, h=IMG_H):
        self.rs = None
        if isinstance(source, str) and source.startswith("rs:"):
            import pyrealsense2 as rs
            self.rs = rs.pipeline(); cfg = rs.config(); cfg.enable_device(source[3:])
            cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30); self.rs.start(cfg)
        else:
            import cv2
            self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            assert self.cap.isOpened(), f"camera {source} not opened"

    def read(self):
        if self.rs is not None:
            f = self.rs.wait_for_frames().get_color_frame(); return np.asanyarray(f.get_data()).copy()
        import cv2
        ok, bgr = self.cap.read(); assert ok, "camera read failed"
        return cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)


class MockCamera:
    def __init__(self, seed): self.i = seed
    def read(self):
        self.i += 1; im = np.zeros((IMG_H, IMG_W, 3), np.uint8); im[:, (self.i * 5) % IMG_W] = 255; return im


# --------------------------------------------------------------------------- robot
class RobotBackend:
    """Implement these four for your stack. Poses are 4x4 numpy in the robot base frame."""
    def read_state(self) -> dict:   # {"q": (7,), "dq": (7,), "ee_pose": (4,4), "gripper_width": float}
        raise NotImplementedError
    def move_delta(self, dpos_m, drot_rad):  # relative Cartesian step, non-blocking or short-blocking
        raise NotImplementedError
    def set_gripper(self, close: bool):
        raise NotImplementedError
    def go_home(self):
        raise NotImplementedError


class FrankyBackend(RobotBackend):
    def __init__(self, ip):
        from franky import Robot, Gripper, Affine, CartesianMotion, ReferenceType, JointMotion
        self.R, self.G = Robot(ip), Gripper(ip)
        self.R.relative_dynamics_factor = 0.15
        self._Affine, self._CM, self._RT, self._JM = Affine, CartesianMotion, ReferenceType, JointMotion
        self.closed = False

    def read_state(self):
        st = self.R.state
        return {"q": np.asarray(st.q, np.float64), "dq": np.asarray(st.dq, np.float64),
                "ee_pose": np.asarray(st.O_T_EE, np.float64).reshape(4, 4).T,  # libfranka is column-major
                "gripper_width": float(self.G.width)}

    def move_delta(self, dpos_m, drot_rad):
        from scipy.spatial.transform import Rotation as Rot
        quat = Rot.from_euler("xyz", drot_rad).as_quat()
        self.R.move(self._CM(self._Affine(list(dpos_m), list(quat)), self._RT.Relative), asynchronous=True)

    def set_gripper(self, close):
        if close != self.closed:
            self.closed = close
            (self.G.grasp_async(GRIP_CLOSE_W, 0.05, 20.0) if close else self.G.open_async(0.05))

    def go_home(self):
        self.R.move(self._JM(HOME_Q)); self.set_gripper(False); time.sleep(0.5)


class MockBackend(RobotBackend):
    def __init__(self):
        self.q = np.array(HOME_Q, np.float64); self.pos = np.array([0.4, 0.0, 0.3]); self.rpy = np.array([np.pi, 0, 0]); self.w = GRIP_OPEN_W
    def read_state(self):
        from scipy.spatial.transform import Rotation as Rot
        T = np.eye(4); T[:3, :3] = Rot.from_euler("xyz", self.rpy).as_matrix(); T[:3, 3] = self.pos
        return {"q": self.q.copy(), "dq": np.zeros(7), "ee_pose": T, "gripper_width": self.w}
    def move_delta(self, dpos_m, drot_rad): self.pos += np.asarray(dpos_m); self.rpy += np.asarray(drot_rad)
    def set_gripper(self, close): self.w = GRIP_CLOSE_W if close else GRIP_OPEN_W
    def go_home(self): self.pos[:] = [0.4, 0.0, 0.3]; self.rpy[:] = [np.pi, 0, 0]; self.w = GRIP_OPEN_W


# --------------------------------------------------------------------------- keyboard
class Keys:
    """Current key state (held keys) + one-shot events, via pynput."""
    MOVE = {"w": (0, +1), "s": (0, -1), "a": (1, +1), "d": (1, -1), "r": (2, +1), "f": (2, -1), "q": (5, +1), "e": (5, -1)}

    def __init__(self):
        from pynput import keyboard
        self.held, self.events, self.lock = set(), [], threading.Lock()
        self.kb = keyboard
        keyboard.Listener(on_press=self._on, on_release=self._off).start()

    def _name(self, k):
        try: return k.char
        except AttributeError: return {self.kb.Key.space: "space", self.kb.Key.enter: "enter", self.kb.Key.backspace: "backspace", self.kb.Key.esc: "esc"}.get(k)

    def _on(self, k):
        n = self._name(k)
        if n is None: return
        with self.lock:
            if n in self.MOVE: self.held.add(n)
            else: self.events.append(n)

    def _off(self, k):
        n = self._name(k)
        with self.lock: self.held.discard(n)

    def pop_events(self):
        with self.lock: ev, self.events = self.events, []; return ev

    def delta(self, step_m, step_rad):
        d = np.zeros(6)
        with self.lock:
            for n in self.held:
                ax, sgn = self.MOVE[n]; d[ax] += sgn * (step_rad if ax >= 3 else step_m)
        return d


# --------------------------------------------------------------------------- recording
class Episode:
    def __init__(self): self.buf = {k: [] for k in ("t", "base_rgb", "wrist_rgb", "q", "dq", "ee_pose", "gripper_width", "gripper_cmd", "action")}
    def add(self, t, base, wrist, st, grip_cmd, action):
        b = self.buf; b["t"].append(t); b["base_rgb"].append(base); b["wrist_rgb"].append(wrist); b["q"].append(st["q"]); b["dq"].append(st["dq"])
        b["ee_pose"].append(st["ee_pose"]); b["gripper_width"].append(st["gripper_width"]); b["gripper_cmd"].append(grip_cmd); b["action"].append(action)
    def __len__(self): return len(self.buf["t"])
    def save(self, path: Path, **meta):
        arr = {k: np.asarray(v) for k, v in self.buf.items()}
        arr["base_rgb"] = arr["base_rgb"].astype(np.uint8); arr["wrist_rgb"] = arr["wrist_rgb"].astype(np.uint8)
        np.savez_compressed(path, **arr, **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in meta.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data_root"); ap.add_argument("--task", default="T1")
    ap.add_argument("--instruction", default="put the green tape in the tray")
    ap.add_argument("--layout", default="{}", help='json: object -> [grid_x, grid_y], e.g. {"tape":[3,4],"tray":[8,2]}')
    ap.add_argument("--robot-ip", default="172.16.0.2"); ap.add_argument("--base-cam", default="0"); ap.add_argument("--wrist-cam", default="2")
    ap.add_argument("--calib", action="store_true", help="calibration mode: press c to save a base frame + EE pose")
    ap.add_argument("--check-camera", action="store_true", help="compare each episode's first base frame with the very first one and warn if it moved")
    ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    cams = (MockCamera(0), MockCamera(100)) if a.dry_run else (Camera(a.base_cam), Camera(a.wrist_cam))
    robot = MockBackend() if a.dry_run else FrankyBackend(a.robot_ip)
    keys = Keys(); step_m, step_rad = STEP_M, STEP_RAD
    meta_p = root / "meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {"rate_hz": RATE_HZ, "img_wh": [IMG_W, IMG_H], "home_q": HOME_Q, "tasks": {}, "notes": "grid paper 5 cm; grid origin = taped corner"}
    meta["tasks"][a.task] = a.instruction; json.dump(meta, open(meta_p, "w"), indent=1)

    if a.calib:
        cdir = root / f"calib_{time.strftime('%Y%m%d_%H%M%S')}"; cdir.mkdir(); poses = []; i = 0
        print(f"[calib] move the TCP to 25-30 spread-out spots; press c at each; q to quit  -> {cdir}")
        while True:
            t0 = time.time(); d = keys.delta(step_m, step_rad)
            if np.any(d): robot.move_delta(d[:3], d[3:])
            for ev in keys.pop_events():
                if ev == "c":
                    import cv2; base = cams[0].read(); st = robot.read_state()
                    cv2.imwrite(str(cdir / f"base_{i:03d}.png"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
                    poses.append({"i": i, "ee_pose": st["ee_pose"].tolist(), "q": st["q"].tolist()}); i += 1
                    json.dump(poses, open(cdir / "poses.json", "w")); print(f"[calib] saved {i}")
                elif ev in ("esc", "q"):
                    print(f"[calib] done, {i} frames"); return
            time.sleep(max(0, 1 / RATE_HZ - (time.time() - t0)))

    tdir = root / a.task; tdir.mkdir(exist_ok=True)
    n_ep = len(list(tdir.glob("episode_*.npz"))); ep = None; grip_closed = False; ref_frame = None
    print(f"[rec] task {a.task}: {n_ep} episodes so far. enter=start  .=save  backspace=discard  esc=quit")
    while True:
        t0 = time.time()
        for ev in keys.pop_events():
            if ev == "enter":
                robot.go_home(); grip_closed = False; ep = Episode(); print(f"[rec] episode {n_ep:03d} started (layout {a.layout})")
            elif ev == "space":
                grip_closed = not grip_closed; robot.set_gripper(grip_closed)
            elif ev == "[": step_m, step_rad = step_m / 2, step_rad / 2; print(f"[rec] step {step_m*100:.2f} cm")
            elif ev == "]": step_m, step_rad = step_m * 2, step_rad * 2; print(f"[rec] step {step_m*100:.2f} cm")
            elif ev == "backspace" and ep is not None:
                ep = None; print("[rec] discarded")
            elif ev == "." and ep is not None and len(ep) > 10:
                p = tdir / f"episode_{n_ep:03d}.npz"
                ep.save(p, instruction=a.instruction, task=a.task, layout=json.loads(a.layout), success=1, home_q=np.array(HOME_Q))
                print(f"[rec] saved {p} ({len(ep)} frames, {len(ep)/RATE_HZ:.1f} s)"); n_ep += 1; ep = None
            elif ev == "esc":
                print("[rec] bye"); return
        base, wrist = cams[0].read(), cams[1].read(); st = robot.read_state()
        d = keys.delta(step_m, step_rad); grip_cmd = 1.0 if grip_closed else -1.0
        if np.any(d): robot.move_delta(d[:3], d[3:])
        if ep is not None:
            if a.check_camera and len(ep) == 0:
                if ref_frame is None: ref_frame = base.astype(np.float32)
                else:
                    diff = float(np.abs(base.astype(np.float32) - ref_frame).mean())
                    if diff > 25: print(f"[WARN] base camera view changed a lot vs the first episode (mean abs diff {diff:.1f}) -- was the camera touched?")
            ep.add(time.time(), base, wrist, st, grip_cmd, np.concatenate([d, [grip_cmd]]))
        time.sleep(max(0, 1 / RATE_HZ - (time.time() - t0)))


if __name__ == "__main__":
    main()
