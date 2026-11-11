#!/usr/bin/env python3
"""Run ONE policy episode on the FR3 (baseline or ShiftBack), record everything, ask the operator for the outcome.

Put this file next to auto_collect.py / robot.py / cameras.py (it reuses AutoCollector for the robot, gripper, cameras,
move_to / gripper / hold / pick_place primitives and the taught poses).  The policy runs on the GPU machine
(openpi serve_policy, websocket); detection runs either on the GPU machine (shiftback_service.py) or locally.

    # pair: same tape placement, baseline first, then ShiftBack
    python3 run_episode.py --mode baseline  --tag tape_x+20 --tape-cell 0.657,-0.200 --policy-host <gpu-ip>
    python3 run_episode.py --mode shiftback --tag tape_x+20 --tape-cell 0.657,-0.200 --policy-host <gpu-ip> --detector http://<gpu-ip>:8010

    # let the robot place the tape at a test cell first (from where it is now: poses.json grasp_current, or --tape-at x,y)
    python3 run_episode.py --set-tape 0.657,-0.200 --only-set-tape
    python3 run_episode.py --retrieve-from-plate --set-tape 0.457,-0.200 --only-set-tape     # after a successful trial

Modes: baseline (nothing), shiftback (start pose + image, both sides), armonly (start pose + carry only), imgonly (image only).
Outcome prompt at the end: success y/n + failure code  A = reached for the memorised spot (grasped air)  B = reached the tape
but failed to grasp  C = grasped, placed wrongly  D = other.  Everything is saved under --results/<tag>/.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from auto_collect import RATE_HZ, Abort, AutoCollector  # noqa: E402
from robot import quat_from_rotvec, quat_mul, quat_to_rotmat  # noqa: E402
from shiftback_real import HttpDetector, ShiftBack, shift_image  # noqa: E402


def rotvec_from_quat(q):
    w, x, y, z = q / np.linalg.norm(q)
    ang = 2 * np.arctan2(np.linalg.norm([x, y, z]), w)
    if ang > np.pi: ang -= 2 * np.pi
    n = np.linalg.norm([x, y, z])
    return np.zeros(3) if n < 1e-9 else np.array([x, y, z]) / n * ang


def quat_angle_deg(qa, qb):
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


class VideoRecorder:
    """Background thread: writes both camera streams to MP4 (640x480, 15 fps) for the whole episode, servo phases included."""
    def __init__(self, cams, stem, fps=15):
        import threading, cv2
        self.cams, self.stem, self.fps = cams, stem, fps; self.stop_flag = False; self.n = {}
        self.w = {r: cv2.VideoWriter(f"{stem}_{n}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480)) for r, n in (("third_person", "base"), ("wrist", "wrist"))}
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start()

    def _loop(self):
        import cv2
        last = {}; t_next = time.time()
        while not self.stop_flag:
            fr = self.cams.get_all()
            for r, wtr in self.w.items():
                img, _, t = fr.get(r, (None, None, 0))
                if img is None:
                    continue
                if img.shape[:2] != (480, 640):
                    img = cv2.resize(img, (640, 480))
                wtr.write(img); self.n[r] = self.n.get(r, 0) + 1; last[r] = t     # fixed-rate write (repeats a frame if the camera is slower)
            t_next += 1 / self.fps; time.sleep(max(0, t_next - time.time()))

    def stop(self):
        self.stop_flag = True; self.t.join(timeout=2)
        for wtr in self.w.values():
            wtr.release()
        print(f"[video] saved {self.stem}_base.mp4 / _wrist.mp4 ({self.n})")


class PolicyRunner(AutoCollector):
    def __init__(self, a, cfg):
        super().__init__(a, cfg)
        self.policy = None
        if not a.only_set_tape:
            self._connect_policy()
            # warm-up: the server compiles on its first request (30-90 s) and drops the connection meanwhile -> retry
            dummy = {"observation/image": self._net_img(np.zeros((480, 640, 3), np.uint8)), "observation/wrist_image": self._net_img(np.zeros((480, 640, 3), np.uint8)),
                     "observation/state": np.zeros(8, np.float32), "prompt": a.instruction}
            t0 = time.time(); self.infer(dummy); print(f"[run] policy warm-up done in {time.time()-t0:.0f}s")
        self.sb = None
        if a.mode != "baseline" or a.diag_detect:
            if a.detector.startswith("http"):
                self.det = HttpDetector(a.detector)
            else:
                from shiftback_real import OwlDetector
                self.det = OwlDetector()
            self.sb = ShiftBack(a.calib, a.anchors, dead_band_m=a.dead_band_m)
        self.replay = None
        if a.replay_npz:
            dz = np.load(a.replay_npz, allow_pickle=True); self.replay = {k: dz[k] for k in ("base_rgb", "wrist_rgb", "action")}
            print(f"[run] REPLAY mode: frames from {a.replay_npz} ({len(self.replay['action'])} frames), robot moves are dry")

    def _net_img(self, img):
        """Letterbox to --send-size before sending (the server applies the same resize_with_pad -> identical input,
        max 1 grey level difference measured; 4x less data on slow links: 700 ms -> 150 ms over a Tailscale relay)."""
        if self.a.send_size <= 0:
            return img
        from openpi_client import image_tools
        return image_tools.resize_with_pad(img, self.a.send_size, self.a.send_size)

    def _connect_policy(self):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.policy = WebsocketClientPolicy(host=self.a.policy_host, port=self.a.policy_port)
        print(f"[run] policy server {self.a.policy_host}:{self.a.policy_port} metadata: {self.policy.get_server_metadata()}")

    def infer(self, obs, retries=8):
        for i in range(retries):
            try:
                return self.policy.infer(obs)
            except Exception as e:   # connection dropped (server busy compiling / network hiccup): reconnect and retry
                print(f"[run] policy call failed ({type(e).__name__}: {str(e)[:80]}), retry {i+1}/{retries} in 10 s")
                time.sleep(10)
                try:
                    self._connect_policy()
                except Exception as e2:
                    print(f"[run] reconnect failed: {e2}")
        raise Abort("policy server unreachable")

    # ------------------------------------------------------------------ frames / state
    def frames(self, i):
        if self.replay is not None:
            j = min(i, len(self.replay["action"]) - 1)
            return self.replay["base_rgb"][j], self.replay["wrist_rgb"][j]
        import cv2
        fr = self.cams.get_all()
        b, w = fr["third_person"][0], fr["wrist"][0]
        if b is None or w is None or time.time() - fr["third_person"][2] > self.a.max_frame_age:
            raise Abort("camera frame missing / stale")
        return cv2.cvtColor(b, cv2.COLOR_BGR2RGB), cv2.cvtColor(w, cv2.COLOR_BGR2RGB)

    def state_vec(self, st):
        return np.concatenate([st["ee_pos"], rotvec_from_quat(st["ee_quat"]), [self.grip.width if not np.isnan(self.grip.width) else 0.0, self.grip_cmd]]).astype(np.float32)

    # ------------------------------------------------------------------ ShiftBack helpers
    def servo_start(self, delta_xy):
        tgt = self.home.copy(); tgt[:2] += delta_xy
        print(f"[sb] start pose servo: home {np.round(self.home, 3)} -> {np.round(tgt, 3)}")
        self.move_to(tgt, self.home_q)

    def carry_servo(self, extra_xy):
        st = self.robot.get_state(); p = st["ee_pos"].copy()
        z_up = max(p[2] + self.a.carry_lift, self.transit_z)          # at least the demo transit height (clears objects)
        z_dn = p[2] + self.a.carry_lower_margin                        # NOT back to contact: the held tape touching the
        print(f"[sb] carry servo: lift to z={z_up:.3f}, translate {np.round(extra_xy, 3)}, lower to z={z_dn:.3f} "  # table under
              f"({self.a.carry_lower_margin*100:.1f} cm above the grasp height)")                                   # position control
        self.move_to(np.array([p[0], p[1], z_up]), speed=self.a.speed_z)                                           # -> 32 N reflex
        self.move_to(np.array([p[0] + extra_xy[0], p[1] + extra_xy[1], z_up]))
        self.move_to(np.array([p[0] + extra_xy[0], p[1] + extra_xy[1], z_dn]), speed=self.a.speed_z)

    # ------------------------------------------------------------------ the episode
    def run_episode(self):
        a = self.a
        # 1. start from home, gripper open
        self.gripper(False)
        st = self.robot.get_state(); p = st["ee_pos"].copy()
        if p[2] < self.transit_z:
            self.move_to(np.array([p[0], p[1], self.transit_z]), speed=a.speed_z)
        self.move_to(self.home, self.home_q); self.hold(0.5)
        base0, wrist0 = self.frames(0)
        if a.replay_first_frame:   # offline test: pretend the first frame is this image (e.g. tape on the plate)
            from PIL import Image
            base0 = np.asarray(Image.open(a.replay_first_frame).convert("RGB")); print(f"[run] first frame overridden by {a.replay_first_frame}")
        # 2. ShiftBack on the first frame
        sbr = None; du = np.zeros(2, int); carry_extra = np.zeros(2); du_place = np.zeros(2, int)
        if self.sb is not None:
            t0 = time.time(); det = self.det.detect(base0, self.sb.phrases); sbr = self.sb.compute(det); sbr["detect_secs"] = time.time() - t0
            for side in ("pick", "place"):
                r = sbr[side]; print(f"[sb] {side:5s}: det {r['det'] and np.round(r['det']['uv'], 1)} score {r['det'] and round(r['det']['score'], 2)} "
                                     f"delta_raw {np.round(r['delta_raw_m'], 3)} m -> {r['reason']}; du {r['du']} px")
            print(f"[sb] carry_extra {np.round(sbr['carry_extra_m'], 3)} m ({sbr['detect_secs']:.1f}s)")
            if a.mode in ("shiftback", "armonly"):
                if sbr["pick"]["active"]:
                    self.servo_start(np.array(sbr["pick"]["delta_m"]))
                carry_extra = np.array(sbr["carry_extra_m"]) if sbr["carry_active"] else np.zeros(2)
            if a.mode in ("shiftback", "imgonly"):
                du = np.array(sbr["pick"]["du"]) if sbr["pick"]["active"] else np.zeros(2, int)
                du_place = np.array(sbr["place"]["du"]) if sbr["place"]["active"] else np.zeros(2, int)
        # 3. policy loop
        H = a.replan; rec = {k: [] for k in ("t", "base_small", "shifted_small", "wrist_small", "state", "action", "chunk_id", "du", "infer_s", "ee_pos", "grip_cmd", "width")}
        events = []; chunk = None; k = H; chunk_id = -1; closed = False; opened = False; n_close = 0; step_open = None
        self.partial = (base0, wrist0, sbr, rec, events)   # kept so an Abort (force / reflex) is still recorded
        t_next = time.time(); import cv2
        print(f"[run] policy loop (max {a.max_steps} steps, replan {H}, step clamp {a.step_max*1000:.0f} mm / {a.rot_max_deg:.0f} deg)")
        for step in range(a.max_steps):
            base, wrist = self.frames(step)
            shifted = shift_image(base, du[0], du[1]) if (du[0] or du[1]) else base
            st = self.robot.get_state(); state = self.state_vec(st)
            infer_s = 0.0
            if k >= H or chunk is None:
                obs = {"observation/image": self._net_img(shifted), "observation/wrist_image": self._net_img(wrist), "observation/state": state, "prompt": a.instruction}
                t0 = time.time(); chunk = np.asarray(self.infer(obs)["actions"], dtype=np.float64); infer_s = time.time() - t0
                k = 0; chunk_id += 1
                if chunk_id % 10 == 0:
                    print(f"  step {step:4d} chunk {chunk_id:3d} infer {infer_s*1000:.0f} ms | a0 {np.round(chunk[0, :3]*1000, 1)} mm grip {chunk[0, 6]:+.2f} | ee {np.round(st['ee_pos'], 3)}")
            act = chunk[k].copy(); gvote = float(chunk[k:k + a.grip_vote, 6].mean()); k += 1
            # gripper: vote over the next few predicted steps (a single flow-matching sample can flip sign at transitions),
            # hysteresis, one close + one open per episode
            if gvote > a.grip_thr and not closed and n_close < a.max_gripper_cycles:
                print(f"[run] step {step}: CLOSE"); events.append(dict(step=step, event="close", ee=st["ee_pos"].tolist()))
                try:
                    self.gripper(True)
                except Abort as e:
                    print(f"[run] grasp check: {e} (continuing)"); events.append(dict(step=step, event="grasp_empty", msg=str(e)))
                closed = True; n_close += 1
                if a.mode in ("shiftback", "armonly") and np.linalg.norm(carry_extra) > 1e-6:
                    self.carry_servo(carry_extra); events.append(dict(step=step, event="carry_servo", extra=carry_extra.tolist()))
                if a.mode in ("shiftback", "imgonly"):
                    du = du_place; events.append(dict(step=step, event="image_shift_place", du=du.tolist()))
                k = H; continue   # re-infer after the gripper / servo
            if gvote < -a.grip_thr and closed and not opened:
                print(f"[run] step {step}: OPEN"); events.append(dict(step=step, event="open", ee=st["ee_pos"].tolist()))
                self.gripper(False); opened = True; step_open = step; k = H; continue
            # motion: clamp, tilt guard, send (tick clips to the workspace box and checks force / robot mode)
            dp = np.clip(act[:3], -a.step_max, a.step_max); dr = act[3:6]
            n = np.linalg.norm(dr); rmax = np.radians(a.rot_max_deg)
            if n > rmax: dr = dr / n * rmax
            new_tq = quat_mul(quat_from_rotvec(dr), self.tq) if a.use_rotation else self.tq
            if quat_angle_deg(new_tq, self.home_q) > a.tilt_max_deg: new_tq = self.tq
            self.tick(self.tp + dp, new_tq)
            rec["t"].append(time.time()); rec["base_small"].append(cv2.resize(base, (320, 240))); rec["shifted_small"].append(cv2.resize(shifted, (320, 240)))
            rec["wrist_small"].append(cv2.resize(wrist, (320, 240))); rec["state"].append(state); rec["action"].append(np.concatenate([dp, dr, [act[6]]]))
            rec["chunk_id"].append(chunk_id); rec["du"].append(du.copy()); rec["infer_s"].append(infer_s); rec["ee_pos"].append(st["ee_pos"].copy())
            rec["grip_cmd"].append(self.grip_cmd); rec["width"].append(self.grip.width)
            if opened and step - step_open >= a.post_open_steps:
                print("[run] post-open steps done"); break
            t_next += 1 / RATE_HZ; time.sleep(max(0, t_next - time.time()))
        else:
            print("[run] max steps reached")
        # 4. retreat
        st = self.robot.get_state(); p = st["ee_pos"].copy()
        self.move_to(np.array([p[0], p[1], self.transit_z]), speed=a.speed_z)
        self.move_to(self.home, self.home_q)
        return base0, wrist0, sbr, rec, events

    # ------------------------------------------------------------------ tape placement (robot does it: exact metric cells)
    def set_tape(self):
        a = self.a
        src = np.array([float(v) for v in a.tape_at.split(",")] + [self.grasp0[2]]) if a.tape_at else self.grasp_cur.copy()
        if a.retrieve_from_plate:
            src = self.place.copy(); src[2] -= a.reset_grasp_dz
        dst = np.array([float(v) for v in a.set_tape.split(",")] + [self.grasp0[2]])
        print(f"[set] moving the tape {np.round(src, 3)} -> {np.round(dst, 3)}")
        self.pick_place(src, dst, record=False)
        self.grasp_cur = dst
        self.poses["grasp_current"] = {"pos": dst.tolist(), "quat_wxyz": self.grasp_q.tolist(), "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(self.poses, open(a.poses, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    # --- AutoCollector arguments (same meaning as auto_collect.py)
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml")); ap.add_argument("--poses", default=os.path.join(HERE, "poses.json"))
    ap.add_argument("--root", default=os.path.join(HERE, "data_root")); ap.add_argument("--task", default="T1")
    ap.add_argument("--instruction", default="put the tape on the plate")
    ap.add_argument("--lift", type=float, default=0.05); ap.add_argument("--speed", type=float, default=0.05); ap.add_argument("--speed-z", type=float, default=0.03)
    ap.add_argument("--max-force", type=float, default=25.0); ap.add_argument("--min-grasp-width", type=float, default=0.0015)
    ap.add_argument("--settle-timeout", type=float, default=3.0); ap.add_argument("--push-max", type=float, default=0.012); ap.add_argument("--grip-wait", type=float, default=0.8)
    ap.add_argument("--reset-grasp-dz", type=float, default=0.02); ap.add_argument("--seed", type=int, default=None); ap.add_argument("--max-frame-age", type=float, default=0.3)
    ap.add_argument("--preview", action="store_true"); ap.add_argument("--backend", default="cart", choices=["cart", "ros2"])
    ap.add_argument("--no-go-home-first", dest="go_home_first", action="store_false"); ap.add_argument("--joint-home", action="store_true")
    ap.add_argument("--max-q-dev", type=float, default=0.35); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--mock-robot", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.0); ap.add_argument("--n", type=int, default=1); ap.add_argument("--max-cam-retries", type=int, default=8)
    ap.add_argument("--reset-last", action="store_true"); ap.add_argument("--tape-on-plate", action="store_true"); ap.add_argument("--test-home", action="store_true")
    # --- policy / ShiftBack
    ap.add_argument("--send-size", type=int, default=224, help="letterbox images to this size before sending (0 = raw 640x480)")
    ap.add_argument("--policy-host", default="127.0.0.1"); ap.add_argument("--policy-port", type=int, default=8000)
    ap.add_argument("--mode", default="baseline", choices=["baseline", "shiftback", "armonly", "imgonly"])
    ap.add_argument("--detector", default="http://127.0.0.1:8010", help="shiftback_service URL, or 'local' (OWLv2 on this machine)")
    ap.add_argument("--calib", default=os.path.join(HERE, "calib", "P_manual.npy")); ap.add_argument("--anchors", default=os.path.join(HERE, "calib", "anchors.json"))
    ap.add_argument("--dead-band-m", type=float, default=0.04); ap.add_argument("--carry-lift", type=float, default=0.05); ap.add_argument("--carry-lower-margin", type=float, default=0.015)
    ap.add_argument("--diag-detect", action="store_true", help="baseline: still run the detector and log what ShiftBack WOULD do")
    ap.add_argument("--replan", type=int, default=10, help="execute this many steps of each predicted chunk (<= action horizon 10)")
    ap.add_argument("--max-steps", type=int, default=750); ap.add_argument("--post-open-steps", type=int, default=45)
    ap.add_argument("--step-max", type=float, default=0.01, help="per-step translation clamp (m); demos move 2.2 mm/step")
    ap.add_argument("--rot-max-deg", type=float, default=2.0); ap.add_argument("--tilt-max-deg", type=float, default=15.0)
    ap.add_argument("--no-rotation", dest="use_rotation", action="store_false", help="ignore the policy's rotation deltas")
    ap.add_argument("--grip-thr", type=float, default=0.5); ap.add_argument("--max-gripper-cycles", type=int, default=1)
    ap.add_argument("--grip-vote", type=int, default=5, help="gripper decision = mean over this many upcoming predicted steps")
    # --- bookkeeping
    ap.add_argument("--tag", default="trial"); ap.add_argument("--tape-cell", default="", help="where the tape IS (x,y m), for the log")
    ap.add_argument("--plate-cell", default="", help="where the plate IS (x,y m) if moved, for the log"); ap.add_argument("--note", default="")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--repeat", type=int, default=1, help="run N episodes back to back (operator resets the scene in between)")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds to wait at home between repeated episodes")
    ap.add_argument("--no-prompt", action="store_true", help="do not ask success/code (recorded as '?', operator keeps the list)")
    ap.add_argument("--video", type=int, default=1, help="1 = record both cameras to MP4 for every episode")
    ap.add_argument("--set-tape", default="", help="x,y: robot moves the tape here before the episode"); ap.add_argument("--tape-at", default="", help="x,y where the tape is now (default poses.json grasp_current)")
    ap.add_argument("--retrieve-from-plate", action="store_true"); ap.add_argument("--only-set-tape", action="store_true")
    ap.add_argument("--replay-npz", default="", help="offline test: feed a recorded episode's frames instead of the cameras (use with --dry-run)")
    ap.add_argument("--replay-first-frame", default="", help="offline test: png used as the first frame for detection")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    r = PolicyRunner(a, cfg)
    vid = None
    try:
        if a.set_tape:
            r.set_tape()
            if a.only_set_tape:
                return
        for rep in range(a.repeat):
            out = os.path.join(a.results, a.tag); os.makedirs(out, exist_ok=True)
            stem = os.path.join(out, f"{a.mode}_{time.strftime('%Y%m%d_%H%M%S')}")
            if a.repeat > 1:
                print(f"\n[run] ===== episode {rep + 1} / {a.repeat}  ({stem}) =====")
            r.partial = None
            if a.video and r.cams is not None and r.replay is None:
                vid = VideoRecorder(r.cams, stem)
            try:
                base0, wrist0, sbr, rec, events = r.run_episode()
                print("\n[run] episode finished. events:", [e["event"] for e in events])
                save_trial(a, stem, base0, wrist0, sbr, rec, events)
            except Abort as e:
                print(f"\n[run] ABORT: {e}")
                if getattr(r, "partial", None) is not None and not a.replay_npz:
                    a.note = (a.note + " | " if a.note else "") + f"ABORT: {e}"
                    save_trial(a, stem, *r.partial)
                raise
            finally:
                if vid is not None:
                    vid.stop(); vid = None
            if rep + 1 < a.repeat:
                print(f"[run] at home, waiting {a.pause:.0f} s for the scene reset ...", flush=True); time.sleep(a.pause)
    except Abort as e:
        print(f"[run] batch stopped by abort: {e}")
    except KeyboardInterrupt:
        print("\n[run] Ctrl+C")
    finally:
        r.close()


def save_trial(a, stem, base0, wrist0, sbr, rec, events):
    if True:
        if a.replay_npz:
            succ, code = "replay", ""
        elif a.no_prompt:
            succ, code = "?", ""          # operator keeps the 0/1 list; filled into trials.csv afterwards
        else:
            succ = input("success? [y/n]: ").strip().lower(); code = "" if succ == "y" else input("failure code [A=memorised spot / B=at tape but no grasp / C=placed wrongly / D=other]: ").strip().upper()
        fn = stem; ts = os.path.basename(stem).split("_", 1)[1]
        np.savez_compressed(fn + ".npz", base0=base0, wrist0=wrist0, **{k: np.asarray(v) for k, v in rec.items()},
                            events=json.dumps(events), shiftback=json.dumps(sbr), args=json.dumps(vars(a)), success=succ, code=code)
        row = dict(time=ts, tag=a.tag, mode=a.mode, tape_cell=a.tape_cell, plate_cell=a.plate_cell, success=succ, code=code, steps=len(rec["t"]),
                   events=" ".join(e["event"] for e in events), pick_delta=(sbr and sbr["pick"]["delta_raw_m"]), place_delta=(sbr and sbr["place"]["delta_raw_m"]), note=a.note, file=fn + ".npz")
        csvp = os.path.join(a.results, "trials.csv"); new = not os.path.exists(csvp)
        with open(csvp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new: w.writeheader()
            w.writerow(row)
        print("[run] saved", fn + ".npz", "->", csvp)


if __name__ == "__main__":
    main()
