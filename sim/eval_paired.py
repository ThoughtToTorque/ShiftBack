#!/usr/bin/env python3
"""Paired evaluation of a frozen policy with and without ShiftBack on LIBERO / LIBERO-Pro.

Every episode is run under identical initial state and identical policy noise once per mode, so
an episode can only be gained (frozen fails, ShiftBack succeeds) or lost. The policy is served by
an openpi policy server (``scripts/serve_policy.py``) and is never changed.

    python sim/eval_paired.py --suite libero_object_swap --task-ids 0-9 --trials 10 --port 8000 \
        --modes frozen,shiftback --out outputs/object_swap

ShiftBack inputs (all shipped in ``data/libero/``):
  --anchors   anchors_grasp.json   demonstrated grasp point per prompt (gripper position at closure) and its pixel
  --stages    anchors_stages.json  per-stage grasp / release points and the fraction of the demo before the first closure
  --hold      hold_height.json     gripper-to-surface height at release per prompt (release surface height)
  --pixels    localiser_pixels_owlv2.json  the fine-tuned localiser's pixels on every first frame (``live``) and
              on the demonstrated layouts (``mem``), precomputed with build_localiser_pixels.py; used for the
              paper's numbers. Alternatively ``--localiser <checkpoint dir>`` runs the localiser online on the
              first frame (memorised pixel = projected demonstrated grasp point) and ``--localiser zero-shot``
              uses the released OWLv2 with box centres.
Ablation switches: --no-arm-shift, --no-image-shift, --no-place, --place-carry-only, --place-image-only,
--fill {edge,gray,mirror}, --tau, --calib-perturb focal=0.05,yaw_deg=2, --pixel-noise SIGMA.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shiftback import lift, perturb_projection, shift_image  # noqa: E402
from shiftback.libero_phrases import place_phrases, stage_phrases  # noqa: E402
from sim.sim_camera import camera_world_to_pixel  # noqa: E402

RES = 256                                 # LIBERO render resolution (the policy input is resized to 224)
DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def default_max_steps(suite):
    for k, v in MAX_STEPS.items():
        if suite == k or suite.startswith(k + "_"):
            return v
    return 220


def quat2axisangle(quat):
    quat = np.asarray(quat, np.float64).copy()
    quat[3] = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def parse_ids(s, n):
    out = []
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def ensure_libero_config(libero_root):
    """LIBERO reads its asset paths from $LIBERO_CONFIG_PATH/config.yaml; write it for a checkout."""
    if not libero_root:
        return
    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH") or os.path.join(os.path.expanduser("~"), ".libero")
    os.environ["LIBERO_CONFIG_PATH"] = cfg_dir
    os.makedirs(cfg_dir, exist_ok=True)
    lp = os.path.join(libero_root, "libero", "libero")
    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        f.write(f"benchmark_root: {lp}\nbddl_files: {lp}/bddl_files\ninit_states: {lp}/init_files\n"
                f"datasets: {lp}/../datasets\nassets: {lp}/assets\n")


class Localiser:
    """Pixels of the object and the placement surface on the first frame, from one of three sources."""

    def __init__(self, a, anchors):
        self.mode, self.a, self.anchors = None, a, anchors
        self.pix = None
        if a.pixels:
            self.pix = json.load(open(a.pixels)); self.mode = "table"
        elif a.localiser:
            from shiftback.localiser import OwlLocaliser
            self.loc = OwlLocaliser(None if a.localiser == "zero-shot" else a.localiser)
            self.mode = "zero-shot" if a.localiser == "zero-shot" else "point"
        self.rng = np.random.default_rng(a.pixel_noise_seed)

    def __call__(self, key, prompt, frame256):
        """-> (live_obj, mem_obj, live_place, mem_place), pixels in the 256 frame or None."""
        if self.mode == "table":
            live = self.pix["live"].get(key); mem = self.pix["mem"].get(prompt)
            lp = (self.pix.get("live_place") or {}).get(key) or [None]
            mp = (self.pix.get("mem_place") or {}).get(prompt) or [None]
            out = [live, mem, lp[0], mp[0]]
        else:
            ph = (stage_phrases(prompt) or [None])[0]
            pp = (place_phrases(prompt) or [None])[0]
            phrases = [p for p in (ph, pp) if p]
            if not phrases:
                return None, None, None, None
            det = self.loc.locate(frame256, phrases, mode="box" if self.mode == "zero-shot" else "point")
            an = self.anchors.get(prompt) or {}
            mem = None if "uv" not in an else (np.asarray(an["uv"]) * (RES / 224.0)).tolist()
            st = (self.a.stage_table.get(prompt) or {}).get("stages") or [{}]
            mem_place = None if not st[0].get("release_uv") else (np.asarray(st[0]["release_uv"]) * (RES / 224.0)).tolist()
            d_o = det.get(ph) if ph else None
            d_p = det.get(pp) if pp else None
            out = [None if d_o is None else d_o["uv"], mem, None if d_p is None else d_p["uv"], mem_place]
        if self.a.pixel_noise > 0:
            out = [None if v is None else (np.asarray(v) + self.rng.normal(0, self.a.pixel_noise, 2)).tolist() for v in out]
        return tuple(out)


def run_episode(env, init_state, client, prompt, key, task_id, ep, a, P, anchors, stage, localiser, shiftback):
    from openpi_client import image_tools
    obs = env.set_init_state(init_state)
    log = {"mode": "shiftback" if shiftback else "frozen"}
    done = False
    delta = np.zeros(3); duv = np.zeros(2)          # pick side
    place_delta = None; place_duv = None            # place side (set at closure)
    phase = "pick"
    plan = collections.deque()
    frames_raw, frames_in = [], []
    replan_index = 0
    closed_seen = reopened_seen = False
    place_armed = None; place_done = False
    max_steps = a.max_steps or default_max_steps(a.suite)
    P_lift = perturb_projection(P, (RES, RES), **a.calib_perturb) if a.calib_perturb else P

    def servo(goal, grip, steps=60):
        nonlocal obs, done
        for _ in range(steps):
            err = goal - np.asarray(obs["robot0_eef_pos"], np.float64)
            if np.linalg.norm(err) < 0.005 or done:
                break
            act = np.clip(err / 0.05, -1.0, 1.0).tolist() + [0.0, 0.0, 0.0, float(grip)]
            obs, _r, done, _i = env.step(act)
        return float(np.linalg.norm(goal - np.asarray(obs["robot0_eef_pos"], np.float64)))

    def frame_of(obs):
        return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

    def request_of(obs, img):
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        return {
            "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(img, a.resize, a.resize)),
            "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, a.resize, a.resize)),
            "observation/state": np.concatenate((obs["robot0_eef_pos"] - (delta if a.shift_state and phase == "pick" else (place_delta if a.shift_state and place_delta is not None else 0.0)),
                                                 quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
            "prompt": str(prompt),
        }

    for t in range(max_steps + a.wait_steps):
        if t < a.wait_steps:
            obs, _r, done, _i = env.step(DUMMY)
            if shiftback and t == a.wait_steps - 1:
                # ---- first frame: detect, lift, decide ----
                g = anchors.get(prompt)
                frac = (stage or {}).get("first_grasp_frac")
                if g is None or (a.skip_late_grasp > 0 and frac is not None and frac > a.skip_late_grasp):
                    log["skipped"] = "no anchor" if g is None else f"first grasp late in the demo ({frac:.2f})"
                    continue
                live, mem, live_pl, mem_pl = localiser(key, prompt, frame_of(obs))
                log.update(uv_live=live, uv_mem=mem, uv_live_place=live_pl, uv_mem_place=mem_pl)
                if live is None or mem is None:
                    log["skipped"] = "no detection"; continue
                z = float(g["g"][2])
                d = lift(P_lift, live, z) - lift(P_lift, mem, z); d[2] = 0.0
                duv0 = np.asarray(mem, np.float64) - np.asarray(live, np.float64)
                log["delta_raw"] = d.tolist()
                if np.linalg.norm(d[:2]) < a.tau:
                    log["inside_threshold"] = True
                else:
                    delta, duv = d, duv0
                    log["delta"] = delta.tolist(); log["duv"] = duv.tolist()
                    if not a.no_arm_shift:
                        log["arm_residual_m"] = servo(np.asarray(obs["robot0_eef_pos"], np.float64) + delta, -1.0)
                    if a.no_image_shift:
                        duv = np.zeros(2)
                    # placement side, decided now from the same first frame, applied at closure
                    rel = (stage or {}).get("stages", [{}])[0].get("release_g")
                    if not a.no_place and rel is not None and live_pl is not None and mem_pl is not None:
                        zr = float(rel[2]) - float(((a.hold.get(prompt) or {}).get("0", 0.0)))
                        pd = lift(P_lift, live_pl, zr) - lift(P_lift, mem_pl, zr); pd[2] = 0.0
                        log["place_delta_raw"] = pd.tolist()
                        if np.linalg.norm(pd[:2]) >= a.tau and np.linalg.norm(pd[:2]) <= a.max_shift:
                            place_delta = pd
                            place_duv = np.asarray(mem_pl, np.float64) - np.asarray(live_pl, np.float64)
            continue

        # ---- gripper state from the finger opening (for the placement switch) ----
        q = np.asarray(obs["robot0_gripper_qpos"], np.float32)
        width = float(abs(q[-1] - q[0]))
        if width < 0.06:
            closed_seen = True
        elif closed_seen and width > 0.07:
            reopened_seen = True
        if place_delta is not None and not place_done:
            if not closed_seen:
                place_armed = None
            elif place_armed is None:
                place_armed = t + a.place_settle
            elif t >= place_armed:
                place_done = True
                carry = place_delta - delta
                log["carry"] = carry.tolist()
                if not a.place_image_only:
                    cur = np.asarray(obs["robot0_eef_pos"], np.float64); goal = cur + carry
                    for leg in ([cur[0], cur[1], cur[2] + a.place_lift], [goal[0], goal[1], cur[2] + a.place_lift], goal):
                        servo(np.asarray(leg, np.float64), 1.0)
                    log["carry_residual_m"] = float(np.linalg.norm(goal - np.asarray(obs["robot0_eef_pos"], np.float64)))
                    plan.clear()
                if not (a.place_carry_only or a.no_image_shift):
                    duv = place_duv
                phase = "place"
                if done:
                    break

        img = frame_of(obs)
        frames_raw.append(img)
        img_in = shift_image(img, duv[0], duv[1], a.fill) if np.any(duv != 0) else img
        frames_in.append(img_in)
        req = request_of(obs, img_in)
        if not plan:
            if a.deterministic_policy_noise:
                rng = np.random.default_rng(np.random.SeedSequence([a.policy_noise_seed, task_id, ep, replan_index]))
                req["_openpi_noise"] = rng.standard_normal((a.action_horizon, a.action_dim), dtype=np.float32)
            chunk = np.asarray(client.infer(req)["actions"])
            replan_index += 1
            plan.extend(chunk[:a.replan_steps])
        act = plan.popleft()
        obs, _r, done, _i = env.step(act.tolist())
        if done:
            break
    log.update(closed_seen=closed_seen, reopened_seen=reopened_seen, phase_at_end=phase)
    return bool(done), frames_raw, frames_in, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object_swap")
    ap.add_argument("--task-ids", default="0-9")
    ap.add_argument("--trials", type=int, default=10, help="initial states per task (the official protocol uses 50)")
    ap.add_argument("--modes", default="frozen,shiftback")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT"), help="LIBERO / LIBERO-Pro checkout (writes its config.yaml)")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--wait-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--deterministic-policy-noise", action="store_true", default=True)
    ap.add_argument("--policy-noise-seed", type=int, default=20260722)
    ap.add_argument("--action-horizon", type=int, default=10); ap.add_argument("--action-dim", type=int, default=32)
    ap.add_argument("--save-videos", action="store_true")
    # ShiftBack inputs
    D = os.path.join(ROOT, "data", "libero")
    ap.add_argument("--anchors", default=os.path.join(D, "anchors_grasp.json"))
    ap.add_argument("--stages", default=os.path.join(D, "anchors_stages.json"))
    ap.add_argument("--hold", default=os.path.join(D, "hold_height.json"))
    ap.add_argument("--pixels", default=os.path.join(D, "localiser_pixels_owlv2.json"), help="precomputed localiser pixels ('' to disable)")
    ap.add_argument("--localiser", default=None, help="OWLv2 checkpoint dir for online localisation, or 'zero-shot'")
    ap.add_argument("--tau", type=float, default=0.04); ap.add_argument("--max-shift", type=float, default=0.55)
    ap.add_argument("--skip-late-grasp", type=float, default=0.48)
    ap.add_argument("--fill", default="edge", choices=["edge", "gray", "mirror"])
    ap.add_argument("--no-arm-shift", action="store_true"); ap.add_argument("--no-image-shift", action="store_true")
    ap.add_argument("--no-place", action="store_true"); ap.add_argument("--place-carry-only", action="store_true")
    ap.add_argument("--place-image-only", action="store_true")
    ap.add_argument("--shift-state", action="store_true", help="also express the arm-state reading in the demonstrated frame (policies that read it)")
    ap.add_argument("--place-settle", type=int, default=8); ap.add_argument("--place-lift", type=float, default=0.10)
    ap.add_argument("--calib-perturb", default=None, help="e.g. focal=0.05,yaw_deg=2,t=0,0,0")
    ap.add_argument("--pixel-noise", type=float, default=0.0, help="Gaussian noise (px) added to the detected pixels")
    ap.add_argument("--pixel-noise-seed", type=int, default=0)
    a = ap.parse_args()
    if a.localiser:
        a.pixels = ""
    raw_perturb, a.calib_perturb = a.calib_perturb, None
    if raw_perturb:
        cp = {}
        for kv in raw_perturb.split(","):
            k, v = kv.split("="); cp[k] = float(v)
        a.calib_perturb = {"focal": cp.get("focal", 0.0), "yaw_deg": cp.get("yaw_deg", 0.0), "t": (cp.get("tx", 0.0), cp.get("ty", 0.0), cp.get("tz", 0.0))}

    ensure_libero_config(a.libero_root)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    import imageio

    anchors = json.load(open(a.anchors))
    a.stage_table = json.load(open(a.stages)) if a.stages and os.path.exists(a.stages) else {}
    a.hold = json.load(open(a.hold)) if a.hold and os.path.exists(a.hold) else {}
    localiser = Localiser(a, anchors)
    client = websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    modes = a.modes.split(",")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed)

    suite = benchmark.get_benchmark_dict()[a.suite]()
    task_ids = parse_ids(a.task_ids, suite.n_tasks)
    rows = []
    t0 = time.time()
    for tid in task_ids:
        task = suite.get_task(tid)
        prompt = task.language
        inits = suite.get_task_init_states(tid)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        env.seed(a.seed); env.reset()
        P = camera_world_to_pixel(env, resolution=RES)
        stage = a.stage_table.get(prompt)
        for ep in range(min(a.trials, len(inits))):
            key = f"{a.suite}/{tid}/{ep}"
            res = {}
            for mode in modes:
                env.reset()
                ok, fr_raw, fr_in, log = run_episode(env, inits[ep], client, prompt, key, tid, ep, a, P, anchors, stage, localiser, mode == "shiftback")
                res[mode] = ok
                rows.append(dict(suite=a.suite, task_id=tid, episode_index=ep, prompt=prompt, mode=mode, success=ok, frames=len(fr_raw), log=log))
                if a.save_videos and fr_raw:
                    vd = out / "videos"; vd.mkdir(exist_ok=True)
                    stem = f"{a.suite}_t{tid:02d}_e{ep:02d}_{mode}_{'success' if ok else 'failure'}"
                    imageio.mimwrite(vd / f"{stem}.mp4", fr_raw, fps=20, macro_block_size=None)
                    if mode == "shiftback":
                        imageio.mimwrite(vd / f"{stem}_policy_view.mp4", fr_in, fps=20, macro_block_size=None)
                print(f"[{a.suite} t{tid} e{ep}] {mode:9s} {'success' if ok else 'failure':8s} {len(fr_raw):4d} steps  "
                      f"{log.get('delta', log.get('skipped', log.get('inside_threshold', '')))}", flush=True)
            if len(modes) == 2:
                g = (not res[modes[0]]) and res[modes[1]]; l = res[modes[0]] and not res[modes[1]]
                print(f"    paired: {'GAINED' if g else 'LOST' if l else 'same'}", flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1))
        env.close()

    report = {"suite": a.suite, "task_ids": task_ids, "trials": a.trials, "modes": modes, "args": {k: v for k, v in vars(a).items() if k not in ("stage_table", "hold")},
              "elapsed_s": time.time() - t0}
    for mode in modes:
        r = [x for x in rows if x["mode"] == mode]
        report[mode] = {"episodes": len(r), "successes": sum(x["success"] for x in r), "success_rate": float(np.mean([x["success"] for x in r])) if r else None}
    if len(modes) == 2:
        key_of = lambda x: (x["task_id"], x["episode_index"])
        base = {key_of(x): x["success"] for x in rows if x["mode"] == modes[0]}
        cand = {key_of(x): x["success"] for x in rows if x["mode"] == modes[1]}
        gained = sum(1 for k in base if not base[k] and cand[k]); lost = sum(1 for k in base if base[k] and not cand[k])
        n = gained + lost
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n)
        report["paired"] = {"gained": gained, "lost": lost, "exact_binomial_p": p}
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in modes + (["paired"] if "paired" in report else [])}, indent=1))


if __name__ == "__main__":
    main()
