#!/usr/bin/env python3
"""Dump the first policy frame of every (suite, task, initial state) of LIBERO / LIBERO-Pro.

For each initial state the environment is reset, ``--wait-steps`` dummy actions are stepped (as
the evaluation does before the first policy query) and the agentview frame as the policy sees it
(256 px, flipped both ways), the wrist frame (resized to ``--resize``), the proprioceptive state and
the camera matrix are stored. These frames are what the localiser is scored on and what
``build_localiser_pixels.py`` reads.

``--sim-truth`` additionally stores the simulator position of the task's first object of interest
(``true``), used ONLY to score the localiser (``localiser/finetune_owlv2.py``,
``build_localiser_pixels.py --sim-truth``); the method never reads it.

Output npz: ``imgs`` (N,256,256,3) uint8, ``wrist`` (N,resize,resize,3), ``state`` (N,8) and ``meta``
(json list with suite, task, init, prompt, w2p, eef and optionally ooi / true).

Example
    python sim/dump_first_frames.py --suites libero_object_swap libero_spatial_swap libero_goal_swap \
        --n-init 10 --out outputs/first_frames/swap_frames.npz --libero-root $LIBERO_ROOT
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.eval_paired import DUMMY, RES, ensure_libero_config, quat2axisangle  # noqa: E402
from sim.sim_camera import camera_world_to_pixel  # noqa: E402


def sim_object_position(env, name):
    """Simulator world position of an object of interest (validation only). None for table regions."""
    bodies = env.env.obj_body_id
    body = name if name in bodies else next((b for b in bodies if name.startswith(b)), None)
    if body is None:
        return None
    return np.asarray(env.sim.data.body_xpos[bodies[body]], np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suites", nargs="+", required=True, help="e.g. libero_object_swap libero_spatial_swap")
    ap.add_argument("--n-init", type=int, default=10, help="initial states per task (50 = the official protocol)")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "first_frames", "first_frames.npz"))
    ap.add_argument("--libero-root", default=os.environ.get("LIBERO_ROOT"), help="LIBERO / LIBERO-Pro checkout (writes its config.yaml)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--wait-steps", type=int, default=10)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--sim-truth", action="store_true", help="store the simulator object position (scoring only)")
    a = ap.parse_args()

    ensure_libero_config(a.libero_root)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools

    bd = benchmark.get_benchmark_dict()
    imgs, wrists, states, meta = [], [], [], []
    for suite in a.suites:
        ts = bd[suite]()
        for tid in range(ts.n_tasks):
            task = ts.get_task(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
            env.seed(a.seed); env.reset()
            w2p = camera_world_to_pixel(env, resolution=RES)
            for k, st in enumerate(ts.get_task_init_states(tid)[:a.n_init]):
                env.reset(); obs = env.set_init_state(st)
                for _ in range(a.wait_steps):
                    obs = env.step(DUMMY)[0]
                frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, a.resize, a.resize))
                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                imgs.append(frame.copy()); wrists.append(np.asarray(wrist).copy()); states.append(np.asarray(state, np.float32))
                m = dict(suite=suite, task=tid, init=k, prompt=task.language, w2p=np.asarray(w2p).tolist(),
                         eef=np.asarray(obs["robot0_eef_pos"]).tolist())
                if a.sim_truth:
                    ooi = list(env.obj_of_interest)
                    true = sim_object_position(env, ooi[0]) if ooi else None
                    m.update(ooi=ooi, true=None if true is None else true.tolist())
                meta.append(m)
            env.close(); print(suite, tid, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, imgs=np.stack(imgs), wrist=np.stack(wrists), state=np.stack(states), meta=json.dumps(meta))
    print(f"-> {a.out}  ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
