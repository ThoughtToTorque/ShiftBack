"""Checks the pure-numpy core against the numbers recorded on the robot (run: python -m pytest tests)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shiftback import (ShiftBackPlanner, anchors_from_demos, fit_projection, gripper_events, lift,  # noqa: E402
                       load_projection, project, refine_projection, reprojection_error, shift_image)
from shiftback.geometry import finger_tips  # noqa: E402
from shiftback.localiser import phrases_from_instruction  # noqa: E402

DATA = os.path.join(ROOT, "data", "real_sample")


def test_lift_inverts_project():
    P = load_projection(os.path.join(DATA, "calib", "P.npy"))
    X = np.array([0.45, -0.2, 0.066])
    uv = project(P, X)
    assert np.allclose(lift(P, uv, X[2]), X, atol=1e-9)


def test_planner_matches_robot_trials():
    P = load_projection(os.path.join(DATA, "calib", "P.npy"))
    A = json.load(open(os.path.join(DATA, "calib", "anchors.json")))
    dets = json.load(open(os.path.join(DATA, "live", "detections.json")))
    for scene, d in dets.items():
        pl = ShiftBackPlanner(P, A["tape_uv_mem"], A["z_grasp"], A["plate_uv_mem"], A["z_release"], tau=0.04, min_score=0.1)
        plan = pl.plan(d["pick"]["det"], d["place"]["det"])
        rec = d["recorded"]
        assert np.allclose(plan.pick.delta[:2], rec["pick_delta_m"], atol=1e-6), scene
        assert np.allclose(plan.place.delta[:2], rec["place_delta_m"], atol=1e-6), scene
        assert np.allclose(np.round(plan.pick.duv), rec["pick_du"]), scene
        assert np.allclose(plan.carry[:2], rec["carry_extra_m"], atol=1e-6), scene


def test_shift_image_edge_fill():
    img = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    out = shift_image(img, 2, -1)
    assert out.shape == img.shape
    assert np.array_equal(out[0, 2:], img[1, :5])      # content moved right by 2 and up by 1
    assert np.array_equal(out[:, 0], out[:, 1])        # left strip repeats the edge column
    assert np.array_equal(out[-1], out[-2])            # bottom strip repeats the edge row


def test_gripper_events_three_step_rule():
    cmd = [-1] * 5 + [1] * 2 + [-1] * 3 + [1] * 10 + [-1] * 6
    ev = gripper_events(cmd, debounce=3)
    assert ev == [(10, 1.0), (20, -1.0)]               # the 2-step blip is ignored


def test_anchors_from_shipped_demos():
    log = json.load(open(os.path.join(DATA, "demo_gripper_log.json")))
    P = load_projection(os.path.join(DATA, "calib", "P.npy"))
    A = json.load(open(os.path.join(DATA, "calib", "anchors.json")))
    an = anchors_from_demos(log, P)
    assert an["K"] == 1 and an["n_kept"] >= 35
    assert np.allclose(an["g"], A["grasp_tcp_mem"], atol=0.01)
    assert np.allclose(an["r"], A["release_tcp_mem"], atol=0.01)


def test_calibration_from_clicked_fingertips():
    rows = json.load(open(os.path.join(DATA, "calib", "clicked_frames.json")))
    X, uv = [], []
    for r in rows:
        tl, tr = finger_tips(r["tcp"], r["R"], r["width"])
        X += [tl, tr]; uv += [r["click_L"], r["click_R"]]
    P = refine_projection(fit_projection(X, uv), X, uv)
    err = reprojection_error(P, X, uv)
    assert np.median(err) < 5.0
    P_ref = load_projection(os.path.join(DATA, "calib", "P.npy"))
    x = lift(P, [112.3, 251.6], 0.066)[:2]
    x_ref = lift(P_ref, [112.3, 251.6], 0.066)[:2]
    assert np.linalg.norm(x - x_ref) < 0.02           # same table point within 2 cm


def test_phrases():
    assert phrases_from_instruction("pick up the alphabet soup and place it in the basket") == ("alphabet soup", "basket")
    assert phrases_from_instruction("put the tape on the plate") == ("tape", "plate")
    assert phrases_from_instruction("put the bowl on the stove") == ("bowl", "stove")
