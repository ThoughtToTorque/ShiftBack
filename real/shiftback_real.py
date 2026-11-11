"""ShiftBack on the real robot: OWLv2 detection + the planner, shared by the robot-side runner and the GPU-side service.

Geometry (all in the robot base frame, metres; z never changes)
  P (3x4)       world -> fixed-camera pixel (640x480), from ``calibrate_from_demos.py``
  anchors.json  demonstrated pixels of the object / placement surface (mean of the detections on the
                demonstrations' first frames) and the grasp / release heights (gripper at closure / opening)
  pick  : delta = lift(det_obj)   - lift(anchor_obj)   -> start pose += delta,  image moved by anchor - det
  place : delta = lift(det_place) - lift(anchor_place) -> at gripper closure carry by (place - pick), image switches
Threshold: a displacement below ``tau`` (4 cm in every experiment) is treated as zero on that side.
"""
import base64
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shiftback import ShiftBackPlanner, load_projection, shift_image  # noqa: E402,F401
from shiftback.localiser import OwlLocaliser  # noqa: E402


def encode_png_b64(img_rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode_png_b64(s):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


class OwlDetector:
    """Zero-shot OWLv2 boxes (released checkpoint, no fine-tuning), as used on the robot."""

    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", device=None):
        self.loc = OwlLocaliser(model_id, device)

    def detect(self, img_rgb, phrases):
        return self.loc.boxes(img_rgb, phrases)


class HttpDetector:
    """Client of shiftback_service.py (POST /detect)."""

    def __init__(self, url):
        self.url = url.rstrip("/")

    def detect(self, img_rgb, phrases):
        import urllib.request
        body = json.dumps({"image_png_b64": encode_png_b64(img_rgb), "phrases": list(phrases)}).encode()
        req = urllib.request.Request(self.url + "/detect", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["detections"]


class ShiftBack:
    """First-frame detections -> both shifts, in the dict format the robot runner logs."""

    def __init__(self, calib_npy, anchors_json, dead_band_m=0.04, min_score=0.1):
        self.P = load_projection(calib_npy)
        self.A = json.load(open(anchors_json))
        self.dead_band, self.min_score = dead_band_m, min_score

        def _lst(k):
            v = self.A.get(k + "s") or self.A.get(k)
            return list(v) if isinstance(v, (list, tuple)) else [v]
        self.tape_phrases, self.plate_phrases = _lst("tape_phrase"), _lst("plate_phrase")
        self.phrases = self.tape_phrases + [p for p in self.plate_phrases if p not in self.tape_phrases]
        self.planner = ShiftBackPlanner(self.P, self.A["tape_uv_mem"], self.A["z_grasp"],
                                        self.A.get("plate_uv_mem"), self.A.get("z_release"),
                                        tau=dead_band_m, min_score=min_score)

    def compute(self, det):
        """det: {phrase: {uv, score, box} | None} on the FIRST frame -> dict with both sides and the carry correction."""
        def best(phrases):
            c = [(det[p]["score"], p) for p in phrases if det.get(p)]
            return (max(c)[1], det[max(c)[1]]) if c else (phrases[0], None)
        ph_o, d_o = best(self.tape_phrases)
        ph_p, d_p = best(self.plate_phrases)
        plan = self.planner.plan(d_o, d_p, ph_o, ph_p)
        out = {"dead_band_m": self.dead_band}
        for side in ("pick", "place"):
            s = getattr(plan, side)
            out[side] = dict(phrase=s.phrase, anchor_uv=s.anchor_uv, z=s.z, det=det.get(s.phrase),
                             active=s.active, reason=s.reason, delta_raw_m=s.delta_raw[:2], delta_m=s.delta[:2],
                             du=[int(round(s.duv[0])), int(round(s.duv[1]))], norm_m=s.norm)
        out["carry_extra_m"] = plan.carry[:2].tolist()
        out["carry_active"] = bool(np.linalg.norm(plan.carry[:2]) >= 1e-6)
        return out
