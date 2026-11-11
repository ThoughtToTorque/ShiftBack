"""ShiftBack for one episode: from the first-frame detections to the two shifts, pick and place.

The planner is policy-agnostic. The caller detects the object (and the placement surface) on the
first frame, hands the pixels to :meth:`ShiftBackPlanner.plan`, moves the arm by ``plan.pick.delta``
before the first policy query, passes every third-person frame through
:meth:`ShiftBackPlanner.shift_frame` and, when the gripper closes, carries the held object by
``plan.carry`` and switches the image shift to the placement side. ``libero/eval_paired.py`` and
``real/run_episode.py`` show the two concrete loops.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import displacement
from .shift import apply_threshold, shift_image


@dataclass
class Side:
    """One side of the correction (pick or place)."""
    phrase: Optional[str] = None
    anchor_uv: Optional[List[float]] = None
    z: Optional[float] = None
    det_uv: Optional[List[float]] = None
    det_score: Optional[float] = None
    delta_raw: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    delta: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    duv: List[float] = field(default_factory=lambda: [0.0, 0.0])
    active: bool = False
    reason: str = "not planned"

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.delta_raw[:2]))


@dataclass
class Plan:
    pick: Side
    place: Side
    tau: float

    @property
    def carry(self) -> np.ndarray:
        """Extra arm move at gripper closure: the placement shift minus what the pick shift already gave."""
        return np.asarray(self.place.delta, np.float64) - np.asarray(self.pick.delta, np.float64)

    def to_dict(self) -> Dict:
        return {"tau": self.tau, "pick": self.pick.__dict__, "place": self.place.__dict__,
                "carry": self.carry.tolist()}


class ShiftBackPlanner:
    """Turns first-frame detections into the start-pose shift and the image shift of each side.

    Parameters
    ----------
    P : 3x4 projection matrix of the fixed camera (robot base frame -> pixels of the frames you
        pass in; use the same resolution for the anchors and the detections).
    grasp_uv, grasp_z : demonstrated grasp pixel and grasp height (gripper position at closure).
    release_uv, release_z : demonstrated release pixel and the height of the *surface* the object
        is released onto (gripper height at opening minus the hold height), or ``None`` when the
        placement side is not corrected.
    tau : minimum-shift threshold in metres; smaller measured displacements are ignored.
    max_shift : displacements above this (metres) are treated as detector failures and ignored.
    min_score : detections below this score are ignored.
    """

    def __init__(self, P: np.ndarray, grasp_uv: Sequence[float], grasp_z: float,
                 release_uv: Optional[Sequence[float]] = None, release_z: Optional[float] = None,
                 tau: float = 0.04, max_shift: float = 0.55, min_score: float = 0.0, fill: str = "edge"):
        self.P = np.asarray(P, np.float64)[:3]
        self.grasp_uv, self.grasp_z = np.asarray(grasp_uv, np.float64), float(grasp_z)
        self.release_uv = None if release_uv is None else np.asarray(release_uv, np.float64)
        self.release_z = None if release_z is None else float(release_z)
        self.tau, self.max_shift, self.min_score, self.fill = float(tau), float(max_shift), float(min_score), fill
        self.plan_: Optional[Plan] = None

    def _side(self, det: Optional[Dict], anchor_uv: Optional[np.ndarray], z: Optional[float], phrase: Optional[str]) -> Side:
        s = Side(phrase=phrase, anchor_uv=None if anchor_uv is None else anchor_uv.tolist(), z=z)
        if anchor_uv is None or z is None:
            s.reason = "no anchor"
            return s
        if det is None or det.get("uv") is None:
            s.reason = "no detection"
            return s
        if det.get("score") is not None and det["score"] < self.min_score:
            s.reason = f"score {det['score']:.2f} < {self.min_score}"
            return s
        s.det_uv, s.det_score = [float(x) for x in det["uv"]], det.get("score")
        delta, duv = displacement(self.P, s.det_uv, anchor_uv, z)
        s.delta_raw = delta.tolist()
        if np.linalg.norm(delta[:2]) > self.max_shift:
            s.reason = f"displacement {np.linalg.norm(delta[:2]):.2f} m above max_shift"
            return s
        delta, duv, active = apply_threshold(delta, duv, self.tau)
        s.delta, s.duv, s.active = delta.tolist(), duv.tolist(), active
        s.reason = "ok" if active else "inside threshold"
        return s

    def plan(self, det_obj: Optional[Dict], det_place: Optional[Dict] = None,
             obj_phrase: Optional[str] = None, place_phrase: Optional[str] = None) -> Plan:
        """``det_*`` are ``{"uv": [u, v], "score": s}`` dicts (or ``None``) from the first frame."""
        pick = self._side(det_obj, self.grasp_uv, self.grasp_z, obj_phrase)
        place = self._side(det_place, self.release_uv, self.release_z, place_phrase)
        self.plan_ = Plan(pick=pick, place=place, tau=self.tau)
        return self.plan_

    def shift_frame(self, img: np.ndarray, phase: str = "pick") -> np.ndarray:
        """Third-person frame as the policy should see it in the given phase (``"pick"`` or ``"place"``)."""
        assert self.plan_ is not None, "call plan() first"
        side = self.plan_.pick if phase == "pick" else self.plan_.place
        return shift_image(img, side.duv[0], side.duv[1], self.fill)

    def shift_state(self, state: np.ndarray, phase: str = "pick") -> np.ndarray:
        """Arm position reading expressed in the demonstrated frame (for policies that read it)."""
        assert self.plan_ is not None
        side = self.plan_.pick if phase == "pick" else self.plan_.place
        s = np.asarray(state, np.float64).copy()
        s[:3] -= np.asarray(side.delta, np.float64)
        return s


class GripperStageMachine:
    """Online closing/opening detector over the policy's commanded gripper channel.

    The same three-step rule as :func:`shiftback.anchors.gripper_events`, run step by step. A sign
    flip is committed after ``debounce`` consecutive commands of the new sign, judged against the
    last accepted sign, so events strictly alternate. ``closed`` tells the caller whether the
    object is held (pick phase vs place phase).
    """

    def __init__(self, debounce: int = 3, start_sign: float = -1.0):
        self.debounce, self.sign, self.run = int(debounce), float(start_sign), 0
        self.closed = False
        self.events: List[Tuple[int, float]] = []
        self.just_closed = False
        self.just_opened = False

    def step(self, grip_cmd: float, t: int) -> None:
        self.just_closed = self.just_opened = False
        s = 1.0 if float(grip_cmd) > 0.0 else -1.0
        if s != self.sign:
            self.run += 1
            if self.run >= self.debounce:
                self.sign, self.run = s, 0
                self.events.append((t, s))
                if s > 0:
                    self.closed, self.just_closed = True, True
                else:
                    self.closed, self.just_opened = False, True
        else:
            self.run = 0

    @property
    def n_closures(self) -> int:
        return sum(1 for _, s in self.events if s > 0)


__all__ = ["Side", "Plan", "ShiftBackPlanner", "GripperStageMachine"]
