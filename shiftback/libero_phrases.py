"""Target phrase of the FIRST grasp for every LIBERO prompt (text only, no scene knowledge).

The phrase is what an open-vocabulary detector is queried with. Spatial prompts keep their relation ("black bowl
next to the plate") because the two bowls are identical; "put both A and B" prompts are excluded (the text does not
say which object is grasped first, and we refuse to look it up in the simulator)."""
from __future__ import annotations

import re

_OVERRIDE = {
    "put the yellow and white mug in the microwave and close it": "yellow and white mug",
    "turn on the stove and put the moka pot on it": "stove",
    "turn on the stove": "stove",
    "open the top drawer and put the bowl inside": "top drawer",
    "put both moka pots on the stove": None,
    "put both the alphabet soup and the cream cheese box in the basket": None,
    "put both the alphabet soup and the tomato sauce in the basket": None,
    "put both the cream cheese box and the butter in the basket": None,
}


def first_grasp_phrase(prompt: str) -> str | None:
    if prompt in _OVERRIDE:
        return _OVERRIDE[prompt]
    m = re.match(r"pick up the (.+?) and place", prompt)
    if m:
        return m.group(1)
    m = re.match(r"(?:put|push|open|close) the (.+?) (?:on|in|to|of|into|from|inside)\b", prompt)
    return m.group(1) if m else None


_STAGES = {
    "put the white mug on the left plate and put the yellow and white mug on the right plate": ["white mug", "yellow and white mug"],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": ["white mug", "chocolate pudding"],
    "turn on the stove and put the moka pot on it": ["stove", "moka pot"],
    # "put both A and B": the text does not say which is grasped first; step108 resolves the order by matching the
    # detector's pixels on the BASE first frames to the label-side stage pixels (gripper pose only, no sim lookup).
    "put both the alphabet soup and the cream cheese box in the basket": ["alphabet soup", "cream cheese box"],
    "put both the alphabet soup and the tomato sauce in the basket": ["alphabet soup", "tomato sauce"],
    "put both moka pots on the stove": ["moka pot", "moka pot"],
    "put both the cream cheese box and the butter in the basket": ["cream cheese box", "butter"],
}


def stage_phrases(prompt: str) -> list[str] | None:
    """One detector phrase per grasp stage, in TEXT order (unresolved for 'put both'); None if unknown."""
    if prompt in _STAGES:
        return list(_STAGES[prompt])
    p = first_grasp_phrase(prompt)
    return [p] if p else None


# ---- RELEASE (place) anchors, 2026-10-02 ------------------------------------------------------------------------
# The second anchor of the two-anchor re-basing: WHERE THE GRIPPER OPENS. Same label source as the grasp anchor
# (gripper pose at a gripper event, fixed camera), same localiser, and the phrase is lifted verbatim from the
# prompt's destination clause. The "release spot" prefix is the ROLE token: it lets one network answer both
# "grasp the stove" (the knob, in "turn on the stove") and "release spot on the stove" (the burner) in the same scene.
_PLACE = {
    "put the white mug on the left plate and put the yellow and white mug on the right plate":
        ["release spot on the left plate", "release spot on the right plate"],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate":
        ["release spot on the plate", "release spot to the right of the plate"],
    "put the yellow and white mug in the microwave and close it": ["release spot in the microwave"],
    "turn on the stove and put the moka pot on it": [None, "release spot on the stove"],
    "put both the alphabet soup and the cream cheese box in the basket": ["release spot in the basket"] * 2,
    "put both the alphabet soup and the tomato sauce in the basket": ["release spot in the basket"] * 2,
    "put both moka pots on the stove": ["release spot on the stove"] * 2,
    "put both the cream cheese box and the butter in the basket": ["release spot in the basket"] * 2,
    "put the black bowl in the bottom drawer of the cabinet and close it": ["release spot in the bottom drawer"],
    "pick up the book and place it in the back compartment of the caddy": ["release spot in the back compartment of the caddy"],
    "put the bowl on the plate": ["release spot on the plate"],
    "put the wine bottle on the rack": ["release spot on the rack"],
    "open the top drawer and put the bowl inside": [None],          # stage 0 = drawer handle; no placement anchor
    "put the cream cheese in the bowl": ["release spot in the bowl"],
    "put the wine bottle on top of the cabinet": ["release spot on top of the cabinet"],
    "push the plate to the front of the stove": [None],
    "turn on the stove": [None],
    "put the bowl on the stove": ["release spot on the stove"],
    "put the bowl on top of the cabinet": ["release spot on top of the cabinet"],
    "open the middle drawer of the cabinet": [None],
}


def place_phrases(prompt: str) -> list[str | None] | None:
    """One release-anchor phrase per grasp stage (None = that stage places nothing); None if prompt unknown."""
    if prompt in _PLACE:
        return list(_PLACE[prompt])
    m = re.match(r"pick up the .+? and place it (in|on) the (.+)$", prompt)
    if m:
        return [f"release spot {m.group(1)} the {m.group(2)}"]
    sp = stage_phrases(prompt)
    return [None] * len(sp) if sp else None
