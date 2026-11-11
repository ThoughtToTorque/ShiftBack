"""ShiftBack: training-free correction of layout changes for a frozen manipulation policy.

Core pieces (pure numpy unless noted):

- :mod:`shiftback.geometry`  pixel <-> table-plane points through a fixed, calibrated camera
- :mod:`shiftback.anchors`   grasp/release anchors from the gripper events of the demonstrations
- :mod:`shiftback.shift`     the image shift with edge filling, and the minimum-shift threshold
- :mod:`shiftback.localiser` phrase extraction and the OWLv2 localiser (needs torch + transformers)
- :mod:`shiftback.planner`   first-frame detections -> start-pose shift, image shift, carry at closure
"""
from .anchors import anchors_from_demos, gripper_events, stage_events, start_pose_from_demos
from .geometry import (displacement, fit_projection, lift, load_projection, perturb_projection,
                       pixel_to_table_jacobian, project, refine_projection, reprojection_error)
from .planner import GripperStageMachine, Plan, ShiftBackPlanner, Side
from .shift import apply_threshold, kept_box, shift_image

__version__ = "1.0.0"
__all__ = [
    "anchors_from_demos", "gripper_events", "stage_events", "start_pose_from_demos",
    "displacement", "fit_projection", "lift", "load_projection", "perturb_projection",
    "pixel_to_table_jacobian", "project", "refine_projection", "reprojection_error",
    "GripperStageMachine", "Plan", "ShiftBackPlanner", "Side",
    "apply_threshold", "kept_box", "shift_image",
]
