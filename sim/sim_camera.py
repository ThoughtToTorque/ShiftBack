"""The fixed camera of the LIBERO simulator as a 3x4 projection matrix.

In simulation the "calibration" is read from the simulator once per task: the camera's field of
view and pose. The matrix maps robot-base (world) coordinates to pixels of the ``agentview``
image as the policy sees it (LIBERO renders upside-down, so the image is flipped both ways before
it is shown to the policy; the matrix accounts for that).
"""
import numpy as np


def camera_world_to_pixel(env, camera_name: str = "agentview", resolution: int = 256) -> np.ndarray:
    """4x4 homogeneous world -> pixel matrix for the policy's (flipped) view at ``resolution``."""
    sim = env.sim
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = float(sim.model.cam_fovy[cam_id])
    f = 0.5 * resolution / np.tan(fovy * np.pi / 360.0)
    K = np.eye(4)
    K[:3, :3] = [[f, 0.0, resolution / 2.0], [0.0, f, resolution / 2.0], [0.0, 0.0, 1.0]]
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(sim.data.cam_xmat[cam_id]).reshape(3, 3)
    pose[:3, 3] = np.asarray(sim.data.cam_xpos[cam_id])
    axis_fix = np.diag([1.0, -1.0, -1.0, 1.0])  # MuJoCo camera looks down -z
    P = K @ np.linalg.inv(pose @ axis_fix)
    # the rendered frame is rotated by 180 degrees before it reaches the policy: a horizontal
    # flip of the pixel coordinates (the vertical flip is absorbed by the axis correction above)
    flip = np.eye(4)
    flip[0, 0], flip[0, 2] = -1.0, float(resolution - 1)
    return flip @ P


__all__ = ["camera_world_to_pixel"]
