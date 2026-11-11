"""Pinhole geometry for ShiftBack: pixel <-> table-plane points through a fixed camera.

Everything here works on a 3x4 projection matrix ``P`` (world -> homogeneous pixel) expressed in the
robot base frame, so that a point lifted from a pixel is directly a gripper position. ``P`` comes
from any standard calibration (see :func:`fit_projection` for the demonstration-only fit used on
our robot, and ``libero/sim_camera.py`` for the simulator).
"""
from typing import Optional, Sequence, Tuple

import numpy as np


def project(P: np.ndarray, X) -> np.ndarray:
    """World point(s) (N,3) or (3,) -> pixel(s) (N,2) or (2,)."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1) @ np.asarray(P, np.float64).T
    uv = Xh[:, :2] / Xh[:, 2:3]
    return uv[0] if uv.shape[0] == 1 else uv


def lift(P: np.ndarray, uv, z: float) -> np.ndarray:
    """Pixel(s) -> the point(s) where the line of sight meets the horizontal plane Z = z.

    This is the planar lift of the paper: it replaces a depth camera with a height the
    demonstrations already provide (the height at which the gripper closed).
    """
    P = np.asarray(P, np.float64)[:3]
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    out = np.zeros((len(uv), 3))
    for i, (u, v) in enumerate(uv):
        r0 = P[0] - u * P[2]
        r1 = P[1] - v * P[2]
        A = np.array([[r0[0], r0[1]], [r1[0], r1[1]]])
        b = -np.array([r0[2] * z + r0[3], r1[2] * z + r1[3]])
        xy = np.linalg.solve(A, b)
        out[i] = [xy[0], xy[1], z]
    return out[0] if out.shape[0] == 1 else out


def displacement(P: np.ndarray, uv_live, uv_demo, z: float) -> Tuple[np.ndarray, np.ndarray]:
    """Object displacement on the table and the matching pixel offset.

    Returns ``(delta, duv)`` with ``delta = lift(uv_live) - lift(uv_demo)`` (z set to 0, metres,
    live minus demonstrated) and ``duv = uv_demo - uv_live`` (pixels), the offset by which the image
    has to be moved so that the object reappears at its demonstrated pixel.
    """
    delta = lift(P, uv_live, z) - lift(P, uv_demo, z)
    delta[2] = 0.0
    duv = np.asarray(uv_demo, np.float64) - np.asarray(uv_live, np.float64)
    return delta, duv


def pixel_to_table_jacobian(P: np.ndarray, uv, z: float, eps: float = 0.5) -> np.ndarray:
    """d(lift)/d(uv) at a pixel (2x2, metres per pixel), by central differences."""
    uv = np.asarray(uv, np.float64)
    J = np.zeros((2, 2))
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps
        J[:, k] = (lift(P, uv + d, z)[:2] - lift(P, uv - d, z)[:2]) / (2 * eps)
    return J


def perturb_projection(P: np.ndarray, image_size: Tuple[int, int], focal: float = 0.0,
                       yaw_deg: float = 0.0, t: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Return ``P`` with a deliberate calibration error (used for the robustness sweep).

    ``focal`` is a relative focal-length error, ``yaw_deg`` a rotation about the world z axis and
    ``t`` a translation of the world frame (metres). ``image_size`` is (width, height).
    """
    P = np.asarray(P, np.float64).copy()[:3]
    if focal:
        c = np.array([[image_size[0] / 2.0], [image_size[1] / 2.0]])
        P[:2] = (P[:2] - c * P[2]) * (1.0 + focal) + c * P[2]
    T = np.eye(4)
    a = np.deg2rad(yaw_deg)
    if a:
        T[:3, :3] = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    t = np.asarray(t, np.float64)
    if np.any(t):
        T2 = np.eye(4)
        T2[:3, 3] = -t
        T = T @ T2
    return P @ T


def fit_projection(X, uv, normalise: bool = True) -> np.ndarray:
    """Direct linear transform: 3x4 ``P`` from >= 6 world <-> pixel correspondences.

    On our robot the correspondences are the logged gripper positions and the pixels at which the
    gripper (finger tips) appears in the corresponding demonstration frames, so no calibration
    target is needed (``real/calibrate_from_demos.py``).
    """
    X = np.asarray(X, np.float64)
    uv = np.asarray(uv, np.float64)
    assert len(X) >= 6 and len(X) == len(uv)
    # Hartley normalisation keeps the linear system well conditioned.
    if normalise:
        mx, su = X.mean(0), X.std(0).mean() + 1e-12
        mu, sv = uv.mean(0), uv.std(0).mean() + 1e-12
        Tx = np.diag([1 / su, 1 / su, 1 / su, 1.0]); Tx[:3, 3] = -mx / su
        Tu = np.diag([1 / sv, 1 / sv, 1.0]); Tu[:2, 2] = -mu / sv
        Xn = (X - mx) / su
        uvn = (uv - mu) / sv
    else:
        Tx, Tu, Xn, uvn = np.eye(4), np.eye(3), X, uv
    A = []
    for (x, y, z), (u, v) in zip(Xn, uvn):
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Pn = Vt[-1].reshape(3, 4)
    P = np.linalg.inv(Tu) @ Pn @ Tx
    return P / np.linalg.norm(P[2, :3])


def reprojection_error(P: np.ndarray, X, uv) -> np.ndarray:
    """Per-point pixel error of ``P`` on correspondences."""
    return np.linalg.norm(project(P, X).reshape(-1, 2) - np.asarray(uv, np.float64), axis=1)


def refine_projection(P0: np.ndarray, X, uv, iters: int = 50) -> np.ndarray:
    """Gauss-Newton refinement of the 11 free entries of ``P`` on the reprojection error.

    Pure numpy; good enough after the DLT initialisation. ``P[2, 3]`` is held at its initial value
    to fix the scale.
    """
    P = np.asarray(P0, np.float64).copy()
    X = np.asarray(X, np.float64)
    uv = np.asarray(uv, np.float64)
    idx = [(i, j) for i in range(3) for j in range(4) if not (i == 2 and j == 3)]
    for _ in range(iters):
        r = (project(P, X).reshape(-1, 2) - uv).ravel()
        J = np.zeros((r.size, len(idx)))
        for k, (i, j) in enumerate(idx):
            d = np.zeros_like(P); d[i, j] = 1e-6
            J[:, k] = ((project(P + d, X).reshape(-1, 2) - uv).ravel() - r) / 1e-6
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        if np.linalg.norm(step) < 1e-10:
            break
        for k, (i, j) in enumerate(idx):
            P[i, j] += step[k]
    return P


def finger_tips(tcp: np.ndarray, R: np.ndarray, width: float, axis: int = 1, tip_offset: float = 0.008) -> Tuple[np.ndarray, np.ndarray]:
    """3-D positions of the two finger tips of a parallel gripper from the logged TCP pose.

    The fingers slide along one axis of the hand frame (``axis`` = 1 for the Franka hand) and the
    clicked point is the outer face of each pad, ``tip_offset`` beyond the commanded half width.
    Returns ``(left tip, right tip)`` as they appear in a camera facing the robot:
    ``tcp -/+ (width/2 + tip_offset) * R[:, axis]``. Clicking these tips in a few demonstration
    frames gives the correspondences for :func:`fit_projection`.
    """
    tcp = np.asarray(tcp, np.float64)
    d = np.asarray(R, np.float64)[:, axis] * (0.5 * width + tip_offset)
    return tcp - d, tcp + d


def load_projection(path: str) -> np.ndarray:
    """Load a 3x4 (or 4x4, top three rows used) projection matrix saved with ``np.save``."""
    P = np.load(path)
    return np.asarray(P, np.float64)[:3]


__all__ = [
    "project", "lift", "displacement", "pixel_to_table_jacobian", "perturb_projection",
    "fit_projection", "refine_projection", "reprojection_error", "finger_tips", "load_projection",
]
