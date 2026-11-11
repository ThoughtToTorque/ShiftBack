"""The two shifts: moving the camera image back, and the minimum-shift threshold."""
from typing import Tuple

import numpy as np


def shift_image(img: np.ndarray, du: float, dv: float, fill: str = "edge") -> np.ndarray:
    """Move the image content by (du, dv) pixels (+u right, +v down).

    ``fill`` decides what goes into the strip the move leaves empty: ``"edge"`` repeats the border
    pixels (what the paper uses), ``"gray"`` fills with mid-grey, ``"mirror"`` reflects the image.
    The result has the same shape as ``img``.
    """
    h, w = img.shape[:2]
    ix, iy = int(round(du)), int(round(dv))
    if ix == 0 and iy == 0:
        return img
    if fill == "edge":
        ys = np.clip(np.arange(h) - iy, 0, h - 1)
        xs = np.clip(np.arange(w) - ix, 0, w - 1)
        return np.ascontiguousarray(img[ys][:, xs])
    if fill == "gray":
        out = np.full_like(img, 128)
        ys = np.arange(h) - iy
        xs = np.arange(w) - ix
        my = (ys >= 0) & (ys < h)
        mx = (xs >= 0) & (xs < w)
        out[np.ix_(my, mx)] = img[np.ix_(ys[my], xs[mx])]
        return np.ascontiguousarray(out)
    if fill == "mirror":
        py, px = abs(iy), abs(ix)
        pad_w = ((py, py), (px, px), (0, 0)) if img.ndim == 3 else ((py, py), (px, px))
        pad = np.pad(img, pad_w, mode="symmetric")
        return np.ascontiguousarray(pad[py - iy:py - iy + h, px - ix:px - ix + w])
    raise ValueError(f"unknown fill {fill!r}")


def apply_threshold(delta: np.ndarray, duv: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Minimum-shift threshold: below ``tau`` metres (horizontal norm) nothing is corrected.

    Returns ``(delta, duv, active)``; when inactive both are zero.
    """
    delta = np.asarray(delta, np.float64).copy()
    duv = np.asarray(duv, np.float64).copy()
    if np.linalg.norm(delta[:2]) < tau:
        return np.zeros_like(delta), np.zeros_like(duv), False
    return delta, duv, True


def kept_box(width: int, height: int, du: float, dv: float) -> Tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of the original frame that survives a shift by (du, dv).

    Handy for drawing: the same box, moved by (du, dv), is where that content lands in the
    shifted frame; the rest of the shifted frame is the filled strip.
    """
    du, dv = int(round(du)), int(round(dv))
    x0, y0 = max(0, -du), max(0, -dv)
    x1, y1 = min(width, width - du), min(height, height - dv)
    return x0, y0, x1, y1


__all__ = ["shift_image", "apply_threshold", "kept_box"]
