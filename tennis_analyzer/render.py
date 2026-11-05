"""Drawing: skeleton overlay, annotated video, and side-by-side comparison."""
from __future__ import annotations

import cv2
import numpy as np

from .pose import CONNECTIONS

LEFT_COLOR = (80, 170, 255)  # RGB: subject's left side
RIGHT_COLOR = (255, 120, 60)  # RGB: subject's right side
MID_COLOR = (235, 235, 235)
LEFT_IDS = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_IDS = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}


def _side_color(a: int, b: int):
    if a in LEFT_IDS and b in LEFT_IDS:
        return LEFT_COLOR
    if a in RIGHT_IDS and b in RIGHT_IDS:
        return RIGHT_COLOR
    return MID_COLOR


def draw_skeleton(rgb: np.ndarray, lm: np.ndarray, scale: float = 1.0, min_vis: float = 0.5,
                  thickness: int | None = None, highlight: tuple[int, ...] = ()) -> np.ndarray:
    """Draw pose landmarks (33x5, full-res pixel coords) onto rgb in place.

    Landmarks below min_vis are skipped, so low-confidence joints are never drawn.
    """
    if lm is None or np.isnan(lm[0, 0]):
        return rgb
    h = rgb.shape[0]
    t = thickness or max(2, int(round(h / 360)))
    pts = lm[:, :2] * scale
    ok = lm[:, 3] >= min_vis
    for a, b in CONNECTIONS:
        if a <= 10 or b <= 10:  # skip face mesh lines
            continue
        if ok[a] and ok[b]:
            cv2.line(rgb, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                     _side_color(a, b), t, cv2.LINE_AA)
    for j in range(11, 33):
        if ok[j]:
            r = t + 2 if j in highlight else t
            cv2.circle(rgb, tuple(int(v) for v in pts[j]), r, (255, 255, 0) if j in highlight else (255, 255, 255),
                       -1, cv2.LINE_AA)
    return rgb


def put_label(rgb: np.ndarray, text: str, org: tuple[int, int], size: float | None = None,
              color=(255, 255, 255), bg=(0, 0, 0)) -> None:
    h = rgb.shape[0]
    s = size or max(0.5, h / 1400)
    th = max(1, int(round(s * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, th)
    x, y = org
    cv2.rectangle(rgb, (x - 4, y - tht - 6), (x + tw + 4, y + base + 2), bg, -1)
    cv2.putText(rgb, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, th, cv2.LINE_AA)
