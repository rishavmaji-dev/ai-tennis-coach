"""Swing detection (peak wrist speed) and forehand/backhand classification."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from . import kinematics as K
from .pose import L_HIP, L_WRIST, R_HIP, R_WRIST, PoseTrack

# Detection thresholds, in torso lengths per second (2D image speed / torso length).
MIN_PEAK_SPEED = 3.5  # a real swing's wrist goes well above this; fidgeting stays below
MIN_PROMINENCE = 2.0
REL_PEAK_FLOOR = 0.4  # drop peaks under 40% of the session's median swing peak
MIN_SWING_GAP_S = 1.2  # two contacts can't be closer than this
CLASSIFY_WINDOW_S = (-0.6, -0.05)  # backswing + forward swing, relative to contact


@dataclass
class CleanTrack:
    """Pose track after confidence masking, glitch removal, gap filling and smoothing."""
    times: np.ndarray
    xy: np.ndarray  # (N,33,2) px, NaN where unreliable
    world: np.ndarray  # (N,33,3) m, NaN where unreliable
    fps: float
    torso_px: float


@dataclass
class Swing:
    number: int  # 1-based, in time order
    contact_idx: int  # index into the analysis track
    contact_time: float
    peak_speed: float  # torso lengths / s (2D)
    stroke: str  # "forehand" | "backhand" | "unknown"
    side_score: float  # mean wrist offset toward the dominant side (m); >0 forehand, <0 backhand
    two_handed: bool | None
    notes: list[str] = field(default_factory=list)


def clean_track(track: PoseTrack) -> CleanTrack:
    times = track.times
    fps = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 30.0
    xy, world = K.mask_low_confidence(track)
    torso = K.torso_length_px(xy)
    if not np.isfinite(torso) or torso <= 0:
        return CleanTrack(times, xy, world, fps, float("nan"))
    xy = K.hampel(xy, min_dev=0.25 * torso)
    world = K.hampel(world, min_dev=0.10)
    xy = K.smooth(K.fill_gaps(xy, times), fps)
    world = K.smooth(K.fill_gaps(world, times), fps)
    return CleanTrack(times, xy, world, fps, torso)


def wrist_ids(handedness: str) -> tuple[int, int]:
    """(dominant wrist, other wrist) landmark ids."""
    return (R_WRIST, L_WRIST) if handedness == "right" else (L_WRIST, R_WRIST)


def hand_speed(ct: CleanTrack, handedness: str) -> np.ndarray:
    """Dominant-wrist 2D speed in torso lengths/s; falls back to the other wrist where the dominant one is lost."""
    dom, other = wrist_ids(handedness)
    s_dom = K.speed(ct.xy[:, dom], ct.times) / ct.torso_px
    s_oth = K.speed(ct.xy[:, other], ct.times) / ct.torso_px
    s = np.where(np.isnan(s_dom), s_oth, s_dom)
    return s


def detect_contacts(ct: CleanTrack, handedness: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (contact frame indices, robust speed signal)."""
    s = hand_speed(ct, handedness)
    s_f = median_filter(np.nan_to_num(s, nan=0.0), size=5, mode="nearest")
    dist = max(1, int(MIN_SWING_GAP_S * ct.fps))
    peaks, props = find_peaks(s_f, height=MIN_PEAK_SPEED, prominence=MIN_PROMINENCE, distance=dist)
    if len(peaks):
        floor = REL_PEAK_FLOOR * float(np.median(props["peak_heights"]))
        peaks = peaks[props["peak_heights"] >= floor]
    return peaks, s_f


def _side_axis(world: np.ndarray, handedness: str) -> np.ndarray:
    """Unit vector in the horizontal plane pointing from the non-dominant hip to the dominant hip."""
    d = world[:, R_HIP] - world[:, L_HIP]
    if handedness == "left":
        d = -d
    d = d.copy()
    d[:, 1] = 0.0  # horizontal plane (y is vertical)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.where(n > 1e-6, n, np.nan)


def classify(ct: CleanTrack, contact_idx: int, handedness: str) -> tuple[str, float, bool | None]:
    """Forehand if the dominant wrist travels on the dominant side of the torso before contact."""
    dom, other = wrist_ids(handedness)
    t0 = ct.times[contact_idx]
    w = (ct.times >= t0 + CLASSIFY_WINDOW_S[0]) & (ct.times <= t0 + CLASSIFY_WINDOW_S[1])
    if w.sum() < 3:
        return "unknown", float("nan"), None
    axis = _side_axis(ct.world[w], handedness)
    hip_mid = (ct.world[w, L_HIP] + ct.world[w, R_HIP]) / 2
    rel = ct.world[w, dom] - hip_mid
    lateral = np.sum(rel * axis, axis=1)
    lateral = lateral[np.isfinite(lateral)]
    if len(lateral) < 3:
        return "unknown", float("nan"), None
    score = float(np.median(lateral))
    # two hands on the racket: wrists stay close through the forward swing
    sep = np.linalg.norm(ct.world[w, dom] - ct.world[w, other], axis=1)
    sep = sep[np.isfinite(sep)]
    two_handed = bool(np.median(sep) < 0.15) if len(sep) >= 3 else None
    if abs(score) < 0.03:
        return "unknown", score, two_handed
    return ("forehand" if score > 0 else "backhand"), score, two_handed


def find_swings(ct: CleanTrack, handedness: str) -> list[Swing]:
    peaks, s_f = detect_contacts(ct, handedness)
    swings = []
    for n, i in enumerate(peaks, start=1):
        stroke, score, two = classify(ct, int(i), handedness)
        swings.append(Swing(number=n, contact_idx=int(i), contact_time=float(ct.times[i]),
                            peak_speed=float(s_f[i]), stroke=stroke, side_score=score, two_handed=two))
    return swings
