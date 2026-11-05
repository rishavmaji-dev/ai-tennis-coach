"""Measurements at the contact frame of one swing (from the Heavy-model window).

3D angles use MediaPipe world landmarks (metres, hip-centred), so they don't depend on where the
camera stands. Every metric is None when a landmark it needs was low-confidence at contact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import kinematics as K
from .pose import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, R_ANKLE, R_ELBOW, R_HIP,
                   R_KNEE, R_SHOULDER, R_WRIST)
from .swings import CleanTrack, wrist_ids

COIL_SEARCH_S = 0.7  # look this far before contact for the most-coiled position
CONTACT_REFINE_S = 0.15  # re-pick contact as peak wrist speed within +/- this of the Lite estimate


@dataclass
class ContactMetrics:
    elbow_angle_deg: float | None  # 180 = straight
    elbow_arm: str | None  # "right"/"left": hitting arm, or the top-hand arm on two-handed strokes if
    #                        the hitting arm's elbow was hidden
    knee_bend_deg: float | None  # flexion of the more-bent knee, 0 = straight leg
    left_knee_bend_deg: float | None
    right_knee_bend_deg: float | None
    shoulder_rotation_deg: float | None  # shoulder-line turn from most-coiled position to contact
    hip_rotation_deg: float | None  # hip-line turn from most-coiled position to contact
    separation_at_coil_deg: float | None  # shoulder vs hip line angle at the most-coiled position
    contact_in_front_cm: float | None  # wrist ahead of hip centre along the swing direction
    hip_lead_ms: float | None  # hips reach half their turn this long before shoulders do (+ = hips first)
    forward_swing_ms: float | None  # slowest point of the backswing to contact
    wrist_speed_mps: float | None  # dominant wrist speed at contact (3D, relative to hips)
    missing: list[str]  # plain-language reasons for any metric that couldn't be measured

    def to_dict(self) -> dict:
        return asdict(self)


def refine_contact(ct: CleanTrack, approx_idx: int, handedness: str) -> int:
    """Peak dominant-wrist speed near the Lite contact estimate, using the Heavy track."""
    dom, _ = wrist_ids(handedness)
    s = K.speed(ct.xy[:, dom], ct.times)
    t0 = ct.times[approx_idx]
    w = np.flatnonzero(np.abs(ct.times - t0) <= CONTACT_REFINE_S)
    if len(w) == 0 or np.all(np.isnan(s[w])):
        return approx_idx
    return int(w[np.nanargmax(s[w])])


def _f(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


def _rotation_from_coil(yaw: np.ndarray, times: np.ndarray, c: int) -> tuple[float, int] | None:
    """Largest turn (deg) between any frame in the coil-search window and contact, plus that frame."""
    w = np.flatnonzero((times >= times[c] - COIL_SEARCH_S) & (times <= times[c]))
    y = yaw[w]
    ok = np.isfinite(y)
    if ok.sum() < 3 or not np.isfinite(yaw[c]):
        return None
    diff = np.abs(yaw[c] - y)
    diff[~ok] = -1
    k = int(np.argmax(diff))
    return float(diff[k]), int(w[k])


def _half_turn_time(yaw: np.ndarray, times: np.ndarray, c: int, coil_idx: int) -> float | None:
    """First time after the coil frame that the segment has completed half its turn to contact."""
    if coil_idx >= c:
        return None
    seg = yaw[coil_idx:c + 1]
    if np.isnan(seg).any():
        return None
    progress = (seg - seg[0]) / (seg[-1] - seg[0])
    k = int(np.argmax(progress >= 0.5))
    if k == 0:
        return float(times[coil_idx])
    # linear interpolation between the bracketing frames
    p0, p1 = progress[k - 1], progress[k]
    t0, t1 = times[coil_idx + k - 1], times[coil_idx + k]
    return float(t0 + (0.5 - p0) / (p1 - p0) * (t1 - t0)) if p1 != p0 else float(t1)


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def measure(ct: CleanTrack, c: int, handedness: str, forward: np.ndarray | None,
            two_handed: bool | None = None) -> ContactMetrics:
    W = ct.world
    dom_w, oth_w = wrist_ids(handedness)
    arms = {"right": (R_SHOULDER, R_ELBOW, R_WRIST), "left": (L_SHOULDER, L_ELBOW, L_WRIST)}
    hitting = handedness
    top_hand = "left" if handedness == "right" else "right"
    missing: list[str] = []

    elbow_arm = hitting
    elbow = _f(K.angle_3pt(*(W[c, j] for j in arms[hitting])))
    if elbow is None and two_handed:
        elbow_arm = top_hand
        elbow = _f(K.angle_3pt(*(W[c, j] for j in arms[top_hand])))
    if elbow is None:
        elbow_arm = None
        missing.append("Elbow angle: the arm joints were hidden or tracked with low confidence at contact.")

    knees = {}
    for name, (h, k, a) in {"left": (L_HIP, L_KNEE, L_ANKLE), "right": (R_HIP, R_KNEE, R_ANKLE)}.items():
        ang = _f(K.angle_3pt(W[c, h], W[c, k], W[c, a]))
        knees[name] = None if ang is None else 180.0 - ang
    bends = [v for v in knees.values() if v is not None]
    knee = max(bends) if bends else None
    if knee is None:
        missing.append("Knee bend: hips, knees or ankles were not tracked confidently at contact.")

    sh_yaw = K.unwrap_deg(K.yaw_deg(W[:, L_SHOULDER], W[:, R_SHOULDER]))
    hip_yaw = K.unwrap_deg(K.yaw_deg(W[:, L_HIP], W[:, R_HIP]))
    sh_rot = _rotation_from_coil(sh_yaw, ct.times, c)
    hip_rot = _rotation_from_coil(hip_yaw, ct.times, c)
    if sh_rot is None:
        missing.append("Shoulder rotation: shoulders were not tracked confidently through the backswing.")
    if hip_rot is None:
        missing.append("Hip rotation: hips were not tracked confidently through the backswing.")
    separation = None
    if sh_rot is not None:
        ci = sh_rot[1]
        separation = _f(abs(_wrap180(sh_yaw[ci] - hip_yaw[ci])))

    # Contact in front: hitting wrist ahead of the hip centre along the direction the player faces
    # between swings (toward the target), in cm.
    in_front = None
    hip_mid = (W[c, L_HIP] + W[c, R_HIP]) / 2
    rel = W[c, dom_w] - hip_mid
    if forward is not None and np.all(np.isfinite(rel)):
        in_front = float(np.dot(rel, forward) * 100.0)
    if in_front is None:
        missing.append("Contact point: hitting wrist or hips were not tracked confidently at contact"
                       if forward is not None else
                       "Contact point: couldn't tell which way the player faces between swings.")

    hip_lead = None
    if sh_rot is not None and hip_rot is not None:
        th = _half_turn_time(hip_yaw, ct.times, c, hip_rot[1])
        ts = _half_turn_time(sh_yaw, ct.times, c, sh_rot[1])
        if th is not None and ts is not None:
            hip_lead = (ts - th) * 1000.0
    if hip_lead is None:
        missing.append("Hip-shoulder timing: hips or shoulders were not tracked through the forward swing.")

    fwd = None
    ws = K.speed(ct.xy[:, dom_w], ct.times)
    w = np.flatnonzero((ct.times >= ct.times[c] - COIL_SEARCH_S) & (ct.times <= ct.times[c] - 0.1))
    if len(w) >= 3 and np.isfinite(ws[w]).sum() >= 3:
        fwd = (ct.times[c] - ct.times[w[np.nanargmin(ws[w])]]) * 1000.0

    wrist_speed = None
    lo, hi = max(c - 2, 0), min(c + 3, len(ct.times))
    if hi - lo >= 3:
        rel_w = W[lo:hi, dom_w] - (W[lo:hi, L_HIP] + W[lo:hi, R_HIP]) / 2
        sp = K.speed(rel_w, ct.times[lo:hi])
        wrist_speed = _f(sp[c - lo])

    return ContactMetrics(
        elbow_angle_deg=elbow, elbow_arm=elbow_arm, knee_bend_deg=_f(knee), left_knee_bend_deg=knees["left"],
        right_knee_bend_deg=knees["right"],
        shoulder_rotation_deg=None if sh_rot is None else sh_rot[0],
        hip_rotation_deg=None if hip_rot is None else hip_rot[0],
        separation_at_coil_deg=separation, contact_in_front_cm=_f(in_front),
        hip_lead_ms=_f(hip_lead), forward_swing_ms=_f(fwd), wrist_speed_mps=wrist_speed, missing=missing,
    )
