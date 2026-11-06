"""Coaching guideline ranges used to rank fixes. These are general groundstroke guidelines, not
measurements. They're the only hand-set numbers in the analyzer, and you can edit them here.

Contact here means the frame of peak hitting-wrist speed. Shadow swings have no ball, so that frame
stands in for impact.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    tolerance: float  # how far outside the range counts as one "unit" of severity
    spread_limit: float  # swing-to-swing range above this is flagged as inconsistent
    fmt: str = "{:.0f}"
    physical_max: float | None = None  # a range end at this value is a body limit, not a coaching edge

    def show(self, v: float) -> str:
        if self.key == "hip_lead_ms":  # signed timing reads better in words
            return f"hips first by {v:.0f} ms" if v >= 0 else f"shoulders first by {abs(v):.0f} ms"
        return self.fmt.format(v) + self.unit

    def show_range(self, lo: float, hi: float) -> str:
        return f"{self.fmt.format(lo)}–{self.fmt.format(hi)}{self.unit}"


METRICS: dict[str, Metric] = {m.key: m for m in [
    Metric("elbow_angle_deg", "Elbow angle", "°", 10, 35, physical_max=180),  # 180 = fully straight
    Metric("knee_bend_deg", "Knee bend", "°", 10, 35),
    Metric("shoulder_rotation_deg", "Shoulder rotation", "°", 10, 40),
    Metric("hip_rotation_deg", "Hip rotation", "°", 10, 40),
    Metric("contact_in_front_cm", "Contact in front", " cm", 8, 30),
    Metric("hip_lead_ms", "Hip lead", " ms", 25, 90),
]}
# Measured and shown, but not judged (no well-defined guideline for shadow swings):
INFO_METRICS = {
    "separation_at_coil_deg": ("Hip-shoulder separation at coil", "°"),
    "forward_swing_ms": ("Forward swing time", " ms"),
    "wrist_speed_mps": ("Hand speed at contact", " m/s"),
}

# (low, high) per stroke style. A missing key means "not judged for this stroke".
RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "forehand": {
        "elbow_angle_deg": (100, 180),  # double-bend to straight-arm; below 100° is cramped
        "knee_bend_deg": (15, 60),  # loaded but driving up through contact
        "shoulder_rotation_deg": (60, 120),  # unit turn unwinding through contact
        "hip_rotation_deg": (30, 80),
        "contact_in_front_cm": (15, 60),  # hand ahead of the hips toward the target
        "hip_lead_ms": (0, 120),  # hips start the unwind before the shoulders
    },
    "backhand_two_handed": {
        "elbow_angle_deg": (100, 180),
        "knee_bend_deg": (15, 60),
        "shoulder_rotation_deg": (50, 110),
        "hip_rotation_deg": (30, 80),
        "contact_in_front_cm": (15, 60),
        "hip_lead_ms": (0, 120),
    },
    "backhand_one_handed": {
        "elbow_angle_deg": (150, 180),  # one-hander hits with a straight arm
        "knee_bend_deg": (15, 60),
        "shoulder_rotation_deg": (50, 110),
        "hip_rotation_deg": (20, 70),
        "contact_in_front_cm": (20, 70),  # one-hander meets the ball further in front
        "hip_lead_ms": (0, 120),
    },
}


def style_key(stroke: str, two_handed: bool | None) -> str | None:
    if stroke == "forehand":
        return "forehand"
    if stroke == "backhand":
        return "backhand_one_handed" if two_handed is False else "backhand_two_handed"
    return None


def style_label(key: str) -> str:
    return {"forehand": "forehands", "backhand_two_handed": "two-handed backhands",
            "backhand_one_handed": "one-handed backhands"}[key]
