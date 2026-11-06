"""Reference-swing choice and the session's 3 fixes. Every fix cites measured values and frames."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .pipeline import SessionResult, SwingResult, frame_ref
from .targets import METRICS, RANGES, style_key, style_label

N_FIXES = 3
MIN_JUDGED_FOR_BEST = 4  # a "best swing" must have at least this many judged metrics measured


@dataclass
class Reference:
    source: str  # "own_best" | "uploaded"
    style: str
    swing: SwingResult
    session: SessionResult  # the session the swing belongs to (for rendering)

    @property
    def describe(self) -> str:
        who = "your best" if self.source == "own_best" else "reference clip"
        return (f"{who} {style_label(self.style)[:-1]} (swing #{self.swing.number}, "
                f"{frame_ref(self.swing.contact_frame, self.swing.contact_time)})")

    def to_dict(self) -> dict:
        return {"source": self.source, "style": self.style, "swing_number": self.swing.number,
                "frame": self.swing.contact_frame, "time": self.swing.contact_time,
                "video": self.session.video.path}


@dataclass
class Fix:
    rank: int
    metric: str
    style: str
    kind: str  # "out_of_range" | "inconsistent" | "near_edge"
    title: str
    detail: str
    session_value: float  # median over the style's swings
    swing_number: int  # the swing the tip points at
    frame: int
    time: float
    swing_value: float
    target: tuple[float, float]
    reference_value: float | None
    severity: float

    def to_dict(self) -> dict:
        return asdict(self)


def groups(session: SessionResult) -> dict[str, list[SwingResult]]:
    out: dict[str, list[SwingResult]] = {}
    for s in session.swings:
        k = style_key(s.stroke, s.two_handed)
        if k:
            out.setdefault(k, []).append(s)
    return out


def _value(s: SwingResult, key: str) -> float | None:
    return getattr(s.metrics, key)


def _outside(v: float, lo: float, hi: float) -> float:
    return lo - v if v < lo else (v - hi if v > hi else 0.0)


def swing_deviation(s: SwingResult, style: str) -> tuple[float, int]:
    """(total severity in tolerance units, number of judged metrics measured).

    An unmeasured metric counts as one unit, so a swing can't look 'best' by hiding joints.
    """
    total, n = 0.0, 0
    for key, (lo, hi) in RANGES[style].items():
        v = _value(s, key)
        if v is None:
            total += 1.0
            continue
        n += 1
        total += _outside(v, lo, hi) / METRICS[key].tolerance
    return total, n


def best_swing(swings: list[SwingResult], style: str) -> SwingResult | None:
    scored = [(swing_deviation(s, style), s) for s in swings]
    scored = [((dev, -s.peak_speed), s) for (dev, n), s in scored if n >= MIN_JUDGED_FOR_BEST]
    if not scored:
        return None
    return min(scored, key=lambda t: t[0])[1]


def choose_references(session: SessionResult, ref_session: SessionResult | None = None) -> tuple[dict[str, Reference], list[str]]:
    """Per stroke style: best swing from an uploaded reference clip, else the player's own best swing."""
    refs: dict[str, Reference] = {}
    notes: list[str] = []
    ref_groups = groups(ref_session) if ref_session else {}
    for style, swings in groups(session).items():
        if style in ref_groups:
            b = best_swing(ref_groups[style], style)
            if b:
                refs[style] = Reference("uploaded", style, b, ref_session)
                continue
            notes.append(f"The reference clip's {style_label(style)} weren't measured fully enough to compare.")
        elif ref_session is not None:
            notes.append(f"The reference clip has no {style_label(style)}; using your own best one instead.")
        if len(swings) < 2:
            notes.append(f"Only one {style_label(style)[:-1]} in this clip, so there's no own-best swing to "
                         "compare it with. Upload a reference clip to get a side-by-side.")
            continue
        b = best_swing(swings, style)
        if b:
            refs[style] = Reference("own_best", style, b, session)
    return refs, notes


# ---- tip wording -------------------------------------------------------------------------------

def _arm(s: SwingResult) -> str:
    return f"{s.metrics.elbow_arm} " if s.metrics.elbow_arm else ""


TIPS = {
    ("elbow_angle_deg", "low"): ("Give the ball room so your arm can extend",
        "Your {arm}elbow was bent to {med} at contact (median of {n} {style}). Swing #{k} at {fref} was the most "
        "cramped at {val}. Aim for {rng}: set up a step farther from the contact point."),
    ("elbow_angle_deg", "high"): ("Keep a little flex in the hitting arm",
        "Your {arm}elbow measured {med} at contact (median of {n} {style}); swing #{k} at {fref} reached {val}. "
        "Aim for {rng}."),
    ("knee_bend_deg", "low"): ("Load your legs before contact",
        "Your more-bent knee was only {med} at contact (median of {n} {style}). Swing #{k} at {fref} was the "
        "straightest at {val}. Bend to {rng} so the legs can drive the swing."),
    ("knee_bend_deg", "high"): ("Drive up through contact",
        "Your knees were still bent {med} at contact (median of {n} {style}). Swing #{k} at {fref} was the "
        "deepest at {val}. Push up into the ball: aim for {rng} of knee bend at contact."),
    ("shoulder_rotation_deg", "low"): ("Turn your shoulders more",
        "Your shoulders rotated {med} from the coiled backswing to contact (median of {n} {style}). Swing #{k} "
        "at {fref} turned only {val}. Aim for {rng}: show your back to the target on the takeback, then "
        "unwind fully."),
    ("shoulder_rotation_deg", "high"): ("Stop over-spinning the shoulders",
        "Your shoulders rotated {med} from coil to contact (median of {n} {style}); swing #{k} at {fref} turned "
        "{val}. Over-rotating drags the racket across the ball. Aim for {rng}."),
    ("hip_rotation_deg", "low"): ("Use your hips",
        "Your hips rotated {med} from the coiled backswing to contact (median of {n} {style}). Swing #{k} at "
        "{fref} turned only {val}. Aim for {rng}: let the hips open first and pull the arm through."),
    ("hip_rotation_deg", "high"): ("Keep the hips more stable",
        "Your hips rotated {med} from coil to contact (median of {n} {style}); swing #{k} at {fref} turned "
        "{val}. Opening that far leaks power sideways. Aim for {rng}."),
    ("contact_in_front_cm", "low"): ("Meet the ball farther in front",
        "At contact your hand was {med} ahead of your hips (median of {n} {style}). Swing #{k} at {fref} was "
        "the latest at {val}. Aim for {rng} in front so the racket meets the ball before it gets beside you."),
    ("contact_in_front_cm", "high"): ("Let the ball come to you",
        "At contact your hand was {med} ahead of your hips (median of {n} {style}); swing #{k} at {fref} "
        "reached {val}. That's reaching. Aim for {rng}."),
    ("hip_lead_ms", "low"): ("Start the forward swing with your hips",
        "{lead_text} (median of {n} {style}). Swing #{k} at {fref}: {val}. Let the hips lead by {rng} so "
        "the shoulders and arm whip through after them."),
    ("hip_lead_ms", "high"): ("Bring the shoulders through sooner",
        "{lead_text} (median of {n} {style}); swing #{k} at {fref}: {val}. Too long a gap stalls the swing. "
        "Aim for {rng}."),
}


SPREAD_CUE = {
    "elbow_angle_deg": "Set up the same distance from the ball every time. ",
    "knee_bend_deg": "Use the same leg load on every swing. ",
    "shoulder_rotation_deg": "Make the same full unit turn on every backswing. ",
    "hip_rotation_deg": "Open the hips the same amount on every swing. ",
    "contact_in_front_cm": "Meet the ball at the same spot in front every time. ",
    "hip_lead_ms": "Start every forward swing the same way: hips first. ",
}


def _lead_text(med: float) -> str:
    if med < 0:
        return f"Your shoulders reached half their turn {abs(med):.0f} ms before your hips"
    return f"Your hips led your shoulders by only {med:.0f} ms" if med < 60 else \
        f"Your hips led your shoulders by {med:.0f} ms"


def _candidates(session: SessionResult, refs: dict[str, Reference]) -> list[Fix]:
    cands: list[Fix] = []
    for style, swings in groups(session).items():
        for key, (lo, hi) in RANGES[style].items():
            M = METRICS[key]
            vals = [(s, _value(s, key)) for s in swings if _value(s, key) is not None]
            if not vals:
                continue
            v = np.array([x for _, x in vals])
            med = float(np.median(v))
            ref = refs.get(style)
            ref_val = _value(ref.swing, key) if ref else None
            base = dict(metric=key, style=style, session_value=med, target=(lo, hi), reference_value=ref_val)
            out = _outside(med, lo, hi)
            if out > 0:
                direction = "low" if med < lo else "high"
                s_k, v_k = (min if direction == "low" else max)(vals, key=lambda t: t[1])
                cands.append(Fix(rank=0, kind="out_of_range", title="", detail="", swing_number=s_k.number,
                                 frame=s_k.contact_frame, time=s_k.contact_time, swing_value=v_k,
                                 severity=1.0 + out / M.tolerance, **base))
                cands[-1].detail = _render(cands[-1], direction, len(vals), s_k, ref)
                continue
            spread = float(v.max() - v.min())
            if len(vals) >= 3 and spread > M.spread_limit:
                s_k, v_k = max(vals, key=lambda t: abs(t[1] - med))
                cands.append(Fix(rank=0, kind="inconsistent", title="", detail="", swing_number=s_k.number,
                                 frame=s_k.contact_frame, time=s_k.contact_time, swing_value=v_k,
                                 severity=0.5 + (spread - M.spread_limit) / M.tolerance * 0.5, **base))
                cands[-1].detail = _render_spread(cands[-1], vals, ref)
                continue
            # Only warn about range ends that are coaching limits. A straight arm (180°) is never "too straight".
            margin_lo = (med - lo) / (hi - lo)
            margin_hi = (hi - med) / (hi - lo) if M.physical_max is None or hi < M.physical_max else np.inf
            margin = min(margin_lo, margin_hi)
            if margin < 0.2:
                direction = "low" if margin_lo <= margin_hi else "high"
                s_k, v_k = (min if direction == "low" else max)(vals, key=lambda t: t[1])
                cands.append(Fix(rank=0, kind="near_edge", title="", detail="", swing_number=s_k.number,
                                 frame=s_k.contact_frame, time=s_k.contact_time, swing_value=v_k,
                                 severity=0.4 * (0.2 - margin) / 0.2, **base))
                cands[-1].detail = _render(cands[-1], direction, len(vals), s_k, ref, near_edge=True)
    return cands


def _ref_suffix(fix: Fix, ref: Reference | None) -> str:
    if ref is None or fix.reference_value is None:
        return ""
    if ref.source == "own_best" and ref.swing.number == fix.swing_number:
        return f" This is also your best-measured {style_label(fix.style)[:-1]} overall."
    return f" Reference, {ref.describe}: {METRICS[fix.metric].show(fix.reference_value)}."


def _render(fix: Fix, direction: str, n: int, s_k: SwingResult, ref: Reference | None,
            near_edge: bool = False) -> str:
    M = METRICS[fix.metric]
    title, body = TIPS[(fix.metric, direction)]
    lo, hi = fix.target
    text = body.format(arm=_arm(s_k), med=M.show(fix.session_value), n=n, style=style_label(fix.style),
                       k=fix.swing_number, fref=frame_ref(fix.frame, fix.time), val=M.show(fix.swing_value),
                       rng=M.show_range(lo, hi), lead_text=_lead_text(fix.session_value))
    if near_edge:
        title = f"{title} (you're at the edge of the range)"
        text = (f"Inside the {M.show_range(lo, hi)} guideline, but close to the {direction} end. " + text)
    fix.title = title
    return text + _ref_suffix(fix, ref)


def _render_spread(fix: Fix, vals: list, ref: Reference | None) -> str:
    M = METRICS[fix.metric]
    (s_lo, v_lo), (s_hi, v_hi) = min(vals, key=lambda t: t[1]), max(vals, key=lambda t: t[1])
    fix.title = f"Make your {M.label.lower()} repeatable"
    text = (f"Across {len(vals)} {style_label(fix.style)} your {M.label.lower()} at contact ranged from "
            f"{M.show(v_lo)} (swing #{s_lo.number}, {frame_ref(s_lo.contact_frame, s_lo.contact_time)}) to "
            f"{M.show(v_hi)} (swing #{s_hi.number}, {frame_ref(s_hi.contact_frame, s_hi.contact_time)}), a "
            f"{M.show(v_hi - v_lo)} spread. {SPREAD_CUE.get(fix.metric, '')}The median {M.show(fix.session_value)} "
            f"is inside the {M.show_range(*fix.target)} guideline, so repeat that.")
    return text + _ref_suffix(fix, ref)


def make_fixes(session: SessionResult, refs: dict[str, Reference]) -> tuple[list[Fix], str | None]:
    """Exactly N_FIXES fixes when the data supports it, each on a different metric."""
    cands = sorted(_candidates(session, refs), key=lambda f: -f.severity)
    chosen: list[Fix] = []
    used: set[str] = set()
    for f in cands:
        if f.metric in used:
            continue
        chosen.append(f)
        used.add(f.metric)
        if len(chosen) == N_FIXES:
            break
    for i, f in enumerate(chosen, 1):
        f.rank = i
    note = None
    if len(chosen) < N_FIXES:
        measured = {k for s in session.swings for k in METRICS if _value(s, k) is not None}
        note = (f"Only {len(chosen)} fix{'es' if len(chosen) != 1 else ''} this session: the other measured "
                f"metrics ({', '.join(METRICS[k].label.lower() for k in sorted(measured - used)) or 'none'}) are "
                "comfortably inside their guideline ranges and consistent from swing to swing.")
    return chosen, note
