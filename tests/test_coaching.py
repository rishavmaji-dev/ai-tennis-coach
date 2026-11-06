"""Fix ranking/wording rules on hand-built sessions (logic tests; the product never uses these numbers)."""
import re

from tennis_analyzer.coaching import N_FIXES, best_swing, choose_references, make_fixes
from tennis_analyzer.metrics import ContactMetrics
from tennis_analyzer.pipeline import SessionResult, SwingResult
from tennis_analyzer.video_io import VideoInfo

INFO = VideoInfo(path="x.mp4", codec="h264", width=1080, height=1920, rotation=0, frame_count=600,
                 duration_s=10, nominal_fps=60, average_fps=60, capture_fps=None)


def metrics(**kw):
    base = dict(elbow_angle_deg=150, elbow_arm="right", knee_bend_deg=35, left_knee_bend_deg=30,
                right_knee_bend_deg=35, shoulder_rotation_deg=85, hip_rotation_deg=50, separation_at_coil_deg=30,
                contact_in_front_cm=30, hip_lead_ms=40, forward_swing_ms=500, wrist_speed_mps=6, missing=[])
    base.update(kw)
    return ContactMetrics(**base)


def swing(n, **kw):
    return SwingResult(number=n, stroke="forehand", two_handed=False, contact_frame=60 * n, contact_time=float(n),
                       peak_speed=8, side_score=0.5, metrics=metrics(**kw))


def session(swings):
    return SessionResult(video=INFO, handedness="right", mode="accurate", swings=swings, forward=[1, 0, 0])


def test_three_distinct_fixes_each_citing_value_and_frame():
    s = session([swing(1, knee_bend_deg=5, contact_in_front_cm=2, hip_lead_ms=-40),
                 swing(2, knee_bend_deg=8, contact_in_front_cm=5, hip_lead_ms=-30, elbow_angle_deg=90),
                 swing(3, knee_bend_deg=6, contact_in_front_cm=4, hip_lead_ms=-35)])
    refs, _ = choose_references(s)
    fixes, note = make_fixes(s, refs)
    assert len(fixes) == N_FIXES and note is None
    assert len({f.metric for f in fixes}) == N_FIXES
    for f in fixes:
        assert re.search(r"frame \d+ \(\d+:\d\d\.\d\d\)", f.detail), f.detail
        assert re.search(r"\d+(°| cm| ms)", f.detail), f.detail


def test_fewer_than_three_when_everything_is_in_range_and_consistent():
    s = session([swing(1), swing(2), swing(3)])
    fixes, note = make_fixes(s, choose_references(s)[0])
    assert len(fixes) < N_FIXES and "Only" in note


def test_missing_metrics_cannot_make_a_swing_best():
    hidden = swing(1, elbow_angle_deg=None, contact_in_front_cm=None, hip_lead_ms=None)
    full = swing(2, knee_bend_deg=12)  # slightly out of range but fully measured
    assert best_swing([hidden, full], "forehand").number == 2


def test_single_swing_has_no_own_reference():
    refs, notes = choose_references(session([swing(1)]))
    assert refs == {} and any("Only one" in n for n in notes)


def test_straight_arm_is_never_told_to_bend():
    one_hander = [SwingResult(number=n, stroke="backhand", two_handed=False, contact_frame=60 * n,
                              contact_time=float(n), peak_speed=8, side_score=-0.3,
                              metrics=metrics(elbow_angle_deg=v)) for n, v in ((1, 175), (2, 176), (3, 177))]
    forehands = [swing(n, elbow_angle_deg=v) for n, v in ((4, 165), (5, 166), (6, 167))]
    s = session(one_hander + forehands)
    fixes, _ = make_fixes(s, choose_references(s)[0])
    assert not any(f.metric == "elbow_angle_deg" for f in fixes)
    assert not any("flex" in f.title.lower() or "bend" in f.detail.lower() and "elbow" in f.detail.lower()
                   for f in fixes)


def test_cramped_elbow_near_low_edge_still_flagged():
    s = session([swing(n, elbow_angle_deg=v) for n, v in ((1, 104), (2, 105), (3, 106))])
    fixes, _ = make_fixes(s, choose_references(s)[0])
    assert any(f.metric == "elbow_angle_deg" and f.kind == "near_edge" for f in fixes)
