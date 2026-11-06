"""End-to-end analysis of one clip. The CLI and the Streamlit app both call analyze_video()."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .metrics import HEAVY_AFTER_S, HEAVY_BEFORE_S, ContactMetrics, measure, refine_contact
from .pose import PoseTrack, track_pose
from .swings import CleanTrack, Swing, clean_track, facing_direction, find_swings, hand_speed, wrist_ids
from .video_io import VideoError, VideoInfo, probe

HEAVY_WINDOW_S = (-HEAVY_BEFORE_S, HEAVY_AFTER_S)  # around each Lite contact: full coil search to just after contact
MIN_DETECTED_FRACTION = 0.05
WARN_DETECTED_FRACTION = 0.6

ProgressFn = Callable[[float, str], None]


class AnalysisError(Exception):
    """Plain-language failure the UI can show as-is."""


@dataclass
class SwingResult:
    number: int
    stroke: str
    two_handed: bool | None
    contact_frame: int  # 0-based index of the decoded frame in the source clip
    contact_time: float  # seconds
    peak_speed: float  # torso lengths / s at contact (Lite, 2D)
    side_score: float
    metrics: ContactMetrics

    @property
    def label(self) -> str:
        return f"#{self.number} {self.stroke}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = self.metrics.to_dict()
        return d


@dataclass
class SessionResult:
    video: VideoInfo
    handedness: str
    mode: str  # "accurate" (Lite + Heavy) or "fast" (Full)
    swings: list[SwingResult]
    forward: list[float] | None
    warnings: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    # Kept for rendering, not serialized:
    track: PoseTrack | None = None  # whole-clip track (Lite or Full)
    windows: dict[int, PoseTrack] = field(default_factory=dict)  # swing number -> Heavy window track

    def to_dict(self) -> dict:
        return {
            "video": self.video.to_dict(), "handedness": self.handedness, "mode": self.mode,
            "swings": [s.to_dict() for s in self.swings], "forward": self.forward,
            "warnings": self.warnings, "timings_s": self.timings_s,
        }

    def by_number(self, n: int) -> SwingResult:
        return next(s for s in self.swings if s.number == n)


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def frame_ref(frame: int, t: float) -> str:
    """How every tip/UI line cites a frame."""
    return f"frame {frame} ({_fmt_t(t)})"


def _handedness_warning(ct: CleanTrack, swings: list[Swing], handedness: str) -> str | None:
    """Warn when the swings look like they came from a player of the other hand.

    One-handed swings: within +/-0.3 s of contact the hitting wrist reaches the higher peak speed. Measured on
    the samples, the other/chosen ratio is 0.36-0.43 with the correct setting and 1.60-2.35 with the wrong one.
    Two-handed swings: there the top hand legitimately moves as fast or faster, so they vote by label instead.
    A flipped setting turns two-handed backhands into two-handed forehands, which are rare.
    """
    from scipy.ndimage import median_filter
    from . import kinematics as K
    dom, other = wrist_ids(handedness)

    def robust_speed(j: int) -> np.ndarray:
        return median_filter(np.nan_to_num(K.speed(ct.xy[:, j], ct.times) / ct.torso_px, nan=0.0), 5)

    s_dom, s_oth = robust_speed(dom), robust_speed(other)
    votes = faster = two_hand_fh = 0
    for s in swings:
        if s.two_handed:
            if s.stroke in ("forehand", "backhand"):
                votes += 1
                two_hand_fh += s.stroke == "forehand"
            continue
        w = np.abs(ct.times - s.contact_time) <= 0.3
        ratio = s_oth[w].max() / max(s_dom[w].max(), 1e-6)
        if ratio > 1.25:
            votes += 1
            faster += 1
        elif ratio < 0.8:
            votes += 1
    wrong = faster + two_hand_fh
    if votes < 2 or wrong / votes <= 0.5:
        return None
    other_hand = "left" if handedness == "right" else "right"
    reasons = []
    if faster:
        reasons.append(f"on {faster} one-handed swings the {other_hand} wrist moved fastest")
    if two_hand_fh:
        reasons.append(f"{two_hand_fh} came out as two-handed forehands, which are rare")
    return (f"These swings look {other_hand}-handed: " + "; ".join(reasons) + ". "
            f"If the player is {other_hand}-handed, switch the setting and analyze again. Stroke labels, arm "
            "measurements and fixes all depend on it.")


def analyze_video(path: str | Path, handedness: str, fast: bool = False,
                  progress: ProgressFn | None = None, use_cache: bool = True) -> SessionResult:
    if handedness not in ("right", "left"):
        raise AnalysisError("Choose right- or left-handed.")
    progress = progress or (lambda f, m: None)
    timings: dict[str, float] = {}
    t0 = time.time()
    try:
        info = probe(path)
    except VideoError as e:
        raise AnalysisError(str(e)) from e
    if info.duration_s and info.duration_s < 1.0:
        raise AnalysisError(f"The clip is only {info.duration_s:.1f} s long. Upload at least one full swing.")

    first_model = "full" if fast else "lite"
    pass_share = 1.0 if fast else 0.6
    try:
        track = track_pose(path, info, first_model, max_side=960, use_cache=use_cache,
                           progress=lambda f, m: progress(f * pass_share, m))
    except VideoError as e:
        raise AnalysisError(str(e)) from e
    timings["pose_pass1"] = time.time() - t0

    # Frame rate as actually recorded (timestamps), shown in the UI next to the container value.
    dts = np.diff(track.times)
    dts = dts[dts > 0]
    if len(dts):
        info.measured_fps = float(1.0 / np.median(dts))
        info.variable_frame_rate = bool(np.std(dts) > 0.1 * np.median(dts) or
                                        abs(info.average_fps - info.nominal_fps) > 0.2)
    info.frame_count = len(track)

    detected = float(track.detected.mean())
    warnings: list[str] = []
    if detected < MIN_DETECTED_FRACTION:
        raise AnalysisError("No player was found in the video. Film the whole body, side-on or face-on, "
                            "with the player filling at least a third of the frame height.")
    if detected < WARN_DETECTED_FRACTION:
        warnings.append(f"The player was only tracked in {detected:.0%} of frames. Swings in the gaps are missed.")

    ct = clean_track(track)
    if not np.isfinite(ct.torso_px):
        raise AnalysisError("Shoulders and hips were never tracked clearly, so swings can't be measured. "
                            "Make sure the player's whole body is in frame.")
    swings = find_swings(ct, handedness)
    if not swings:
        peak = float(np.nanmax(hand_speed(ct, handedness))) if len(ct.times) else 0.0
        raise AnalysisError(
            f"No swings found. The fastest hand movement was {peak:.1f} torso lengths per second; a swing "
            "needs at least 3.5. Check that the clip shows full-speed groundstrokes and the hitting arm is visible.")
    hw = _handedness_warning(ct, swings, handedness)
    if hw:
        warnings.append(hw)
    forward = facing_direction(ct, handedness)

    results: list[SwingResult] = []
    windows: dict[int, PoseTrack] = {}
    t1 = time.time()
    for k, s in enumerate(swings):
        if fast:
            win_ct, c_win = ct, s.contact_idx
        else:
            progress(pass_share + (1 - pass_share) * k / len(swings),
                     f"Measuring swing {k + 1} of {len(swings)} (heavy model)")
            try:
                wtrack = track_pose(path, info, "heavy", max_side=1280, use_cache=use_cache,
                                    start_s=max(0.0, s.contact_time + HEAVY_WINDOW_S[0]),
                                    end_s=s.contact_time + HEAVY_WINDOW_S[1])
            except VideoError as e:
                raise AnalysisError(str(e)) from e
            win_ct = clean_track(wtrack)
            win_ct.torso_px = ct.torso_px  # whole-clip body scale
            c_win = refine_contact(win_ct, int(np.argmin(np.abs(win_ct.times - s.contact_time))), handedness,
                                   s.two_handed)
            windows[s.number] = wtrack
        m = measure(win_ct, c_win, handedness, forward, s.two_handed)
        t_contact = float(win_ct.times[c_win])
        frame = int(np.argmin(np.abs(track.times - t_contact)))
        results.append(SwingResult(number=s.number, stroke=s.stroke, two_handed=s.two_handed,
                                   contact_frame=frame, contact_time=t_contact, peak_speed=s.peak_speed,
                                   side_score=s.side_score, metrics=m))
    timings["measure"] = time.time() - t1
    if forward is None:
        warnings.append("Couldn't tell which way the player faces between swings, so contact distance in front "
                        "isn't measured.")
    unknown = [s.number for s in results if s.stroke == "unknown"]
    if unknown:
        warnings.append(f"Swing(s) {', '.join(f'#{n}' for n in unknown)} couldn't be labeled forehand or backhand: "
                        "the hitting wrist stayed close to the body's midline.")
    timings["total"] = time.time() - t0
    progress(1.0, "Analysis complete")
    return SessionResult(video=info, handedness=handedness, mode="fast" if fast else "accurate",
                         swings=results, forward=None if forward is None else [float(v) for v in forward],
                         warnings=warnings, timings_s=timings, track=track, windows=windows)
