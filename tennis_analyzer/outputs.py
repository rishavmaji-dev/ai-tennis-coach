"""Rendered outputs: annotated full clip, contact stills, and side-by-side comparison (all H.264)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .pose import L_ELBOW, L_KNEE, L_WRIST, R_ELBOW, R_KNEE, R_WRIST
from .render import draw_skeleton, put_label
from .video_io import H264Writer, iter_frames, prefetch

OUT_MAX_SIDE = 1280
COMPARE_HEIGHT = 960
COMPARE_WINDOW_S = (-1.0, 0.6)
COMPARE_SLOWMO = 2.0  # comparison plays at half speed
CONTACT_HOLD_S = 0.5  # freeze on the synced contact frame
PANEL_AFTER_S = 1.2  # metrics panel stays up this long after contact
ACCENT = (255, 220, 90)

ProgressFn = Callable[[float, str], None]


def best_landmarks(session) -> np.ndarray:
    """Whole-clip landmarks (N,33,5): Heavy-window landmarks where available, else the first pass."""
    lm = session.track.img.copy()
    times = session.track.times
    for w in session.windows.values():
        idx = np.clip(np.searchsorted(times, w.times), 0, len(times) - 1)
        prev = np.clip(idx - 1, 0, len(times) - 1)
        idx = np.where(np.abs(times[prev] - w.times) < np.abs(times[idx] - w.times), prev, idx)
        ok = np.abs(times[idx] - w.times) < 0.004
        lm[idx[ok]] = w.img[ok]
    return lm


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def metric_lines(m) -> list[str]:
    def f(v, unit):
        return "-" if v is None else f"{v:.0f}{unit}"
    arm = f" ({m.elbow_arm})" if m.elbow_arm else ""
    lead = "-" if m.hip_lead_ms is None else (
        f"hips first {m.hip_lead_ms:.0f} ms" if m.hip_lead_ms >= 0 else f"shoulders first {abs(m.hip_lead_ms):.0f} ms")
    return [f"Elbow{arm}: {f(m.elbow_angle_deg, ' deg')}",
            f"Knee bend: {f(m.knee_bend_deg, ' deg')}",
            f"Shoulder turn: {f(m.shoulder_rotation_deg, ' deg')}",
            f"Hip turn: {f(m.hip_rotation_deg, ' deg')}",
            f"In front: {f(m.contact_in_front_cm, ' cm')}",
            f"Timing: {lead}"]


def _panel(rgb: np.ndarray, lines: list[str], org: tuple[int, int], title: str | None = None) -> None:
    s = max(0.45, rgb.shape[0] / 1700)
    th = max(1, int(round(s * 2)))
    lh = int(32 * s * 1.25)
    rows = ([title] if title else []) + lines
    width = max(cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, s, th)[0][0] for r in rows) + 16
    x, y = org
    x2, y2 = min(x + width, rgb.shape[1] - 1), min(y + lh * len(rows) + 10, rgb.shape[0] - 1)
    roi = rgb[y:y2, x:x2]
    roi[:] = (roi * 0.4).astype(np.uint8)
    for i, r in enumerate(rows):
        c = ACCENT if (title and i == 0) else (255, 255, 255)
        cv2.putText(rgb, r, (x + 8, y + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)


def _angle_tags(rgb: np.ndarray, lm: np.ndarray, scale: float, m) -> None:
    """Write the measured elbow and knee angles next to those joints."""
    if m.elbow_arm and m.elbow_angle_deg is not None:
        j = R_ELBOW if m.elbow_arm == "right" else L_ELBOW
        if lm[j, 3] >= 0.5:
            put_label(rgb, f"elbow {m.elbow_angle_deg:.0f} deg",
                      (int(lm[j, 0] * scale) + 12, int(lm[j, 1] * scale)), color=ACCENT)
    knees = [(bend, j) for bend, j in ((m.right_knee_bend_deg, R_KNEE), (m.left_knee_bend_deg, L_KNEE))
             if bend is not None and lm[j, 3] >= 0.5]
    leftmost = min((j for _, j in knees), key=lambda j: lm[j, 0], default=None) if len(knees) == 2 else None
    for bend, j in knees:
        text = f"bend {bend:.0f} deg"
        x, y = int(lm[j, 0] * scale), int(lm[j, 1] * scale)
        if j == leftmost:  # put the left knee's label on its left so the two don't overlap
            s = max(0.5, rgb.shape[0] / 1400)
            x -= cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, max(1, int(round(s * 2))))[0][0] + 16
        else:
            x += 12
        put_label(rgb, text, (x, y), color=ACCENT)


def render_annotated(session, path: Path, progress: ProgressFn | None = None) -> Path:
    """Whole clip with skeleton, frame counter, swing labels, contact marker and per-swing metrics."""
    progress = progress or (lambda f, m: None)
    info = session.video
    lm_all = best_landmarks(session)
    times = session.track.times
    hit = R_WRIST if session.handedness == "right" else L_WRIST
    n = len(times)
    writer = None
    t0 = 0.0
    try:
        for i, fr in enumerate(prefetch(iter_frames(info.path, info.rotation, max_side=OUT_MAX_SIDE))):
            rgb = fr.rgb
            if writer is None:
                writer = H264Writer(path, rgb.shape[1], rgb.shape[0], info.nominal_fps)
                t0 = fr.time
            h = rgb.shape[0]
            scale = h / info.height
            k = i if i < n and abs(times[i] - fr.time) < 0.004 else int(np.argmin(np.abs(times - fr.time)))
            near = [s for s in session.swings if -0.8 <= fr.time - s.contact_time <= PANEL_AFTER_S]
            draw_skeleton(rgb, lm_all[k], scale, highlight=(hit,) if near else ())
            put_label(rgb, f"frame {k}  {_fmt_t(fr.time)}", (12, h - 20))
            if near:
                s = near[0]
                put_label(rgb, f"Swing #{s.number}: {s.stroke}{' (two-handed)' if s.two_handed else ''}", (12, 40),
                          color=ACCENT)
                if abs(fr.time - s.contact_time) <= 0.1:
                    put_label(rgb, f"CONTACT  frame {s.contact_frame}", (12, 40 + h // 22), color=(20, 20, 20),
                              bg=ACCENT)
                if 0 <= fr.time - s.contact_time <= PANEL_AFTER_S:
                    _panel(rgb, metric_lines(s.metrics), (12, 40 + h // 11), title=f"At contact (frame {s.contact_frame})")
            writer.write(rgb, fr.time - t0)
            if i % 30 == 0:
                progress(min(i / max(n, 1), 1.0), f"Rendering annotated video - frame {i} of {n}")
    finally:
        if writer is not None:
            writer.close()
    return path


def render_contact_stills(session, out_dir: Path) -> dict[int, Path]:
    """One JPEG per swing at its contact frame, with skeleton and measured angles."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    info = session.video
    lm_all = best_landmarks(session)
    paths: dict[int, Path] = {}
    for s in session.swings:
        for fr in iter_frames(info.path, info.rotation, max_side=OUT_MAX_SIDE,
                              start_s=s.contact_time - 0.002, end_s=s.contact_time + 0.03):
            rgb = fr.rgb
            scale = rgb.shape[0] / info.height
            lm = lm_all[s.contact_frame]
            draw_skeleton(rgb, lm, scale)
            _angle_tags(rgb, lm, scale, s.metrics)
            put_label(rgb, f"Swing #{s.number} {s.stroke}  contact frame {s.contact_frame}  {_fmt_t(s.contact_time)}",
                      (12, 40), color=ACCENT)
            p = out_dir / f"swing_{s.number:02d}.jpg"
            Image.fromarray(rgb).save(p, quality=88)
            paths[s.number] = p
            break
    return paths


class _Stream:
    """Walks one clip forward in time and returns the frame nearest each requested time."""

    def __init__(self, session, center: float):
        info = session.video
        lo, hi = center + COMPARE_WINDOW_S[0] - 0.05, center + COMPARE_WINDOW_S[1] + 0.05
        self.it = iter(iter_frames(info.path, info.rotation, max_side=COMPARE_HEIGHT * 2,
                                   start_s=max(0.0, lo), end_s=hi))
        self.prev = None
        self.cur = next(self.it, None)

    def at(self, t: float):
        while self.cur is not None and self.cur.time < t:
            nxt = next(self.it, None)
            if nxt is None:
                break
            self.prev, self.cur = self.cur, nxt
        cands = [f for f in (self.prev, self.cur) if f is not None]
        return min(cands, key=lambda f: abs(f.time - t)) if cands else None


def _fit_height(rgb: np.ndarray, h: int) -> np.ndarray:
    w = max(2, int(round(rgb.shape[1] * h / rgb.shape[0])) // 2 * 2)
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def render_comparison(session, swing_number: int, reference, path: Path,
                      progress: ProgressFn | None = None) -> Path:
    """Player's swing (left) vs reference swing (right), synced at contact, half speed."""
    progress = progress or (lambda f, m: None)
    me = session.by_number(swing_number)
    rs, rsess = reference.swing, reference.session
    ref_title = f"your best #{rs.number}" if reference.source == "own_best" else f"reference clip #{rs.number}"
    sides = [dict(sess=sess, sw=sw, mine=mine, stream=_Stream(sess, sw.contact_time), lm=best_landmarks(sess))
             for sess, sw, mine in ((session, me, True), (rsess, rs, False))]
    fps = min(session.video.fps, rsess.video.fps)
    half = 0.5 / fps
    taus = np.arange(COMPARE_WINDOW_S[0], COMPARE_WINDOW_S[1] + 1e-9, 1.0 / fps)
    blank_w = COMPARE_HEIGHT * 9 // 16 // 2 * 2
    writer = None
    out_t, held = 0.0, False
    try:
        for n_i, tau in enumerate(taus):
            panels = []
            for sd in sides:
                fr = sd["stream"].at(sd["sw"].contact_time + tau)
                if fr is None:
                    panels.append(np.zeros((COMPARE_HEIGHT, blank_w, 3), np.uint8))
                    continue
                rgb = fr.rgb.copy()
                times = sd["sess"].track.times
                k = int(np.argmin(np.abs(times - fr.time)))
                draw_skeleton(rgb, sd["lm"][k], rgb.shape[0] / sd["sess"].video.height)
                rgb = _fit_height(rgb, COMPARE_HEIGHT)
                sw = sd["sw"]
                head = f"YOU  swing #{sw.number}  frame {k}" if sd["mine"] else f"REF  {ref_title}  frame {k}"
                put_label(rgb, head, (10, 34), color=ACCENT)
                at_contact = abs(tau) <= half
                put_label(rgb, "CONTACT" if at_contact else f"{tau:+.2f} s to contact", (10, 34 + COMPARE_HEIGHT // 20),
                          color=(20, 20, 20) if at_contact else (255, 255, 255), bg=ACCENT if at_contact else (0, 0, 0))
                if tau >= -half:
                    _panel(rgb, metric_lines(sw.metrics), (10, int(COMPARE_HEIGHT * 0.13)))
                panels.append(rgb)
            div = np.full((COMPARE_HEIGHT, 8, 3), 40, np.uint8)
            frame = np.concatenate([panels[0], div, panels[1]], axis=1)
            if writer is None:
                writer = H264Writer(path, frame.shape[1], frame.shape[0], fps)
            put_label(frame, "1/2 speed", (frame.shape[1] - 150, frame.shape[0] - 20))
            if frame.shape[1] < writer.width:  # keep a constant canvas if a side changed width
                frame = np.pad(frame, ((0, 0), (0, writer.width - frame.shape[1]), (0, 0)))
            frame = np.ascontiguousarray(frame[: writer.height, : writer.width])
            writer.write(frame, out_t)
            out_t += COMPARE_SLOWMO / fps
            if not held and abs(tau) <= half:
                held = True
                for _ in range(int(CONTACT_HOLD_S * fps)):
                    writer.write(frame, out_t)
                    out_t += 1.0 / fps
            if n_i % 10 == 0:
                progress(n_i / len(taus), "Rendering side-by-side")
    finally:
        if writer is not None:
            writer.close()
    return path


def comparison_swing(report, style: str) -> int | None:
    """Default swing to compare: the one the top fix in this style points at, else the worst-scored swing."""
    from .coaching import groups, swing_deviation
    ref = report.references.get(style)
    for f in report.fixes:
        if f.style == style and not (ref and ref.source == "own_best" and f.swing_number == ref.swing.number):
            return f.swing_number
    cands = [s for s in groups(report.session).get(style, [])
             if not (ref and ref.source == "own_best" and s.number == ref.swing.number)]
    if not cands:
        return None
    return max(cands, key=lambda s: swing_deviation(s, style)[0]).number


def render_outputs(report, out: Path, progress: ProgressFn | None = None) -> dict[str, str]:
    progress = progress or (lambda f, m: None)
    outputs: dict[str, str] = {}
    session = report.session
    n_cmp = len(report.references)
    share = 0.75 if n_cmp else 0.95
    outputs["annotated"] = str(render_annotated(session, out / "annotated.mp4",
                                                progress=lambda f, m: progress(f * share, m)))
    for n, p in render_contact_stills(session, out / "contacts").items():
        outputs[f"contact_{n}"] = str(p)
    for i, (style, ref) in enumerate(report.references.items()):
        num = comparison_swing(report, style)
        if num is None:
            continue
        lo = share + 0.05 + (0.95 - share) * i / n_cmp
        cp = out / f"compare_{style}_swing{num:02d}.mp4"
        render_comparison(session, num, ref, cp, progress=lambda f, m, lo=lo: progress(lo + f * (0.95 - share) / n_cmp, m))
        outputs[f"compare_{style}"] = str(cp)
    return outputs
