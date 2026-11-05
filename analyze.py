"""Command-line tennis stroke analyzer.

    python analyze.py samples/forehand.mp4 --hand right
    python analyze.py samples --hand right            # every clip in a folder
    python analyze.py clip.mp4 --hand left --reference pro.mp4 --reference-hand right
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _progress(frac: float, msg: str, _state={"last": ""}) -> None:
    line = f"\r[{frac * 100:5.1f}%] {msg[:90]:<90}"
    if line != _state["last"]:
        sys.stderr.write(line)
        sys.stderr.flush()
        _state["last"] = line


def _num(v, unit="", fmt="{:.0f}") -> str:
    return "-" if v is None else fmt.format(v) + unit


def print_report(report) -> None:
    from tennis_analyzer.pipeline import frame_ref
    s = report.session
    v = s.video
    print(f"\n=== {Path(v.path).name} ===")
    print(f"Video: {v.codec.upper()} {v.width}x{v.height} ({v.orientation}), {v.duration_s:.1f} s, "
          f"{v.frame_count} frames, {v.describe_fps()}")
    print(f"Player: {s.handedness}-handed | mode: {s.mode} | "
          + " | ".join(f"{k} {t:.1f}s" for k, t in s.timings_s.items()))
    print(f"Swings found: {len(s.swings)}")
    print(f"  {'#':>3} {'stroke':<9} {'grip':<4} {'contact':<22} {'elbow':>9} {'knee':>5} {'shRot':>6} "
          f"{'hipRot':>6} {'front':>7} {'hipLead':>8} {'fwdSwing':>8} {'hand':>8}")
    for w in s.swings:
        m = w.metrics
        grip = "2H" if w.two_handed else ("1H" if w.two_handed is False else "?")
        elbow = _num(m.elbow_angle_deg, "°") + (f"({m.elbow_arm[0].upper()})" if m.elbow_arm else "")
        print(f"  {w.number:>3} {w.stroke:<9} {grip:<4} {frame_ref(w.contact_frame, w.contact_time):<22} "
              f"{elbow:>9} {_num(m.knee_bend_deg, '°'):>5} {_num(m.shoulder_rotation_deg, '°'):>6} "
              f"{_num(m.hip_rotation_deg, '°'):>6} {_num(m.contact_in_front_cm, 'cm'):>7} "
              f"{_num(m.hip_lead_ms, 'ms'):>8} {_num(m.forward_swing_ms, 'ms'):>8} "
              f"{_num(m.wrist_speed_mps, 'm/s', '{:.1f}'):>8}")
        for miss in m.missing:
            print(f"        not measured - {miss}")
    for style, r in report.references.items():
        print(f"Reference for {style}: {r.describe}")
    for n in report.notes + s.warnings:
        print(f"Note: {n}")
    print(f"\nTop {len(report.fixes)} fixes:")
    for f in report.fixes:
        print(f"  {f.rank}. {f.title}\n     {f.detail}")
    if report.fixes_note:
        print(f"  {report.fixes_note}")
    for k, p in report.outputs.items():
        print(f"Output {k}: {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze tennis groundstrokes in a phone video.")
    ap.add_argument("videos", nargs="+", help="video file(s) or folder(s)")
    ap.add_argument("--hand", required=True, choices=["right", "left"], help="player's dominant hand")
    ap.add_argument("--fast", action="store_true", help="use the Full model throughout (faster, less precise)")
    ap.add_argument("--reference", help="optional reference clip to compare against")
    ap.add_argument("--reference-hand", choices=["right", "left"], help="reference player's hand (default: --hand)")
    ap.add_argument("--out", default="out", help="output folder (default: out/)")
    ap.add_argument("--no-video", action="store_true", help="skip rendering annotated/comparison videos")
    ap.add_argument("--no-cache", action="store_true", help="recompute pose instead of using out/cache")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # degree signs etc. on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    from tennis_analyzer.api import run
    from tennis_analyzer.pipeline import AnalysisError

    paths: list[Path] = []
    for v in args.videos:
        p = Path(v)
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXT)
        else:
            paths.append(p)
    if not paths:
        print("No video files found.", file=sys.stderr)
        return 2
    failures = 0
    for p in paths:
        try:
            report = run(p, args.hand, Path(args.out) / p.stem, fast=args.fast, reference=args.reference,
                         reference_handedness=args.reference_hand, render_videos=not args.no_video,
                         progress=_progress, use_cache=not args.no_cache)
            sys.stderr.write("\n")
            print_report(report)
        except AnalysisError as e:
            sys.stderr.write("\n")
            print(f"\n=== {p.name} ===\nCould not analyze: {e}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
