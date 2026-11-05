"""Streamlit UI. All analysis happens in tennis_analyzer.api.run(). This file only collects input
and displays results.

    .venv\\Scripts\\python -m streamlit run app.py
"""
from __future__ import annotations

import hashlib
import logging
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from tennis_analyzer.api import Report, run
from tennis_analyzer.coaching import groups
from tennis_analyzer.outputs import comparison_swing, render_comparison
from tennis_analyzer.pipeline import AnalysisError, frame_ref
from tennis_analyzer.targets import INFO_METRICS, METRICS, RANGES, style_label

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "tmp" / "uploads"
OUT = ROOT / "out" / "ui"
VIDEO_TYPES = ["mp4", "mov", "m4v"]
log = logging.getLogger("tennis_analyzer.app")

st.set_page_config(page_title="Tennis Stroke Analyzer", page_icon="🎾", layout="wide")


def save_upload(up) -> Path:
    data = up.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:16]
    suffix = Path(up.name).suffix.lower() if Path(up.name).suffix.lower() in {f".{t}" for t in VIDEO_TYPES} else ".mp4"
    path = UPLOADS / digest / f"input{suffix}"  # never trust the uploaded filename as a path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def fmt(v, unit="", digits=0) -> str:
    return "–" if v is None else f"{v:.{digits}f}{unit}"


# ---- input ------------------------------------------------------------------------------------

st.title("Tennis stroke analyzer")
st.write("Upload a phone video of one player hitting groundstrokes. Every swing is found, labeled forehand "
         "or backhand, and measured at contact, then you get the three fixes that matter most.")

left, right = st.columns([3, 2], gap="large")
with left:
    up = st.file_uploader("Swing video", type=VIDEO_TYPES,
                          help="MP4 or MOV straight from the phone. HEVC (H.265) and 60–240 fps clips are fine.")
with right:
    hand = st.segmented_control("Is the player right- or left-handed?", ["Right-handed", "Left-handed"],
                                default=None, key="hand")
    with st.expander("Options"):
        mode = st.radio("Analysis mode", ["Accurate", "Fast"], horizontal=True,
                        captions=["Lite model finds swings, Heavy model measures each contact",
                                  "Full model throughout, about twice as quick"])
        ref_up = st.file_uploader("Reference clip (optional)", type=VIDEO_TYPES, key="ref",
                                  help="A coach or pro clip to compare against. Without one, your own best swing "
                                       "of each stroke is the reference.")
        ref_hand = st.segmented_control("Reference player hits", ["Right-handed", "Left-handed"],
                                        default="Right-handed", key="ref_hand")

ready = up is not None and hand is not None
go = st.button("Analyze", type="primary", disabled=not ready, icon=":material/play_arrow:")
if up is None:
    st.info("Upload a clip to start.")
elif hand is None:
    st.info("Choose right- or left-handed so forehands and backhands are labeled correctly.")

if go and ready:
    handedness = "right" if hand.startswith("Right") else "left"
    video_path = save_upload(up)
    ref_path = save_upload(ref_up) if ref_up is not None else None
    key = "_".join([video_path.parent.name, handedness, mode.lower()] +
                   ([ref_path.parent.name, "right" if (ref_hand or "").startswith("Right") else "left"] if ref_path else []))
    with st.status("Analyzing your swings…", expanded=True) as status:
        bar = st.progress(0.0, text="Reading video")
        try:
            report = run(video_path, handedness, OUT / key, fast=(mode == "Fast"), reference=ref_path,
                         reference_handedness=("right" if (ref_hand or "").startswith("Right") else "left"),
                         progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
            st.session_state["report"] = report
            st.session_state["out_dir"] = str(OUT / key)
            st.session_state["clip_name"] = up.name
            status.update(label=f"Done: {len(report.session.swings)} swings analyzed", state="complete",
                          expanded=False)
        except AnalysisError as e:
            st.session_state.pop("report", None)
            status.update(label="Couldn't analyze this clip", state="error", expanded=True)
            st.error(str(e))
        except Exception as e:  # never show a raw stack trace in the UI
            log.error("Unexpected failure\n%s", traceback.format_exc())
            st.session_state.pop("report", None)
            status.update(label="Couldn't analyze this clip", state="error", expanded=True)
            st.error(f"Something went wrong while analyzing this clip ({e.__class__.__name__}: {e}). "
                     "Try re-exporting it from the phone as MP4, or trim it to the swings.")

report: Report | None = st.session_state.get("report")
if report is None:
    st.stop()

# ---- results ----------------------------------------------------------------------------------

s = report.session
v = s.video
st.divider()
st.subheader(f"{st.session_state.get('clip_name', Path(v.path).name)}: {len(s.swings)} swings")
counts = {k: len(sw) for k, sw in groups(s).items()}
st.caption(" · ".join(f"{n} {style_label(k) if n != 1 else style_label(k)[:-1]}" for k, n in counts.items())
           + f" · {s.handedness}-handed · {s.mode} mode")

c1, c2, c3, c4 = st.columns(4)
fps_help = (f"Measured from frame timestamps: {v.fps:.2f} fps. Container says {v.nominal_fps:.2f} fps"
            + (f"; phone capture tag {v.capture_fps:g} fps" if v.capture_fps else "")
            + (". Frame timing varies, so all timing uses real timestamps." if v.variable_frame_rate else "."))
c1.metric("Frame rate", f"{v.fps:.2f} fps", help=fps_help, border=True)
c2.metric("Resolution", f"{v.width}×{v.height}", delta=v.orientation, delta_color="off", delta_arrow="off",
          border=True)
c3.metric("Codec", v.codec.upper(), border=True)
c4.metric("Length", f"{v.duration_s:.1f} s", delta=f"{v.frame_count} frames", delta_color="off",
          delta_arrow="off", border=True)

for w in s.warnings + report.notes:
    st.warning(w)

st.header("Your 3 fixes")
st.caption("Ranked by how far each measurement sits outside general coaching ranges, or how much it varies "
           "between swings. Contact is the frame of peak hand speed.")
if not report.fixes:
    st.info(report.fixes_note or "No fixes could be measured.")
for f in report.fixes:
    with st.container(border=True):
        text_col, img_col = st.columns([3, 1], gap="medium")
        with text_col:
            st.markdown(f"#### {f.rank}. {f.title}")
            st.write(f.detail)
            st.caption(f"{METRICS[f.metric].label} · {style_label(f.style)} · guideline "
                       f"{METRICS[f.metric].show_range(*f.target)}")
        with img_col:
            still = report.outputs.get(f"contact_{f.swing_number}")
            if still and Path(still).exists():
                st.image(still, caption=f"Swing #{f.swing_number}, {frame_ref(f.frame, f.time)}", width="stretch")
if report.fixes and report.fixes_note:
    st.caption(report.fixes_note)

st.header("Side-by-side at contact")
if not report.references:
    st.info("No reference swing available: " + (" ".join(report.notes) or "add a reference clip."))
for style, ref in report.references.items():
    swings = groups(s).get(style, [])
    choices = [sw.number for sw in swings if not (ref.source == "own_best" and sw.number == ref.swing.number)]
    if not choices:
        continue
    default = comparison_swing(report, style)
    st.markdown(f"**{style_label(style).capitalize()}**: compared with {ref.describe}")
    pick = st.selectbox("Your swing", choices, index=choices.index(default) if default in choices else 0,
                        format_func=lambda n: f"Swing #{n} ({frame_ref(s.by_number(n).contact_frame, s.by_number(n).contact_time)})",
                        key=f"pick_{style}")
    path = Path(st.session_state["out_dir"]) / f"compare_{style}_swing{pick:02d}.mp4"
    if not path.exists():
        with st.spinner(f"Rendering swing #{pick} next to the reference…"):
            try:
                render_comparison(s, pick, ref, path)
            except Exception as e:
                log.error("Comparison render failed\n%s", traceback.format_exc())
                st.error(f"Couldn't render this comparison ({e.__class__.__name__}).")
    if path.exists():
        vcol, tcol = st.columns([3, 2], gap="medium")
        with vcol:
            st.video(str(path), loop=True, autoplay=True, muted=True)
        with tcol:
            me = s.by_number(pick).metrics
            rm = ref.swing.metrics
            rows = []
            for k in RANGES[style]:
                lo, hi = RANGES[style][k]
                rows.append({"Measurement": METRICS[k].label,
                             f"Swing #{pick}": METRICS[k].show(getattr(me, k)) if getattr(me, k) is not None else "–",
                             "Reference": METRICS[k].show(getattr(rm, k)) if getattr(rm, k) is not None else "–",
                             "Guideline": METRICS[k].show_range(lo, hi)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Half speed. Both clips are aligned so contact happens at the same moment, "
                       "and the video holds on that frame for half a second.")

st.header("Annotated video and every swing")
ann_col, table_col = st.columns([2, 5] if v.orientation == "portrait" else [1, 1], gap="large")
with ann_col:
    ann = report.outputs.get("annotated")
    if ann and Path(ann).exists():
        st.video(ann)
        st.caption("Low-confidence joints are left out of the skeleton. The frame number in the corner matches the "
                   "frames cited in the fixes.")
rows = []
for sw in s.swings:
    m = sw.metrics
    rows.append({
        "#": sw.number, "Stroke": sw.stroke, "Grip": {True: "two-handed", False: "one-handed"}.get(sw.two_handed, "–"),
        "Contact frame": sw.contact_frame, "Time (s)": round(sw.contact_time, 2),
        "Elbow (°)": fmt(m.elbow_angle_deg) + (f" ({m.elbow_arm})" if m.elbow_arm else ""),
        "Knee bend (°)": fmt(m.knee_bend_deg), "Shoulder turn (°)": fmt(m.shoulder_rotation_deg),
        "Hip turn (°)": fmt(m.hip_rotation_deg), "In front (cm)": fmt(m.contact_in_front_cm),
        "Hip lead (ms)": fmt(m.hip_lead_ms), INFO_METRICS["forward_swing_ms"][0] + " (ms)": fmt(m.forward_swing_ms),
        "Not measured": " ".join(m.missing),
    })
table_col.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=38 + 35 * len(rows))
table_col.caption("Angles come from 3D pose (MediaPipe world landmarks). Hip lead: positive means the hips started "
           "unwinding before the shoulders.")
rep = report.outputs.get("report")
if rep and Path(rep).exists():
    table_col.download_button("Download full report (JSON)", Path(rep).read_bytes(), file_name="swing_report.json",
                       mime="application/json")
