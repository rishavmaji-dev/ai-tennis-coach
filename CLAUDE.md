# Tennis Stroke Analyzer: project rules

## Non-negotiable rules
1. **Never fake data or hardcode results.** Every number shown or written comes from the pose
   measurements of the uploaded clip. No placeholder metrics, no canned tips, and no fallback values
   that look like real measurements. If something can't be measured, say so.
2. **Run every change on `/samples` and show real output.** After any change to analysis code, run the
   CLI on every clip in `samples/` and paste the actual output (swings found, metrics, tips).
3. **Fail gracefully with clear messages.** Unreadable video, no person found, no swings detected,
   low-confidence landmarks, and wrong handedness all produce a plain-language message that says
   what went wrong and what to try. Never show a raw stack trace to the user.
4. **Every feedback tip cites a measurement.** Each tip names the measured angle, distance, or
   timing and the swing number + frame (and timestamp) it came from.
5. **Plain, confident UI copy.** Short sentences, no hedging filler, no jargon without explanation.

## Working conventions
- Python 3.11, virtual env in `.venv/`. Pinned dependencies in `requirements.txt`.
- Analysis lives in the `tennis_analyzer/` package and runs from the CLI (`analyze.py`) first;
  the Streamlit app (`app.py`) only calls the package. It never re-implements analysis.
- Temporary/scratch files go in `tmp/` (gitignored). Analysis outputs go in `out/` (gitignored).
- Sample videos in `samples/` are not committed (too large).
- Read the true frame rate and per-frame timestamps from the container. Never assume 30/60 fps.
- Any video shown in the UI is H.264 (yuv420p, faststart) MP4.
- Landmarks below the visibility threshold are treated as missing, never used as-is.
- Commit after every working step. Update `NEXT_STEPS.md` after every major step.
