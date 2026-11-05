# NEXT_STEPS

_Resume point for the tennis stroke analyzer. Update after every major step._

## Decisions (agreed with user)
- **Reference swing:** the player's own best-measured swing of the same stroke type in the session is
  the default reference; an optional uploaded reference video (CLI `--reference`) is analyzed the same way.
- **Pose model, two-pass:** MediaPipe **Lite** on every frame finds the swings. Then **Heavy** runs on a window
  of about 1 s around each detected contact, and the measurements come from that pass. The `--fast` option uses **Full** throughout.
- **Fixes:** ranked by deviation from coaching guideline ranges (editable, in `tennis_analyzer/targets.py`).
  When a reference exists, the tip also cites the reference value.

## Environment facts
- **Python:** 3.11.4 in `.venv`. Create it with `py -3.11 -m venv .venv`.
- **Packages:** mediapipe 1.2.2 (Tasks API; `mp.solutions` was removed in 0.10.30), av 18.1.0 (the last version with Python 3.11 wheels), streamlit 1.66.1.
- **OpenCV:** mediapipe pulls in `opencv-contrib-python`. Never also install `opencv-python`. OpenCV can't write H.264, so PyAV (libx264) does all encoding.
- **Samples:** both are HEVC 1920x1080 with a -90° display matrix, so the upright frames are portrait 1080x1920. They come from a Pixel 9a.
  `backhand.mp4` has a variable frame rate (nominal 59.94, average 59.55), so all timing uses per-frame timestamps.
- **Rotation:** `np.rot90(img, frame.rotation // 90)` gives the upright frame.
- **Measured speeds:** decode at 960px long side runs at ~143 fps. Lite runs at ~90 fps, Full at ~60 fps, Heavy at ~20 fps.
- **Unavailable this session:** the Context7 and Playwright MCP servers failed to connect. Docs came from official sites instead, and the screenshot uses the Python `playwright` package.

## Plan
1. [ ] Video I/O: probe metadata, upright frame iterator, H.264 writer. Pose tracking with cache. CLI stage 1.
2. [ ] Swing detection (peak dominant-wrist speed) and forehand/backhand classification.
3. [ ] Heavy pass around contacts and contact metrics: elbow, knee, shoulder/hip rotation, contact in front.
4. [ ] Coaching: target ranges, best-swing reference, exactly 3 fixes citing value and frame.
5. [ ] Rendering: annotated H.264 video, side-by-side swing vs reference synced at contact.
6. [ ] Streamlit UI calling the package.
7. [ ] Playwright screenshot of the UI with a sample uploaded.
8. [ ] Review pass and fixes.

## Done
- CLAUDE.md, .gitignore, git init.

## Known issues
- (none yet)
