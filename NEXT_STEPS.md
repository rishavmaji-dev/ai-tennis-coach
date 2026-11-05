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
1. [x] Video I/O: probe metadata, upright frame iterator, H.264 writer. Pose tracking with cache. CLI stage 1.
2. [x] Swing detection (peak dominant-wrist speed) and forehand/backhand classification.
3. [x] Heavy pass around contacts and contact metrics: elbow, knee, shoulder/hip rotation, contact in front.
4. [x] Coaching: target ranges, best-swing reference, exactly 3 fixes citing value and frame.
5. [x] Rendering: annotated H.264 video, side-by-side swing vs reference synced at contact.
6. [x] Streamlit UI calling the package.
7. [x] Playwright screenshot of the UI with a sample uploaded.
8. [ ] Review pass and fixes.

## Done
- CLAUDE.md, .gitignore, git init.
- `tennis_analyzer/`:
  - `video_io` (PyAV probe, upright frames, prefetch thread, H264Writer)
  - `pose` (Tasks API, models auto-download to `models/`, cache in `out/cache`)
  - `kinematics` (confidence mask, Hampel, gap fill, Savitzky-Golay)
  - `swings` (detect, classify, two-handed, facing direction)
  - `metrics`
  - `pipeline` (analyze_video)
  - `targets` (guideline ranges)
  - `coaching` (references, 3 fixes)
  - `api.run`
- CLI: `.venv/Scripts/python analyze.py samples --hand right [--no-video] [--fast] [--reference clip]`.
- Results on the samples:
  - forehand.mp4: 14 forehands.
  - backhand.mp4: 11 two-handed backhands.
  - Metrics look plausible, and the 3 fixes cite their values and frames.

- Rendering (`outputs.py`), all H.264 High, yuv420p, faststart:
  - `out/<clip>/annotated.mp4` keeps true timestamps and shows a frame counter on every frame.
  - `contacts/swing_NN.jpg` has the measured angles on the joints.
  - `compare_<style>_swingNN.mp4` is half speed, synced at contact, and holds 0.5 s on the contact frame.
- **Full CLI on the samples:** about 2 min with cached pose, about 4 min cold.
  - Cold time: Lite at ~80 fps over the whole clip, Heavy about 50 s per clip, rendering about 60 s per clip.

- **UI** (`app.py`): run `.venv/Scripts/python -m streamlit run app.py`. Uploads are saved to `tmp/uploads/<sha>/` and outputs to `out/ui/<key>/`.
- **Playwright** (`scripts/screenshot_ui.py`, real Chrome channel): run the app, then `.venv/Scripts/python scripts/screenshot_ui.py samples/forehand.mp4 --hand right`.
  - It uploads the clip, analyzes it, and saves screenshots to `docs/screenshots/`.
  - It checks that every `<video>` reaches readyState 4 in Chrome. Both samples pass.

## How the metrics are defined (see `metrics.py`)
- **Contact:** the frame of peak hitting-wrist speed, re-picked on the Heavy window within ±0.15 s of the Lite estimate.
- **Elbow and knee:** 3D angles from world landmarks. Knee bend is 180 minus the knee angle, taking the more-bent knee.
  - On a two-handed stroke, if the hitting elbow is hidden, the top-hand elbow is used and labelled with its arm.
- **Shoulder and hip rotation:** the largest change in the segment's yaw (horizontal plane) between any frame in the 0.7 s before contact and contact.
- **Contact in front:** the wrist minus the hip centre, projected on the session's facing direction. That direction is the hip-line normal averaged over quiet frames between swings.
- **Hip lead:** the time the shoulders reach half their coil-to-contact turn, minus the time the hips do.

## Known issues
- (none yet)
