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
8. [~] Review pass and fixes. **Stopped partway at the user's request on 2026-11-04. See "Where the review left off".**

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

## Run it
- App: `.venv\Scripts\python -m streamlit run app.py`, then open http://localhost:8501
- CLI: `.venv\Scripts\python analyze.py samples --hand right`
- Tests: `.venv\Scripts\python -m pytest -q`. 18 pass.

## Edge cases tested by hand (all behave correctly)
- **Clips that fail with a clear message, no stack trace:**
  - corrupt file
  - empty file
  - 0.5 s clip
  - clip with no person
- **Landscape 120 fps H.264 clip** (a sample segment retimed 2x, in `tmp/edge/`): 3 forehands found.
  - Contacts map exactly onto the original clip's contacts.
  - Hand speed doubles as expected, so timing comes from the timestamps.

## Deadline fix list (2026-11-05): only these, in order, with no new features
1. [x] **Backswing cut off (review #1).**
   - The Heavy window now starts at -(COIL_SEARCH_S + CONTACT_REFINE_S + 0.1) = -0.95 s, derived in `metrics.py`.
   - If the coil search would still start before the analyzed frames (for example, a swing at the very start of a clip), rotation, timing and forward swing are marked "not measured". They are never computed from a cut-off backswing.
   - Backhand #10 shoulder/hip went from 20°/17° to 31°/29°. #4 forward swing went from 269 to 639 ms.
   - Every coil search now starts at least 124 ms inside its window.
2. [ ] Straight-arm "bend your arm" tip (review #3).
3. [ ] Wrong-handedness warning (item 9).
4. [ ] Backhand contact-frame timing (review #4).
5. [ ] Re-exported portrait clips not detected (review #5).
6. [ ] 30 fps smoothing (review #2).
7. [ ] Finish the CLAUDE.md rules / failure-handling review.

## Where the review left off
The review ran as a workflow (wf_e2eda3ad-6a8) with three lenses: correctness, rules/failure paths, and video I/O. Each finding then went to a skeptic who tried to refute it.
- The correctness and video reviewers finished.
- The rules/failure-path reviewer had not returned when the run was stopped, so that lens is still open.
- Repro scripts from the reviewers are in `tmp/review/` (gitignored).
- No fixes from the review have been applied yet.

**Confirmed by the skeptic (fix next):**
1. **[medium] Heavy window too short** (`pipeline.py` `HEAVY_WINDOW_S`, `metrics.py` `COIL_SEARCH_S`).
   - Contact refinement can move contact up to 0.15 s earlier. The 0.7 s coil search then runs past the start of the window.
   - Backhand swings #4 and #10 get clipped rotation and forward-swing values. #10's 20°/17° turn drives backhand fixes 1 and 2.
   - Fix: start the window at about -0.95 s, derived from the metric constants. Mark the metric missing when the search would still start before the window or the clip.
2. **[medium] Even Savitzky-Golay window at 30 fps or lower** (`kinematics.smooth`).
   - The window comes out as 4, which shifts every landmark half a frame later.
   - Fix: force an odd window of at least 5. The 60 fps samples are not affected.
3. **[low] Near-edge elbow tip** (`coaching._candidates`).
   - It treats 180° (a straight arm) as a coaching edge and tells straight-arm hitters to bend.
   - Fix: skip the near-edge check against physical limits.

**Rejected by the skeptic:** "knee bend changes meaning when one knee is hidden". Re-measuring the hidden knee changed no fix.

**Found but not yet verified (verify first, then fix):**
4. **[high] `refine_contact` uses only the dominant wrist** (`metrics.py`).
   - On occluded backhands it moves contact by ±0.1 s. The reviewer measured shifts of -84 to +118 ms on backhand swings 1, 2, 4, 8 and 10.
   - Suggested fix: refine with the same `hand_speed` signal used for detection, and keep the Lite time when the wrist is mostly hidden.
5. **[high] `iter_frames` can return row-padded, non-contiguous arrays** (`video_io.py`).
   - This happens when there's no rotation and the width isn't a multiple of 32.
   - MediaPipe ignores the strides and sees a sheared image. The reviewer's portrait-coded copy of forehand.mp4 got 0% detection.
   - Suggested fix: `np.ascontiguousarray` always. Also add a version to the pose cache key so bad cached tracks aren't reused.
6. **[medium] Slow-motion capture tag is read but not applied** (`pipeline.py`).
   - Example: a 240 fps capture stored at 30 fps playback makes speeds 8x too low.
   - Suggested fix: rescale times by measured/capture, or at least warn.
7. **[low] `measured_fps` median of quantized intervals is wrong for some timescales** (`pipeline.py`), giving the wrong fps and a false VFR flag.
8. **[low] Comparison writer declares `rate=fps` while frames are spaced 2/fps** (`outputs.py`), which pushes the H.264 level up on large high-fps comparisons.

**Also found during my own testing (not from the workflow), still to fix:**
9. **The wrong-handedness warning doesn't fire.**
   - Running the samples with `--hand left` mislabels strokes with no warning.
   - The forehand clip comes out as mixed forehand/backhand/unknown; the backhand clip as 11 "two-handed forehands".
   - Cause: the check only votes on one-handed swings, and needs the selected wrist to be visible.
   - Plan:
     - Also vote "wrong hand" when the selected wrist is untracked at contact but the other wrist is tracked.
     - Warn when most swings are two-handed forehands.

**Remaining review work:** rerun the rules/failure-path lens (Streamlit session state, rendering exceptions inside `api.run`, offline model download).

## Known issues
- Contact is the frame of peak hand speed (no ball in the samples). In real strokes it falls slightly before impact, so the hand sits near the hip. Contact-in-front values read low compared with true impact.
- Hip-lead timing is single-camera 3D. Treat values within ±20 ms (about one frame at 60 fps) as "together".
- Some backhand swings in the sample are low-effort (#4, #7, #10). They are real detections, not errors.
- Guideline ranges in `targets.py` are general coaching values, not calibrated to a player.
