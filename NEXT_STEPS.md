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
2. [x] **Straight-arm tip (review #3).**
   - `Metric.physical_max` (the elbow's is 180°) marks a range end as a body limit, and the near-edge rule ignores that end.
   - Cramped elbows are still flagged at the low end.
   - Tests: `test_straight_arm_is_never_told_to_bend` (fails on the old code) and `test_cramped_elbow_near_low_edge_still_flagged`.
   - Sample fixes are unchanged.
3. [x] **Wrong-handedness warning (item 9).**
   - The old check voted at the contact frame, but with the wrong setting the contacts come from the wrong wrist's own peaks.
   - New rule:
     - One-handed swings compare peak wrist speeds within ±0.3 s of contact. The other/chosen ratio is 0.36–0.43 when the setting is right and 1.60–2.35 when it is wrong.
     - Two-handed swings vote by label, since a flipped setting makes them "two-handed forehands".
   - Results on forehand.mp4, backhand.mp4 and the landscape 120 fps clip:
     - `--hand right`: no warning on any clip.
     - `--hand left`: a warning on all three.
4. [x] **Backhand contact timing (review #4): confirmed and fixed.**
   - Before: refinement used only the hitting wrist. On the two-handed backhands that wrist sits just above the visibility threshold and shows glitch spikes, so contact jumped up to ±126 ms (on swings 2, 4, 8 and 10).
   - Now (`metrics.contact_signal`):
     - On two-handed swings, refinement uses the wrist with the higher mean visibility, which is the top hand on the sample.
     - The refine range is ±0.25 s. The Lite peak is a plateau, and on #5 and #11 the true peak sat 0.17–0.22 s away.
     - The Lite estimate is kept when the hands are tracked in under half the window.
   - Verified with plots (`tmp/bh_curves.py`, `tmp/refine_check2.py`): all 11 backhand contacts sit on the clean top-hand speed peak.
   - Forehand contact frames are identical to before, and the forehand fixes are unchanged.
   - The Heavy window is now -1.05 s, derived from the constants.
5. [x] **Re-exported portrait clips not detected (review #5): confirmed and fixed.**
   - Cause: clips stored upright with no rotation flag (WhatsApp, editing apps) decode to row-padded arrays, for example 1632-byte rows for 540 px. MediaPipe ignores strides and saw a sheared image, so 0% detection.
   - Fix:
     - `video_io._upright` always returns packed arrays.
     - `pose.py` re-packs before `mp.Image`.
     - The pose cache key is versioned (`CACHE_VERSION = "v2"`), so empty tracks cached by v1 are never reused.
   - Results:
     - `tmp/review/fh_portrait_coded.mp4` now finds its swing at frame 79 with the same numbers as the original.
     - A cold re-run of both samples gives output identical to step 4.
     - New test `test_frames_are_packed_even_when_pyav_pads_rows` fails on the old code.
6. [x] **30 fps smoothing (review #2).**
   - The Savitzky-Golay window is now always odd and at least 5. Before, it was 4 at 24–30 fps, which shifted every landmark half a frame later.
   - New test `test_smooth_is_centred_at_every_frame_rate` covers 24–240 fps and fails on the old code at 24, 25 and 30.
   - 30 fps check (`tmp/edge/forehand_30fps.mp4`, every other frame of the first 25 s of forehand.mp4):
     - all 6 swings are found at the same times as at 60 fps;
     - most angles agree within a few degrees.
   - The 60 fps samples are unchanged (window still 7) and identical to step 5.
7. [~] **Finish the CLAUDE.md rules / failure-handling review** (workflow wf_b8cdf85e-a65: 2 reviewers plus a skeptic per finding; repros in `tmp/review2/`).
   - 9 findings confirmed and 2 rejected: an offline model download, and the "Your 3 fixes" heading with fewer fixes. Neither could be triggered with real input.
   - **7a. [x] Failure handling (5 findings):**
     - **Rendering or report failures no longer throw away a finished analysis.**
       - `report.json` is written before rendering.
       - A render failure (for example a locked file) becomes a plain note, and the traceback goes to `<out>/error.log`.
       - The CLI catches any per-clip error and continues with the next clip.
       - Repro `tmp/review2/locked.py`: the analysis survives and the run exits 0.
     - **Truncated clip:** the decoded span is compared with the header duration, and a warning appears if the clip was cut short. `trunc60.mp4` warns "Only the first 13.5 s of this 25.0 s clip...".
     - **Files with no timing** (a photo renamed .mov): a clear message replaces the ZeroDivisionError.
     - **Same-name clips** get separate output folders in the CLI.
     - **App messages** name the uploaded file (a sanitized name) instead of `input.mov`, with no exception class names.
     - The original samples are identical to step 6. The teammate clips run with no errors.
   - 7b. [ ] Left/right swap makes impossible rotations (rule 1, also teammate issue A).
   - 7c. [ ] Copy: garbled hip-lead consistency tip, one-swing grammar ("median of 1 forehands"), and the inconsistent tip calling the outlier "your best-measured" swing.

## Teammate clips (2026-11-05): teammate1_fh/teammate1_bh.mov and teammate2_fh/teammate2_bh.MOV, iPhone, H.264, -90° rotation, about 60 fps, right-handed
- Output: `tmp/teammates.txt` (CLI `--no-video`). The clips are gitignored.
- Nothing crashed. Every clip got swings, labels and 3 cited fixes, with no false handedness warning.
- **Looks off (not fixed yet):**
  - **A. Impossible rotations on face-on clips.** Shoulders turned 144–225° on teammate1_fh, and hips 243°/440° on teammate1_bh #1/#5.
    - Cause: MediaPipe swaps left and right as the player turns. The shoulder line jumps 61–81° in one frame (about 4,000°/s; real rotation is at most about 17° per frame), and `unwrap_deg` counts each swap as about 180° of rotation.
    - These values drive wrong top fixes: teammate1_fh #1 "Stop over-spinning the shoulders" and #2, and teammate1_bh #1 "Keep the hips more stable".
    - Clean swings are plausible, for example teammate1_fh #3 at 52°.
    - Fix plan:
      - Track yaw as an axis: unwrap the doubled angle, so a left/right swap isn't counted as rotation.
      - Mark rotation and timing as not measured if the line moves more than about 30° in one frame inside the coil window.
  - **B. False swings from walking, camera panning and ready-position fidgets.**
    - Cases: teammate2_bh #1–#4 (walking away, standing, ready position, half out of frame during a pan) and teammate1_bh #4 (ready position).
    - Also, teammate2_bh #5 is a real two-handed backhand labeled as a forehand.
    - The false swings create a fake "forehand" group in a backhand clip, which drives teammate2_bh fix #3.
    - Cause: detection uses 2D hand speed in image coordinates, so body or camera motion counts as hand speed.
    - Fix plan: detect on hand speed relative to the hip centre. That needs a re-check on all 6 clips.
  - **C. Copy bugs.**
    - The hip-lead consistency tip reads "a hips first by 188 ms spread".
    - With one swing, tips read "median of 1 forehands".

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
- **Per-swing values depend on where the Heavy window starts.** The VIDEO-mode tracker re-initializes from that frame. Moving the window 0.1 s changed forehand #9's hip lead from -29 to +47 ms and elbows by about ±4°. Session medians are stable (hip lead -27 to -25 ms). Trust medians over single-swing timing values.
- **Contact-in-front is sensitive to the exact contact frame.** At about 5 m/s, one frame moves the hand about 8 cm at 60 fps and about 16 cm at 30 fps.
  - On the 30 fps test clip, contact-in-front differed from 60 fps by up to about ±12 cm. Prefer 60 fps or higher for that metric.
- Contact is the frame of peak hand speed (no ball in the samples). In real strokes it falls slightly before impact, so the hand sits near the hip. Contact-in-front values read low compared with true impact.
- Hip-lead timing is single-camera 3D. Treat values within ±20 ms (about one frame at 60 fps) as "together".
- Some backhand swings in the sample are low-effort (#4, #7, #10). They are real detections, not errors.
- Guideline ranges in `targets.py` are general coaching values, not calibrated to a player.
