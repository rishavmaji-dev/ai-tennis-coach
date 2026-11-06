"""Video I/O round trips on small synthetic clips written to a temp folder."""
import av
import numpy as np
import pytest

from tennis_analyzer.video_io import H264Writer, VideoError, iter_frames, probe


def _write_clip(path, w=64, h=48, n=30, fps=120):
    with H264Writer(path, w, h, fps) as wr:
        for i in range(n):
            img = np.zeros((h, w, 3), np.uint8)
            img[:, : (i % w) + 1] = 255
            wr.write(img, i / fps)


def test_h264_writer_output_is_browser_safe(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_clip(p)
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        assert s.codec_context.name == "h264"
        assert s.codec_context.pix_fmt == "yuv420p"
        assert s.frames == 30
    assert b"moov" in p.read_bytes()[:4096]  # faststart: index before media data


def test_probe_reads_true_rate_and_timestamps(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_clip(p, fps=120)
    info = probe(p)
    assert info.nominal_fps == pytest.approx(120, rel=1e-3)
    assert info.orientation == "landscape"
    times = [f.time for f in iter_frames(p, info.rotation)]
    assert np.median(np.diff(times)) == pytest.approx(1 / 120, rel=1e-2)


def test_iter_frames_window_and_scale(tmp_path):
    p = tmp_path / "clip.mp4"
    _write_clip(p, w=640, h=480, n=60, fps=60)
    info = probe(p)
    frames = list(iter_frames(p, info.rotation, max_side=320, start_s=0.25, end_s=0.5))
    assert frames[0].time >= 0.25 - 1e-6 and frames[-1].time <= 0.5 + 1e-6
    assert frames[0].rgb.shape == (240, 320, 3)


def test_rotation_metadata_gives_upright_portrait(tmp_path):
    p = tmp_path / "rot.mp4"
    with av.open(str(p), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = 64, 32, "yuv420p"
        s.set_display_rotation(90)  # phone held upright: stored landscape, displayed portrait
        for _ in range(5):
            for pk in s.encode(av.VideoFrame.from_ndarray(np.zeros((32, 64, 3), np.uint8), format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    info = probe(p)
    assert (info.width, info.height) == (32, 64)
    assert next(iter_frames(p, info.rotation)).rgb.shape[:2] == (64, 32)


@pytest.mark.parametrize("content", [b"", b"not a video at all"])
def test_bad_files_raise_clear_error(tmp_path, content):
    p = tmp_path / "bad.mp4"
    p.write_bytes(content)
    with pytest.raises(VideoError):
        probe(p)


def test_frames_are_packed_even_when_pyav_pads_rows(tmp_path):
    """Upright-coded portrait clip (no rotation flag), scaled so width*3 isn't a multiple of 32."""
    p = tmp_path / "portrait.mp4"
    _write_clip(p, w=1080, h=1920, n=3, fps=30)
    info = probe(p)
    assert info.rotation == 0
    fr = next(iter_frames(p, info.rotation, max_side=960))
    assert fr.rgb.shape == (960, 540, 3)
    assert fr.rgb.flags["C_CONTIGUOUS"]
