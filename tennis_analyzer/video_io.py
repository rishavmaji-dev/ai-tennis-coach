"""Video reading/writing.

PyAV (bundled FFmpeg) decodes HEVC/H.264 reliably on Windows and encodes browser-safe H.264
with libx264. OpenCV is only used for drawing.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import av
import numpy as np


class VideoError(Exception):
    """Raised with a plain-language message the UI can show as-is."""


@dataclass
class VideoInfo:
    path: str
    codec: str
    width: int  # upright (display) width in pixels
    height: int  # upright (display) height in pixels
    rotation: int  # display-matrix rotation in degrees (0, 90, -90, 180)
    frame_count: int  # container estimate until frames are decoded
    duration_s: float
    nominal_fps: float  # frame rate declared by the container
    average_fps: float  # frames / duration from the container
    capture_fps: float | None  # vendor capture tag (e.g. com.android.capture.fps), if present
    measured_fps: float | None = None  # from decoded frame timestamps (median frame interval)
    variable_frame_rate: bool | None = None  # from decoded frame timestamps

    @property
    def orientation(self) -> str:
        return "portrait" if self.height > self.width else "landscape"

    @property
    def fps(self) -> float:
        """Best frame-rate estimate: measured from timestamps when known, else container rate."""
        return self.measured_fps or self.nominal_fps

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orientation"] = self.orientation
        d["fps"] = self.fps
        return d

    def describe_fps(self) -> str:
        parts = [f"{self.fps:.2f} fps"]
        if self.capture_fps:
            parts.append(f"capture tag {self.capture_fps:g} fps")
        parts.append(f"container {self.nominal_fps:.2f} fps")
        if self.variable_frame_rate:
            parts.append("variable frame rate")
        return ", ".join(parts)


@dataclass
class Frame:
    time: float  # seconds from stream start
    rgb: np.ndarray  # upright HxWx3 uint8


def _open(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise VideoError(f"Video file not found: {p}")
    if p.stat().st_size == 0:
        raise VideoError(f"Video file is empty: {p.name}")
    try:
        container = av.open(str(p))
    except av.error.FFmpegError as e:  # InvalidDataError, etc.
        raise VideoError(f"Can't read {p.name} as a video ({e.__class__.__name__}). "
                         "Upload an MP4 or MOV straight from the phone.") from e
    if not container.streams.video:
        container.close()
        raise VideoError(f"{p.name} has no video track.")
    return container


def _capture_fps(meta: dict) -> float | None:
    for key, val in meta.items():
        k = key.lower()
        if "capture.fps" in k or k.endswith("framerate") or "capture_fps" in k:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _upright(arr: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate to display orientation and always return a packed (C-contiguous) array.

    PyAV pads rows (e.g. 540 px wide -> 1632-byte rows), and MediaPipe reads the buffer as packed, so a
    padded frame looks sheared and no person is found. Rotated phone clips hid this because rot90 copies.
    """
    k = (rotation or 0) // 90
    return np.ascontiguousarray(np.rot90(arr, k) if k % 4 else arr)


def probe(path: str | Path) -> VideoInfo:
    container = _open(path)
    try:
        s = container.streams.video[0]
        rotation = 0
        first = None
        try:
            for first in container.decode(s):
                rotation = int(first.rotation or 0)
                break
        except av.error.FFmpegError as e:
            raise VideoError(f"The video track in {Path(path).name} can't be decoded ({e}).") from e
        if first is None:
            raise VideoError(f"{Path(path).name} contains no decodable frames.")
        w, h = s.codec_context.width, s.codec_context.height
        if rotation % 180:
            w, h = h, w
        if s.duration is not None and s.time_base is not None:
            duration = float(s.duration * s.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000
        else:
            duration = 0.0
        nominal = float(s.guessed_rate or s.base_rate or s.average_rate or 0)
        average = float(s.average_rate or 0) or (s.frames / duration if duration and s.frames else nominal)
        if not nominal:
            nominal = average
        if not nominal:
            raise VideoError(f"{Path(path).name} has no usable frame-rate information.")
        meta = {**dict(container.metadata), **dict(s.metadata)}
        return VideoInfo(
            path=str(path), codec=s.codec_context.name, width=w, height=h, rotation=rotation,
            frame_count=int(s.frames or round(duration * nominal)), duration_s=duration,
            nominal_fps=nominal, average_fps=average, capture_fps=_capture_fps(meta),
        )
    finally:
        container.close()


def iter_frames(path: str | Path, rotation: int, max_side: int | None = None,
                start_s: float | None = None, end_s: float | None = None) -> Iterator[Frame]:
    """Yield upright RGB frames with true timestamps, optionally scaled and limited to [start_s, end_s]."""
    container = _open(path)
    try:
        s = container.streams.video[0]
        s.thread_type = "AUTO"
        if start_s and start_s > 0:
            container.seek(int(start_s / s.time_base), stream=s, backward=True)
        fallback_dt = 1.0 / float(s.guessed_rate or s.average_rate or 30)
        last_t = None
        for f in container.decode(s):
            t = f.time
            if t is None:
                t = 0.0 if last_t is None else last_t + fallback_dt
            last_t = t
            if start_s is not None and t < start_s - 1e-6:
                continue
            if end_s is not None and t > end_s + 1e-6:
                break
            if max_side and max(f.width, f.height) > max_side:
                scale = max_side / max(f.width, f.height)
                rgb = f.reformat(width=int(f.width * scale) // 2 * 2, height=int(f.height * scale) // 2 * 2,
                                 format="rgb24").to_ndarray()
            else:
                rgb = f.to_ndarray(format="rgb24")
            yield Frame(time=float(t), rgb=_upright(rgb, rotation))
    except av.error.FFmpegError as e:
        raise VideoError(f"Decoding stopped partway through the video ({e}). The file may be damaged.") from e
    finally:
        container.close()


def prefetch(gen: Iterator, size: int = 32) -> Iterator:
    """Run a generator in a background thread so decoding overlaps with pose inference."""
    q: queue.Queue = queue.Queue(maxsize=size)
    done = object()
    stop = threading.Event()

    def worker():
        try:
            for item in gen:
                if stop.is_set():
                    break
                q.put(item)
        except BaseException as e:  # forwarded to the consumer
            q.put(e)
        finally:
            q.put(done)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while th.is_alive():  # drain so the worker can exit
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


class H264Writer:
    """Browser-safe H.264 MP4 (yuv420p, +faststart) with per-frame timestamps."""

    TIME_BASE = Fraction(1, 90000)

    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 23):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width - width % 2, height - height % 2
        self.container = av.open(str(self.path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        self.stream.width, self.stream.height, self.stream.pix_fmt = self.width, self.height, "yuv420p"
        self.stream.time_base = self.TIME_BASE
        self.stream.codec_context.time_base = self.TIME_BASE
        self.stream.options = {"crf": str(crf), "preset": "veryfast", "profile": "high"}
        self._fps = fps
        self._n = 0
        self._last_pts = -1

    def write(self, rgb: np.ndarray, t: float | None = None) -> None:
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            rgb = np.ascontiguousarray(rgb[: self.height, : self.width])
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        pts = int(round((t if t is not None else self._n / self._fps) / self.TIME_BASE))
        if pts <= self._last_pts:
            pts = self._last_pts + 1
        frame.pts, frame.time_base = pts, self.TIME_BASE
        self._last_pts = pts
        self._n += 1
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def transcode_to_h264(src: str | Path, dst: str | Path, max_side: int = 1280) -> Path:
    """Upright, browser-playable copy of any input clip (keeps true timestamps)."""
    info = probe(src)
    writer = None
    try:
        t0 = None
        for fr in prefetch(iter_frames(src, info.rotation, max_side=max_side)):
            if writer is None:
                writer = H264Writer(dst, fr.rgb.shape[1], fr.rgb.shape[0], info.nominal_fps)
                t0 = fr.time
            writer.write(fr.rgb, fr.time - t0)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise VideoError(f"{Path(src).name} contains no decodable frames.")
    return Path(dst)
