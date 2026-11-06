"""MediaPipe Tasks PoseLandmarker (VIDEO mode) over a clip or a time window, with on-disk cache."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .video_io import VideoError, VideoInfo, iter_frames, prefetch

os.environ.setdefault("GLOG_minloglevel", "2")  # quiet MediaPipe C++ logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import mediapipe as mp  # noqa: E402  (after env vars)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "out" / "cache"
CACHE_VERSION = "v2"  # bump when frame decoding changes; v1 could cache empty tracks from padded frames
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{m}/float16/latest/pose_landmarker_{m}.task")
MODELS = ("lite", "full", "heavy")

# Landmark indices (subject's own left/right).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

CONNECTIONS = [(c.start, c.end) for c in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS]

ProgressFn = Callable[[float, str], None]


@dataclass
class PoseTrack:
    """Per-frame pose. Missing detections are NaN with visibility 0."""
    times: np.ndarray  # (N,) seconds
    img: np.ndarray  # (N,33,5) x_px, y_px (upright full-res frame), z, visibility, presence
    world: np.ndarray  # (N,33,3) metres, hip-centred, camera-aligned axes (x right, y down, z away)
    model: str
    frame_size: tuple[int, int]  # (width, height) upright, full resolution

    @property
    def detected(self) -> np.ndarray:
        return ~np.isnan(self.img[:, 0, 0])

    def __len__(self) -> int:
        return len(self.times)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=self.times, img=self.img, world=self.world,
                            model=np.array(self.model), frame_size=np.array(self.frame_size))

    @classmethod
    def load(cls, path: Path) -> "PoseTrack":
        d = np.load(path)
        return cls(times=d["times"], img=d["img"], world=d["world"], model=str(d["model"]),
                   frame_size=tuple(int(v) for v in d["frame_size"]))


def ensure_model(name: str) -> Path:
    if name not in MODELS:
        raise ValueError(f"Unknown pose model '{name}'. Choose one of {MODELS}.")
    path = MODELS_DIR / f"pose_landmarker_{name}.task"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL.format(m=name), tmp)
    except OSError as e:
        raise VideoError(f"Couldn't download the MediaPipe '{name}' pose model ({e}). "
                         f"Connect to the internet once, or place pose_landmarker_{name}.task in {MODELS_DIR}.") from e
    tmp.replace(path)
    return path


def file_fingerprint(path: str | Path) -> str:
    """Content hash (size + sampled chunks) so the cache survives renames/re-uploads."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(p, "rb") as f:
        for off in np.linspace(0, max(size - 1_048_576, 0), 16).astype(int):
            f.seek(int(off))
            h.update(f.read(1_048_576))
    return h.hexdigest()[:16]


def _landmarker(model: str):
    V = mp.tasks.vision
    opts = V.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=ensure_model(model).read_bytes()),
        running_mode=V.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return V.PoseLandmarker.create_from_options(opts)


def track_pose(path: str | Path, info: VideoInfo, model: str = "lite", max_side: int = 960,
               start_s: float | None = None, end_s: float | None = None,
               progress: ProgressFn | None = None, use_cache: bool = True) -> PoseTrack:
    """Run pose on every frame of the clip (or of [start_s, end_s])."""
    key = f"{CACHE_VERSION}_{file_fingerprint(path)}_{model}_{max_side}"
    if start_s is not None or end_s is not None:
        key += f"_{start_s or 0:.3f}-{end_s or 0:.3f}"
    cache = CACHE_DIR / f"{key}.npz"
    if use_cache and cache.exists():
        try:
            return PoseTrack.load(cache)
        except Exception:  # corrupt cache -> recompute
            cache.unlink(missing_ok=True)

    W, H = info.width, info.height
    expected = max(1, info.frame_count) if start_s is None and end_s is None else \
        max(1, int(((end_s or info.duration_s) - (start_s or 0)) * info.nominal_fps))
    times, img, world = [], [], []
    last_ms = -1
    with _landmarker(model) as lm:
        for i, fr in enumerate(prefetch(iter_frames(path, info.rotation, max_side, start_s, end_s))):
            ms = int(round(fr.time * 1000))
            if ms <= last_ms:  # detect_for_video needs strictly increasing ms (matters at 240 fps)
                ms = last_ms + 1
            last_ms = ms
            rgb = np.ascontiguousarray(fr.rgb)  # MediaPipe ignores strides
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
            row = np.full((33, 5), np.nan, dtype=np.float32)
            wrow = np.full((33, 3), np.nan, dtype=np.float32)
            if res.pose_landmarks:
                for j, p in enumerate(res.pose_landmarks[0]):
                    row[j] = (p.x * W, p.y * H, p.z or 0.0, p.visibility or 0.0, p.presence or 0.0)
                for j, p in enumerate(res.pose_world_landmarks[0]):
                    wrow[j] = (p.x, p.y, p.z)
            else:
                row[:, 3:] = 0.0
            times.append(fr.time)
            img.append(row)
            world.append(wrow)
            if progress and i % 15 == 0:
                progress(min(i / expected, 1.0), f"Tracking pose ({model}) - frame {i + 1} of ~{expected}")
    if not times:
        raise VideoError("No frames could be decoded from the video.")
    track = PoseTrack(times=np.asarray(times, dtype=np.float64), img=np.stack(img), world=np.stack(world),
                      model=model, frame_size=(W, H))
    if use_cache:
        track.save(cache)
    return track
