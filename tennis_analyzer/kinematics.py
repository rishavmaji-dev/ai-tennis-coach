"""Signal cleaning and geometry on pose tracks.

Low-confidence landmarks (visibility < MIN_VIS) are dropped to NaN before anything is computed.
Short gaps are interpolated; longer gaps stay NaN so they can't produce a measurement.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import savgol_filter

from .pose import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, PoseTrack

MIN_VIS = 0.5
MAX_GAP_S = 0.12  # interpolate gaps up to this long


def mask_low_confidence(track: PoseTrack, min_vis: float = MIN_VIS) -> tuple[np.ndarray, np.ndarray]:
    """Return (img_xy (N,33,2), world (N,33,3)) with low-visibility landmarks set to NaN."""
    bad = ~(track.img[:, :, 3] >= min_vis)  # NaN visibility counts as bad
    xy = track.img[:, :, :2].astype(np.float64).copy()
    w = track.world.astype(np.float64).copy()
    xy[bad] = np.nan
    w[bad] = np.nan
    return xy, w


def fill_gaps(x: np.ndarray, times: np.ndarray, max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """Linearly interpolate NaN runs no longer than max_gap_s along axis 0 (any trailing shape)."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(len(x), -1).copy()
    for c in range(flat.shape[1]):
        col = flat[:, c]
        nan = np.isnan(col)
        if not nan.any() or nan.all():
            continue
        idx = np.flatnonzero(~nan)
        # run boundaries of NaN segments
        edges = np.diff(np.concatenate([[0], nan.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):  # NaN run is [s, e)
            if s == 0 or e == len(col):
                continue  # don't extrapolate past the ends
            if times[e] - times[s - 1] <= max_gap_s + 1e-9:
                col[s:e] = np.interp(times[s:e], times[[s - 1, e]], col[[s - 1, e]])
        flat[:, c] = col
    return flat.reshape(x.shape)


def smooth(x: np.ndarray, fps: float, window_s: float = 0.1, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0, applied per finite run (NaNs preserved)."""
    x = np.asarray(x, dtype=np.float64)
    win = max(poly + 2, int(round(window_s * fps)) | 1)
    flat = x.reshape(len(x), -1).copy()
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            n = e - s
            w = min(win, n if n % 2 else n - 1)
            if w > poly:
                col[s:e] = savgol_filter(col[s:e], w, poly)
        flat[:, c] = col
    return flat.reshape(x.shape)


def speed(pos: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Speed of a (N,D) trajectory using true timestamps (central differences)."""
    v = np.gradient(pos, times, axis=0)
    return np.linalg.norm(v, axis=1)


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for (...,D) arrays. NaN propagates."""
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def yaw_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Heading (deg) of the left->right segment in the horizontal (x-z) plane of world coords."""
    d = right - left
    return np.degrees(np.arctan2(d[..., 2], d[..., 0]))


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    ok = ~np.isnan(a)
    out[ok] = np.degrees(np.unwrap(np.radians(a[ok])))
    return out


def torso_length_px(xy: np.ndarray) -> float:
    """Median shoulder-mid to hip-mid distance in pixels: body-size scale for 2D speeds."""
    sh = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    hp = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    d = np.linalg.norm(sh - hp, axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else float("nan")


def hampel(x: np.ndarray, min_dev: float, half_window: int = 3, k: float = 3.0) -> np.ndarray:
    """Set single-frame tracking glitches to NaN along axis 0.

    A sample is a glitch when it is more than max(k * rolling MAD, min_dev) away from the rolling
    median. min_dev (same units as x) stops ordinary jitter on a still joint from being rejected.
    """
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(len(x), -1)
    if len(flat) < 2 * half_window + 1:
        return x.copy()
    pad = np.pad(flat, ((half_window, half_window), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half_window + 1, axis=0)  # (N, C, W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        med = np.nanmedian(win, axis=-1)
        mad = 1.4826 * np.nanmedian(np.abs(win - med[..., None]), axis=-1)
    out = flat.copy()
    out[np.abs(flat - med) > np.maximum(k * mad, min_dev)] = np.nan
    return out.reshape(x.shape)
