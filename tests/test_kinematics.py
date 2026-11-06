"""Geometry and signal-cleaning unit tests (synthetic inputs with known answers)."""
import numpy as np
import pytest

from tennis_analyzer import kinematics as K


def test_angle_3pt_right_angle_and_straight():
    a, b = np.array([1.0, 0, 0]), np.array([0.0, 0, 0])
    assert K.angle_3pt(a, b, np.array([0.0, 1, 0])) == pytest.approx(90)
    assert K.angle_3pt(a, b, np.array([-1.0, 0, 0])) == pytest.approx(180)


def test_angle_3pt_nan_propagates():
    assert np.isnan(K.angle_3pt(np.array([np.nan, 0, 0]), np.zeros(3), np.ones(3)))


def test_yaw_and_unwrap():
    left = np.zeros((4, 3))
    right = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0], [0, 0, -1]], float)
    yaw = K.yaw_deg(left, right)
    assert yaw[:2] == pytest.approx([0, 90])
    un = K.unwrap_deg(yaw)
    assert np.all(np.diff(un) > 0)  # a continuous quarter-turn sequence stays continuous
    assert un[-1] - un[0] == pytest.approx(270)


def test_fill_gaps_only_short_runs():
    t = np.arange(20) / 60.0
    x = np.arange(20, dtype=float)
    x[5:7] = np.nan  # 2 frames: filled
    x[10:19] = np.nan  # 9 frames (0.15 s): left alone
    y = K.fill_gaps(x, t, max_gap_s=0.12)
    assert y[5:7] == pytest.approx([5, 6])
    assert np.isnan(y[10:19]).all()


def test_hampel_removes_glitch_keeps_motion():
    x = np.linspace(0, 100, 50)
    x[25] += 80  # single-frame tracking glitch
    y = K.hampel(x, min_dev=10)
    assert np.isnan(y[25])
    assert np.isfinite(np.delete(y, 25)).all()


def test_hampel_keeps_jitter_on_still_joint():
    x = 100 + np.random.default_rng(0).normal(0, 0.5, 60)
    assert np.isfinite(K.hampel(x, min_dev=5)).all()


def test_smooth_preserves_nan_and_shape():
    x = np.sin(np.linspace(0, 6, 100))
    x[40:45] = np.nan
    y = K.smooth(x, fps=60)
    assert y.shape == x.shape
    assert np.isnan(y[40:45]).all() and np.isfinite(y[:40]).all()


def test_speed_uses_true_timestamps():
    t = np.array([0, 0.01, 0.03, 0.04])  # uneven frame spacing
    pos = np.stack([t * 2.0, np.zeros(4)], axis=1)  # 2 units/s
    assert K.speed(pos, t) == pytest.approx([2, 2, 2, 2])


@pytest.mark.parametrize("fps", [24, 25, 30, 60, 120, 240])
def test_smooth_is_centred_at_every_frame_rate(fps):
    """A symmetric bump must stay symmetric: no half-frame shift at low frame rates."""
    x = np.zeros(61)
    x[30] = 1.0
    y = K.smooth(x, fps=fps)
    assert y == pytest.approx(y[::-1], abs=1e-9)
    assert int(np.argmax(y)) == 30
