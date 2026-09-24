"""Unit tests for the landmark preprocessing used to build the training tensors.

The browser re-implements these same steps in JavaScript; tests/js/preprocess.test.mjs
asserts the two agree. These tests pin down what the Python side is supposed to do.
"""

import numpy as np
import pytest

from extract_landmarks import apply_ema_smoothing, create_windows, normalize_landmarks

NUM_LANDMARKS = 33
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def make_frame(offset: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """One frame of a body translated by `offset` and scaled by `scale`."""
    frame = np.zeros((NUM_LANDMARKS, 4))
    frame[:, 3] = 0.9
    frame[LEFT_HIP, :3] = [-0.1 * scale, 0.5 * scale, 0.0]
    frame[RIGHT_HIP, :3] = [0.1 * scale, 0.5 * scale, 0.0]
    frame[LEFT_SHOULDER, :3] = [-0.1 * scale, 0.2 * scale, 0.0]
    frame[RIGHT_SHOULDER, :3] = [0.1 * scale, 0.2 * scale, 0.0]
    frame[0, :3] = [0.0, 0.1 * scale, 0.0]
    frame[:, :3] += offset
    return frame


@pytest.mark.unit
def test_normalization_puts_mid_hip_at_the_origin():
    normalized = normalize_landmarks(make_frame()[None, ...])[0]

    mid_hip = (normalized[LEFT_HIP, :3] + normalized[RIGHT_HIP, :3]) / 2
    assert np.allclose(mid_hip, 0.0, atol=1e-12)


@pytest.mark.unit
def test_normalization_makes_torso_length_one():
    normalized = normalize_landmarks(make_frame()[None, ...])[0]

    mid_hip = (normalized[LEFT_HIP, :3] + normalized[RIGHT_HIP, :3]) / 2
    mid_shoulder = (normalized[LEFT_SHOULDER, :3] + normalized[RIGHT_SHOULDER, :3]) / 2
    assert np.isclose(np.linalg.norm(mid_shoulder - mid_hip), 1.0)


@pytest.mark.unit
def test_normalization_removes_camera_distance_and_subject_size():
    """Two bodies differing only in position and size must normalize to the same pose."""
    near = normalize_landmarks(make_frame()[None, ...])[0]
    far = normalize_landmarks(make_frame(offset=0.3, scale=0.5)[None, ...])[0]

    assert np.allclose(near[:, :3], far[:, :3], atol=1e-12)


@pytest.mark.unit
def test_normalization_leaves_visibility_untouched():
    frame = make_frame()
    normalized = normalize_landmarks(frame[None, ...])[0]

    assert np.allclose(normalized[:, 3], frame[:, 3])


@pytest.mark.unit
def test_normalization_survives_a_collapsed_torso():
    """A zero-length torso must not divide the coordinates by zero."""
    frame = make_frame()
    frame[LEFT_SHOULDER, :3] = frame[LEFT_HIP, :3]
    frame[RIGHT_SHOULDER, :3] = frame[RIGHT_HIP, :3]

    normalized = normalize_landmarks(frame[None, ...])[0]
    assert np.isfinite(normalized).all()


@pytest.mark.unit
def test_ema_passes_the_first_frame_through_unchanged():
    sequence = np.random.default_rng(0).random((6, NUM_LANDMARKS, 4))
    smoothed = apply_ema_smoothing(sequence, alpha=0.35)

    assert np.allclose(smoothed[0], sequence[0])


@pytest.mark.unit
def test_ema_follows_the_recurrence():
    alpha = 0.35
    sequence = np.random.default_rng(1).random((5, NUM_LANDMARKS, 4))
    smoothed = apply_ema_smoothing(sequence, alpha=alpha)

    for i in range(1, len(sequence)):
        expected = alpha * sequence[i] + (1 - alpha) * smoothed[i - 1]
        assert np.allclose(smoothed[i], expected)


@pytest.mark.unit
def test_ema_reduces_variance_of_a_noisy_signal():
    rng = np.random.default_rng(2)
    noisy = np.ones((200, NUM_LANDMARKS, 4)) + rng.normal(0, 0.1, (200, NUM_LANDMARKS, 4))
    smoothed = apply_ema_smoothing(noisy, alpha=0.35)

    assert smoothed[10:].var() < noisy[10:].var()


@pytest.mark.unit
@pytest.mark.parametrize(
    "frames,window,stride,expected",
    [(30, 15, 5, 4), (15, 15, 5, 1), (14, 15, 5, 0), (20, 15, 5, 2)],
)
def test_window_count_follows_frames_window_and_stride(frames, window, stride, expected):
    sequence = np.zeros((frames, NUM_LANDMARKS, 4))
    windows = create_windows(sequence, window_size=window, stride=stride)

    assert len(windows) == expected


@pytest.mark.unit
def test_windows_are_contiguous_slices_at_the_given_stride():
    sequence = np.arange(30 * NUM_LANDMARKS * 4).reshape(30, NUM_LANDMARKS, 4).astype(float)
    windows = create_windows(sequence, window_size=15, stride=5)

    for i, window in enumerate(windows):
        assert np.array_equal(window, sequence[i * 5 : i * 5 + 15])
