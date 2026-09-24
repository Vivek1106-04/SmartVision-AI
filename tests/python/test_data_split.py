"""Unit tests for the video-grouped split.

A random window-level split leaks near-duplicate windows across folds and
reports an accuracy close to 100 percent. These tests pin the property that
makes the reported numbers honest: no source video appears in two folds.
"""

import numpy as np
import pytest

from data_split import assert_disjoint_groups, describe_split, grouped_split

NUM_CLASSES = 3
VIDEOS_PER_CLASS = 12
WINDOWS_PER_VIDEO = 8


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels and source videos shaped like the real extraction output."""
    labels, groups = [], []
    for class_id in range(NUM_CLASSES):
        for video in range(VIDEOS_PER_CLASS):
            labels.extend([class_id] * WINDOWS_PER_VIDEO)
            groups.extend([f"class{class_id}_video{video}.mp4"] * WINDOWS_PER_VIDEO)
    class_names = np.array(["squat", "pushup", "bicepCurl"])
    return np.array(labels), np.array(groups), class_names


@pytest.mark.unit
def test_no_video_appears_in_two_folds(dataset):
    y, groups, _ = dataset
    train_idx, val_idx, test_idx = grouped_split(y, groups)

    train_videos = set(groups[train_idx])
    val_videos = set(groups[val_idx])
    test_videos = set(groups[test_idx])

    assert not train_videos & val_videos
    assert not train_videos & test_videos
    assert not val_videos & test_videos


@pytest.mark.unit
def test_every_window_lands_in_exactly_one_fold(dataset):
    y, groups, _ = dataset
    train_idx, val_idx, test_idx = grouped_split(y, groups)

    combined = np.concatenate([train_idx, val_idx, test_idx])
    assert len(combined) == len(np.unique(combined)) == len(y)


@pytest.mark.unit
def test_every_class_is_represented_in_every_fold(dataset):
    y, groups, _ = dataset
    for idx in grouped_split(y, groups):
        assert set(np.unique(y[idx])) == set(range(NUM_CLASSES))


@pytest.mark.unit
def test_split_is_deterministic_for_a_given_seed(dataset):
    y, groups, _ = dataset
    first = grouped_split(y, groups, seed=7)
    second = grouped_split(y, groups, seed=7)

    for a, b in zip(first, second):
        assert np.array_equal(a, b)


@pytest.mark.unit
def test_a_different_seed_produces_a_different_split(dataset):
    y, groups, _ = dataset
    _, _, test_a = grouped_split(y, groups, seed=1)
    _, _, test_b = grouped_split(y, groups, seed=2)

    assert not np.array_equal(test_a, test_b)


@pytest.mark.unit
def test_held_out_share_is_roughly_a_fifth_of_the_videos(dataset):
    y, groups, _ = dataset
    _, _, test_idx = grouped_split(y, groups)

    share = len(set(groups[test_idx])) / len(set(groups))
    assert 0.1 < share < 0.35


@pytest.mark.unit
def test_leak_detector_rejects_a_shared_video(dataset):
    y, groups, _ = dataset
    train_idx, val_idx, test_idx = grouped_split(y, groups)
    leaked = np.concatenate([train_idx, test_idx[:1]])

    with pytest.raises(ValueError, match="Video leakage"):
        assert_disjoint_groups(groups, leaked, val_idx, test_idx)


@pytest.mark.unit
def test_summary_reports_every_fold_and_class(dataset):
    y, groups, class_names = dataset
    train_idx, val_idx, test_idx = grouped_split(y, groups)

    summary = describe_split(y, groups, class_names, train_idx, val_idx, test_idx)

    for expected in ("train", "val", "test", *class_names):
        assert expected in summary
