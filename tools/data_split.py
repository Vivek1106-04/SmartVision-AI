"""Video-grouped dataset loading and splitting.

Sliding windows are cut with stride 5 from a window size of 15, so consecutive
windows of one clip share 10 of their 15 frames. A random window-level split
therefore puts near-duplicate windows in both train and test and reports an
accuracy close to 100 percent that says nothing about generalisation.

Splitting by source video keeps every window of a clip inside exactly one fold,
which is the honest measurement for this dataset.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# Fold counts chosen so the held-out share is ~20 percent of the videos.
TEST_N_SPLITS = 5
VAL_N_SPLITS = 4
RANDOM_SEED = 42


def load_windows(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an extracted landmark archive, flattened to (N, frames, features).

    Returns X, y, groups (source video per window) and the class names.
    """
    data = np.load(data_path)
    X = data["X"]
    X = X.reshape((X.shape[0], X.shape[1], -1))
    return X, data["y"], data["video_sources"], data["exercise_names"]


def grouped_split(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split window indices into train/val/test with no video shared between them.

    Class balance is preserved as far as whole videos allow.
    """
    outer = StratifiedGroupKFold(n_splits=TEST_N_SPLITS, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(np.zeros(len(y)), y, groups))

    inner = StratifiedGroupKFold(n_splits=VAL_N_SPLITS, shuffle=True, random_state=seed)
    dev_train_pos, dev_val_pos = next(
        inner.split(np.zeros(len(dev_idx)), y[dev_idx], groups[dev_idx])
    )
    train_idx = dev_idx[dev_train_pos]
    val_idx = dev_idx[dev_val_pos]

    assert_disjoint_groups(groups, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def assert_disjoint_groups(
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Fail loudly if any source video appears in more than one fold."""
    train_g = set(groups[train_idx])
    val_g = set(groups[val_idx])
    test_g = set(groups[test_idx])
    overlaps = {
        "train/val": train_g & val_g,
        "train/test": train_g & test_g,
        "val/test": val_g & test_g,
    }
    leaked = {name: sorted(v) for name, v in overlaps.items() if v}
    if leaked:
        raise ValueError(f"Video leakage between folds: {leaked}")


def describe_split(
    y: np.ndarray,
    groups: np.ndarray,
    class_names: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> str:
    """Human-readable summary of the split, for logs and the report."""
    lines = ["Video-grouped split (no clip shared between folds)"]
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        per_class = ", ".join(
            f"{class_names[c]}={int(np.sum(y[idx] == c))}" for c in range(len(class_names))
        )
        lines.append(
            f"  {name:5s}: {len(idx):5d} windows from {len(set(groups[idx])):3d} videos  [{per_class}]"
        )
    return "\n".join(lines)
