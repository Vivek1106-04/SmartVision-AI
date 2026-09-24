"""Measure which body parts the trained classifier actually relies on.

The XAI panel previously showed hand-written constants. This computes real
permutation importance: for each anatomical landmark group, the features of
that group are shuffled across the held-out samples and the resulting drop in
accuracy is recorded. A group the model ignores costs nothing to destroy; a
group it depends on costs a lot.

Outputs:
  results/permutation_importance.csv                        - full measurement
  public/models/exercise_classifier/saliency.json           - what the app loads
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from data_split import grouped_split, load_windows

# MediaPipe BlazePose landmark indices, grouped into parts a reader recognises.
LANDMARK_GROUPS = {
    "Head and Face": tuple(range(0, 11)),
    "Shoulders": (11, 12),
    "Elbows": (13, 14),
    "Wrists": (15, 16),
    "Hands": tuple(range(17, 23)),
    "Hips": (23, 24),
    "Knees": (25, 26),
    "Ankles": (27, 28),
    "Feet": tuple(range(29, 33)),
}

VALUES_PER_LANDMARK = 4  # x, y, z, visibility
DEFAULT_REPEATS = 10
TOP_FEATURES_PER_CLASS = 3
RANDOM_SEED = 42


def feature_columns(landmark_indices: tuple[int, ...]) -> np.ndarray:
    """Column positions in the flattened (frames, 132) feature vector."""
    return np.array([
        index * VALUES_PER_LANDMARK + offset
        for index in landmark_indices
        for offset in range(VALUES_PER_LANDMARK)
    ])


def accuracy_per_class(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Recall for each class, plus overall accuracy in the last position."""
    scores = np.empty(n_classes + 1)
    for class_id in range(n_classes):
        mask = labels == class_id
        scores[class_id] = float(np.mean(predictions[mask] == class_id)) if mask.any() else np.nan
    scores[n_classes] = float(np.mean(predictions == labels))
    return scores


def permutation_importance(model, X, y, n_classes: int, repeats: int) -> pd.DataFrame:
    """Accuracy drop per landmark group, averaged over several shuffles."""
    baseline = accuracy_per_class(np.argmax(model.predict(X, verbose=0), axis=1), y, n_classes)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for group_name, landmark_indices in LANDMARK_GROUPS.items():
        columns = feature_columns(landmark_indices)
        drops = np.empty((repeats, n_classes + 1))

        for repeat in range(repeats):
            shuffled = X.copy()
            # Shuffle this group across samples, keeping each sample's own
            # 15-frame trajectory intact, so only the link between this body
            # part and the label is destroyed.
            order = rng.permutation(len(X))
            shuffled[:, :, columns] = X[order][:, :, columns]
            scores = accuracy_per_class(
                np.argmax(model.predict(shuffled, verbose=0), axis=1), y, n_classes
            )
            drops[repeat] = baseline - scores

        rows.append({
            "group": group_name,
            "landmarks": ",".join(str(i) for i in landmark_indices),
            "drop_mean": drops[:, n_classes].mean(),
            "drop_std": drops[:, n_classes].std(),
            **{f"drop_class_{c}_mean": drops[:, c].mean() for c in range(n_classes)},
            **{f"drop_class_{c}_std": drops[:, c].std() for c in range(n_classes)},
        })
        print(f"  {group_name:15s} accuracy drop {rows[-1]['drop_mean']:+.4f}"
              f" +/- {rows[-1]['drop_std']:.4f}")

    frame = pd.DataFrame(rows).sort_values("drop_mean", ascending=False)
    frame.attrs["baseline"] = baseline
    return frame


def build_saliency(frame: pd.DataFrame, class_names: list[str]) -> dict:
    """Top groups per class, as weights that sum to one for the UI bars."""
    saliency = {}
    for class_id, class_name in enumerate(class_names):
        column = f"drop_class_{class_id}_mean"
        ranked = frame.sort_values(column, ascending=False).head(TOP_FEATURES_PER_CLASS)

        # Negative drops mean the shuffle helped by chance; clamp so the bars
        # stay meaningful, and fall back to an even split if nothing mattered.
        drops = ranked[column].clip(lower=0.0).to_numpy()
        total = drops.sum()
        weights = drops / total if total > 0 else np.full(len(drops), 1.0 / len(drops))

        saliency[class_name] = [
            {"name": group, "weight": round(float(weight), 4)}
            for group, weight in zip(ranked["group"], weights)
        ]
    return saliency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="dataset/processed/landmarks.npz")
    parser.add_argument("--model_path", default="models/exercise_classifier.h5")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_dir", default="public/models/exercise_classifier")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    import tensorflow as tf

    X, y, groups, class_names = load_windows(args.data_path)
    _, _, test_idx = grouped_split(y, groups)
    X_test, y_test = X[test_idx], y[test_idx]
    class_names = [str(name) for name in class_names]

    print(f"Measuring permutation importance on {len(X_test)} held-out windows "
          f"from {len(set(groups[test_idx]))} videos, {args.repeats} shuffles per group")

    model = tf.keras.models.load_model(args.model_path)
    frame = permutation_importance(model, X_test, y_test, len(class_names), args.repeats)

    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "permutation_importance.csv")
    frame.to_csv(csv_path, index=False)
    print(f"\nBaseline test accuracy: {frame.attrs['baseline'][-1]:.4f}")
    print(f"Wrote {csv_path}")

    saliency = {
        "method": "permutation importance, accuracy drop over "
                  f"{args.repeats} shuffles of the held-out fold",
        "baselineAccuracy": round(float(frame.attrs["baseline"][-1]), 4),
        "classes": build_saliency(frame, class_names),
    }
    json_path = os.path.join(args.model_dir, "saliency.json")
    with open(json_path, "w") as handle:
        json.dump(saliency, handle, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
