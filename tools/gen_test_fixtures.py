"""Generate train/serve parity fixtures.

The browser re-implements the preprocessing of tools/extract_landmarks.py and
the forward pass of the trained Keras model in plain JavaScript
(src/core/dlClassifier.js).  Two implementations of the same maths drift
silently, so this script records what the Python side produces and
tests/js/*.test.mjs asserts the JavaScript side reproduces it.

Output: tests/fixtures/*.json
"""

import argparse
import json
import os

import numpy as np

from data_split import grouped_split, load_windows
from extract_landmarks import apply_ema_smoothing, normalize_landmarks

NUM_LANDMARKS = 33
SEQUENCE_FRAMES = 20
PREPROCESS_SEED = 20260919
NUM_MODEL_SAMPLES = 24


def synthetic_raw_sequence(seed: int = PREPROCESS_SEED) -> np.ndarray:
    """A deterministic stand-in for a MediaPipe result stream.

    Coordinates sit in the ranges MediaPipe emits (x, y in [0, 1], z roughly
    centred on zero, visibility in [0, 1]) so the normalization path is
    exercised on realistic magnitudes rather than on arbitrary numbers.
    """
    rng = np.random.default_rng(seed)
    sequence = np.empty((SEQUENCE_FRAMES, NUM_LANDMARKS, 4), dtype=np.float64)
    sequence[..., 0] = rng.uniform(0.2, 0.8, (SEQUENCE_FRAMES, NUM_LANDMARKS))
    sequence[..., 1] = rng.uniform(0.1, 0.9, (SEQUENCE_FRAMES, NUM_LANDMARKS))
    sequence[..., 2] = rng.uniform(-0.4, 0.4, (SEQUENCE_FRAMES, NUM_LANDMARKS))
    sequence[..., 3] = rng.uniform(0.6, 1.0, (SEQUENCE_FRAMES, NUM_LANDMARKS))
    return sequence


def preprocess_fixture() -> dict:
    """Input sequence plus the normalized and EMA-smoothed reference output."""
    raw = synthetic_raw_sequence()
    normalized = normalize_landmarks(np.copy(raw))
    smoothed = apply_ema_smoothing(normalized, alpha=0.35)
    return {
        "description": "normalize_landmarks + apply_ema_smoothing from tools/extract_landmarks.py",
        "emaAlpha": 0.35,
        "raw": raw.tolist(),
        "normalized": normalized.tolist(),
        "smoothed": smoothed.tolist(),
    }


def model_fixture(data_path: str, model_path: str) -> dict:
    """Held-out windows and the probabilities Keras produces for them."""
    import tensorflow as tf

    X, y, groups, class_names = load_windows(data_path)
    _, _, test_idx = grouped_split(y, groups)

    # Spread the samples across the test fold so every class is represented.
    picked = test_idx[np.linspace(0, len(test_idx) - 1, NUM_MODEL_SAMPLES).astype(int)]
    windows = X[picked]

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(windows, verbose=0)

    return {
        "description": "Keras forward pass of models/exercise_classifier.h5 on held-out windows",
        "classNames": [str(name) for name in class_names],
        "windows": windows.tolist(),
        "probabilities": probabilities.tolist(),
        "labels": [int(label) for label in y[picked]],
        "sources": [str(source) for source in groups[picked]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="dataset/processed/landmarks.npz")
    parser.add_argument("--model_path", default="models/exercise_classifier.h5")
    parser.add_argument("--output_dir", default="tests/fixtures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fixtures = {
        "preprocess.json": preprocess_fixture(),
        "model_forward.json": model_fixture(args.data_path, args.model_path),
    }

    for filename, payload in fixtures.items():
        path = os.path.join(args.output_dir, filename)
        with open(path, "w") as handle:
            json.dump(payload, handle)
        print(f"Wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
