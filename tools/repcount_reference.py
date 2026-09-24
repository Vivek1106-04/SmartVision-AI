"""Build a reference repetition count for the held-out clips.

The Kaggle clips carry no repetition labels, so the app's counter has nothing
to be scored against. This derives a reference count from the primary joint
angle signal itself: one repetition is one prominent flexion-extension cycle.

The method is deliberately independent of the rule thresholds in
src/exercises/*.js - it uses signal prominence, not the evaluator's angle
gates - so agreement between the two is informative rather than circular.

It is a semi-automatic reference, not hand annotation. The plots written to
results/repcount_plots/ exist so every count can be checked by eye before the
numbers are quoted.

Outputs:
  results/repcount_reference.csv
  results/repcount_plots/<exercise>.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from data_split import grouped_split, load_windows
from extract_landmarks import apply_ema_smoothing

# Joint triplets whose angle each evaluator treats as its primary readout.
PRIMARY_ANGLE_JOINTS = {
    "squat": (23, 25, 27),
    "pushup": (11, 13, 15),
    "bicepCurl": (11, 13, 15),
}

# A repetition has to dip at least this far and last at least this long,
# which rejects jitter and pose-estimation dropouts without tuning per clip.
# The threshold is a judgement call about how shallow a movement still counts,
# so every clip is also counted at a looser and a stricter setting and the
# report quotes the spread rather than pretending one cutoff is objective.
MIN_PROMINENCE_DEG = 30.0
PROMINENCE_SWEEP_DEG = (20.0, 30.0, 40.0)
MIN_SEPARATION_FRAMES = 12
ASSUMED_FPS = 30.0


def joint_angle(frame: np.ndarray, joints: tuple[int, int, int],
                width: int, height: int) -> float:
    """Angle at the middle joint, in true on-screen degrees."""
    a, b, c = (frame[i, :2] * np.array([width, height]) for i in joints)
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return np.nan
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def angle_signal(sequence: np.ndarray, exercise: str, width: int, height: int) -> np.ndarray:
    """Smoothed primary-angle trace for one clip."""
    smoothed = apply_ema_smoothing(sequence.astype(np.float64), alpha=0.35)
    joints = PRIMARY_ANGLE_JOINTS[exercise]
    angles = np.array([joint_angle(frame, joints, width, height) for frame in smoothed])
    # Carry the last valid value across dropouts so find_peaks sees no gaps.
    return pd.Series(angles).ffill().bfill().to_numpy()


def count_repetitions(angles: np.ndarray,
                      prominence: float = MIN_PROMINENCE_DEG) -> tuple[int, np.ndarray]:
    """One repetition per prominent minimum of the angle trace."""
    minima, _ = find_peaks(-angles, prominence=prominence,
                           distance=MIN_SEPARATION_FRAMES)
    return len(minima), minima


def plot_clips(records: list[dict], output_dir: str) -> None:
    """One figure per exercise so each reference count can be checked by eye."""
    os.makedirs(output_dir, exist_ok=True)

    for exercise in sorted({r["exercise"] for r in records}):
        clips = [r for r in records if r["exercise"] == exercise]
        columns = 3
        rows = int(np.ceil(len(clips) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 2.4 * rows),
                                    squeeze=False)

        for axis, clip in zip(axes.flat, clips):
            axis.plot(clip["angles"], linewidth=1.0, color="#2b6cb0")
            axis.plot(clip["minima"], clip["angles"][clip["minima"]], "v",
                      color="#c53030", markersize=6)
            axis.set_title(f"{clip['video'][:28]}\nreference={clip['reference_reps']}",
                           fontsize=8)
            axis.tick_params(labelsize=7)
        for axis in axes.flat[len(clips):]:
            axis.axis("off")

        figure.suptitle(f"{exercise}: primary joint angle, red markers are counted repetitions",
                        fontsize=11)
        figure.tight_layout()
        path = os.path.join(output_dir, f"{exercise}.png")
        figure.savefig(path, dpi=110)
        plt.close(figure)
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", default="dataset/processed/sequences.npz")
    parser.add_argument("--windows_path", default="dataset/processed/landmarks.npz")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    data = np.load(args.sequences, allow_pickle=True)
    _, y, groups, _ = load_windows(args.windows_path)
    _, _, test_idx = grouped_split(y, groups)
    test_videos = set(groups[test_idx])

    records = []
    for name, exercise, width, height, sequence in zip(
        data["videos"], data["exercises"], data["widths"], data["heights"], data["sequences"]
    ):
        if str(name) not in test_videos:
            continue

        angles = angle_signal(sequence, str(exercise), int(width), int(height))
        reps, minima = count_repetitions(angles)
        sweep = {
            f"reference_reps_p{int(prominence)}": count_repetitions(angles, prominence)[0]
            for prominence in PROMINENCE_SWEEP_DEG
        }
        records.append({
            "video": str(name),
            "exercise": str(exercise),
            "frames": len(angles),
            "duration_s": round(len(angles) / ASSUMED_FPS, 2),
            "reference_reps": reps,
            **sweep,
            "angles": angles,
            "minima": minima,
        })

    os.makedirs(args.results_dir, exist_ok=True)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("angles", "minima")}
                          for r in records])
    csv_path = os.path.join(args.results_dir, "repcount_reference.csv")
    frame.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(frame)} clips)")

    print("Plots for manual verification:")
    plot_clips(records, os.path.join(args.results_dir, "repcount_plots"))

    print(f"\nReference repetitions at {MIN_PROMINENCE_DEG:.0f} deg prominence: "
          f"total {int(frame['reference_reps'].sum())}, "
          f"median {frame['reference_reps'].median():.0f} per clip")
    for prominence in PROMINENCE_SWEEP_DEG:
        column = f"reference_reps_p{int(prominence)}"
        print(f"  at {prominence:.0f} deg: {int(frame[column].sum())} repetitions")


if __name__ == "__main__":
    main()
