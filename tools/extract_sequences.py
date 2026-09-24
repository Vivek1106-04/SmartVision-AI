"""Extract full-length, un-normalized landmark sequences per video.

tools/extract_landmarks.py already produces the training tensors, but it only
keeps normalized, EMA-smoothed 15-frame windows.  The offline replay harness
(tools/bench/replay.mjs) needs what the browser actually receives from
MediaPipe: raw image-space landmarks, one entry per frame, un-smoothed, so the
JavaScript CVEngine can apply its own smoothing exactly as it does at runtime.

Output: dataset/processed/sequences.npz  (object array, one entry per video)
"""

import argparse
import json
import os

import numpy as np

from extract_landmarks import create_pose_landmarker, extract_landmarks, get_videos

EXERCISES = ("squat", "pushup", "bicepCurl")
MIN_FRAMES = 15
JSON_COORD_PRECISION = 6


def video_dimensions(video_path: str) -> tuple[int, int]:
    """Frame width and height, needed to undo normalized-coordinate distortion."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def extract_all(dataset_dir: str, model_path: str) -> list[dict]:
    """Run MediaPipe over every video and return one record per usable clip."""
    records = []
    for exercise in EXERCISES:
        exercise_dir = os.path.join(dataset_dir, exercise)
        if not os.path.exists(exercise_dir):
            print(f"WARNING: {exercise_dir} missing, skipping")
            continue

        videos = get_videos(exercise_dir)
        print(f"\n{exercise}: {len(videos)} videos")

        for index, video in enumerate(videos):
            name = os.path.basename(video)
            print(f"  [{index + 1}/{len(videos)}] {name}", end=" ")

            # A fresh landmarker per video resets MediaPipe's tracking state.
            landmarker = create_pose_landmarker(model_path)
            landmarks = extract_landmarks(video, landmarker)

            if len(landmarks) < MIN_FRAMES:
                print(f"SKIP ({len(landmarks)} frames)")
                continue

            width, height = video_dimensions(video)
            records.append({
                "video": name,
                "exercise": exercise,
                "width": width,
                "height": height,
                "landmarks": np.asarray(landmarks, dtype=np.float32),
            })
            print(f"OK {len(landmarks)} frames ({width}x{height})")

    return records


def save_npz(records: list[dict], out_path: str) -> None:
    """Persist variable-length sequences as an object array."""
    np.savez_compressed(
        out_path,
        videos=np.array([r["video"] for r in records]),
        exercises=np.array([r["exercise"] for r in records]),
        widths=np.array([r["width"] for r in records]),
        heights=np.array([r["height"] for r in records]),
        sequences=np.array([r["landmarks"] for r in records], dtype=object),
    )


def save_json(records: list[dict], out_path: str, only: set[str] | None) -> None:
    """Write a JSON copy the Node replay harness can read without numpy."""
    selected = [r for r in records if only is None or r["video"] in only]
    payload = [
        {
            "video": r["video"],
            "exercise": r["exercise"],
            "width": r["width"],
            "height": r["height"],
            "landmarks": np.round(r["landmarks"], JSON_COORD_PRECISION).tolist(),
        }
        for r in selected
    ]
    with open(out_path, "w") as handle:
        json.dump(payload, handle)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Wrote {len(payload)} sequences to {out_path} ({size_mb:.1f} MB)")


def test_fold_videos(data_path: str) -> set[str]:
    """Video names in the held-out test fold, so the JSON stays small."""
    from data_split import grouped_split, load_windows

    _, y, groups, _ = load_windows(data_path)
    _, _, test_idx = grouped_split(y, groups)
    return set(groups[test_idx])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="dataset/raw")
    parser.add_argument("--output_dir", default="dataset/processed")
    parser.add_argument("--model_path", default="tools/pose_landmarker_full.task")
    parser.add_argument("--windows_path", default="dataset/processed/landmarks.npz")
    parser.add_argument("--json_all", action="store_true",
                        help="export every clip to JSON, not only the test fold")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model file not found at {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    records = extract_all(args.dataset_dir, args.model_path)
    if not records:
        raise SystemExit("No sequences extracted")

    npz_path = os.path.join(args.output_dir, "sequences.npz")
    save_npz(records, npz_path)
    print(f"\nSaved {len(records)} sequences to {npz_path}")

    only = None if args.json_all else test_fold_videos(args.windows_path)
    save_json(records, os.path.join(args.output_dir, "sequences_test.json"), only)


if __name__ == "__main__":
    main()
