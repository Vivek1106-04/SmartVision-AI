import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os
import argparse
import glob

# Number of landmarks MediaPipe Pose outputs
NUM_LANDMARKS = 33


def create_pose_landmarker(model_path):
    """Create a PoseLandmarker using the new MediaPipe Tasks API."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_landmarks(video_path, landmarker):
    """Extract 33 pose landmarks from each frame of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠ Could not open {video_path}")
        return np.array([])

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    landmarks_list = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp in milliseconds
        timestamp_ms = int(frame_idx * (1000.0 / fps))

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            frame_idx += 1
            continue

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            pose = result.pose_landmarks[0]  # First (and only) detected pose
            frame_marks = []
            for lm in pose:
                frame_marks.append([lm.x, lm.y, lm.z, lm.visibility])
            landmarks_list.append(frame_marks)

        frame_idx += 1

    cap.release()
    return np.array(landmarks_list)


def normalize_landmarks(landmarks):
    """Translate to mid-hip origin and scale by torso length."""
    if len(landmarks) == 0:
        return landmarks

    normalized = np.copy(landmarks)

    for i in range(len(normalized)):
        frame_lm = normalized[i]

        # Mid-hip as origin
        left_hip = frame_lm[23][:3]
        right_hip = frame_lm[24][:3]
        mid_hip = (left_hip + right_hip) / 2.0

        # Torso length for scale normalization
        left_shoulder = frame_lm[11][:3]
        right_shoulder = frame_lm[12][:3]
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0

        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        if torso_length < 1e-5:
            torso_length = 1.0

        # Translate and scale xyz, keep visibility untouched
        frame_lm[:, :3] = (frame_lm[:, :3] - mid_hip) / torso_length
        normalized[i] = frame_lm

    return normalized


def apply_ema_smoothing(landmarks, alpha=0.35):
    """Exponential Moving Average smoothing — matches cvEngine.js."""
    if len(landmarks) == 0:
        return landmarks

    smoothed = np.copy(landmarks)
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * landmarks[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def create_windows(landmarks, window_size=15, stride=5):
    """Create sliding windows of fixed length for temporal classification."""
    windows = []
    for i in range(0, len(landmarks) - window_size + 1, stride):
        windows.append(landmarks[i : i + window_size])
    return np.array(windows) if len(windows) > 0 else np.array([])


def get_videos(directory):
    """Recursively find all video files in a directory."""
    videos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.webm", "*.MP4", "*.MOV"]:
        videos.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose landmarks from exercise videos")
    parser.add_argument("--dataset_dir", type=str, default="dataset/raw", help="Path to raw dataset")
    parser.add_argument("--output_dir", type=str, default="dataset/processed", help="Path to processed output")
    parser.add_argument("--model_path", type=str, default="tools/pose_landmarker_full.task",
                        help="Path to MediaPipe PoseLandmarker .task model file")
    args = parser.parse_args()

    # Verify model file exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        print("Download it with:")
        print('  curl -L -o tools/pose_landmarker_full.task '
              '"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"')
        return

    print(f"Loading PoseLandmarker model from {args.model_path}...")
    landmarker = create_pose_landmarker(args.model_path)

    X_all, y_all, video_sources_all = [], [], []
    X_raw_all = []  # Un-normalized version for ablation
    exercises = ["squat", "pushup", "bicepCurl"]
    manifest_data = []

    os.makedirs(args.output_dir, exist_ok=True)
    class_map = {ex: i for i, ex in enumerate(exercises)}

    for ex in exercises:
        ex_dir = os.path.join(args.dataset_dir, ex)
        if not os.path.exists(ex_dir):
            print(f"⚠ Warning: {ex_dir} does not exist, skipping.")
            continue

        videos = get_videos(ex_dir)
        print(f"\n{'='*50}")
        print(f"Processing {ex}: {len(videos)} videos found")
        print(f"{'='*50}")

        for idx, vid in enumerate(videos):
            print(f"  [{idx+1}/{len(videos)}] {os.path.basename(vid)}...", end=" ")

            # Need a fresh landmarker per video to reset tracking state
            landmarker = create_pose_landmarker(args.model_path)

            lms = extract_landmarks(vid, landmarker)

            if len(lms) == 0:
                print("✗ no pose detected")
                continue

            if len(lms) < 15:
                print(f"✗ too short ({len(lms)} frames)")
                continue

            # Save raw (un-normalized) windows for ablation
            lms_raw = apply_ema_smoothing(lms, alpha=0.35)
            raw_windows = create_windows(lms_raw, window_size=15, stride=5)
            if len(raw_windows) > 0:
                X_raw_all.append(raw_windows)

            # Normalized pipeline
            lms_norm = normalize_landmarks(lms)
            lms_norm = apply_ema_smoothing(lms_norm, alpha=0.35)
            windows = create_windows(lms_norm, window_size=15, stride=5)

            if len(windows) > 0:
                X_all.append(windows)
                y_all.extend([class_map[ex]] * len(windows))
                video_sources_all.extend([os.path.basename(vid)] * len(windows))
                print(f"✓ {len(lms)} frames → {len(windows)} windows")
            else:
                print("✗ no windows created")

            manifest_data.append({
                "video": os.path.basename(vid),
                "exercise": ex,
                "frames": len(lms),
                "windows": len(windows) if len(windows) > 0 else 0,
            })

    # Save results
    if len(X_all) > 0:
        X = np.concatenate(X_all, axis=0)
        y = np.array(y_all)
        video_sources = np.array(video_sources_all)
        exercise_names = np.array(exercises)

        out_path = os.path.join(args.output_dir, "landmarks.npz")
        np.savez_compressed(out_path, X=X, y=y, exercise_names=exercise_names, video_sources=video_sources)
        print(f"\n✓ Saved normalized landmarks to {out_path}")
        print(f"  X shape: {X.shape}  (samples, frames, landmarks, features)")

        # Save un-normalized version for ablation
        if len(X_raw_all) > 0:
            X_raw = np.concatenate(X_raw_all, axis=0)
            raw_path = os.path.join(args.output_dir, "landmarks_raw.npz")
            np.savez_compressed(raw_path, X=X_raw, y=y, exercise_names=exercise_names, video_sources=video_sources)
            print(f"✓ Saved un-normalized landmarks to {raw_path}")
            print(f"  X_raw shape: {X_raw.shape}")

        # Save manifest
        df_manifest = pd.DataFrame(manifest_data)
        manifest_path = os.path.normpath(os.path.join(args.output_dir, "../manifest.csv"))
        df_manifest.to_csv(manifest_path, index=False)
        print(f"✓ Saved manifest to {manifest_path}")

        # Print summary
        print(f"\n{'='*50}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total videos processed: {len(manifest_data)}")
        print(f"Total sliding windows:  {len(X)}")
        for ex in exercises:
            count = np.sum(y == class_map[ex])
            print(f"  {ex:12s}: {count} windows")
        print(f"{'='*50}")
    else:
        print("\n✗ No data extracted. Check that videos contain visible people.")


if __name__ == "__main__":
    main()
