import os
import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Conv1D, GlobalAveragePooling1D

from data_split import load_windows, grouped_split, describe_split

def build_lstm(input_shape, n_classes):
    return Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True), Dropout(0.3),
        LSTM(32), Dropout(0.3),
        Dense(32, activation='relu'), Dense(n_classes, activation='softmax')
    ])

def build_gru(input_shape, n_classes):
    return Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True), Dropout(0.3),
        GRU(32), Dropout(0.3),
        Dense(32, activation='relu'), Dense(n_classes, activation='softmax')
    ])

def build_1dcnn(input_shape, n_classes):
    return Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu'), Dropout(0.3),
        Conv1D(32, 3, activation='relu'), GlobalAveragePooling1D(),
        Dense(32, activation='relu'), Dense(n_classes, activation='softmax')
    ])

def build_mlp(input_shape, n_classes):
    """MLP over a single frame — the no-temporal-information baseline.

    input_shape is (n_features,): one frame, not a window.
    """
    return Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/processed/landmarks.npz')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of random initialisations per variant')
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Loading normalized data from {args.data_path}...")
    try:
        X, y, groups, exercise_names = load_windows(args.data_path)
    except Exception as e:
        print(f"Data not found: {e}")
        return

    n_classes = len(exercise_names)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=n_classes)

    train_idx, val_idx, test_idx = grouped_split(y, groups)
    print(describe_split(y, groups, exercise_names, train_idx, val_idx, test_idx))

    y_train, y_val, y_test = y_cat[train_idx], y_cat[val_idx], y_cat[test_idx]
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Load un-normalized data for the No-Norm ablation. It is extracted in the
    # same order as the normalized archive, so the same split indices apply and
    # the two variants stay directly comparable.
    raw_data_path = args.data_path.replace('landmarks.npz', 'landmarks_raw.npz')
    X_raw_train, X_raw_val, X_raw_test = X_train, X_val, X_test  # fallback to same data
    if os.path.exists(raw_data_path):
        print(f"Loading un-normalized data from {raw_data_path}...")
        X_raw, y_raw, groups_raw, _ = load_windows(raw_data_path)
        if not np.array_equal(groups_raw, groups) or not np.array_equal(y_raw, y):
            raise ValueError("landmarks_raw.npz is not aligned with landmarks.npz; re-run extract_landmarks.py")
        X_raw_train, X_raw_val, X_raw_test = X_raw[train_idx], X_raw[val_idx], X_raw[test_idx]
    else:
        print("Warning: landmarks_raw.npz not found, using normalized data for No-Norm variant too.")

    # The single-frame baseline sees only the last frame of each window, so any
    # gap to the sequence models is attributable to temporal information.
    frame_shape = (X_train.shape[2],)
    last_frame = lambda arr: arr[:, -1, :]

    # Each variant is (name, builder, input_shape, train, val, test). The builder
    # is re-invoked per seed so every repeat starts from a fresh initialisation.
    variants = [
        ('LSTM (Main)',      build_lstm,  input_shape, X_train, X_val, X_test),
        ('GRU',              build_gru,   input_shape, X_train, X_val, X_test),
        ('1D-CNN',           build_1dcnn, input_shape, X_train, X_val, X_test),
        ('LSTM (No-Norm)',   build_lstm,  input_shape, X_raw_train, X_raw_val, X_raw_test),
        ('Single-Frame MLP', build_mlp,   frame_shape,
         last_frame(X_train), last_frame(X_val), last_frame(X_test)),
    ]

    # Run-to-run spread on this dataset is several accuracy points, so a single
    # seed cannot separate the variants. Every variant is repeated and reported
    # as mean +/- standard deviation.
    runs = []
    for name, builder, shape, X_tr, X_va, X_te in variants:
        print(f"\nTraining {name} ({args.seeds} seeds)...")
        for seed in range(args.seeds):
            tf.keras.utils.set_random_seed(seed)
            model = builder(shape, n_classes)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                          loss='categorical_crossentropy', metrics=['accuracy'])
            started = time.perf_counter()
            model.fit(X_tr, y_train, epochs=30, batch_size=32, verbose=0,
                      validation_data=(X_va, y_val))
            train_seconds = time.perf_counter() - started
            loss, acc = model.evaluate(X_te, y_test, verbose=0)
            print(f"  seed {seed}: acc={acc:.4f} loss={loss:.4f} ({train_seconds:.1f}s)")
            runs.append({
                'Model': name,
                'Seed': seed,
                'Parameters': int(model.count_params()),
                'Test_Accuracy': acc,
                'Test_Loss': loss,
                'Train_Seconds': round(train_seconds, 1),
            })

    df_runs = pd.DataFrame(runs)
    df_runs.to_csv(os.path.join(args.results_dir, 'ablation_runs.csv'), index=False)

    df_results = df_runs.groupby('Model', sort=False).agg(
        Parameters=('Parameters', 'first'),
        Test_Accuracy_Mean=('Test_Accuracy', 'mean'),
        Test_Accuracy_Std=('Test_Accuracy', 'std'),
        Test_Loss_Mean=('Test_Loss', 'mean'),
        Test_Loss_Std=('Test_Loss', 'std'),
        Train_Seconds_Mean=('Train_Seconds', 'mean'),
        Seeds=('Seed', 'count'),
    ).reset_index()
    df_results.to_csv(os.path.join(args.results_dir, 'ablation_results.csv'), index=False)

    print()
    print(df_results.to_string(index=False))
    print("\nAblation study complete. Run tools/generate_plots.py to render figures.")

if __name__ == '__main__':
    main()
