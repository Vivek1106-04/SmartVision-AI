import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from data_split import load_windows, grouped_split, describe_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/processed/landmarks.npz')
    parser.add_argument('--models_dir', type=str, default='models')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_path}...")
    try:
        X, y, groups, exercise_names = load_windows(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    n_classes = len(exercise_names)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=n_classes)

    train_idx, val_idx, test_idx = grouped_split(y, groups)
    split_summary = describe_split(y, groups, exercise_names, train_idx, val_idx, test_idx)
    print(split_summary)
    with open(os.path.join(args.results_dir, 'split_summary.txt'), 'w') as f:
        f.write(split_summary + "\n")

    X_train, y_train = X[train_idx], y_cat[train_idx]
    X_val, y_val = X[val_idx], y_cat[val_idx]
    X_test, y_test = X[test_idx], y_cat[test_idx]


    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_path = os.path.join(args.models_dir, 'exercise_classifier.h5')
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]
    
    print("Training model...")
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=60, batch_size=32,
        class_weight=class_weight_dict, callbacks=callbacks
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Every artifact below is raw measured data. Plot rendering lives in
    # generate_plots.py, which reads these files - no number is retyped by hand.
    pd.DataFrame(history.history).to_csv(
        os.path.join(args.results_dir, 'history.csv'), index_label='epoch')

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    pd.DataFrame(cm, index=exercise_names, columns=exercise_names).to_csv(
        os.path.join(args.results_dir, 'confusion_matrix.csv'), index_label='true_label')

    report_dict = classification_report(
        y_true_classes, y_pred_classes, target_names=exercise_names, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(args.results_dir, 'per_class_metrics.csv'), index_label='label')

    report = classification_report(y_true_classes, y_pred_classes, target_names=exercise_names)
    with open(os.path.join(args.results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    df_metrics = pd.DataFrame({
        'Split': ['train', 'val', 'test'],
        'Loss': [train_loss, val_loss, test_loss],
        'Accuracy': [train_acc, val_acc, test_acc],
    })
    df_metrics.to_csv(os.path.join(args.results_dir, 'metrics_summary.csv'), index=False)
    print(df_metrics.to_string(index=False))
    print("Training complete. Results saved. Run tools/generate_plots.py to render figures.")

if __name__ == '__main__':
    main()
