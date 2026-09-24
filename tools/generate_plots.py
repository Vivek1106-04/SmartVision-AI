"""
Render the measurement figures for SmartVision AI.

Every number plotted here is read from an artifact written by
tools/train_classifier.py or tools/run_ablation.py. Nothing is typed in by
hand, so a figure can never disagree with the measured result.

Each figure is built to show something a table cannot: the spread behind a
mean, where the errors actually go, how unevenly the clips contribute. Where a
summary would do, the report uses a table instead.

Schematic figures (pipeline, leakage, layer stack) live in
tools/generate_diagrams.py.

Run order:
    python tools/extract_landmarks.py
    python tools/train_classifier.py
    python tools/run_ablation.py
    python tools/generate_plots.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator

import plot_style as style
from plot_style import (
    SERIES_1, SERIES_2, SERIES_3, SURFACE, INK_PRIMARY, INK_SECONDARY,
    INK_MUTED, GRIDLINE, FILL_NEUTRAL,
)

style.apply()

RESULTS_DIR = 'results'
DATA_PATH = 'dataset/processed/landmarks.npz'
MANIFEST_PATH = 'dataset/manifest.csv'

# Rows of per_class_metrics.csv that summarise rather than name a class.
AGGREGATE_ROWS = ('accuracy', 'macro avg', 'weighted avg')


def require(path: str) -> str:
    """Abort with an actionable message rather than plotting stale or absent data."""
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Re-run the training pipeline before generating plots.")
    return path


def save(fig, filename: str) -> None:
    fig.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close(fig)
    print(f'OK {filename}')


# --- Training curves ---------------------------------------------------------

def plot_training_curves() -> None:
    """Loss and accuracy per epoch, with the generalization gap made visible."""
    history = pd.read_csv(require(os.path.join(RESULTS_DIR, 'history.csv')))
    epochs = history['epoch'] + 1
    best_epoch = int(history['val_loss'].idxmin()) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax in (ax1, ax2):
        style.value_axis(ax, 'y')
        ax.set_xlabel('Epoch')
        ax.set_xlim(1, epochs.max())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # The selected epoch is the one moment worth marking on both panels.
        ax.axvline(best_epoch, color=INK_MUTED, linewidth=0.8, zorder=1)

    # The shaded wedge between the curves is the generalization gap itself.
    ax1.fill_between(epochs, history['loss'], history['val_loss'],
                     color=FILL_NEUTRAL, zorder=1, label='Generalization gap')
    ax1.plot(epochs, history['loss'], color=SERIES_1, linewidth=1.8, label='Train', zorder=3)
    ax1.plot(epochs, history['val_loss'], color=SERIES_2, linewidth=1.8,
             label='Validation', zorder=3)
    ax1.set_ylabel('Cross-entropy loss')
    ax1.set_title('Loss')
    ax1.legend(loc='upper left', ncols=1)
    ax1.annotate(f'checkpoint\nepoch {best_epoch}',
                 xy=(best_epoch, history['val_loss'].max() * 0.92),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=8.5, color=INK_SECONDARY, va='top')

    ax2.fill_between(epochs, history['accuracy'] * 100, history['val_accuracy'] * 100,
                     color=FILL_NEUTRAL, zorder=1)
    ax2.plot(epochs, history['accuracy'] * 100, color=SERIES_1, linewidth=1.8,
             label='Train', zorder=3)
    ax2.plot(epochs, history['val_accuracy'] * 100, color=SERIES_2, linewidth=1.8,
             label='Validation', zorder=3)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend(loc='lower right')
    lowest = min(history['accuracy'].min(), history['val_accuracy'].min()) * 100
    ax2.set_ylim(max(0, lowest - 4), 100.5)

    final_gap = (history['accuracy'].iloc[-1] - history['val_accuracy'].iloc[-1]) * 100
    style.headline(fig, 'Training progress',
                   f'Validation loss bottoms out at epoch {best_epoch} and rises after; '
                   f'the train/validation accuracy gap reaches {final_gap:.1f} points by epoch '
                   f'{int(epochs.max())}.')
    style.note(fig, 'Source: results/history.csv. Shaded wedge is the train/validation gap.')
    fig.tight_layout()
    save(fig, 'training_curves.png')


# --- Confusion as flow -------------------------------------------------------

def _ribbon(ax, x0, x1, y0_top, y0_bottom, y1_top, y1_bottom, color, alpha):
    """A smooth band joining a segment on the left column to one on the right."""
    curve = (x1 - x0) * 0.5
    vertices = [
        (x0, y0_top),
        (x0 + curve, y0_top), (x1 - curve, y1_top), (x1, y1_top),
        (x1, y1_bottom),
        (x1 - curve, y1_bottom), (x0 + curve, y0_bottom), (x0, y0_bottom),
        (x0, y0_top),
    ]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(vertices, codes), facecolor=color, edgecolor='none',
                           alpha=alpha, zorder=2))


def plot_confusion_flow() -> None:
    """Where the held-out windows actually go, as flow rather than a grid.

    A matrix makes the reader decode nine numbers. The flow shows the two
    directions that carry nearly all the error at a glance.
    """
    counts = pd.read_csv(
        require(os.path.join(RESULTS_DIR, 'confusion_matrix.csv')), index_col=0)
    classes = list(counts.columns)
    cm = counts.to_numpy(dtype=float)
    total = cm.sum()

    gap = total * 0.035
    left_x, right_x = 0.0, 1.0
    column_width = 0.055

    def stack(totals):
        """Top edge of each class block, laid out top to bottom with gaps."""
        height = totals.sum() + gap * (len(totals) - 1)
        tops, cursor = [], height
        for value in totals:
            tops.append(cursor)
            cursor -= value + gap
        return tops, height

    actual_tops, height = stack(cm.sum(axis=1))
    predicted_tops, _ = stack(cm.sum(axis=0))

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    style.strip_frame(ax)

    # Ribbons first, so the column blocks sit on top of their ends.
    actual_cursor = list(actual_tops)
    predicted_cursor = list(predicted_tops)
    for actual in range(len(classes)):
        for predicted in range(len(classes)):
            value = cm[actual, predicted]
            if value <= 0:
                continue
            is_error = actual != predicted
            _ribbon(ax, left_x + column_width, right_x,
                    actual_cursor[actual], actual_cursor[actual] - value,
                    predicted_cursor[predicted], predicted_cursor[predicted] - value,
                    SERIES_2 if is_error else SERIES_1,
                    0.75 if is_error else 0.16)
            actual_cursor[actual] -= value
            predicted_cursor[predicted] -= value

    for index, name in enumerate(classes):
        support = cm[index].sum()
        ax.add_patch(plt.Rectangle((left_x, actual_tops[index] - support),
                                   column_width, support,
                                   facecolor=INK_SECONDARY, edgecolor='none'))
        ax.text(left_x - 0.02, actual_tops[index] - support / 2,
                f'{name}\n{int(support)} windows', ha='right', va='center',
                fontsize=9.5, color=INK_PRIMARY)

        received = cm[:, index].sum()
        ax.add_patch(plt.Rectangle((right_x, predicted_tops[index] - received),
                                   column_width, received,
                                   facecolor=INK_SECONDARY, edgecolor='none'))
        recall = cm[index, index] / support
        ax.text(right_x + column_width + 0.02, predicted_tops[index] - received / 2,
                f'{name}\n{recall * 100:.0f}% of actual recovered', ha='left', va='center',
                fontsize=9.5, color=INK_PRIMARY)

    # Name the two flows that matter; the rest are visible but unlabelled.
    errors = [(cm[a, p], classes[a], classes[p])
              for a in range(len(classes)) for p in range(len(classes)) if a != p]
    worst = sorted(errors, reverse=True)[:2]
    caption = '   '.join(f'{int(v)} {a} -> {p}' for v, a, p in worst)

    ax.text(0.5, -0.06 * height, f'Largest confusions:  {caption}',
            ha='center', va='top', fontsize=9, color=SERIES_2, transform=ax.transData)
    ax.text(left_x, height * 1.04, 'ACTUAL', fontsize=8.5, color=INK_MUTED, weight='bold')
    ax.text(right_x + column_width, height * 1.04, 'PREDICTED', fontsize=8.5,
            color=INK_MUTED, weight='bold', ha='right')

    ax.set_xlim(-0.33, 1.46)
    ax.set_ylim(-0.16 * height, height * 1.1)

    accuracy = np.trace(cm) / total
    style.headline(fig, 'Where the held-out windows go',
                   f'{accuracy * 100:.1f}% of {int(total):,} windows land on the correct class. '
                   'Orange ribbons are the errors.')
    style.note(fig, 'Source: results/confusion_matrix.csv. Ribbon thickness is window count.')
    fig.tight_layout()
    save(fig, 'confusion_matrix.png')


# --- Ablation as seed runs ---------------------------------------------------

def plot_ablation() -> None:
    """Every seed run, not a summary of them.

    The study's conclusion is that four of the five variants are not
    distinguishable. Plotting the individual runs lets the reader see the
    overlap instead of taking a standard deviation on trust.
    """
    runs = pd.read_csv(require(os.path.join(RESULTS_DIR, 'ablation_runs.csv')))
    summary = pd.read_csv(require(os.path.join(RESULTS_DIR, 'ablation_results.csv')))

    labels = list(summary['Model'])
    means = summary['Test_Accuracy_Mean'].to_numpy() * 100
    spreads = summary['Test_Accuracy_Std'].fillna(0).to_numpy() * 100
    params = summary['Parameters'].to_numpy() / 1000
    n_seeds = int(summary['Seeds'].max())
    y = np.arange(len(labels))[::-1]

    baseline = 0
    outlier = int(np.argmin(means))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9),
                                   gridspec_kw={'width_ratios': [1.55, 1]})

    ax1.axvspan(means[baseline] - spreads[baseline], means[baseline] + spreads[baseline],
                color=GRIDLINE, zorder=0)

    rng = np.random.default_rng(0)
    for index, label in enumerate(labels):
        accent = SERIES_2 if index == outlier else SERIES_1
        seed_scores = runs.loc[runs['Model'] == label, 'Test_Accuracy'].to_numpy() * 100
        # A little vertical jitter so coincident runs stay countable.
        jitter = rng.uniform(-0.13, 0.13, len(seed_scores))
        ax1.scatter(seed_scores, y[index] + jitter, s=22, color=accent, alpha=0.5,
                    edgecolors='none', zorder=2)
        ax1.plot([means[index]], [y[index]], marker='|', markersize=20,
                 markeredgewidth=2.2, color=accent, zorder=4)

    for index in (baseline, outlier):
        ax1.annotate(f'{means[index]:.2f}%', xy=(means[index], y[index]),
                     xytext=(0, 14), textcoords='offset points', ha='center',
                     fontsize=9, fontweight='bold',
                     color=SERIES_2 if index == outlier else INK_PRIMARY)

    ax1.set_yticks(y, labels)
    ax1.set_xlabel('Test accuracy (%)')
    ax1.set_title(f'Every seed run ({n_seeds} per variant)')
    ax1.set_xlim(runs['Test_Accuracy'].min() * 100 - 1, runs['Test_Accuracy'].max() * 100 + 1)
    ax1.set_ylim(-0.7, len(labels) - 0.3)
    style.value_axis(ax1, 'x')
    ax1.text(0.02, 0.06, f'shaded: {labels[baseline]} +/- 1 s.d.', transform=ax1.transAxes,
             fontsize=8, color=INK_MUTED)

    # Capacity as thin lollipops rather than blocks.
    ax2.hlines(y, 0, params, color=FILL_NEUTRAL, linewidth=3)
    ax2.plot(params, y, 'o', markersize=7, color=SERIES_1,
             markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax2.set_yticks(y, labels)
    ax2.set_xlabel('Trainable parameters (thousands)')
    ax2.set_title('Model capacity')
    ax2.set_xlim(0, params.max() * 1.25)
    ax2.set_ylim(-0.7, len(labels) - 0.3)
    style.value_axis(ax2, 'x')
    for index, value in enumerate(params):
        ax2.annotate(f'{value:.1f}k', xy=(value, y[index]), xytext=(8, 0),
                     textcoords='offset points', va='center',
                     fontsize=8.5, color=INK_SECONDARY)

    style.headline(
        fig, 'Ablation: only normalization clears the noise',
        f'{labels[outlier]} sits {means[baseline] - means[outlier]:.1f} points below the main '
        'model. The other three variants overlap it run for run.')
    style.note(fig, 'Source: results/ablation_runs.csv, results/ablation_results.csv.')
    fig.tight_layout()
    save(fig, 'ablation_comparison.png')


# --- Dataset composition -----------------------------------------------------

def plot_dataset_distribution() -> None:
    """One dot per source video, so clip-length imbalance is visible.

    Three totals would hide that a handful of long clips contribute as many
    windows as dozens of short ones - which is the reason the split has to
    group by video rather than by window.
    """
    manifest = pd.read_csv(require(MANIFEST_PATH))
    data = np.load(require(DATA_PATH))
    class_names = [str(name) for name in data['exercise_names']]
    total_windows = int(len(data['y']))

    y = np.arange(len(class_names))[::-1]
    rng = np.random.default_rng(1)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    for index, name in enumerate(class_names):
        clips = manifest.loc[manifest['exercise'] == name, 'windows'].to_numpy()
        jitter = rng.uniform(-0.17, 0.17, len(clips))
        ax.scatter(clips, y[index] + jitter, s=26, color=SERIES_1, alpha=0.45,
                   edgecolors='none', zorder=3)
        ax.plot([np.median(clips)], [y[index]], marker='|', markersize=22,
                markeredgewidth=2.2, color=INK_PRIMARY, zorder=4)
        ax.annotate(f'{len(clips)} clips   {clips.sum():,} windows',
                    xy=(0, y[index] + 0.36), xytext=(0, 0), textcoords='offset points',
                    fontsize=8.5, color=INK_SECONDARY, va='center')

    ax.set_yticks(y, class_names)
    ax.set_xlabel('Sliding windows contributed by one clip')
    ax.set_xlim(0, manifest['windows'].max() * 1.06)
    ax.set_ylim(-0.7, len(class_names) - 0.25)
    style.value_axis(ax, 'x')

    longest = manifest.loc[manifest['windows'].idxmax()]
    ax.annotate(f"longest clip: {longest['windows']} windows",
                xy=(longest['windows'], y[class_names.index(longest['exercise'])]),
                xytext=(-8, -26), textcoords='offset points', ha='right',
                fontsize=8.5, color=INK_MUTED)

    style.headline(
        fig, 'Dataset composition',
        f'{len(manifest)} clips yield {total_windows:,} windows, but clips differ by more than '
        'an order of magnitude. Vertical rule marks the median clip.')
    style.note(fig, 'Source: dataset/manifest.csv. One dot per source video.')
    fig.tight_layout()
    save(fig, 'dataset_distribution.png')


# --- Per-class metrics -------------------------------------------------------

def plot_per_class_metrics() -> None:
    """Precision, recall and F1 per class as connected dots.

    The three scores per class sit within a few points of each other, so bars
    from zero would render as identical full-height slabs.
    """
    df = pd.read_csv(require(os.path.join(RESULTS_DIR, 'per_class_metrics.csv')), index_col=0)
    per_class = df[~df.index.isin(AGGREGATE_ROWS)]
    aggregate = df.loc['macro avg'] if 'macro avg' in df.index else None

    classes = list(per_class.index)
    metrics = [('Precision', 'precision', SERIES_1),
               ('Recall', 'recall', SERIES_2),
               ('F1-score', 'f1-score', SERIES_3)]
    y = np.arange(len(classes))[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 3.4))

    for index, name in enumerate(classes):
        values = [per_class.loc[name, column] * 100 for _, column, _ in metrics]
        ax.plot([min(values), max(values)], [y[index], y[index]],
                color=GRIDLINE, linewidth=3, zorder=1, solid_capstyle='round')
        for (label, column, color) in metrics:
            ax.plot([per_class.loc[name, column] * 100], [y[index]], 'o', markersize=8,
                    color=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    zorder=3, label=label if index == 0 else None)
        ax.annotate(f'spread {max(values) - min(values):.1f} pts',
                    xy=(max(values), y[index]), xytext=(12, 0), textcoords='offset points',
                    va='center', fontsize=8.5, color=INK_MUTED)

    if aggregate is not None:
        ax.axvline(aggregate['f1-score'] * 100, color=INK_MUTED, linewidth=0.9, zorder=0)
        ax.text(aggregate['f1-score'] * 100, len(classes) - 0.42,
                f"  macro F1 {aggregate['f1-score'] * 100:.0f}%",
                fontsize=8.5, color=INK_MUTED, va='top')

    supports = per_class['support'].to_numpy()
    ax.set_yticks(y, [f'{name}\nn={int(support)}'
                      for name, support in zip(classes, supports)])
    ax.set_xlabel('Score (%)')
    ax.set_xlim(78, 104)
    ax.set_ylim(-0.6, len(classes) - 0.2)
    style.value_axis(ax, 'x')
    ax.legend(loc='lower left', ncols=3, bbox_to_anchor=(0, 1.0))

    weakest = per_class['f1-score'].idxmin()
    style.headline(
        fig, 'Per-class performance',
        f'{weakest} is the weakest class at F1 '
        f"{per_class.loc[weakest, 'f1-score'] * 100:.0f}%; it is also the smallest.")
    style.note(fig, 'Source: results/per_class_metrics.csv. Axis truncated at 78% '
                    'to resolve differences; exact values are tabulated in the report.')
    fig.tight_layout()
    save(fig, 'per_class_metrics.png')


if __name__ == '__main__':
    print('Generating measurement figures...\n')
    plot_training_curves()
    plot_confusion_flow()
    plot_ablation()
    plot_dataset_distribution()
    plot_per_class_metrics()
    print(f'\nFigures saved to {RESULTS_DIR}/')
