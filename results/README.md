# Results

Every file here is generated. Nothing is hand-edited. The commands that produce
each one are listed below and in Appendix B of `SmartVision_AI_Report.md`.

## Model performance

| File | Contents |
|---|---|
| `split_summary.txt` | Video-grouped fold sizes, proof no clip spans two folds |
| `classification_report.txt` | Per-class precision, recall, F1 on the held-out fold |
| `per_class_metrics.csv` / `.png` | The same numbers as data and as a figure |
| `confusion_matrix.csv` / `.png` | Confusion matrix on the held-out fold |
| `history.csv`, `training_curves.png` | Loss and accuracy per epoch |
| `metrics_summary.csv` | Headline accuracy figures |
| `ablation_results.csv`, `ablation_runs.csv`, `ablation_comparison.png` | Architecture comparison, 5 seeds per variant |
| `dataset_distribution.png` | Windows per class |

Produced by `tools/train_classifier.py`, `tools/run_ablation.py` and
`tools/generate_plots.py`.

## Explainability

| File | Contents |
|---|---|
| `permutation_importance.csv` | Accuracy drop when each landmark group is shuffled, 10 shuffles, overall and per class |

Produced by `tools/permutation_importance.py`, which also writes
`public/models/exercise_classifier/saliency.json` - the file the XAI panel
loads at runtime. The panel shows measured attribution or nothing at all.

## System-level metrics

These come from replaying the held-out clips through the actual application
modules, so they measure the shipped system rather than the model alone.

| File | Contents |
|---|---|
| `repcount_reference.csv` | Reference repetition count per clip, from angle-signal prominence |
| `repcount_system.csv` | Repetitions the rule-based evaluators counted |
| `repcount_accuracy.csv` | Per-clip counted vs reference, with error |
| `repcount_sensitivity.csv` | The same score at 20, 30 and 40 degree prominence |
| `repcount_plots/*.png` | Angle traces with counted repetitions marked, for eyeball verification |
| `angle_validation.csv` | Difference between the displayed joint angle and the true on-screen angle |
| `agreement.csv` | Per-clip classifier verdict and whether the rule engine corroborates it |
| `latency_js.csv` | Cost of the preprocessing and LSTM stages, measured in Node |
| `skew.csv` | Accuracy under the training-faithful and the shipped preprocessing routes |

Produced by `tools/extract_sequences.py`, `tools/bench/replay.mjs`,
`tools/repcount_reference.py` and `tools/repcount_score.py`.

## Screenshots

`screenshots/*.png` with `screenshots/manifest.csv` giving the clip, the
moment and the live HUD state for each figure. Captured from the running
application by `tools/bench/screenshots.mjs`; nothing is staged or mocked.

## Not yet measured

`latency_browser.csv` is absent. End-to-end browser FPS and MediaPipe pose
latency depend on the GPU of the machine running the app, and a headless or
software-rendered browser measures the renderer rather than the product. Run
the app with `?bench=1`, play a clip to the end, and the app downloads that
file itself. See the report, Section 8.5.

## Reproducing

```bash
# Model, metrics, figures
venv/bin/python tools/train_classifier.py
venv/bin/python tools/run_ablation.py --seeds 5
venv/bin/python tools/generate_plots.py

# Explainability
PYTHONPATH=tools venv/bin/python tools/permutation_importance.py

# System metrics (extraction takes about 20 minutes)
PYTHONPATH=tools venv/bin/python tools/extract_sequences.py
npm run bench
PYTHONPATH=tools venv/bin/python tools/repcount_reference.py
venv/bin/python tools/repcount_score.py

# Screenshots (needs `npm run dev` in another shell)
npm run screenshots
```
