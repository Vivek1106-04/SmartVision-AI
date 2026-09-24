# SmartVision AI: Real-Time Exercise Classification with Temporal Deep Learning

**SmartVision AI** is a Deep Learning project: a custom-trained LSTM network that
classifies exercises (squat, pushup, bicep curl) from sequences of body pose landmarks,
deployed in the browser for real-time inference. A lightweight computer vision layer
supplies the pose input and on-screen feedback.

---

## 🎯 Key Project Features

### 1. Deep Learning Core (primary)
* **Temporal Sequence Classifier**: A 2-layer LSTM trained by us on 140 Kaggle exercise videos (4,613 sliding windows of 15 frames) to classify the exercise being performed. 92.5% test accuracy on 26 unseen videos under a leakage-free, video-grouped split.
* **Ablation Study**: LSTM vs GRU vs 1D-CNN vs single-frame MLP vs un-normalized LSTM, each repeated over 5 random seeds, isolating the value of spatial normalization and of temporal modelling.
* **Explainable AI (XAI)**: Per-class feature attribution measured by permutation importance over the held-out fold (`tools/permutation_importance.py`) and loaded at runtime from `public/models/exercise_classifier/saliency.json`. Until the model produces a prediction the panel shows nothing rather than an invented ranking.
* **In-Browser Deployment**: Keras model exported to TF.js; the browser forward pass matches Keras to 1.19e-7, pinned by train/serve parity tests.
* **Pretrained Feature Extractor**: MediaPipe BlazePose (frozen) supplies 33 3D keypoints per frame as the network input. No landmark training is performed.

### 2. Computer Vision Support (secondary)
* **Joint Angle Calculation**: Vector trigonometry ($\theta = \arccos\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}$) drives rule-based rep counting and form checks.
* **EMA Jitter Reduction**: Low-pass keypoint smoothing ($S_t = \alpha Y_t + (1 - \alpha) S_{t-1}, \alpha = 0.35$).
* **HUD Visualization**: Skeleton overlay, motion trails, ROI boxes, angular velocity telemetry, voice coaching and downloadable session reports.

---

## 🧠 Dataset & Training
**Dataset Source**: 140 videos from Kaggle 'Workout/Exercises Video' dataset (squat: 29, pushup: 56, bicepCurl: 62).
**Data Processing**: Extracted 4,613 sliding windows (15 frames each, stride 5).

The raw videos (~616MB) are not committed. Download the dataset and place clips under
`dataset/raw/<squat|pushup|bicepCurl>/`, matching `dataset/manifest.csv`. The extracted
landmark tensors in `dataset/processed/` are committed, so training and evaluation run
without the raw videos.

### Model Architecture
```text
Input (15, 132)  ← 33 landmarks × 4 values
  → LSTM(64, return_sequences=True)
  → Dropout(0.3)
  → LSTM(32)
  → Dropout(0.3)
  → Dense(32, ReLU)
  → Dense(3, Softmax)  ← squat / pushup / bicepCurl
```

### Evaluation Protocol
Sliding windows overlap by 10 of 15 frames, so a random window-level split leaks
near-duplicate frames into the test set and reports a meaningless 100% accuracy. All
results below use a **video-grouped split**: every window of a clip stays in one fold
(86 train / 28 validation / 26 test videos).

### Training Results
* **LSTM Test Accuracy**: 92.5% (macro F1 0.92) on 26 unseen videos
* **Parameters**: 64,003
* **Model Size**: 260 KB (TF.js)
* **Ablation Study** (mean +/- std over 5 seeds, 30-epoch fixed schedule):
  - LSTM: 93.44% +/- 0.70
  - GRU: 92.83% +/- 1.07
  - 1D-CNN: 92.59% +/- 1.89
  - LSTM-No-Norm: 90.51% +/- 1.43
  - Single-Frame MLP: 93.27% +/- 0.62

Spatial normalization is worth +2.9 points. Temporal modelling is worth +0.2 points,
which is inside seed noise: exercise-type classification turns out to be close to a
static pose problem. See the report for the full discussion.

### What the Model Actually Uses
Permutation importance over the held-out fold, 10 shuffles per landmark group:

| Group | Accuracy drop |
|---|---:|
| Hands | +18.4% |
| Head and Face | +11.1% |
| Feet | +5.9% |
| Wrists | +1.6% |
| Shoulders, ankles, elbows, knees, hips | within noise of zero |

The classifier ignores the knee, hip and elbow angles that every rule-based form check
is built on. It separates the three exercises from where the extremities sit relative to
the hips, which is consistent with the ablation finding that the task is close to static.

---

## Measured System Behaviour

Model accuracy is not system accuracy. Replaying the held-out clips through the real
application modules (`npm run bench`) gives:

| Metric | Value |
|---|---|
| Window-level accuracy | 92.5% |
| Clip-level accuracy (majority vote) | 96.2% (25/26 clips) |
| Model and rule engine corroborate | 92.3% (24/26 clips) |
| Repetition count MAE | 0.92 per clip, 55 counted vs 75 reference |
| Repetition exact-match rate | 53.8% (76.9% within one) |
| Displayed vs on-screen joint angle | 11.35 deg mean difference |
| LSTM forward pass | 0.79 ms mean per window (Apple M4) |
| JavaScript vs Keras probabilities | max difference 1.19e-7 |

Two known defects fall out of these numbers and are documented rather than hidden:
the repetition counter **systematically undercounts** because its angle gates are too
narrow, and the HUD measures joint angles in normalized square coordinates, which
distorts them by roughly 11 degrees on 16:9 video. See Section 8.5 of the report.

End-to-end browser FPS is not yet measured on real hardware; load the app with
`?bench=1` and play a clip to the end to export per-frame timings.

---

## Testing

```bash
npm test          # 31 JavaScript tests
npm run test:py   # 21 Python tests
```

The JavaScript suite includes **train/serve parity tests**: the browser re-implements
the training preprocessing and the Keras forward pass by hand, so fixtures recorded from
Python (`tools/gen_test_fixtures.py`) pin the two implementations together. These tests
found a weight-export defect that had left the in-browser classifier returning an empty
probability vector on every frame.

---

## 🛠️ Project Architecture & Directory Structure

```text
SmartVision-AI/
├── index.html                  # Main dark HUD dashboard layout
├── src/
│   ├── style.css               # Futuristic dark glassmorphic CSS design system
│   ├── main.js                 # Main application orchestrator
│   ├── core/
│   │   ├── cameraManager.js    # Webcam, video upload & synthetic demo generator
│   │   ├── poseDetector.js     # MediaPipe Pose neural network runner
│   │   ├── cvEngine.js         # Computer Vision spatial geometry & velocity engine
│   │   ├── dlClassifier.js     # LSTM temporal classifier & XAI saliency engine
│   │   ├── benchmark.js        # Per-stage frame timing, enabled with ?bench=1
│   │   ├── renderer.js         # High-FPS canvas HUD rendering engine
│   │   └── audioEngine.js      # Web Speech API real-time voice coaching
│   ├── exercises/
│   │   ├── exerciseBase.js     # Abstract exercise evaluator base class
│   │   ├── squatEvaluator.js   # Squat depth, knee valgus & spinal flexion rules
│   │   ├── pushupEvaluator.js  # Pushup depth, elbow angle & plank alignment
│   │   ├── bicepCurlEvaluator.js # Range of motion & shoulder momentum detection
│   │   └── postureEvaluator.js # Ergonomic forward head incline & slouch index
│   ├── ui/
│   │   ├── telemetryChart.js   # Real-time Chart.js kinematic telemetry graph
│   │   ├── xaiPanel.js         # Explainable AI confidence & feature attribution card
│   │   └── theoryModal.js      # Academic CV/DL mathematical specifications popup
│   └── utils/
│       └── math.js             # Trigonometry, vector algebra & EMA filter
├── tools/
│   ├── extract_landmarks.py    # MediaPipe landmark extraction pipeline
│   ├── extract_sequences.py    # Full-length sequences for the replay harness
│   ├── train_classifier.py     # LSTM model training script
│   ├── run_ablation.py         # Architecture comparison study
│   ├── permutation_importance.py # Measured feature attribution for the XAI panel
│   ├── repcount_reference.py   # Signal-derived reference repetition counts
│   ├── repcount_score.py       # Scores the counter against the reference
│   ├── gen_test_fixtures.py    # Records Python/Keras output as parity fixtures
│   ├── convert_to_tfjs.py      # Keras -> TF.js weight export
│   └── bench/
│       ├── replay.mjs          # Replays clips through the real app modules
│       ├── pipelines.mjs       # Training-faithful vs as-shipped preprocessing
│       └── screenshots.mjs     # Captures HUD figures from the running app
├── tests/
│   ├── js/                     # Parity, geometry and rep-counting tests
│   ├── python/                 # Preprocessing and data-split tests
│   └── fixtures/               # Recorded Python/Keras reference output
├── dataset/
│   ├── processed/              # Extracted landmark tensors (.npz)
│   └── manifest.csv            # Dataset metadata
├── models/                     # Trained Keras models (.h5)
├── results/                    # Metrics, plots, confusion matrices
└── public/models/              # TF.js model for browser inference
```

Not committed: `dataset/raw/` (source videos, see Dataset & Training above) and
`dist/` (production bundle, created by `npm run build`).

---

## ⚡ Quick Start & Live Demo Instructions

### 1. Install Dependencies & Run Local Dev Server
```bash
git clone https://github.com/Vivek1106-04/SmartVision-AI.git
cd SmartVision-AI
npm install
npm run dev
```

### 2. Live Demo Presentation Walkthrough
1. **Open the App**: Launch the local server URL (e.g. `http://localhost:5173`) in any modern browser.
2. **Select Exercise Mode**: Choose between **Squat Coach**, **Pushup Coach**, **Bicep Curl**, or **Ergonomic Posture**.
3. **Choose Input Source**:
   * Click **Start Webcam Stream** for live camera interaction.
   * Click **Upload Video File** to test pre-recorded footage.
   * Click **Run Synthetic Demo Video** to run a camera-less automated demonstration for professors.
4. **Academic Specifications Modal**: Click **CV/DL Academic Specs** in the top navigation bar to open the popup showing mathematical formulas, neural network loss functions, and PCK metrics.
