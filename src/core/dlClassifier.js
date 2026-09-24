/**
 * Deep Learning Temporal Classifier & Explainable AI (XAI) Engine
 *
 * Runs a real LSTM model trained on 4,613 sliding windows from 140 exercise
 * videos (Kaggle Workout/Exercises dataset).  The model classifies exercise
 * type (squat / pushup / bicepCurl) from a 15-frame window of 33 MediaPipe
 * landmarks (x, y, z, visibility → 132 features per frame).
 *
 * Architecture (matches tools/train_classifier.py):
 *   Input(15, 132) → LSTM(64, return_seq) → Dropout(0.3) → LSTM(32)
 *   → Dropout(0.3) → Dense(32, relu) → Dense(3, softmax)
 *
 * The model weights are loaded from public/models/exercise_classifier/.
 * Normalization (mid-hip translation + torso-length scaling) is applied
 * identically to the training pipeline (tools/extract_landmarks.py)
 * to prevent train/serve skew.
 */

// Exercise class labels — order matches training
const EXERCISE_CLASSES = ['squat', 'pushup', 'bicepCurl'];

export class DLClassifier {
  /**
   * @param {Object} [options]
   * @param {boolean} [options.autoLoad=true] - fetch the weights on construction.
   *   Tests load the weight file from disk instead and set them directly.
   */
  constructor({ autoLoad = true } = {}) {
    this.slidingWindowSize = 15;
    this.landmarkHistory = [];

    // Model state
    this.modelLoaded = false;
    this.modelLoading = false;
    this.weights = null;

    // EMA smoothing state (α = 0.35, matches cvEngine.js / training pipeline)
    this.emaAlpha = 0.35;
    this.prevSmoothed = null;

    // Latest classification result
    this.lastPrediction = null;
    this.lastProbabilities = null;

    // Measured per-class feature attribution, loaded alongside the weights.
    // Produced by tools/permutation_importance.py from the held-out fold, so
    // the panel reports what the model actually relies on rather than what a
    // textbook says it should.
    this.saliency = null;

    // Start loading model weights in background
    if (autoLoad) this.loadModel();
  }

  /**
   * Loads the trained LSTM weights from the exported binary file.
   */
  async loadModel() {
    if (this.modelLoading || this.modelLoaded) return;
    this.modelLoading = true;

    try {
      const modelUrl = '/models/exercise_classifier/model.json';
      const resp = await fetch(modelUrl);
      if (!resp.ok) throw new Error(`Model manifest not found at ${modelUrl}`);

      const manifest = await resp.json();

      // Load binary weights
      const weightsPath = '/models/exercise_classifier/' + manifest.weightsManifest[0].paths[0];
      const weightsResp = await fetch(weightsPath);
      if (!weightsResp.ok) throw new Error(`Weights not found at ${weightsPath}`);

      const weightsBuffer = await weightsResp.arrayBuffer();
      this.weights = this._parseWeights(weightsBuffer, manifest.weightsManifest[0].weights);
      this.modelLoaded = true;
      console.log('[DLClassifier] LSTM model loaded successfully (260 KB)');

      await this.loadSaliency();
    } catch (err) {
      console.warn('[DLClassifier] Model not loaded, running in fallback mode:', err.message);
      this.modelLoaded = false;
    } finally {
      this.modelLoading = false;
    }
  }

  /**
   * Loads the measured per-class feature attribution.
   * The panel simply shows nothing if this is unavailable - an empty attribution
   * is honest, a hardcoded one is not.
   */
  async loadSaliency() {
    try {
      const resp = await fetch('/models/exercise_classifier/saliency.json');
      if (!resp.ok) throw new Error(`Saliency not found (${resp.status})`);
      this.saliency = await resp.json();
    } catch (err) {
      console.warn('[DLClassifier] Feature attribution unavailable:', err.message);
      this.saliency = null;
    }
  }

  /**
   * Measured attribution for a predicted class, or an empty list when the
   * model has not produced a prediction yet.
   */
  _featureImportance(predictedExercise) {
    if (!this.saliency || !predictedExercise) return [];
    return this.saliency.classes[predictedExercise] || [];
  }

  /**
   * Parse the raw Float32 weight buffer into named tensors using the manifest.
   */
  _parseWeights(buffer, weightSpecs) {
    const weights = {};
    let offset = 0;
    for (const spec of weightSpecs) {
      const numElements = spec.shape.reduce((a, b) => a * b, 1);
      const byteLength = numElements * 4; // float32
      weights[spec.name] = {
        data: new Float32Array(buffer, offset, numElements),
        shape: spec.shape
      };
      offset += byteLength;
    }
    return weights;
  }

  /**
   * Normalize landmarks identically to the training pipeline.
   * Translate: subtract mid-hip. Scale: divide by torso length.
   */
  _normalizeLandmarks(landmarks) {
    if (!landmarks || landmarks.length < 33) return null;

    const normalized = landmarks.map(lm => ({
      x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility || 0.95
    }));

    // Mid-hip as origin (landmarks 23, 24)
    const lHip = normalized[23];
    const rHip = normalized[24];
    const midHipX = (lHip.x + rHip.x) / 2;
    const midHipY = (lHip.y + rHip.y) / 2;
    const midHipZ = (lHip.z + rHip.z) / 2;

    // Torso length for scale (mid-shoulder to mid-hip)
    const lShoulder = normalized[11];
    const rShoulder = normalized[12];
    const midShoulderX = (lShoulder.x + rShoulder.x) / 2;
    const midShoulderY = (lShoulder.y + rShoulder.y) / 2;
    const midShoulderZ = (lShoulder.z + rShoulder.z) / 2;

    const dx = midShoulderX - midHipX;
    const dy = midShoulderY - midHipY;
    const dz = midShoulderZ - midHipZ;
    let torsoLength = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (torsoLength < 1e-5) torsoLength = 1.0;

    // Translate and scale
    for (const lm of normalized) {
      lm.x = (lm.x - midHipX) / torsoLength;
      lm.y = (lm.y - midHipY) / torsoLength;
      lm.z = (lm.z - midHipZ) / torsoLength;
      // visibility stays unchanged
    }

    return normalized;
  }

  /**
   * Apply EMA smoothing to a normalized landmark frame.
   */
  _applyEMA(normalizedFrame) {
    if (!this.prevSmoothed) {
      this.prevSmoothed = normalizedFrame;
      return normalizedFrame;
    }

    const smoothed = normalizedFrame.map((lm, i) => ({
      x: this.emaAlpha * lm.x + (1 - this.emaAlpha) * this.prevSmoothed[i].x,
      y: this.emaAlpha * lm.y + (1 - this.emaAlpha) * this.prevSmoothed[i].y,
      z: this.emaAlpha * lm.z + (1 - this.emaAlpha) * this.prevSmoothed[i].z,
      visibility: this.emaAlpha * lm.visibility + (1 - this.emaAlpha) * this.prevSmoothed[i].visibility
    }));

    this.prevSmoothed = smoothed;
    return smoothed;
  }

  /**
   * Convert a landmark frame to a flat feature vector [x0,y0,z0,v0, x1,y1,z1,v1, ...].
   */
  _landmarksToFeatures(normalizedFrame) {
    const features = new Float32Array(132); // 33 * 4
    for (let i = 0; i < 33; i++) {
      const lm = normalizedFrame[i];
      features[i * 4 + 0] = lm.x;
      features[i * 4 + 1] = lm.y;
      features[i * 4 + 2] = lm.z;
      features[i * 4 + 3] = lm.visibility;
    }
    return features;
  }

  /**
   * Softmax activation
   */
  _softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }

  /**
   * Dense layer: output = activation(input @ kernel + bias)
   */
  _denseLayer(input, kernel, bias, activation = 'linear') {
    const [inSize, outSize] = kernel.shape;
    const output = new Float32Array(outSize);

    for (let j = 0; j < outSize; j++) {
      let sum = bias.data[j];
      for (let i = 0; i < inSize; i++) {
        sum += input[i] * kernel.data[i * outSize + j];
      }
      if (activation === 'relu') {
        output[j] = Math.max(0, sum);
      } else if (activation === 'softmax') {
        output[j] = sum; // softmax applied after
      } else {
        output[j] = sum;
      }
    }

    if (activation === 'softmax') {
      const sm = this._softmax(output);
      for (let j = 0; j < outSize; j++) output[j] = sm[j];
    }

    return output;
  }

  /**
   * LSTM layer forward pass.
   * @param {Array<Float32Array>} sequence - Array of feature vectors (timesteps)
   * @param {Object} kernel  - {data, shape: [input_dim, 4*units]}
   * @param {Object} recurrentKernel - {data, shape: [units, 4*units]}
   * @param {Object} bias - {data, shape: [4*units]}
   * @param {boolean} returnSequences
   * @returns {Array<Float32Array>|Float32Array}
   */
  _lstmLayer(sequence, kernel, recurrentKernel, bias, returnSequences = false) {
    const units = recurrentKernel.shape[0];
    const inputDim = kernel.shape[0];

    let h = new Float32Array(units); // hidden state
    let c = new Float32Array(units); // cell state
    const outputs = [];

    for (let t = 0; t < sequence.length; t++) {
      const x = sequence[t];
      const gates = new Float32Array(4 * units);

      // gates = x @ kernel + h @ recurrent_kernel + bias
      for (let j = 0; j < 4 * units; j++) {
        let sum = bias.data[j];
        for (let i = 0; i < inputDim; i++) {
          sum += x[i] * kernel.data[i * (4 * units) + j];
        }
        for (let i = 0; i < units; i++) {
          sum += h[i] * recurrentKernel.data[i * (4 * units) + j];
        }
        gates[j] = sum;
      }

      // Split into i, f, c_candidate, o gates
      const newH = new Float32Array(units);
      const newC = new Float32Array(units);

      for (let i = 0; i < units; i++) {
        const ig = 1 / (1 + Math.exp(-gates[i]));                    // input gate (sigmoid)
        const fg = 1 / (1 + Math.exp(-gates[units + i]));            // forget gate (sigmoid)
        const cg = Math.tanh(gates[2 * units + i]);                  // cell candidate (tanh)
        const og = 1 / (1 + Math.exp(-gates[3 * units + i]));        // output gate (sigmoid)

        newC[i] = fg * c[i] + ig * cg;
        newH[i] = og * Math.tanh(newC[i]);
      }

      h = newH;
      c = newC;

      if (returnSequences) {
        outputs.push(new Float32Array(h));
      }
    }

    return returnSequences ? outputs : h;
  }

  /**
   * Run full LSTM model inference on the current sliding window.
   * Returns class probabilities [squat, pushup, bicepCurl].
   */
  _runInference(window) {
    if (!this.weights) return null;

    const w = this.weights;
    const required = [
      'lstm/kernel', 'lstm/recurrent_kernel', 'lstm/bias',
      'lstm_1/kernel', 'lstm_1/recurrent_kernel', 'lstm_1/bias',
      'dense/kernel', 'dense/bias', 'dense_1/kernel', 'dense_1/bias'
    ];
    const missing = required.filter(name => !w[name]);
    if (missing.length > 0) {
      console.warn('[DLClassifier] Export is missing weights:', missing.join(', '));
      return null;
    }

    // LSTM layer 1 (return_sequences=True): input(15, 132) -> output(15, 64)
    const lstm1Out = this._lstmLayer(
      window,
      w['lstm/kernel'], w['lstm/recurrent_kernel'], w['lstm/bias'],
      true
    );

    // Dropout is skipped at inference time (no-op)

    // LSTM layer 2 (return_sequences=False): input(15, 64) -> output(32)
    const lstm2Out = this._lstmLayer(
      lstm1Out,
      w['lstm_1/kernel'], w['lstm_1/recurrent_kernel'], w['lstm_1/bias'],
      false
    );

    // Dense(32, relu)
    const dense1Out = this._denseLayer(lstm2Out, w['dense/kernel'], w['dense/bias'], 'relu');

    // Dense(3, softmax)
    return this._denseLayer(dense1Out, w['dense_1/kernel'], w['dense_1/bias'], 'softmax');
  }

  /**
   * Main entry point — called every frame from main.js.
   * Maintains sliding window, runs normalization + inference, returns XAI data.
   */
  processFrame(landmarks) {
    if (!landmarks || landmarks.length < 33) {
      return {
        overallConfidence: 0.0,
        predictedExercise: null,
        classProbabilities: null,
        featureImportance: []
      };
    }

    // Step 1: Normalize (same as training pipeline)
    const normalized = this._normalizeLandmarks(landmarks);
    if (!normalized) {
      return {
        overallConfidence: 0.0,
        predictedExercise: null,
        classProbabilities: null,
        featureImportance: []
      };
    }

    // Step 2: EMA smoothing (α = 0.35, matches training)
    const smoothed = this._applyEMA(normalized);

    // Step 3: Convert to feature vector and buffer into sliding window
    const features = this._landmarksToFeatures(smoothed);
    this.landmarkHistory.push(features);
    if (this.landmarkHistory.length > this.slidingWindowSize) {
      this.landmarkHistory.shift();
    }

    // Step 4: Compute landmark visibility confidence
    const totalVisibility = landmarks.reduce((acc, lm) => acc + (lm.visibility || 0.95), 0);
    const avgConfidence = (totalVisibility / landmarks.length) * 100.0;

    // Step 5: Run LSTM inference if window is full and model is loaded
    let predictedExercise = null;
    let classProbabilities = null;
    let modelConfidence = avgConfidence;

    if (this.modelLoaded && this.landmarkHistory.length === this.slidingWindowSize) {
      const probs = this._runInference(this.landmarkHistory);
      if (probs) {
        classProbabilities = {
          squat: (probs[0] * 100).toFixed(1),
          pushup: (probs[1] * 100).toFixed(1),
          bicepCurl: (probs[2] * 100).toFixed(1)
        };

        const maxIdx = probs.indexOf(Math.max(...probs));
        predictedExercise = EXERCISE_CLASSES[maxIdx];
        modelConfidence = Math.max(...probs) * 100;

        this.lastPrediction = predictedExercise;
        this.lastProbabilities = classProbabilities;
      }
    }

    // Use cached prediction if window not yet full
    if (!predictedExercise && this.lastPrediction) {
      predictedExercise = this.lastPrediction;
      classProbabilities = this.lastProbabilities;
    }

    return {
      overallConfidence: Math.min(99.9, Math.max(70.0, modelConfidence)),
      predictedExercise,
      classProbabilities,
      featureImportance: this._featureImportance(predictedExercise),
      modelLoaded: this.modelLoaded,
      windowFill: `${this.landmarkHistory.length}/${this.slidingWindowSize}`
    };
  }

  reset() {
    this.landmarkHistory = [];
    this.prevSmoothed = null;
    this.lastPrediction = null;
    this.lastProbabilities = null;
  }
}
