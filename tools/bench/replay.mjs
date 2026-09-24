/**
 * Offline replay harness for the system-level metrics.
 *
 * Feeds the real held-out clips through the real application modules - the same
 * CVEngine, evaluators and DLClassifier the browser runs - and records what the
 * system does end to end. MediaPipe has already run offline, so what is
 * measured here is everything downstream of pose estimation.
 *
 * Writes into results/:
 *   repcount_system.csv    - reps counted per clip by the rule-based evaluators
 *   angle_validation.csv   - joint angle error from the normalized-coordinate convention
 *   agreement.csv          - per-clip classifier verdict and rule/model corroboration
 *   latency_js.csv         - per-frame cost of each processing stage
 *   skew.csv               - accuracy under the training-faithful vs shipped pipeline
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

import { DLClassifier } from '../../src/core/dlClassifier.js';
import { CVEngine } from '../../src/core/cvEngine.js';
import { SquatEvaluator } from '../../src/exercises/squatEvaluator.js';
import { PushupEvaluator } from '../../src/exercises/pushupEvaluator.js';
import { BicepCurlEvaluator } from '../../src/exercises/bicepCurlEvaluator.js';
import { calculate2DAngle } from '../../src/utils/math.js';
import { loadWeightsFromDisk, loadSaliencyFromDisk, PROJECT_ROOT } from '../../tests/js/helpers.mjs';
import { trainingFaithful, asShipped, toLandmarks } from './pipelines.mjs';

const CLASSES = ['squat', 'pushup', 'bicepCurl'];
const WINDOW_SIZE = 15;
const RESULTS_DIR = join(PROJECT_ROOT, 'results');

const EVALUATORS = {
  squat: () => new SquatEvaluator(),
  pushup: () => new PushupEvaluator(),
  bicepCurl: () => new BicepCurlEvaluator()
};

// The joints whose angle each evaluator shows as its primary readout.
const PRIMARY_ANGLE_JOINTS = {
  squat: [23, 25, 27],
  pushup: [11, 13, 15],
  bicepCurl: [11, 13, 15]
};

function newClassifier() {
  const classifier = new DLClassifier({ autoLoad: false });
  loadWeightsFromDisk(classifier);
  classifier.saliency = loadSaliencyFromDisk();
  return classifier;
}

function writeCsv(filename, rows) {
  if (rows.length === 0) return;
  const columns = Object.keys(rows[0]);
  const lines = [columns.join(',')];
  for (const row of rows) {
    lines.push(columns.map(c => row[c]).join(','));
  }
  writeFileSync(join(RESULTS_DIR, filename), lines.join('\n') + '\n');
  console.log(`  results/${filename} (${rows.length} rows)`);
}

const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
const round = (value, places = 4) => Number(value.toFixed(places));

/** Run the rule-based evaluator over a clip exactly as the app would. */
function replayRules(clip) {
  const cvEngine = new CVEngine();
  const evaluator = EVALUATORS[clip.exercise]();
  const [a, b, c] = PRIMARY_ANGLE_JOINTS[clip.exercise];
  const angleErrors = [];
  let framesWithPose = 0;

  for (const rawFrame of clip.landmarks) {
    const smoothed = cvEngine.smoothLandmarks(toLandmarks(rawFrame));
    evaluator.evaluate(smoothed, {});
    framesWithPose++;

    // The evaluators call calculate2DAngle without the aspect arguments, so the
    // angle is measured in normalized [0,1] space. On a non-square frame that
    // is not the angle a protractor would read on screen.
    const asShown = calculate2DAngle(smoothed[a], smoothed[b], smoothed[c]);
    const onScreen = calculate2DAngle(smoothed[a], smoothed[b], smoothed[c], clip.width, clip.height);
    angleErrors.push(Math.abs(asShown - onScreen));
  }

  return {
    reps: evaluator.repCount,
    frames: framesWithPose,
    angleMae: mean(angleErrors),
    angleMax: Math.max(...angleErrors)
  };
}

/** Run the classifier over a clip with one of the preprocessing routes. */
function replayModel(clip, classifier, makePipeline) {
  const pipeline = makePipeline();
  const history = [];
  const votes = new Array(CLASSES.length).fill(0);
  const stageTimes = { preprocess: [], inference: [] };
  let windows = 0;
  let correctWindows = 0;

  for (const rawFrame of clip.landmarks) {
    const preprocessStart = process.hrtime.bigint();
    const features = pipeline(toLandmarks(rawFrame));
    stageTimes.preprocess.push(Number(process.hrtime.bigint() - preprocessStart) / 1e6);

    history.push(features);
    if (history.length > WINDOW_SIZE) history.shift();
    if (history.length < WINDOW_SIZE) continue;

    const inferenceStart = process.hrtime.bigint();
    const probabilities = classifier._runInference(history);
    stageTimes.inference.push(Number(process.hrtime.bigint() - inferenceStart) / 1e6);

    const predicted = probabilities.indexOf(Math.max(...probabilities));
    votes[predicted]++;
    windows++;
    if (CLASSES[predicted] === clip.exercise) correctWindows++;
  }

  const majority = votes.indexOf(Math.max(...votes));
  return {
    windows,
    windowAccuracy: windows > 0 ? correctWindows / windows : 0,
    predicted: windows > 0 ? CLASSES[majority] : null,
    confidence: windows > 0 ? votes[majority] / windows : 0,
    stageTimes
  };
}

function main() {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const clipsPath = join(PROJECT_ROOT, 'dataset', 'processed', 'sequences_test.json');
  const clips = JSON.parse(readFileSync(clipsPath, 'utf8'));
  console.log(`Replaying ${clips.length} held-out clips through the application modules\n`);

  const classifier = newClassifier();
  const repRows = [];
  const angleRows = [];
  const agreementRows = [];
  const latency = { preprocess: [], inference: [] };
  const skew = { faithful: [], shipped: [] };

  for (const clip of clips) {
    const rules = replayRules(clip);
    const faithful = replayModel(clip, classifier, trainingFaithful);
    const shipped = replayModel(clip, classifier, asShipped);

    repRows.push({
      video: JSON.stringify(clip.video),
      exercise: clip.exercise,
      frames: rules.frames,
      reps_counted: rules.reps
    });

    angleRows.push({
      video: JSON.stringify(clip.video),
      exercise: clip.exercise,
      width: clip.width,
      height: clip.height,
      aspect_ratio: round(clip.width / clip.height, 3),
      angle_mae_deg: round(rules.angleMae, 2),
      angle_max_deg: round(rules.angleMax, 2)
    });

    agreementRows.push({
      video: JSON.stringify(clip.video),
      true_exercise: clip.exercise,
      predicted_exercise: shipped.predicted,
      clip_correct: shipped.predicted === clip.exercise ? 1 : 0,
      window_accuracy: round(shipped.windowAccuracy),
      vote_share: round(shipped.confidence),
      rules_counted_reps: rules.reps,
      rules_corroborate: shipped.predicted === clip.exercise && rules.reps > 0 ? 1 : 0
    });

    latency.preprocess.push(...shipped.stageTimes.preprocess);
    latency.inference.push(...shipped.stageTimes.inference);
    skew.faithful.push(faithful.windowAccuracy);
    skew.shipped.push(shipped.windowAccuracy);
  }

  console.log('Writing results:');
  writeCsv('repcount_system.csv', repRows);
  writeCsv('angle_validation.csv', angleRows);
  writeCsv('agreement.csv', agreementRows);

  const percentile = (values, p) => {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
  };
  writeCsv('latency_js.csv', [
    { stage: 'preprocess', frames: latency.preprocess.length,
      mean_ms: round(mean(latency.preprocess), 4),
      p95_ms: round(percentile(latency.preprocess, 0.95), 4) },
    { stage: 'lstm_inference', frames: latency.inference.length,
      mean_ms: round(mean(latency.inference), 4),
      p95_ms: round(percentile(latency.inference, 0.95), 4) }
  ]);
  writeCsv('skew.csv', [
    { pipeline: 'training_faithful', clip_mean_window_accuracy: round(mean(skew.faithful)) },
    { pipeline: 'as_shipped', clip_mean_window_accuracy: round(mean(skew.shipped)) }
  ]);

  const correct = agreementRows.filter(r => r.clip_correct).length;
  const corroborated = agreementRows.filter(r => r.rules_corroborate).length;
  const clipsWithReps = repRows.filter(r => r.reps_counted > 0).length;
  console.log('\nSummary');
  console.log(`  clip-level accuracy (majority vote): ${correct}/${clips.length}`);
  console.log(`  clips where a rep was detected:      ${clipsWithReps}/${clips.length}`);
  console.log(`  model and rules corroborate:         ${corroborated}/${clips.length}`);
  console.log(`  angle convention MAE:                ${round(mean(angleRows.map(r => r.angle_mae_deg)), 2)} deg`);
  console.log(`  window accuracy faithful vs shipped: ${round(mean(skew.faithful))} vs ${round(mean(skew.shipped))}`);
}

main();
