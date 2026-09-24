/**
 * Train/serve parity for the classifier itself.
 *
 * src/core/dlClassifier.js re-implements the Keras forward pass in plain
 * JavaScript. This asserts it reproduces the probabilities Keras produces for
 * the same held-out windows, so the accuracy reported in results/ is the
 * accuracy the browser actually delivers.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { DLClassifier } from '../../src/core/dlClassifier.js';
import { loadFixture, loadWeightsFromDisk, loadSaliencyFromDisk, maxAbsDiff } from './helpers.mjs';

// Keras accumulates in float32, the JavaScript port in float64, so the two
// disagree in the last few digits. Anything larger is a real defect.
const PROBABILITY_TOLERANCE = 1e-4;
const fixture = loadFixture('model_forward.json');

function windowToSequence(window) {
  return window.map(frame => Float32Array.from(frame));
}

function buildClassifier() {
  const classifier = new DLClassifier({ autoLoad: false });
  loadWeightsFromDisk(classifier);
  return classifier;
}

test('exported weight manifest describes the trained architecture', () => {
  const classifier = new DLClassifier({ autoLoad: false });
  loadWeightsFromDisk(classifier);

  const expectedShapes = {
    'lstm/kernel': [132, 256],
    'lstm/recurrent_kernel': [64, 256],
    'lstm/bias': [256],
    'lstm_1/kernel': [64, 128],
    'lstm_1/recurrent_kernel': [32, 128],
    'lstm_1/bias': [128],
    'dense/kernel': [32, 32],
    'dense/bias': [32],
    'dense_1/kernel': [32, 3],
    'dense_1/bias': [3]
  };

  for (const [name, shape] of Object.entries(expectedShapes)) {
    const weight = classifier.weights[name];
    assert.ok(weight, `missing weight ${name}`);
    assert.deepEqual(weight.shape, shape, `${name} has the wrong shape`);
  }
});

test('JavaScript inference reproduces the Keras probabilities', () => {
  const classifier = buildClassifier();

  fixture.windows.forEach((window, index) => {
    const probabilities = classifier._runInference(windowToSequence(window));
    const expected = fixture.probabilities[index];

    assert.notEqual(probabilities, null, `sample ${index} returned no prediction`);
    const diff = maxAbsDiff(probabilities, expected);
    assert.ok(diff < PROBABILITY_TOLERANCE,
      `sample ${index} (${fixture.sources[index]}) differs by ${diff}: ` +
      `js=${Array.from(probabilities).map(p => p.toFixed(5))} keras=${expected.map(p => p.toFixed(5))}`);
  });
});

test('JavaScript inference picks the same class as Keras', () => {
  const classifier = buildClassifier();

  fixture.windows.forEach((window, index) => {
    const probabilities = Array.from(classifier._runInference(windowToSequence(window)));
    const expected = fixture.probabilities[index];
    const argmax = arr => arr.indexOf(Math.max(...arr));

    assert.equal(argmax(probabilities), argmax(expected),
      `sample ${index} (${fixture.sources[index]}) classified differently`);
  });
});

test('probabilities form a distribution', () => {
  const classifier = buildClassifier();

  fixture.windows.forEach((window, index) => {
    const probabilities = Array.from(classifier._runInference(windowToSequence(window)));
    const sum = probabilities.reduce((total, p) => total + p, 0);
    assert.ok(Math.abs(sum - 1) < 1e-5, `sample ${index} sums to ${sum}`);
    assert.ok(probabilities.every(p => p >= 0 && p <= 1), `sample ${index} has out-of-range values`);
  });
});

test('inference is refused until the weights are present', () => {
  const classifier = new DLClassifier({ autoLoad: false });
  assert.equal(classifier._runInference(windowToSequence(fixture.windows[0])), null);
});

test('processFrame buffers frames and only predicts on a full window', () => {
  const classifier = buildClassifier();
  const frame = Array.from({ length: 33 }, (_, i) => ({
    x: 0.4 + i * 0.01, y: 0.3 + i * 0.012, z: i * 0.001, visibility: 0.9
  }));

  for (let i = 1; i < classifier.slidingWindowSize; i++) {
    const result = classifier.processFrame(frame, 'squat');
    assert.equal(result.windowFill, `${i}/15`);
    assert.equal(result.classProbabilities, null, 'must not predict on a partial window');
  }

  const result = classifier.processFrame(frame, 'squat');
  assert.equal(result.windowFill, '15/15');
  assert.notEqual(result.classProbabilities, null, 'a full window must produce a prediction');

  classifier.reset();
  assert.equal(classifier.landmarkHistory.length, 0);
  assert.equal(classifier.prevSmoothed, null);
});

test('feature attribution comes from the measured saliency file', () => {
  const classifier = buildClassifier();
  classifier.saliency = loadSaliencyFromDisk();

  // No prediction yet: nothing measured, so nothing claimed.
  assert.deepEqual(classifier._featureImportance(null), []);

  for (const className of fixture.classNames) {
    const features = classifier._featureImportance(className);
    assert.ok(features.length > 0, `no attribution recorded for ${className}`);

    const total = features.reduce((sum, f) => sum + f.weight, 0);
    assert.ok(Math.abs(total - 1) < 0.01, `${className} weights sum to ${total}`);
    assert.ok(features.every(f => typeof f.name === 'string' && f.weight >= 0),
      `${className} has a malformed entry`);
  }
});

test('an unknown exercise yields no attribution rather than a guess', () => {
  const classifier = buildClassifier();
  classifier.saliency = loadSaliencyFromDisk();

  // The app still offers a posture mode, which the model was never trained on.
  assert.deepEqual(classifier._featureImportance('posture'), []);
});
