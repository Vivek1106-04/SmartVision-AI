/**
 * Train/serve parity: the browser must normalize and smooth landmarks exactly
 * as tools/extract_landmarks.py did when the training tensors were built.
 * Any drift here silently shifts the model's input distribution.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { DLClassifier } from '../../src/core/dlClassifier.js';
import { EMAFilter } from '../../src/utils/math.js';
import { loadFixture, toLandmarkObjects, maxAbsDiff } from './helpers.mjs';

const PARITY_TOLERANCE = 1e-9;
const fixture = loadFixture('preprocess.json');

test('normalization matches the Python training pipeline', () => {
  const classifier = new DLClassifier({ autoLoad: false });

  fixture.raw.forEach((frame, frameIndex) => {
    const normalized = classifier._normalizeLandmarks(toLandmarkObjects(frame));
    const expected = fixture.normalized[frameIndex];

    assert.notEqual(normalized, null, `frame ${frameIndex} failed to normalize`);
    normalized.forEach((landmark, i) => {
      const [x, y, z, visibility] = expected[i];
      const diff = maxAbsDiff(
        [landmark.x, landmark.y, landmark.z, landmark.visibility],
        [x, y, z, visibility]
      );
      assert.ok(diff < PARITY_TOLERANCE,
        `frame ${frameIndex} landmark ${i} differs by ${diff}`);
    });
  });
});

test('EMA smoothing matches the Python training pipeline', () => {
  const classifier = new DLClassifier({ autoLoad: false });

  fixture.normalized.forEach((frame, frameIndex) => {
    const smoothed = classifier._applyEMA(toLandmarkObjects(frame));
    const expected = fixture.smoothed[frameIndex];

    smoothed.forEach((landmark, i) => {
      const diff = maxAbsDiff(
        [landmark.x, landmark.y, landmark.z, landmark.visibility],
        expected[i]
      );
      assert.ok(diff < PARITY_TOLERANCE,
        `frame ${frameIndex} landmark ${i} differs by ${diff}`);
    });
  });
});

test('mid-hip becomes the origin and torso length becomes one', () => {
  const classifier = new DLClassifier({ autoLoad: false });
  const normalized = classifier._normalizeLandmarks(toLandmarkObjects(fixture.raw[0]));

  const midHipX = (normalized[23].x + normalized[24].x) / 2;
  const midHipY = (normalized[23].y + normalized[24].y) / 2;
  const midHipZ = (normalized[23].z + normalized[24].z) / 2;
  assert.ok(Math.abs(midHipX) < 1e-12 && Math.abs(midHipY) < 1e-12 && Math.abs(midHipZ) < 1e-12,
    'mid-hip should sit at the origin after translation');

  const midShoulderX = (normalized[11].x + normalized[12].x) / 2;
  const midShoulderY = (normalized[11].y + normalized[12].y) / 2;
  const midShoulderZ = (normalized[11].z + normalized[12].z) / 2;
  const torso = Math.hypot(midShoulderX - midHipX, midShoulderY - midHipY, midShoulderZ - midHipZ);
  assert.ok(Math.abs(torso - 1) < 1e-12, `torso length should be 1, got ${torso}`);
});

test('normalization rejects a frame with too few landmarks', () => {
  const classifier = new DLClassifier({ autoLoad: false });
  assert.equal(classifier._normalizeLandmarks(null), null);
  assert.equal(classifier._normalizeLandmarks([{ x: 0, y: 0, z: 0 }]), null);
});

test('degenerate torso length falls back to a scale of one', () => {
  const classifier = new DLClassifier({ autoLoad: false });
  // Shoulders and hips at the same point: torso length is zero.
  const collapsed = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  collapsed[7] = { x: 0.9, y: 0.2, z: 0, visibility: 1 };

  const normalized = classifier._normalizeLandmarks(collapsed);
  assert.equal(normalized[7].x, 0.9 - 0.5, 'coordinates should be translated but not rescaled');
});

test('EMA filter reproduces S_t = a*Y_t + (1-a)*S_{t-1}', () => {
  const filter = new EMAFilter(0.35);
  assert.equal(filter.filter(10), 10, 'first sample passes through unchanged');
  assert.ok(Math.abs(filter.filter(20) - (0.35 * 20 + 0.65 * 10)) < 1e-12);

  filter.reset();
  assert.equal(filter.filter(4), 4, 'reset clears the smoothing state');
});
