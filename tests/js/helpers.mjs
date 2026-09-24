/**
 * Shared helpers for the parity and unit tests.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
export const PROJECT_ROOT = join(HERE, '..', '..');

export function loadFixture(name) {
  return JSON.parse(readFileSync(join(PROJECT_ROOT, 'tests', 'fixtures', name), 'utf8'));
}

/** Convert one fixture frame [[x,y,z,v], ...] into the landmark objects the app uses. */
export function toLandmarkObjects(frame) {
  return frame.map(([x, y, z, visibility]) => ({ x, y, z, visibility }));
}

/**
 * Load the exported weights straight from disk.
 * The browser fetches these same two files over HTTP.
 */
export function loadWeightsFromDisk(classifier) {
  const modelDir = join(PROJECT_ROOT, 'public', 'models', 'exercise_classifier');
  const manifest = JSON.parse(readFileSync(join(modelDir, 'model.json'), 'utf8'));
  const entry = manifest.weightsManifest[0];
  const file = readFileSync(join(modelDir, entry.paths[0]));
  // Copy into a standalone ArrayBuffer: Node may hand back a pooled Buffer
  // whose byteOffset is not zero, which breaks the Float32Array views.
  const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  classifier.weights = classifier._parseWeights(buffer, entry.weights);
  classifier.modelLoaded = true;
  return manifest;
}

/** Read the measured feature attribution the app fetches at runtime. */
export function loadSaliencyFromDisk() {
  const path = join(PROJECT_ROOT, 'public', 'models', 'exercise_classifier', 'saliency.json');
  return JSON.parse(readFileSync(path, 'utf8'));
}

/**
 * Largest absolute difference between two flat numeric sequences.
 * Returns Infinity on a length mismatch or a non-finite value, so a truncated
 * or NaN-filled result can never be mistaken for a perfect match.
 */
export function maxAbsDiff(actual, expected) {
  if (actual == null || actual.length !== expected.length) return Infinity;

  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(diff)) return Infinity;
    if (diff > worst) worst = diff;
  }
  return worst;
}
