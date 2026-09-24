/**
 * The two preprocessing routes a landmark frame can take before it reaches the
 * classifier.
 *
 * `trainingFaithful` is what tools/extract_landmarks.py did when the training
 * tensors were built: normalize first, then smooth once.
 *
 * `asShipped` is what src/main.js actually does: CVEngine smooths the raw
 * image-space landmarks, and DLClassifier then normalizes and smooths again.
 * Comparing the two measures the train/serve skew this introduces.
 */
import { CVEngine } from '../../src/core/cvEngine.js';

const EMA_ALPHA = 0.35;

/** Normalize a frame the way both the training pipeline and the app do. */
function normalize(frame) {
  const midHip = ['x', 'y', 'z'].map(axis => (frame[23][axis] + frame[24][axis]) / 2);
  const midShoulder = ['x', 'y', 'z'].map(axis => (frame[11][axis] + frame[12][axis]) / 2);

  let torso = Math.hypot(
    midShoulder[0] - midHip[0], midShoulder[1] - midHip[1], midShoulder[2] - midHip[2]
  );
  if (torso < 1e-5) torso = 1.0;

  return frame.map(lm => ({
    x: (lm.x - midHip[0]) / torso,
    y: (lm.y - midHip[1]) / torso,
    z: (lm.z - midHip[2]) / torso,
    visibility: lm.visibility
  }));
}

/** One EMA step over a whole frame, returning the new smoothed frame. */
function smoothFrame(frame, previous, alpha = EMA_ALPHA) {
  if (!previous) return frame;
  return frame.map((lm, i) => ({
    x: alpha * lm.x + (1 - alpha) * previous[i].x,
    y: alpha * lm.y + (1 - alpha) * previous[i].y,
    z: alpha * lm.z + (1 - alpha) * previous[i].z,
    visibility: alpha * lm.visibility + (1 - alpha) * previous[i].visibility
  }));
}

function toFeatures(frame) {
  const features = new Float32Array(132);
  frame.forEach((lm, i) => {
    features[i * 4] = lm.x;
    features[i * 4 + 1] = lm.y;
    features[i * 4 + 2] = lm.z;
    features[i * 4 + 3] = lm.visibility;
  });
  return features;
}

/** Normalize then smooth once - matches the training tensors exactly. */
export function trainingFaithful() {
  let previous = null;
  return frame => {
    const smoothed = smoothFrame(normalize(frame), previous);
    previous = smoothed;
    return toFeatures(smoothed);
  };
}

/** Smooth raw coordinates, normalize, then smooth again - what the app ships. */
export function asShipped() {
  const cvEngine = new CVEngine();
  let previous = null;
  return frame => {
    const normalized = normalize(cvEngine.smoothLandmarks(frame));
    const smoothed = smoothFrame(normalized, previous);
    previous = smoothed;
    return toFeatures(smoothed);
  };
}

/** Landmark objects for one frame of the extracted JSON sequences. */
export function toLandmarks(frame) {
  return frame.map(([x, y, z, visibility]) => ({ x, y, z, visibility }));
}
