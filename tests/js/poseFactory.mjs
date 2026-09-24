/**
 * Builds synthetic 33-landmark frames whose joint angles are known exactly,
 * so the rep-counting state machines can be driven through a controlled
 * movement instead of a recorded video.
 */
const NUM_LANDMARKS = 33;

/** Rotate the unit vector pointing "up the screen" by a given angle. */
function armAt(origin, degrees, length) {
  const radians = (degrees * Math.PI) / 180;
  // (0,-1) rotated by `degrees`; at 180 degrees the limb points straight down.
  return {
    x: origin.x + length * Math.sin(radians),
    y: origin.y - length * Math.cos(radians)
  };
}

function blankFrame() {
  return Array.from({ length: NUM_LANDMARKS }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.99 }));
}

/**
 * A standing/squatting body whose knee flexion angle is exactly `kneeAngle`.
 * Shoulders sit directly above the hips, so the torso-lean rule stays quiet,
 * and both knees track over both ankles, so the valgus rule stays quiet.
 */
export function squatFrame(kneeAngle) {
  const frame = blankFrame();
  const segment = 0.15;

  for (const [side, offset] of [['left', -0.05], ['right', 0.05]]) {
    const knee = { x: 0.5 + offset, y: 0.6 };
    const hip = { x: knee.x, y: knee.y - segment };
    const shoulder = { x: hip.x, y: hip.y - segment };
    const ankle = armAt(knee, kneeAngle, segment);

    const [shoulderIdx, hipIdx, kneeIdx, ankleIdx] =
      side === 'left' ? [11, 23, 25, 27] : [12, 24, 26, 28];
    frame[shoulderIdx] = { ...frame[shoulderIdx], ...shoulder };
    frame[hipIdx] = { ...frame[hipIdx], ...hip };
    frame[kneeIdx] = { ...frame[kneeIdx], ...knee };
    frame[ankleIdx] = { ...frame[ankleIdx], ...ankle };
  }
  return frame;
}

/**
 * An upper body whose elbow flexion angle is exactly `elbowAngle`.
 * The shoulder sits directly above the hip, keeping the momentum rule quiet.
 */
export function armFrame(elbowAngle) {
  const frame = blankFrame();
  const segment = 0.15;

  for (const [side, offset] of [['left', -0.05], ['right', 0.05]]) {
    const elbow = { x: 0.5 + offset, y: 0.5 };
    const shoulder = { x: elbow.x, y: elbow.y - segment };
    const hip = { x: shoulder.x, y: shoulder.y + segment * 2 };
    const ankle = { x: hip.x, y: hip.y + segment * 2 };
    const wrist = armAt(elbow, elbowAngle, segment);

    const [shoulderIdx, elbowIdx, wristIdx, hipIdx, ankleIdx] =
      side === 'left' ? [11, 13, 15, 23, 27] : [12, 14, 16, 24, 28];
    frame[shoulderIdx] = { ...frame[shoulderIdx], ...shoulder };
    frame[elbowIdx] = { ...frame[elbowIdx], ...elbow };
    frame[wristIdx] = { ...frame[wristIdx], ...wrist };
    frame[hipIdx] = { ...frame[hipIdx], ...hip };
    frame[ankleIdx] = { ...frame[ankleIdx], ...ankle };
  }
  return frame;
}

/** Feed a sequence of angles through an evaluator and return the last result. */
export function driveAngles(evaluator, frameFactory, angles) {
  let result = null;
  for (const angle of angles) {
    result = evaluator.evaluate(frameFactory(angle), {});
  }
  return result;
}

/** Angles for one descend-and-return cycle between two extremes. */
export function repCycle(top, bottom, steps = 12) {
  const down = Array.from({ length: steps }, (_, i) => top + ((bottom - top) * i) / (steps - 1));
  return [...down, ...down.slice().reverse()];
}
