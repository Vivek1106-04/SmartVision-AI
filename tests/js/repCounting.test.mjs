/**
 * Rep counting is the number the user sees and the number the system metrics
 * in results/ are measured against, so each state machine is driven through
 * controlled movements with a known correct count.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { SquatEvaluator } from '../../src/exercises/squatEvaluator.js';
import { PushupEvaluator } from '../../src/exercises/pushupEvaluator.js';
import { BicepCurlEvaluator } from '../../src/exercises/bicepCurlEvaluator.js';
import { squatFrame, armFrame, driveAngles, repCycle } from './poseFactory.mjs';

test('the pose factory produces the joint angle it was asked for', () => {
  const evaluator = new SquatEvaluator();
  for (const target of [170, 120, 85]) {
    const result = evaluator.evaluate(squatFrame(target), {});
    assert.equal(result.metrics.angle1, target,
      `factory produced ${result.metrics.angle1} degrees instead of ${target}`);
  }
});

test('squat counts one rep per full-depth cycle', () => {
  const evaluator = new SquatEvaluator();
  const cycle = repCycle(175, 80);

  for (let rep = 1; rep <= 3; rep++) {
    const result = driveAngles(evaluator, squatFrame, cycle);
    assert.equal(result.repCount, rep, `expected ${rep} reps after ${rep} cycles`);
  }
});

test('squat ignores a movement that never reaches depth', () => {
  const evaluator = new SquatEvaluator();
  driveAngles(evaluator, squatFrame, repCycle(175, 130));
  assert.equal(evaluator.repCount, 0, 'a shallow bend must not count as a rep');
});

test('squat reports knee valgus when the knees collapse inward', () => {
  const evaluator = new SquatEvaluator();
  const frame = squatFrame(95);
  // Pull both knees toward the midline while the ankles stay put.
  frame[25] = { ...frame[25], x: 0.495 };
  frame[26] = { ...frame[26], x: 0.505 };

  const result = evaluator.evaluate(frame, {});
  assert.match(result.feedback.message, /Knee Valgus/);
  assert.equal(result.feedback.status, 'error');
});

test('pushup counts one rep per full-depth cycle', () => {
  const evaluator = new PushupEvaluator();
  driveAngles(evaluator, armFrame, repCycle(170, 75));
  assert.equal(evaluator.repCount, 1);

  driveAngles(evaluator, armFrame, repCycle(170, 75));
  assert.equal(evaluator.repCount, 2);
});

test('pushup ignores a movement that never reaches depth', () => {
  const evaluator = new PushupEvaluator();
  driveAngles(evaluator, armFrame, repCycle(170, 120));
  assert.equal(evaluator.repCount, 0);
});

test('bicep curl counts one rep per full contraction cycle', () => {
  const evaluator = new BicepCurlEvaluator();
  driveAngles(evaluator, armFrame, repCycle(170, 35));
  assert.equal(evaluator.repCount, 1);

  driveAngles(evaluator, armFrame, repCycle(170, 35));
  assert.equal(evaluator.repCount, 2);
});

test('bicep curl ignores a partial contraction', () => {
  const evaluator = new BicepCurlEvaluator();
  driveAngles(evaluator, armFrame, repCycle(170, 70));
  assert.equal(evaluator.repCount, 0);
});

test('evaluators report a searching state instead of throwing on no pose', () => {
  for (const evaluator of [new SquatEvaluator(), new PushupEvaluator(), new BicepCurlEvaluator()]) {
    const result = evaluator.evaluate(null, {});
    assert.equal(result.phase, 'SEARCHING');
    assert.equal(result.repCount, 0);
  }
});

test('reset clears the count and the score history', () => {
  const evaluator = new SquatEvaluator();
  driveAngles(evaluator, squatFrame, repCycle(175, 80));
  assert.equal(evaluator.repCount, 1);

  evaluator.reset();
  assert.equal(evaluator.repCount, 0);
  assert.equal(evaluator.getAverageFormScore(), 100);
});
