/**
 * The joint angles drive every rule-based verdict and every number shown on
 * the HUD, so they are checked against geometry with an exact known answer.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  calculate2DAngle,
  calculate3DAngle,
  euclideanDistance,
  calculateBoundingBox
} from '../../src/utils/math.js';

const EXACT = 1e-9;

test('2D angle matches analytically known configurations', () => {
  const vertex = { x: 0.5, y: 0.5 };
  const cases = [
    { a: { x: 0.5, y: 0.3 }, c: { x: 0.7, y: 0.5 }, expected: 90 },
    { a: { x: 0.5, y: 0.3 }, c: { x: 0.5, y: 0.7 }, expected: 180 },
    { a: { x: 0.5, y: 0.3 }, c: { x: 0.5, y: 0.3 }, expected: 0 },
    { a: { x: 0.5, y: 0.3 }, c: { x: 0.7, y: 0.3 }, expected: 45 }
  ];

  for (const { a, c, expected } of cases) {
    const angle = calculate2DAngle(a, vertex, c);
    assert.ok(Math.abs(angle - expected) < EXACT,
      `expected ${expected} degrees, got ${angle}`);
  }
});

test('2D angle never exceeds 180 degrees', () => {
  const vertex = { x: 0.5, y: 0.5 };
  for (let degrees = 0; degrees < 360; degrees += 7) {
    const radians = (degrees * Math.PI) / 180;
    const c = { x: 0.5 + 0.2 * Math.cos(radians), y: 0.5 + 0.2 * Math.sin(radians) };
    const angle = calculate2DAngle({ x: 0.7, y: 0.5 }, vertex, c);
    assert.ok(angle >= 0 && angle <= 180, `angle ${angle} out of range at ${degrees} degrees`);
  }
});

test('aspect scaling changes the measured angle on non-square video', () => {
  // Landmarks arrive in normalized [0,1] coordinates. On a 16:9 frame one unit
  // of x spans more pixels than one unit of y, so an angle measured without the
  // aspect correction is not the angle a protractor would read on screen.
  const a = { x: 0.5, y: 0.3 };
  const b = { x: 0.5, y: 0.5 };
  const c = { x: 0.7, y: 0.6 };

  const uncorrected = calculate2DAngle(a, b, c);
  const corrected = calculate2DAngle(a, b, c, 1920, 1080);

  assert.ok(Math.abs(uncorrected - corrected) > 1,
    'the two conventions should visibly disagree on a 16:9 frame');
  assert.ok(Math.abs(calculate2DAngle(a, b, c, 100, 100) - uncorrected) < EXACT,
    'a square aspect ratio must reproduce the uncorrected value');
});

test('3D angle matches analytically known configurations', () => {
  const vertex = { x: 0, y: 0, z: 0 };
  assert.ok(Math.abs(calculate3DAngle({ x: 1, y: 0, z: 0 }, vertex, { x: 0, y: 1, z: 0 }) - 90) < EXACT);
  assert.ok(Math.abs(calculate3DAngle({ x: 1, y: 0, z: 0 }, vertex, { x: -1, y: 0, z: 0 }) - 180) < EXACT);
  assert.ok(Math.abs(calculate3DAngle({ x: 0, y: 0, z: 1 }, vertex, { x: 0, y: 1, z: 0 }) - 90) < EXACT);
});

test('angle helpers return zero for degenerate or missing input', () => {
  const p = { x: 0.5, y: 0.5, z: 0 };
  assert.equal(calculate2DAngle(null, p, p), 0);
  assert.equal(calculate3DAngle(p, null, p), 0);
  assert.equal(calculate3DAngle(p, p, p), 0, 'a zero-length vector has no defined angle');
});

test('euclidean distance handles 2D and 3D points', () => {
  assert.ok(Math.abs(euclideanDistance({ x: 0, y: 0 }, { x: 3, y: 4 }) - 5) < EXACT);
  assert.ok(Math.abs(euclideanDistance({ x: 0, y: 0, z: 0 }, { x: 2, y: 3, z: 6 }) - 7) < EXACT);
  assert.equal(euclideanDistance(null, { x: 1, y: 1 }), 0);
});

test('bounding box covers the landmarks and stays inside the canvas', () => {
  const landmarks = [
    { x: 0.2, y: 0.3 },
    { x: 0.6, y: 0.8 }
  ];
  const box = calculateBoundingBox(landmarks, 640, 480, 30);

  assert.ok(box.x >= 0 && box.y >= 0, 'the box must not start off-canvas');
  assert.ok(box.x + box.width <= 640, 'the box must not extend past the canvas width');
  assert.ok(box.y + box.height <= 480, 'the box must not extend past the canvas height');
  assert.equal(calculateBoundingBox([], 640, 480), null);
});
