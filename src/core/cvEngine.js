import { calculateBoundingBox, EMAFilter } from '../utils/math.js';

/**
 * Computer Vision Engine
 * Performs spatial processing, keypoint smoothing, kinetic angular velocity derivations,
 * optical motion trail buffering, and spatial ROI bounding box calculations.
 */
export class CVEngine {
  constructor() {
    this.emaFilters = new Map(); // Index -> EMAFilter instance
    this.motionTrailBuffer = []; // Historical positions of key joints
    this.maxTrailLength = 20;
    this.previousAngle = null;
    this.previousTimestamp = performance.now();
  }

  /**
   * Applies Exponential Moving Average (EMA) noise reduction to keypoint coordinates
   */
  smoothLandmarks(landmarks) {
    if (!landmarks) return null;

    return landmarks.map((lm, idx) => {
      if (!this.emaFilters.has(idx)) {
        this.emaFilters.set(idx, {
          x: new EMAFilter(0.35),
          y: new EMAFilter(0.35),
          z: new EMAFilter(0.35)
        });
      }

      const filterSet = this.emaFilters.get(idx);
      return {
        ...lm,
        x: filterSet.x.filter(lm.x),
        y: filterSet.y.filter(lm.y),
        z: filterSet.z.filter(lm.z)
      };
    });
  }

  /**
   * Calculates Kinetic Angular Velocity: w = d(theta) / dt in degrees per second (deg/s)
   */
  calculateKineticVelocity(currentAngle) {
    const now = performance.now();
    const dt = (now - this.previousTimestamp) / 1000.0;
    this.previousTimestamp = now;

    if (this.previousAngle === null || dt <= 0 || !currentAngle) {
      this.previousAngle = currentAngle;
      return 0.0;
    }

    const angularChange = Math.abs(currentAngle - this.previousAngle);
    const angularVelocity = angularChange / dt;

    this.previousAngle = currentAngle;
    return Math.min(500.0, angularVelocity); // Cap velocity spikes
  }

  /**
   * Updates motion trail buffer for key joint visualization
   */
  updateMotionTrail(landmarks, canvasWidth, canvasHeight) {
    if (!landmarks) return;

    // Track key joints: Left Wrist (15), Right Wrist (16), Left Ankle (27), Right Ankle (28)
    const trackedKeypoints = [15, 16, 25, 26].map(idx => {
      const lm = landmarks[idx];
      return lm ? { x: lm.x * canvasWidth, y: lm.y * canvasHeight, id: idx } : null;
    }).filter(Boolean);

    this.motionTrailBuffer.push(trackedKeypoints);
    if (this.motionTrailBuffer.length > this.maxTrailLength) {
      this.motionTrailBuffer.shift();
    }
  }

  /**
   * Computes Spatial ROI Bounding Box with dynamic padding
   */
  getSpatialROI(landmarks, width, height) {
    return calculateBoundingBox(landmarks, width, height, 40);
  }

  reset() {
    this.emaFilters.clear();
    this.motionTrailBuffer = [];
    this.previousAngle = null;
  }
}
