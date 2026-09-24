/**
 * Vector Mathematics & Signal Processing Utilities for Computer Vision & Deep Learning
 */

/**
 * Calculates 2D angle (in degrees) formed at joint 'b' by keypoints 'a', 'b', and 'c'
 * Optional aspect width/height scaling prevents geometric distortion on non-square video streams (e.g. 16:9).
 */
export function calculate2DAngle(a, b, c, aspectWidth = 1, aspectHeight = 1) {
  if (!a || !b || !c) return 0;

  const ax = a.x * aspectWidth, ay = a.y * aspectHeight;
  const bx = b.x * aspectWidth, by = b.y * aspectHeight;
  const cx = c.x * aspectWidth, cy = c.y * aspectHeight;

  const radians = Math.atan2(cy - by, cx - bx) - Math.atan2(ay - by, ax - bx);
  let angle = Math.abs((radians * 180.0) / Math.PI);

  if (angle > 180.0) {
    angle = 360.0 - angle;
  }

  return angle;
}

/**
 * Calculates 3D Spatial Vector Angle between 3 keypoints considering Z depth
 */
export function calculate3DAngle(a, b, c) {
  if (!a || !b || !c) return 0;

  // Vector ba
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
  // Vector bc
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };

  // Dot product
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  
  // Magnitudes
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

  if (mag1 * mag2 === 0) return 0;

  const cosTheta = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return (Math.acos(cosTheta) * 180.0) / Math.PI;
}

/**
 * Calculates Euclidean Distance between two 2D/3D points
 */
export function euclideanDistance(p1, p2) {
  if (!p1 || !p2) return 0;
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Exponential Moving Average (EMA) low-pass filter for keypoint jitter reduction
 * S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
 */
export class EMAFilter {
  constructor(alpha = 0.35) {
    this.alpha = alpha;
    this.previousValue = null;
  }

  filter(currentValue) {
    if (this.previousValue === null) {
      this.previousValue = currentValue;
      return currentValue;
    }
    const filtered = this.alpha * currentValue + (1 - this.alpha) * this.previousValue;
    this.previousValue = filtered;
    return filtered;
  }

  reset() {
    this.previousValue = null;
  }
}

/**
 * Computes Spatial ROI Bounding Box around landmarks with safe canvas boundary clamping
 */
export function calculateBoundingBox(landmarks, width, height, padding = 30) {
  if (!landmarks || landmarks.length === 0) return null;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  landmarks.forEach(lm => {
    const px = lm.x * width;
    const py = lm.y * height;
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
  });

  const clampedX = Math.max(0, minX - padding);
  const clampedY = Math.max(0, minY - padding);

  return {
    x: clampedX,
    y: clampedY,
    width: Math.min(width - clampedX, (maxX - minX) + padding * 2),
    height: Math.min(height - clampedY, (maxY - minY) + padding * 2)
  };
}
