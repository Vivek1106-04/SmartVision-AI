/**
 * HUD Canvas Renderer Engine
 * Draws live video feed, skeleton wireframe, joint keypoint nodes, spatial angle arcs,
 * motion vector trails, and spatial ROI bounding boxes with high contrast formatting.
 */

// MediaPipe 33-Keypoint Skeleton Topology
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Upper body & arms
  [11, 23], [12, 24], [23, 24],                    // Hips & torso
  [23, 25], [25, 27], [24, 26], [26, 28],          // Legs & knees
  [27, 31], [28, 32]                               // Feet
];

export class CanvasHUDRenderer {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');

    // Render Options
    this.showSkeleton = true;
    this.showAngles = true;
    this.showMotionTrails = true;
    this.showBoundingBox = true;
  }

  resize(width, height) {
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /**
   * Main Render Pipeline Loop
   */
  render(videoSource, landmarks, evaluationResult, cvEngine, spatialROI) {
    const width = this.canvas.width;
    const height = this.canvas.height;
    this.clear();

    // 1. Draw video frame image onto canvas background
    if (videoSource && videoSource.videoWidth > 0 && videoSource.readyState >= 2) {
      this.ctx.drawImage(videoSource, 0, 0, width, height);
    } else {
      // Dark background for synthetic or video-less mode
      this.ctx.fillStyle = '#0F172A';
      this.ctx.fillRect(0, 0, width, height);
    }

    if (!landmarks || landmarks.length < 33) return;

    // 2. Draw Motion Vector Trails
    if (this.showMotionTrails && cvEngine.motionTrailBuffer.length > 0) {
      this.drawMotionTrails(cvEngine.motionTrailBuffer);
    }

    // 3. Draw Spatial ROI Bounding Box
    if (this.showBoundingBox && spatialROI) {
      this.drawBoundingBox(spatialROI, evaluationResult.feedback.status);
    }

    // 4. Draw Skeleton & Joint Nodes
    if (this.showSkeleton) {
      this.drawSkeleton(landmarks, width, height, evaluationResult.feedback.status);
    }

    // 5. Draw Joint Angle Text Badges
    if (this.showAngles && evaluationResult.metrics) {
      this.drawAngleHUD(landmarks, width, height, evaluationResult.metrics);
    }
  }

  /**
   * Draws Motion Path Trails
   */
  drawMotionTrails(trailBuffer) {
    const ctx = this.ctx;
    ctx.lineWidth = 2;

    trailBuffer.forEach((framePoints, frameIdx) => {
      const alpha = (frameIdx + 1) / trailBuffer.length;
      ctx.strokeStyle = `rgba(37, 99, 235, ${alpha * 0.4})`;

      framePoints.forEach(pt => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4 * alpha, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(37, 99, 235, ${alpha * 0.6})`;
        ctx.fill();
      });
    });
  }

  /**
   * Draws Spatial Bounding Box
   */
  drawBoundingBox(roi, status) {
    const ctx = this.ctx;
    ctx.strokeStyle = status === 'error' ? '#DC2626' : status === 'warning' ? '#D97706' : '#2563EB';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);

    ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
    ctx.setLineDash([]); // Reset line dash

    // Corner markers
    const markerLen = 12;
    ctx.strokeStyle = '#2563EB';
    ctx.lineWidth = 2.5;

    // Top-Left corner
    ctx.beginPath();
    ctx.moveTo(roi.x, roi.y + markerLen);
    ctx.lineTo(roi.x, roi.y);
    ctx.lineTo(roi.x + markerLen, roi.y);
    ctx.stroke();

    // Bottom-Right corner
    ctx.beginPath();
    ctx.moveTo(roi.x + roi.width, roi.y + roi.height - markerLen);
    ctx.lineTo(roi.x + roi.width, roi.y + roi.height);
    ctx.lineTo(roi.x + roi.width - markerLen, roi.y + roi.height);
    ctx.stroke();
  }

  /**
   * Draws High-Contrast Skeleton Wireframe & Keypoint Nodes
   */
  drawSkeleton(landmarks, width, height, status) {
    const ctx = this.ctx;

    // Wireframe color based on status
    const wireframeColor = status === 'error' ? '#DC2626' : status === 'warning' ? '#D97706' : '#2563EB';

    // Connections
    ctx.lineWidth = 3;
    ctx.strokeStyle = wireframeColor;

    POSE_CONNECTIONS.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];

      if (p1 && p2 && (p1.visibility || 1) > 0.5 && (p2.visibility || 1) > 0.5) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    });

    // Keypoint Joint Nodes
    landmarks.forEach((lm, idx) => {
      if ([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(idx)) {
        if ((lm.visibility || 1) > 0.5) {
          const cx = lm.x * width;
          const cy = lm.y * height;

          // Solid node core
          ctx.beginPath();
          ctx.arc(cx, cy, 6, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();

          ctx.beginPath();
          ctx.arc(cx, cy, 6, 0, Math.PI * 2);
          ctx.strokeStyle = wireframeColor;
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }
      }
    });
  }

  /**
   * Draws Angle Text Badges
   */
  drawAngleHUD(landmarks, width, height, metrics) {
    const ctx = this.ctx;
    ctx.font = '600 13px "Inter", sans-serif';

    const targetIdx = metrics.angle1Name.includes('Knee') ? 25 : 13;
    const lm = landmarks[targetIdx];

    if (lm && (lm.visibility || 1) > 0.5) {
      const x = lm.x * width + 12;
      const y = lm.y * height - 8;

      ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.strokeStyle = '#CBD5E1';
      ctx.lineWidth = 1;

      const text = `${metrics.angle1Name}: ${metrics.angle1}°`;
      const textWidth = ctx.measureText(text).width;

      ctx.fillRect(x - 4, y - 14, textWidth + 10, 20);
      ctx.strokeRect(x - 4, y - 14, textWidth + 10, 20);

      ctx.fillStyle = '#0F172A';
      ctx.fillText(text, x, y);
    }
  }
}
