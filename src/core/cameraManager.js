/**
 * Camera & Video Input Stream Manager
 * Handles live webcam media streams, uploaded video files, and synthetic video stream fallback.
 */
export class CameraManager {
  constructor(videoElement) {
    this.video = videoElement;
    this.currentStream = null;
    this.isSyntheticDemo = false;
    this.onVideoEndedCallback = null;

    if (this.video) {
      this.video.addEventListener('ended', () => {
        if (this.onVideoEndedCallback) {
          this.onVideoEndedCallback();
        }
      });
    }
  }

  /**
   * Starts live webcam camera stream
   */
  async startWebcam() {
    this.stopStream();
    this.isSyntheticDemo = false;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });

      this.currentStream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      return true;
    } catch (err) {
      console.error('Webcam stream access error:', err);
      throw new Error('Unable to access webcam. Please check browser permissions or upload a video.');
    }
  }

  /**
   * Loads an uploaded video file (loop = false to stop on video completion)
   */
  async loadVideoFile(file) {
    this.stopStream();
    this.isSyntheticDemo = false;

    const url = URL.createObjectURL(file);
    this.video.srcObject = null;
    this.video.src = url;
    this.video.loop = false; // Stop on completion
    await this.video.play();
  }

  /**
   * Runs synthetic skeleton simulation demo (for camera-less grading presentations)
   */
  runSyntheticDemo(onFrame) {
    this.stopStream();
    this.isSyntheticDemo = true;

    let t = 0;
    const maxTicks = 180; // ~6 seconds synthetic simulation cycle
    let currentTick = 0;

    const simulateLoop = () => {
      if (!this.isSyntheticDemo) return;
      t += 0.05;
      currentTick++;

      if (currentTick > maxTicks) {
        this.isSyntheticDemo = false;
        if (this.onVideoEndedCallback) {
          this.onVideoEndedCallback();
        }
        return;
      }

      // Generate synthetic 33-landmark pose array reproducing a squat cycle
      const kneeY = 0.7 + Math.sin(t) * 0.08;
      const hipY = 0.5 + Math.sin(t) * 0.15;

      const syntheticLandmarks = Array.from({ length: 33 }, (_, i) => ({
        x: 0.5 + (i % 2 === 0 ? -0.1 : 0.1),
        y: 0.3 + (i * 0.015),
        z: 0,
        visibility: 0.98
      }));

      // Key landmark mapping for synthetic squat
      syntheticLandmarks[23] = { x: 0.45, y: hipY, z: 0, visibility: 0.99 }; // L Hip
      syntheticLandmarks[24] = { x: 0.55, y: hipY, z: 0, visibility: 0.99 }; // R Hip
      syntheticLandmarks[25] = { x: 0.43, y: kneeY, z: 0, visibility: 0.99 }; // L Knee
      syntheticLandmarks[26] = { x: 0.57, y: kneeY, z: 0, visibility: 0.99 }; // R Knee
      syntheticLandmarks[27] = { x: 0.44, y: 0.9, z: 0, visibility: 0.99 }; // L Ankle
      syntheticLandmarks[28] = { x: 0.56, y: 0.9, z: 0, visibility: 0.99 }; // R Ankle
      syntheticLandmarks[11] = { x: 0.42, y: 0.3, z: 0, visibility: 0.99 }; // L Shoulder
      syntheticLandmarks[12] = { x: 0.58, y: 0.3, z: 0, visibility: 0.99 }; // R Shoulder

      onFrame(syntheticLandmarks);
      setTimeout(simulateLoop, 33); // ~30 FPS
    };

    simulateLoop();
  }

  stopStream() {
    this.isSyntheticDemo = false;
    if (this.currentStream) {
      this.currentStream.getTracks().forEach(track => track.stop());
      this.currentStream = null;
    }
    if (this.video) {
      this.video.pause();
      this.video.srcObject = null;
    }
  }
}
