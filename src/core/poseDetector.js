/**
 * MediaPipe Pose Detector Module
 * Configures the BlazePose deep learning model and manages frame inference.
 */
export class PoseDetector {
  constructor() {
    this.pose = null;
    this.isInitialized = false;
    this.onResultsCallback = null;
  }

  async initialize(onResults) {
    this.onResultsCallback = onResults;

    const PoseClass = window.Pose || (await import('@mediapipe/pose')).Pose;

    this.pose = new PoseClass({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
      }
    });

    this.pose.setOptions({
      modelComplexity: 1, // 0: Lite, 1: Full (Recommended), 2: Heavy
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: false,
      minDetectionConfidence: 0.65,
      minTrackingConfidence: 0.65
    });

    this.pose.onResults((results) => {
      if (this.onResultsCallback) {
        this.onResultsCallback(results);
      }
    });

    await this.pose.initialize();
    this.isInitialized = true;
  }

  async processFrame(imageSource) {
    if (this.isInitialized && this.pose && imageSource) {
      await this.pose.send({ image: imageSource });
    }
  }
}
