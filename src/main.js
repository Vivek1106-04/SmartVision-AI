import { CameraManager } from './core/cameraManager.js';
import { PoseDetector } from './core/poseDetector.js';
import { CVEngine } from './core/cvEngine.js';
import { DLClassifier } from './core/dlClassifier.js';
import { CanvasHUDRenderer } from './core/renderer.js';
import { AudioEngine } from './core/audioEngine.js';
import { ReportGenerator } from './core/reportGenerator.js';
import { Benchmark } from './core/benchmark.js';

import { SquatEvaluator } from './exercises/squatEvaluator.js';
import { PushupEvaluator } from './exercises/pushupEvaluator.js';
import { BicepCurlEvaluator } from './exercises/bicepCurlEvaluator.js';
import { PostureEvaluator } from './exercises/postureEvaluator.js';

import { TelemetryChart } from './ui/telemetryChart.js';
import { XAIPanel } from './ui/xaiPanel.js';
import { TheoryModal } from './ui/theoryModal.js';

/**
 * Main Application Orchestrator
 */
class SmartVisionApp {
  constructor() {
    // Core Modules
    this.videoElement = document.getElementById('webcamVideo');
    this.canvasElement = document.getElementById('outputCanvas');

    this.cameraManager = new CameraManager(this.videoElement);
    this.poseDetector = new PoseDetector();
    this.cvEngine = new CVEngine();
    this.dlClassifier = new DLClassifier();
    this.renderer = new CanvasHUDRenderer(this.canvasElement);
    this.audioEngine = new AudioEngine();
    this.reportGenerator = new ReportGenerator();

    // UI Controllers
    this.telemetryChart = new TelemetryChart('telemetryChart');
    this.xaiPanel = new XAIPanel();
    this.theoryModal = new TheoryModal();

    // Exercise Evaluators
    this.evaluators = {
      squat: new SquatEvaluator(),
      pushup: new PushupEvaluator(),
      bicepCurl: new BicepCurlEvaluator(),
      posture: new PostureEvaluator()
    };
    this.activeExerciseKey = 'squat';

    // Performance Metrics
    this.frameCount = 0;
    this.fps = 0;
    this.lastFpsTimestamp = performance.now();
    this.lastInferenceTime = 0;
    this.latestXAI = null;
    this.benchmark = Benchmark.fromLocation();

    // DOM Element References
    this.fpsDisplay = document.getElementById('fpsDisplay');
    this.latencyDisplay = document.getElementById('latencyDisplay');
    this.phaseDisplay = document.getElementById('phaseDisplay');
    this.feedbackBanner = document.getElementById('feedbackBanner');
    this.feedbackMessage = document.getElementById('feedbackMessage');
    this.repCountDisplay = document.getElementById('repCountDisplay');
    this.scorePercentDisplay = document.getElementById('scorePercentDisplay');
    this.scoreFill = document.getElementById('scoreFill');
    this.systemStatusText = document.getElementById('systemStatusText');

    this.angle1NameEl = document.getElementById('angle1Name');
    this.angle1ValEl = document.getElementById('angle1Val');
    this.angle2NameEl = document.getElementById('angle2Name');
    this.angle2ValEl = document.getElementById('angle2Val');
    this.velocityValEl = document.getElementById('velocityVal');

    this.initEvents();
    this.initPoseDetector();
  }

  /**
   * Initializes MediaPipe Pose neural network
   */
  async initPoseDetector() {
    try {
      this.systemStatusText.textContent = 'Loading Neural Network...';
      await this.poseDetector.initialize((results) => this.onPoseResults(results));
      this.systemStatusText.textContent = 'Model Ready';
      this.systemStatusText.previousElementSibling.className = 'dot green';
    } catch (err) {
      console.error('MediaPipe initialization failure:', err);
      this.systemStatusText.textContent = 'Model Initialization Error';
    }
  }

  /**
   * Main Frame Callback from MediaPipe Pose Inference
   */
  onPoseResults(results) {
    const startTime = performance.now();
    const rawLandmarks = results.poseLandmarks;

    // 1. Apply Exponential Moving Average (EMA) keypoint noise reduction
    const landmarks = this.cvEngine.smoothLandmarks(rawLandmarks);

    // 2. Compute Canvas Spatial Scaling, Motion Trails & ROI Box
    const videoWidth = this.videoElement.videoWidth || 640;
    const videoHeight = this.videoElement.videoHeight || 480;
    this.renderer.resize(videoWidth, videoHeight);

    this.cvEngine.updateMotionTrail(landmarks, videoWidth, videoHeight);
    const spatialROI = this.cvEngine.getSpatialROI(landmarks, videoWidth, videoHeight);
    const afterCv = performance.now();

    // 3. Evaluate Active Exercise Biomechanics & Rep State Machine
    const activeEvaluator = this.evaluators[this.activeExerciseKey];
    const evalResult = activeEvaluator.evaluate(landmarks, {});
    const afterEvaluate = performance.now();

    // Record telemetry for output.md report
    if (evalResult.metrics && evalResult.metrics.angle1) {
      this.reportGenerator.recordMetrics(evalResult.metrics.angle1, evalResult.metrics.angle2);
    }
    if (evalResult.feedback && ['warning', 'error'].includes(evalResult.feedback.status)) {
      const currentTimeSec = (performance.now() - this.reportGenerator.sessionStart) / 1000.0;
      this.reportGenerator.logFault(currentTimeSec, evalResult.feedback.message);
    }

    // 4. Compute Kinetic Angular Velocity (deg / s)
    const primaryAngle = evalResult.metrics ? evalResult.metrics.angle1 : 0;
    const kineticVelocity = this.cvEngine.calculateKineticVelocity(primaryAngle);

    // 5. Run Deep Learning Temporal Classification & Explainable AI (XAI)
    const beforeClassify = performance.now();
    const xaiData = this.dlClassifier.processFrame(landmarks);
    this.latestXAI = xaiData;
    const afterClassify = performance.now();

    // 6. Render Canvas Overlays with Video Frame Background
    this.renderer.render(this.videoElement, landmarks, evalResult, this.cvEngine, spatialROI);
    const afterRender = performance.now();

    // 7. Trigger Speech Synthesis Audio Coaching
    if (evalResult.feedback && evalResult.feedback.message) {
      this.audioEngine.speak(evalResult.feedback.message);
    }

    // 8. Update UI HUD & Metrics Panel
    this.updateHUD(evalResult, kineticVelocity, xaiData, Boolean(landmarks));

    // Calculate Performance FPS & Inference Latency
    this.frameCount++;
    const now = performance.now();
    this.lastInferenceTime = now - startTime;
    if (now - this.lastFpsTimestamp >= 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.lastFpsTimestamp = now;
    }

    if (this.benchmark) {
      this.benchmark.mark('cv', afterCv - startTime);
      this.benchmark.mark('evaluate', afterEvaluate - afterCv);
      this.benchmark.mark('classify', afterClassify - beforeClassify);
      this.benchmark.mark('render', afterRender - afterClassify);
      this.benchmark.mark('total', now - startTime);
      this.benchmark.endFrame();
    }
  }

  /**
   * Updates all HUD telemetry displays and charts
   */
  updateHUD(evalResult, velocity, xaiData, hasLandmarks) {
    if (this.fpsDisplay) this.fpsDisplay.textContent = this.fps || '--';
    if (this.latencyDisplay) this.latencyDisplay.textContent = `${this.lastInferenceTime.toFixed(1)} ms`;
    if (this.phaseDisplay) this.phaseDisplay.textContent = evalResult.phase || 'IDLE';

    if (this.repCountDisplay) this.repCountDisplay.textContent = evalResult.repCount;

    // Score display logic: show score or searching if no landmarks detected
    const scoreVal = hasLandmarks ? evalResult.formScore : '--';
    if (this.scorePercentDisplay) this.scorePercentDisplay.textContent = hasLandmarks ? `${evalResult.formScore}%` : 'Searching body...';
    if (this.scoreFill) this.scoreFill.style.width = hasLandmarks ? `${evalResult.formScore}%` : '0%';

    // Feedback Alert Banner
    if (this.feedbackMessage && evalResult.feedback) {
      this.feedbackMessage.textContent = evalResult.feedback.message;
      this.feedbackBanner.className = `feedback-toast ${evalResult.feedback.status || 'normal'}`;
    }

    // Metrics Grid
    if (evalResult.metrics) {
      if (this.angle1NameEl) this.angle1NameEl.textContent = evalResult.metrics.angle1Name || 'Primary Angle';
      if (this.angle1ValEl) this.angle1ValEl.textContent = hasLandmarks ? `${evalResult.metrics.angle1 || 0}°` : '--°';
      if (this.angle2NameEl) this.angle2NameEl.textContent = evalResult.metrics.angle2Name || 'Secondary Angle';
      if (this.angle2ValEl) this.angle2ValEl.textContent = hasLandmarks ? `${evalResult.metrics.angle2 || 0}°` : '--°';

      // Update Chart Stream
      if (hasLandmarks && evalResult.metrics.angle1) {
        this.telemetryChart.update(evalResult.metrics.angle1, evalResult.metrics.angle1Name);
      }
    }

    if (this.velocityValEl) this.velocityValEl.textContent = hasLandmarks ? `${velocity.toFixed(1)} °/s` : '0.0 °/s';

    // XAI Panel
    if (hasLandmarks) {
      this.xaiPanel.update(xaiData);
    }
  }

  /**
   * Binds UI Event Listeners
   */
  initEvents() {
    // FIX: Exercise Selector Buttons queried using .selector-btn (matches index.html)
    const exerciseBtns = document.querySelectorAll('.selector-btn');
    exerciseBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const targetBtn = e.currentTarget;
        const exerciseKey = targetBtn.getAttribute('data-exercise');

        if (!exerciseKey || !this.evaluators[exerciseKey]) return;

        exerciseBtns.forEach(b => b.classList.remove('active'));
        targetBtn.classList.add('active');

        this.activeExerciseKey = exerciseKey;
        this.evaluators[exerciseKey].reset();
        this.cvEngine.reset();
        this.dlClassifier.reset();
        this.reportGenerator = new ReportGenerator();

        if (this.phaseDisplay) this.phaseDisplay.textContent = 'IDLE';
        if (this.feedbackMessage) this.feedbackMessage.textContent = `Mode switched to ${this.evaluators[exerciseKey].name}. Start video.`;
      });
    });

    // Handle Video End Completion (Auto-Stop & Generate output.md Report)
    this.cameraManager.onVideoEndedCallback = () => {
      if (this.benchmark) this.benchmark.download();
      this.systemStatusText.textContent = 'Video Completed';
      if (this.phaseDisplay) this.phaseDisplay.textContent = 'COMPLETED';
      if (this.feedbackMessage) this.feedbackMessage.textContent = 'Video processing complete. Report generated.';
      if (this.feedbackBanner) this.feedbackBanner.className = 'feedback-toast good';

      const activeEval = this.evaluators[this.activeExerciseKey];
      this.reportGenerator.downloadReport(
        activeEval.name,
        activeEval.repCount,
        activeEval.getAverageFormScore(),
        this.latestXAI
      );
    };

    // Export Report Button
    const exportReportBtn = document.getElementById('exportReportBtn');
    if (exportReportBtn) {
      exportReportBtn.addEventListener('click', () => {
        const activeEval = this.evaluators[this.activeExerciseKey];
        this.reportGenerator.downloadReport(
          activeEval.name,
          activeEval.repCount,
          activeEval.getAverageFormScore(),
          this.latestXAI
        );
      });
    }

    // Input Controls: Webcam
    const webcamBtn = document.getElementById('webcamBtn');
    if (webcamBtn) {
      webcamBtn.addEventListener('click', async () => {
        try {
          this.systemStatusText.textContent = 'Starting Camera...';
          this.reportGenerator = new ReportGenerator();
          await this.cameraManager.startWebcam();
          this.systemStatusText.textContent = 'Live Webcam Active';
          this.startProcessingLoop();
        } catch (err) {
          alert(err.message);
        }
      });
    }

    // Input Controls: File Upload
    const videoFileInput = document.getElementById('videoFileInput');
    if (videoFileInput) {
      videoFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
          this.systemStatusText.textContent = `Playing ${file.name}...`;
          this.reportGenerator = new ReportGenerator();
          await this.cameraManager.loadVideoFile(file);
          this.startProcessingLoop();
        }
      });
    }

    // Input Controls: Synthetic Demo Video
    const demoVideoBtn = document.getElementById('demoVideoBtn');
    if (demoVideoBtn) {
      demoVideoBtn.addEventListener('click', () => {
        this.systemStatusText.textContent = 'Running Synthetic Simulation...';
        this.reportGenerator = new ReportGenerator();
        this.cameraManager.runSyntheticDemo((landmarks) => {
          this.onPoseResults({ poseLandmarks: landmarks });
        });
      });
    }

    // Input Controls: Stop Stream & Reset
    const stopStreamBtn = document.getElementById('stopStreamBtn');
    if (stopStreamBtn) {
      stopStreamBtn.addEventListener('click', () => {
        this.cameraManager.stopStream();
        this.renderer.clear();
        this.cvEngine.reset();
        this.dlClassifier.reset();
        this.evaluators[this.activeExerciseKey].reset();

        this.systemStatusText.textContent = 'Stream Stopped';
        if (this.phaseDisplay) this.phaseDisplay.textContent = 'STOPPED';
        if (this.feedbackMessage) this.feedbackMessage.textContent = 'Stream stopped. Select an input source to resume.';
        if (this.feedbackBanner) this.feedbackBanner.className = 'feedback-toast';
      });
    }

    // HUD Canvas Overlay Toggles
    const showSkeletonToggle = document.getElementById('showSkeletonToggle');
    if (showSkeletonToggle) {
      showSkeletonToggle.addEventListener('change', (e) => {
        this.renderer.showSkeleton = e.target.checked;
      });
    }

    const showAnglesToggle = document.getElementById('showAnglesToggle');
    if (showAnglesToggle) {
      showAnglesToggle.addEventListener('change', (e) => {
        this.renderer.showAngles = e.target.checked;
      });
    }

    const showMotionTrailsToggle = document.getElementById('showMotionTrailsToggle');
    if (showMotionTrailsToggle) {
      showMotionTrailsToggle.addEventListener('change', (e) => {
        this.renderer.showMotionTrails = e.target.checked;
      });
    }

    const showBoundingBoxToggle = document.getElementById('showBoundingBoxToggle');
    if (showBoundingBoxToggle) {
      showBoundingBoxToggle.addEventListener('change', (e) => {
        this.renderer.showBoundingBox = e.target.checked;
      });
    }

    // Audio Toggle
    const audioToggleBtn = document.getElementById('audioToggleBtn');
    const audioIconOn = document.getElementById('audioIconOn');
    const audioIconOff = document.getElementById('audioIconOff');
    if (audioToggleBtn) {
      audioToggleBtn.addEventListener('click', () => {
        const isEnabled = this.audioEngine.toggleAudio();
        if (isEnabled) {
          audioIconOn.classList.remove('hidden');
          audioIconOff.classList.add('hidden');
        } else {
          audioIconOn.classList.add('hidden');
          audioIconOff.classList.remove('hidden');
        }
      });
    }
  }

  /**
   * Continuous Processing Loop for Video Stream
   */
  startProcessingLoop() {
    const processFrame = async () => {
      if (!this.videoElement.paused && !this.videoElement.ended && !this.cameraManager.isSyntheticDemo) {
        const poseStart = performance.now();
        await this.poseDetector.processFrame(this.videoElement);
        if (this.benchmark) this.benchmark.mark('pose', performance.now() - poseStart);
        requestAnimationFrame(processFrame);
      }
    };
    processFrame();
  }
}

// Initialize Application on DOM Ready
window.addEventListener('DOMContentLoaded', () => {
  window.app = new SmartVisionApp();
});
