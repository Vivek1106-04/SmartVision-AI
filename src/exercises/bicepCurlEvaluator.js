import { ExerciseBase } from './exerciseBase.js';
import { calculate2DAngle } from '../utils/math.js';

export class BicepCurlEvaluator extends ExerciseBase {
  constructor() {
    super('Bicep Curl');
    this.stage = 'DOWN';
    this.minElbowAngle = 180;
    this.maxElbowAngle = 0;
  }

  evaluate(landmarks, cvMetrics) {
    if (!landmarks || landmarks.length < 33) {
      return {
        repCount: this.repCount,
        phase: 'SEARCHING',
        formScore: 100,
        feedback: { message: 'Position upper body in frame', status: 'normal' },
        metrics: { angle1: 0, angle2: 0 }
      };
    }

    const lShoulder = landmarks[11], rShoulder = landmarks[12];
    const lElbow = landmarks[13], rElbow = landmarks[14];
    const lWrist = landmarks[15], rWrist = landmarks[16];
    const lHip = landmarks[23], rHip = landmarks[24];

    const leftVis = (lElbow.visibility || 1) + (lWrist.visibility || 1);
    const rightVis = (rElbow.visibility || 1) + (rWrist.visibility || 1);
    const useLeft = leftVis >= rightVis;

    const shoulder = useLeft ? lShoulder : rShoulder;
    const elbow = useLeft ? lElbow : rElbow;
    const wrist = useLeft ? lWrist : rWrist;
    const hip = useLeft ? lHip : rHip;

    // Biomechanical Angles
    const elbowAngle = Math.round(calculate2DAngle(shoulder, elbow, wrist));
    const shoulderStabilityAngle = Math.round(calculate2DAngle(hip, shoulder, elbow));

    let feedback = { message: 'Pin elbow to torso and flex bicep', status: 'normal' };
    let frameDeduction = 0;

    if (elbowAngle < this.minElbowAngle) this.minElbowAngle = elbowAngle;
    if (elbowAngle > this.maxElbowAngle) this.maxElbowAngle = elbowAngle;

    // Check Momentum / Swinging Elbow Fault
    if (shoulderStabilityAngle > 30) {
      feedback = { message: '[WARNING] Shoulder momentum detected. Keep elbow stationary.', status: 'warning' };
      frameDeduction += 15;
    }

    // State Machine
    if (elbowAngle < 50 && this.stage === 'DOWN') {
      this.currentPhase = 'PEAK CONTRACTION';
      this.stage = 'UP';
      feedback = { message: '[OPTIMAL] Peak bicep contraction achieved.', status: 'good' };
    }

    if (elbowAngle > 150 && this.stage === 'UP') {
      this.currentPhase = 'FULL EXTENSION';
      this.stage = 'DOWN';

      if (this.minElbowAngle > 60) {
        feedback = { message: '[WARNING] Incomplete ROM. Flex arm completely.', status: 'warning' };
        this.currentFormScore = Math.max(50, this.currentFormScore - 15);
      } else {
        this.repCount++;
        feedback = { message: `[SUCCESS] Bicep Curl Rep ${this.repCount} Completed.`, status: 'good' };
      }

      this.formScoreHistory.push(this.currentFormScore);
      this.minElbowAngle = 180;
    }

    if (elbowAngle >= 140 && this.stage === 'DOWN') {
      this.currentPhase = 'ARM EXTENDED';
    }

    return {
      repCount: this.repCount,
      phase: this.currentPhase,
      formScore: Math.max(60, 100 - frameDeduction),
      feedback: feedback,
      metrics: {
        angle1Name: 'Elbow Curl Angle',
        angle1: elbowAngle,
        angle2Name: 'Shoulder Stability',
        angle2: shoulderStabilityAngle
      }
    };
  }
}
