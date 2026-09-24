import { ExerciseBase } from './exerciseBase.js';
import { calculate2DAngle } from '../utils/math.js';

export class PushupEvaluator extends ExerciseBase {
  constructor() {
    super('Pushup');
    this.stage = 'UP';
    this.minElbowAngle = 180;
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
    const lAnkle = landmarks[27], rAnkle = landmarks[28];

    const leftVis = (lElbow.visibility || 1) + (lWrist.visibility || 1);
    const rightVis = (rElbow.visibility || 1) + (rWrist.visibility || 1);
    const useLeft = leftVis >= rightVis;

    const shoulder = useLeft ? lShoulder : rShoulder;
    const elbow = useLeft ? lElbow : rElbow;
    const wrist = useLeft ? lWrist : rWrist;
    const hip = useLeft ? lHip : rHip;
    const ankle = useLeft ? lAnkle : rAnkle;

    const elbowAngle = Math.round(calculate2DAngle(shoulder, elbow, wrist));
    const bodyPlankAngle = Math.round(calculate2DAngle(shoulder, hip, ankle));

    let feedback = { message: 'Maintain core contraction & flex elbows', status: 'normal' };
    let frameDeduction = 0;

    if (elbowAngle < this.minElbowAngle) {
      this.minElbowAngle = elbowAngle;
    }

    // Check Plank Line Alignment (Sagging or Piking Hips)
    if (bodyPlankAngle < 155 || bodyPlankAngle > 195) {
      feedback = { message: '[WARNING] Spinal alignment deviation. Maintain rigid plank position.', status: 'warning' };
      frameDeduction += 15;
    }

    // State Machine
    if (elbowAngle < 95) {
      this.currentPhase = 'ECCENTRIC (BOTTOM)';
      if (this.stage === 'UP') {
        this.stage = 'DOWN';
      }
      if (elbowAngle <= 80) {
        feedback = { message: '[OPTIMAL] Full chest depth achieved.', status: 'good' };
      }
    }

    if (elbowAngle > 155 && this.stage === 'DOWN') {
      this.currentPhase = 'CONCENTRIC (TOP)';
      this.stage = 'UP';

      if (this.minElbowAngle > 100) {
        feedback = { message: '[WARNING] Incomplete depth. Flex elbows to 90 degrees.', status: 'warning' };
        this.currentFormScore = Math.max(50, this.currentFormScore - 15);
      } else {
        this.repCount++;
        feedback = { message: `[SUCCESS] Pushup Rep ${this.repCount} Completed.`, status: 'good' };
      }

      this.formScoreHistory.push(this.currentFormScore);
      this.minElbowAngle = 180;
    }

    if (elbowAngle >= 150 && this.stage === 'UP') {
      this.currentPhase = 'TOP PLANK';
    }

    return {
      repCount: this.repCount,
      phase: this.currentPhase,
      formScore: Math.max(60, 100 - frameDeduction),
      feedback: feedback,
      metrics: {
        angle1Name: 'Elbow Flexion',
        angle1: elbowAngle,
        angle2Name: 'Plank Spine Line',
        angle2: bodyPlankAngle
      }
    };
  }
}
