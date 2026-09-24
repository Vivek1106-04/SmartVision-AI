import { ExerciseBase } from './exerciseBase.js';
import { calculate2DAngle } from '../utils/math.js';

export class SquatEvaluator extends ExerciseBase {
  constructor() {
    super('Squat');
    this.stage = 'UP';
    this.minKneeAngle = 180;
  }

  evaluate(landmarks, cvMetrics) {
    if (!landmarks || landmarks.length < 33) {
      return {
        repCount: this.repCount,
        phase: 'SEARCHING',
        formScore: 100,
        feedback: { message: 'Position entire body in frame', status: 'normal' },
        metrics: { angle1: 0, angle2: 0 }
      };
    }

    // MediaPipe Landmarks: 23=L_Hip, 24=R_Hip, 25=L_Knee, 26=R_Knee, 27=L_Ankle, 28=R_Ankle, 11=L_Shoulder, 12=R_Shoulder
    const lHip = landmarks[23], rHip = landmarks[24];
    const lKnee = landmarks[25], rKnee = landmarks[26];
    const lAnkle = landmarks[27], rAnkle = landmarks[28];
    const lShoulder = landmarks[11], rShoulder = landmarks[12];

    // Determine primary visible side
    const leftVis = (lHip.visibility || 1) + (lKnee.visibility || 1) + (lAnkle.visibility || 1);
    const rightVis = (rHip.visibility || 1) + (rKnee.visibility || 1) + (rAnkle.visibility || 1);

    const useLeft = leftVis >= rightVis;
    const hip = useLeft ? lHip : rHip;
    const knee = useLeft ? lKnee : rKnee;
    const ankle = useLeft ? lAnkle : rAnkle;
    const shoulder = useLeft ? lShoulder : rShoulder;

    // Calculate Biomechanical Angles
    const kneeAngle = Math.round(calculate2DAngle(hip, knee, ankle));
    const hipHingeAngle = Math.round(calculate2DAngle(shoulder, hip, knee));

    let feedback = { message: 'Maintain good posture & lower hips', status: 'normal' };
    let frameDeduction = 0;

    // Track minimum knee angle during squat descent
    if (kneeAngle < this.minKneeAngle) {
      this.minKneeAngle = kneeAngle;
    }

    // Form Check 1: Excessive Forward Lean (Spinal flexion fault)
    if (hipHingeAngle < 70 && this.stage === 'DOWN') {
      feedback = { message: '[WARNING] Torso leaning too far forward. Elevate chest.', status: 'warning' };
      frameDeduction += 15;
    }

    // Form Check 2: Knee Valgus (Knees caving inward)
    const kneeDistance = Math.abs(lKnee.x - rKnee.x);
    const ankleDistance = Math.abs(lAnkle.x - rAnkle.x);
    if (kneeDistance < ankleDistance * 0.75 && kneeAngle < 120) {
      feedback = { message: '[ERROR] Knee Valgus detected. Track knees over toes.', status: 'error' };
      frameDeduction += 20;
    }

    // State Machine & Rep Counter
    if (kneeAngle < 100) {
      this.currentPhase = 'ECCENTRIC (BOTTOM)';
      if (this.stage === 'UP') {
        this.stage = 'DOWN';
      }
      if (kneeAngle <= 90) {
        feedback = { message: '[OPTIMAL] Parallel depth achieved.', status: 'good' };
      }
    }

    if (kneeAngle > 160 && this.stage === 'DOWN') {
      this.currentPhase = 'CONCENTRIC (TOP)';
      this.stage = 'UP';
      
      // Check depth on rep completion
      if (this.minKneeAngle > 105) {
        feedback = { message: '[WARNING] Partial range of motion. Squat below 90 degrees.', status: 'warning' };
        this.currentFormScore = Math.max(50, this.currentFormScore - 15);
      } else {
        this.repCount++;
        feedback = { message: `[SUCCESS] Rep ${this.repCount} Completed with optimal depth.`, status: 'good' };
      }
      
      this.formScoreHistory.push(this.currentFormScore);
      this.minKneeAngle = 180;
    }

    if (kneeAngle >= 150 && this.stage === 'UP') {
      this.currentPhase = 'STANDING';
    }

    return {
      repCount: this.repCount,
      phase: this.currentPhase,
      formScore: Math.max(60, 100 - frameDeduction),
      feedback: feedback,
      metrics: {
        angle1Name: 'Knee Flexion',
        angle1: kneeAngle,
        angle2Name: 'Hip Hinge',
        angle2: hipHingeAngle
      }
    };
  }
}
