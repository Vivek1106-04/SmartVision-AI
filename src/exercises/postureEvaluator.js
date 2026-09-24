import { ExerciseBase } from './exerciseBase.js';
import { calculate2DAngle } from '../utils/math.js';

export class PostureEvaluator extends ExerciseBase {
  constructor() {
    super('Ergonomic Posture');
  }

  evaluate(landmarks, cvMetrics) {
    if (!landmarks || landmarks.length < 33) {
      return {
        repCount: 0,
        phase: 'MONITORING',
        formScore: 100,
        feedback: { message: 'Position head and shoulders in camera view', status: 'normal' },
        metrics: { angle1: 0, angle2: 0 }
      };
    }

    const lEar = landmarks[7], rEar = landmarks[8];
    const lShoulder = landmarks[11], rShoulder = landmarks[12];
    const lHip = landmarks[23], rHip = landmarks[24];

    const ear = (lEar.visibility || 1) > (rEar.visibility || 1) ? lEar : rEar;
    const shoulder = (lShoulder.visibility || 1) > (rShoulder.visibility || 1) ? lShoulder : rShoulder;
    const hip = (lHip.visibility || 1) > (rHip.visibility || 1) ? lHip : rHip;

    // Synthetic vertical reference point above shoulder
    const verticalPoint = { x: shoulder.x, y: shoulder.y - 0.5, z: shoulder.z };
    
    // Neck forward inclination (Ear - Shoulder vs Vertical)
    const neckAngle = Math.round(calculate2DAngle(ear, shoulder, verticalPoint));
    // Spine alignment angle (Hip - Shoulder vs Vertical)
    const spineAngle = Math.round(calculate2DAngle(hip, shoulder, verticalPoint));

    // Shoulder Level Slant Check
    const shoulderSlant = Math.abs(lShoulder.y - rShoulder.y) * 100;

    let feedback = { message: 'Optimal Ergonomic Posture Maintained', status: 'good' };
    let score = 100;

    if (neckAngle > 22) {
      feedback = { message: '[WARNING] Forward Head Posture ("Tech Neck") detected. Retract chin.', status: 'error' };
      score -= 25;
    } else if (spineAngle > 20) {
      feedback = { message: '[WARNING] Slouching Detected. Retract shoulders & maintain upright spine.', status: 'warning' };
      score -= 20;
    } else if (shoulderSlant > 4) {
      feedback = { message: '[WARNING] Uneven Shoulder Level. Align left and right shoulders.', status: 'warning' };
      score -= 15;
    }

    this.currentFormScore = Math.max(40, score);
    this.formScoreHistory.push(this.currentFormScore);

    return {
      repCount: 0,
      phase: score > 85 ? 'UPRIGHT POSTURE' : 'POOR ERGONOMICS',
      formScore: this.currentFormScore,
      feedback: feedback,
      metrics: {
        angle1Name: 'Forward Neck Incline',
        angle1: neckAngle,
        angle2Name: 'Spine Slouch Angle',
        angle2: spineAngle
      }
    };
  }
}
