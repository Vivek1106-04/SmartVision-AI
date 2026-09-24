/**
 * Base Abstract Evaluator Class for Biomechanical Exercise Form Analysis
 */
export class ExerciseBase {
  constructor(name) {
    this.name = name;
    this.repCount = 0;
    this.currentPhase = 'IDLE'; // 'ECCENTRIC' (down), 'CONCENTRIC' (up), 'ISOMETRIC' (hold), 'IDLE'
    this.formScoreHistory = [];
    this.currentFormScore = 100;
    this.faults = [];
  }

  reset() {
    this.repCount = 0;
    this.currentPhase = 'IDLE';
    this.formScoreHistory = [];
    this.currentFormScore = 100;
    this.faults = [];
  }

  getAverageFormScore() {
    if (this.formScoreHistory.length === 0) return 100;
    const sum = this.formScoreHistory.reduce((acc, val) => acc + val, 0);
    return Math.round(sum / this.formScoreHistory.length);
  }

  /**
   * Abstract evaluation method - must be implemented by subclasses
   */
  evaluate(landmarks, cvMetrics) {
    throw new Error('evaluate() must be implemented by subclass');
  }
}
