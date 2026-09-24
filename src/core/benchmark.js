/**
 * Per-stage performance recorder.
 *
 * Enabled by loading the app with ?bench=1. It times each stage of the frame
 * pipeline on the machine actually running it, which is the only place an
 * honest FPS number can come from - a headless or software-rendered browser
 * measures the renderer, not the product.
 *
 * Results are downloaded as a CSV that drops straight into results/.
 */
const STAGES = ['pose', 'cv', 'evaluate', 'classify', 'render', 'total'];

export class Benchmark {
  /** Reads the ?bench=1 flag; returns null when benchmarking is off. */
  static fromLocation(search = window.location.search) {
    return new URLSearchParams(search).get('bench') === '1' ? new Benchmark() : null;
  }

  constructor() {
    this.rows = [];
    this.marks = {};
    this.lastFrameTimestamp = null;
    console.log('[Benchmark] recording enabled - press B to download the CSV');
    window.addEventListener('keydown', (event) => {
      if (event.key === 'b' || event.key === 'B') this.download();
    });
  }

  /** Record how long one stage took, in milliseconds. */
  mark(stage, milliseconds) {
    this.marks[stage] = milliseconds;
  }

  /** Close off the current frame and start the next. */
  endFrame() {
    const now = performance.now();
    const framePeriod = this.lastFrameTimestamp === null ? null : now - this.lastFrameTimestamp;
    this.lastFrameTimestamp = now;

    this.rows.push({
      frame: this.rows.length,
      frame_period_ms: framePeriod === null ? '' : framePeriod.toFixed(3),
      instantaneous_fps: framePeriod ? (1000 / framePeriod).toFixed(2) : '',
      ...Object.fromEntries(STAGES.map(stage => [
        `${stage}_ms`, this.marks[stage] === undefined ? '' : this.marks[stage].toFixed(3)
      ]))
    });
    this.marks = {};
  }

  /** Save the recorded frames as results/latency_browser.csv. */
  download(filename = 'latency_browser.csv') {
    if (this.rows.length === 0) {
      console.warn('[Benchmark] nothing recorded yet');
      return;
    }

    const columns = Object.keys(this.rows[0]);
    const csv = [
      `# userAgent: ${navigator.userAgent}`,
      `# hardwareConcurrency: ${navigator.hardwareConcurrency}`,
      `# devicePixelRatio: ${window.devicePixelRatio}`,
      columns.join(','),
      ...this.rows.map(row => columns.map(column => row[column]).join(','))
    ].join('\n');

    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
    console.log(`[Benchmark] wrote ${this.rows.length} frames to ${filename}`);
  }
}
