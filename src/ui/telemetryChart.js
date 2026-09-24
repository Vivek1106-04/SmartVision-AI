/**
 * Real-Time Telemetry Stream Chart (Chart.js)
 * Visualizes kinematic joint angles over time for biomechanical analysis.
 */
export class TelemetryChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.chart = null;
    this.maxDataPoints = 40;
    this.labels = Array.from({ length: this.maxDataPoints }, (_, i) => i);
    this.dataStream = Array(this.maxDataPoints).fill(0);
    this.initChart();
  }

  initChart() {
    if (!this.canvas || !window.Chart) return;

    const ctx = this.canvas.getContext('2d');
    this.chart = new window.Chart(ctx, {
      type: 'line',
      data: {
        labels: this.labels,
        datasets: [{
          label: 'Primary Joint Angle (°)',
          data: this.dataStream,
          borderColor: '#06B6D4',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
          fill: true,
          backgroundColor: 'rgba(6, 182, 212, 0.1)'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { display: false },
          y: {
            min: 0,
            max: 180,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9CA3AF', font: { size: 10 } }
          }
        }
      }
    });
  }

  update(angleValue, angleName = 'Primary Joint Angle') {
    if (!this.chart) return;

    this.dataStream.push(angleValue);
    if (this.dataStream.length > this.maxDataPoints) {
      this.dataStream.shift();
    }

    this.chart.data.datasets[0].data = this.dataStream;
    this.chart.data.datasets[0].label = `${angleName} (°)`;
    this.chart.update('none'); // Update without heavy animation
  }
}
