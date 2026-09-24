/**
 * Explainable AI (XAI) UI Panel Controller
 * Displays LSTM model confidence, predicted exercise class probabilities,
 * and per-exercise feature saliency attribution.
 */
export class XAIPanel {
  constructor() {
    this.confidenceValEl = document.getElementById('keypointConfidenceVal');
    this.confidenceFillEl = document.getElementById('confidenceFill');
    this.tagPrimaryEl = document.getElementById('tagPrimary');
    this.tagSecondaryEl = document.getElementById('tagSecondary');
    this.tagTertiaryEl = document.getElementById('tagTertiary');
  }

  update(xaiData) {
    if (!xaiData) return;

    // Overall model confidence
    const conf = xaiData.overallConfidence.toFixed(1);
    if (this.confidenceValEl) this.confidenceValEl.textContent = `${conf}%`;
    if (this.confidenceFillEl) this.confidenceFillEl.style.width = `${conf}%`;

    // Measured feature attribution for the predicted class. Until the model
    // produces a prediction there is nothing measured to show, so the tags say
    // so rather than displaying a stale or invented ranking.
    const features = xaiData.featureImportance || [];
    const tags = [this.tagPrimaryEl, this.tagSecondaryEl, this.tagTertiaryEl];

    tags.forEach((tagEl, i) => {
      if (!tagEl) return;
      const feature = features[i];
      tagEl.textContent = feature
        ? `${feature.name} (${(feature.weight * 100).toFixed(0)}%)`
        : 'Awaiting prediction';
    });
  }
}
