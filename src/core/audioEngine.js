/**
 * Real-Time Audio Coaching Engine
 * Uses Web Speech API (speechSynthesis) to deliver clean, clear audio instructions
 * on exercise technique and form corrections without interrupting video stream execution.
 */
export class AudioEngine {
  constructor() {
    this.enabled = true;
    this.synth = window.speechSynthesis || null;
    this.lastSpokenTime = 0;
    this.cooldownMs = 3500; // Cooldown to prevent speech overlap
    this.lastSpokenMessage = '';
  }

  speak(message) {
    if (!this.enabled || !this.synth || !message) return;

    const now = performance.now();
    // Clean string by stripping brackets or technical tags
    const cleanMessage = message.replace(/\[.*?\]/g, '').trim();

    if (now - this.lastSpokenTime > this.cooldownMs && cleanMessage !== this.lastSpokenMessage) {
      this.synth.cancel(); // Clear queued speech utterances
      const utterance = new SpeechSynthesisUtterance(cleanMessage);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 0.9;
      this.synth.speak(utterance);

      this.lastSpokenTime = now;
      this.lastSpokenMessage = cleanMessage;
    }
  }

  toggleAudio() {
    this.enabled = !this.enabled;
    if (!this.enabled && this.synth) {
      this.synth.cancel();
    }
    return this.enabled;
  }
}
