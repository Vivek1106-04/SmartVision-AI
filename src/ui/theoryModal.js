/**
 * Academic Theory & Specifications Modal Controller
 */
export class TheoryModal {
  constructor() {
    this.modalEl = document.getElementById('theoryModal');
    this.openBtn = document.getElementById('theoryBtn');
    this.closeBtn = document.getElementById('closeTheoryBtn');
    this.initEvents();
  }

  initEvents() {
    if (this.openBtn) {
      this.openBtn.addEventListener('click', () => this.open());
    }
    if (this.closeBtn) {
      this.closeBtn.addEventListener('click', () => this.close());
    }
    if (this.modalEl) {
      this.modalEl.addEventListener('click', (e) => {
        if (e.target === this.modalEl) this.close();
      });
    }
  }

  open() {
    if (this.modalEl) this.modalEl.classList.remove('hidden');
  }

  close() {
    if (this.modalEl) this.modalEl.classList.add('hidden');
  }
}
