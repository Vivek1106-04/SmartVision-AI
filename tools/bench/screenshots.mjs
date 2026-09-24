/**
 * Capture HUD screenshots from the running application.
 *
 * Drives the real app in a real browser: loads a held-out clip through the
 * normal upload path, lets MediaPipe, the evaluators and the classifier run,
 * and captures the HUD at chosen moments. Nothing is staged or mocked, so the
 * figures in the report show what a user actually sees.
 *
 * Usage: npm run dev, then `node tools/bench/screenshots.mjs`
 * Output: results/screenshots/*.png
 */
import { mkdirSync, copyFileSync, existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import puppeteer from 'puppeteer';

import { PROJECT_ROOT } from '../../tests/js/helpers.mjs';

const APP_URL = process.env.APP_URL || 'http://localhost:5173/';
const SAMPLES_DIR = join(PROJECT_ROOT, 'public', 'samples');
const OUTPUT_DIR = join(PROJECT_ROOT, 'results', 'screenshots');
const VIEWPORT = { width: 1600, height: 1000 };

// Held-out clips, one per exercise. `until` decides when the HUD is worth
// capturing: a fixed delay lands on whatever frame happens to be showing,
// which is often one with no pose detected at all.
const CAPTURES = [
  { name: 'squat', exercise: 'squat', clip: 'squat_20.mp4', until: 'firstRep',
    caption: 'Squat mode at the completion of a counted repetition' },
  { name: 'pushup', exercise: 'pushup', clip: 'push-up_54.mp4', until: 'firstRep',
    caption: 'Pushup mode at the completion of a counted repetition' },
  { name: 'bicepcurl', exercise: 'bicepCurl', clip: 'barbell biceps curl_19.mp4',
    until: 'firstRep',
    caption: 'Bicep curl mode at the completion of a counted repetition' },
  { name: 'failure-undercount', exercise: 'bicepCurl', clip: 'barbell biceps curl_60.mp4',
    until: 'videoEnd',
    caption: 'Failure case: six repetitions performed, the angle gates register far fewer' }
];

const CAPTURE_TIMEOUT_MS = 90000;
const POLL_INTERVAL_MS = 100;

const RAW_DIRS = { squat: 'squat', pushup: 'pushup', bicepCurl: 'bicepCurl' };

/** Copy the clips into public/ so the page can fetch them like any asset. */
function stageClips() {
  mkdirSync(SAMPLES_DIR, { recursive: true });
  for (const capture of CAPTURES) {
    const source = join(PROJECT_ROOT, 'dataset', 'raw', RAW_DIRS[capture.exercise], capture.clip);
    if (!existsSync(source)) throw new Error(`Clip not found: ${source}`);
    copyFileSync(source, join(SAMPLES_DIR, capture.clip));
  }
  console.log(`Staged ${CAPTURES.length} clips in public/samples/`);
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

/** Read the live application state without disturbing it. */
function readState() {
  const video = document.getElementById('webcamVideo');
  return {
    reps: Number(document.getElementById('repCountDisplay').textContent) || 0,
    phase: document.getElementById('phaseDisplay').textContent,
    fps: document.getElementById('fpsDisplay').textContent,
    attribution: document.getElementById('tagPrimary').textContent,
    predicted: window.app.latestXAI ? window.app.latestXAI.predictedExercise : null,
    ended: video.ended,
    currentTime: Number(video.currentTime.toFixed(2))
  };
}

/**
 * Wait for a frame worth photographing: a pose is being tracked, the
 * classifier has a prediction, and the requested milestone has been reached.
 */
async function waitForMoment(page, capture) {
  const deadline = Date.now() + CAPTURE_TIMEOUT_MS;

  while (Date.now() < deadline) {
    const state = await page.evaluate(readState);
    const hasPrediction = state.predicted !== null;

    if (capture.until === 'firstRep' && state.reps >= 1 && hasPrediction) return state;
    if (capture.until === 'videoEnd' && state.ended) return state;
    if (state.ended) return state;

    await sleep(POLL_INTERVAL_MS);
  }
  return page.evaluate(readState);
}

/** Select an exercise mode and feed a clip through the app's own upload path. */
async function runClip(page, capture) {
  await page.evaluate(exercise => {
    document.querySelector(`.selector-btn[data-exercise="${exercise}"]`).click();
  }, capture.exercise);

  await page.evaluate(async clip => {
    const response = await fetch(`/samples/${clip}`);
    const file = new File([await response.blob()], clip, { type: 'video/mp4' });
    const transfer = new DataTransfer();
    transfer.items.add(file);

    const input = document.getElementById('videoFileInput');
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, capture.clip);

  return waitForMoment(page, capture);
}

/** Open the app in a new page and wait until the pose model is ready. */
async function openApp(browser) {
  const page = await browser.newPage();
  await page.setViewport(VIEWPORT);
  page.on('console', message => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      console.log(`  [page error] ${message.text()}`);
    }
  });

  await page.goto(APP_URL, { waitUntil: 'networkidle2', timeout: 60000 });
  await page.waitForFunction(
    () => document.getElementById('systemStatusText').textContent === 'Model Ready',
    { timeout: 90000 }
  );
  await page.waitForFunction(() => window.app.dlClassifier.modelLoaded, { timeout: 30000 });
  return page;
}

async function main() {
  stageClips();
  mkdirSync(OUTPUT_DIR, { recursive: true });

  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--autoplay-policy=no-user-gesture-required',
      '--use-gl=angle',
      '--enable-unsafe-swiftshader',
      '--use-fake-ui-for-media-stream'
    ]
  });

  try {
    const manifest = [];
    for (const capture of CAPTURES) {
      console.log(`Capturing ${capture.name} (${capture.clip})`);

      // A fresh page per clip. Loading a second video into a live page leaves
      // MediaPipe tracking the previous stream, so the HUD stops updating.
      const page = await openApp(browser);
      const state = await runClip(page, capture);

      const path = join(OUTPUT_DIR, `${capture.name}.png`);
      await page.screenshot({ path });
      console.log(`  t=${state.currentTime}s reps=${state.reps} phase=${state.phase} ` +
                  `fps=${state.fps} predicted=${state.predicted} attribution="${state.attribution}"`);
      console.log(`  ${path}\n`);
      manifest.push({ ...capture, ...state });
      await page.close();
    }

    // A caption file so each figure can be cited without guessing what it shows.
    const columns = ['name', 'clip', 'exercise', 'caption', 'reps', 'phase',
                     'predicted', 'attribution', 'currentTime'];
    writeFileSync(
      join(OUTPUT_DIR, 'manifest.csv'),
      [columns.join(','),
       ...manifest.map(entry => columns.map(c => JSON.stringify(String(entry[c]))).join(','))
      ].join('\n') + '\n'
    );

    console.log('Captured:');
    for (const entry of manifest) {
      console.log(`  ${entry.name}.png - ${entry.caption}`);
    }
    console.log(`  manifest.csv`);
  } finally {
    await browser.close();
  }
}

main();
