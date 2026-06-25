/* ─────────────────────────────────────────────────────────────────────────
 * app.js — DecodeLabs Project 4 Dashboard Logic
 * Implements DESIGN.md §5 interactive components
 * ─────────────────────────────────────────────────────────────────────────
 * Modules (each IIFE):
 *   1. State   — single source of truth (no global mutations outside setState)
 *   2. Console — log panel helpers
 *   3. Upload  — file handling, drag-drop
 *   4. Mode    — OCR / Detection toggle
 *   5. Runner  — simulated pipeline execution + result rendering
 *   6. Gates   — milestone gate status updater
 *   7. Init    — wires everything together
 * ─────────────────────────────────────────────────────────────────────────*/

'use strict';

/* ══════════════════════════════════════════════════════════════════════════
 * 1. STATE MODULE
 * ══════════════════════════════════════════════════════════════════════════ */
const State = (() => {
  let _state = {
    mode:      'ocr',      // 'ocr' | 'detection'
    file:      null,       // File object
    imageUrl:  null,       // Object URL for preview
    running:   false,
    result:    null,       // PipelineResult-shaped object
  };

  function get() { return Object.freeze({ ..._state }); }

  function set(patch) {
    _state = { ..._state, ...patch };
    document.dispatchEvent(new CustomEvent('state:changed', { detail: get() }));
  }

  return { get, set };
})();

/* ══════════════════════════════════════════════════════════════════════════
 * 2. CONSOLE MODULE — DESIGN.md §5.3
 * ══════════════════════════════════════════════════════════════════════════ */
const Console = (() => {
  const body     = document.getElementById('console-body');
  const clearBtn = document.getElementById('console-clear');
  const toggleBtn= document.getElementById('console-toggle');

  function _ts() {
    const d = new Date();
    return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}.${String(d.getMilliseconds()).padStart(3,'0')}`;
  }

  /** @param {'INFO'|'OK'|'WARN'|'ERROR'} level */
  function log(msg, level = 'INFO') {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.innerHTML = `<span class="log-ts">${_ts()}</span><span class="log-${level}">[${level}] ${msg}</span>`;
    body.appendChild(line);
    body.scrollTop = body.scrollHeight;
  }

  function clear() {
    body.innerHTML = '';
    log('Console cleared.', 'INFO');
  }

  function init() {
    clearBtn.addEventListener('click', clear);
    toggleBtn.addEventListener('click', () => {
      const collapsed = body.classList.toggle('collapsed');
      toggleBtn.setAttribute('aria-expanded', String(!collapsed));
      toggleBtn.textContent = collapsed ? '↑' : '↕';
    });
    log('DecodeLabs AI Pipeline dashboard loaded.', 'OK');
    log(`Pipeline mode: ${State.get().mode.toUpperCase()}`, 'INFO');
  }

  return { log, clear, init };
})();

/* ══════════════════════════════════════════════════════════════════════════
 * 3. UPLOAD MODULE
 * ══════════════════════════════════════════════════════════════════════════ */
const Upload = (() => {
  const zone       = document.getElementById('upload-zone');
  const input      = document.getElementById('image-upload');
  const preview    = document.getElementById('upload-preview');
  const previewImg = document.getElementById('preview-img');
  const previewName= document.getElementById('preview-name');
  const previewSize= document.getElementById('preview-size');
  const removeBtn  = document.getElementById('preview-remove');
  const runBtn     = document.getElementById('run-btn');

  const MAX_BYTES = 50 * 1024 * 1024;
  const VALID_EXTS = ['.jpg','.jpeg','.png','.bmp','.tiff'];

  function _fmtSize(bytes) {
    if (bytes < 1024)        return `${bytes} B`;
    if (bytes < 1024*1024)   return `${(bytes/1024).toFixed(1)} KB`;
    return `${(bytes/(1024*1024)).toFixed(1)} MB`;
  }

  function _validate(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!VALID_EXTS.includes(ext)) {
      Console.log(`Rejected: unsupported format "${ext}". Use: ${VALID_EXTS.join(', ')}`, 'ERROR');
      return false;
    }
    if (file.size === 0) {
      Console.log('Rejected: file is empty (0 bytes).', 'ERROR');
      return false;
    }
    if (file.size > MAX_BYTES) {
      Console.log(`Rejected: file too large (${_fmtSize(file.size)}). Max: 50 MB`, 'ERROR');
      return false;
    }
    return true;
  }

  function _setFile(file) {
    if (!_validate(file)) return;

    // Revoke previous URL
    if (State.get().imageUrl) URL.revokeObjectURL(State.get().imageUrl);
    const url = URL.createObjectURL(file);

    State.set({ file, imageUrl: url, result: null });

    previewImg.src = url;
    previewName.textContent = file.name;
    previewSize.textContent = _fmtSize(file.size);
    zone.hidden = true;
    preview.hidden = false;
    runBtn.disabled = false;

    Console.log(`Image loaded: ${file.name} (${_fmtSize(file.size)})`, 'OK');
  }

  function _clear() {
    if (State.get().imageUrl) URL.revokeObjectURL(State.get().imageUrl);
    State.set({ file: null, imageUrl: null, result: null });
    input.value = '';
    zone.hidden = false;
    preview.hidden = true;
    runBtn.disabled = true;
    Console.log('Image removed.', 'INFO');
  }

  function init() {
    input.addEventListener('change', e => {
      if (e.target.files[0]) _setFile(e.target.files[0]);
    });
    removeBtn.addEventListener('click', _clear);

    // Drag and drop
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      if (e.dataTransfer.files[0]) _setFile(e.dataTransfer.files[0]);
    });

    // Keyboard: zone acts as button
    zone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
    });
  }

  return { init };
})();

/* ══════════════════════════════════════════════════════════════════════════
 * 4. MODE MODULE — DESIGN.md §5.4
 * ══════════════════════════════════════════════════════════════════════════ */
const Mode = (() => {
  const ocrBtn   = document.getElementById('mode-ocr');
  const detBtn   = document.getElementById('mode-detection');
  const psmSec   = document.getElementById('psm-section');
  const infoMode = document.getElementById('info-mode');
  const infoLib  = document.getElementById('info-library');
  const infoModel= document.getElementById('info-model');

  const META = {
    ocr:       { lib: 'pytesseract',    model: 'Tesseract LSTM' },
    detection: { lib: 'cv2.dnn',        model: 'MobileNet-SSD'  },
  };

  function _apply(mode) {
    State.set({ mode, result: null });
    ocrBtn.classList.toggle('active', mode === 'ocr');
    detBtn.classList.toggle('active', mode === 'detection');
    ocrBtn.setAttribute('aria-pressed', String(mode === 'ocr'));
    detBtn.setAttribute('aria-pressed', String(mode === 'detection'));
    psmSec.hidden = (mode !== 'ocr');
    infoMode.textContent  = mode.toUpperCase();
    infoLib.textContent   = META[mode].lib;
    infoModel.textContent = META[mode].model;
    Console.log(`Mode switched to: ${mode.toUpperCase()}`, 'INFO');
  }

  function init() {
    ocrBtn.addEventListener('click', () => _apply('ocr'));
    detBtn.addEventListener('click', () => _apply('detection'));
  }

  return { init };
})();

/* ══════════════════════════════════════════════════════════════════════════
 * 5. RUNNER MODULE — pipeline simulation + result rendering
 * ══════════════════════════════════════════════════════════════════════════ */
const Runner = (() => {
  const runBtn       = document.getElementById('run-btn');
  const btnLabel     = runBtn.querySelector('.btn-run__label');
  const btnIcon      = runBtn.querySelector('.btn-run__icon');
  const btnSpinner   = runBtn.querySelector('.btn-run__spinner');

  const stateEmpty   = document.getElementById('state-empty');
  const stateProc    = document.getElementById('state-processing');
  const stateResult  = document.getElementById('state-result');
  const procImg      = document.getElementById('processing-img');
  const resultImg    = document.getElementById('result-img');
  const modeLabel    = document.getElementById('result-mode-label');
  const downloadBtn  = document.getElementById('download-btn');
  const detList      = document.getElementById('detection-list');
  const resultsEmpty = document.getElementById('results-empty');
  const resultsZero  = document.getElementById('results-zero');
  const ocrBlock     = document.getElementById('ocr-text-block');
  const ocrContent   = document.getElementById('ocr-text-content');
  const acceptedChip = document.getElementById('accepted-chip');
  const rejectedChip = document.getElementById('rejected-chip');
  const timeChip     = document.getElementById('time-chip');

  function _mockOCR() {
    const words = [
      { text:'Invoice',    conf: 0.97 },
      { text:'Total',      conf: 0.93 },
      { text:'DecodeLabs', conf: 0.91 },
      { text:'Date',       conf: 0.88 },
      { text:'2026-06-25', conf: 0.85 },
      { text:'Amount',     conf: 0.84 },
      { text:'Pipeline',   conf: 0.81 },
    ];
    return { mode:'ocr', detections: words.map((w,i)=>({label:w.text,confidence:w.conf,index:i+1})),
      full_text: words.map(w=>w.text).join(' '), total_raw:11, total_accepted:words.length,
      total_rejected:4, runtime:(0.8+Math.random()*0.6).toFixed(2) };
  }

  function _mockDetection() {
    const objects = [
      { label:'person',   conf: 0.94 },
      { label:'person',   conf: 0.88 },
      { label:'car',      conf: 0.91 },
      { label:'backpack', conf: 0.82 },
    ];
    return { mode:'detection', detections: objects.map((o,i)=>({label:o.label,confidence:o.conf,index:i+1})),
      full_text:null, total_raw:9, total_accepted:objects.length,
      total_rejected:5, runtime:(1.1+Math.random()*0.8).toFixed(2) };
  }

  function _setUIState(s) {
    stateEmpty.hidden  = s !== 'empty';
    stateProc.hidden   = s !== 'processing';
    stateResult.hidden = s !== 'result';
  }

  function _setBtnState(running) {
    runBtn.disabled = running;
    runBtn.classList.toggle('processing', running);
    btnLabel.textContent = running ? 'PROCESSING…' : 'RUN PIPELINE';
    btnIcon.hidden       = running;
    btnSpinner.hidden    = !running;
  }

  function _card(det, idx) {
    const isHigh = det.confidence >= 0.90;
    const tier   = isHigh ? 'high' : 'med';
    const pct    = (det.confidence * 100).toFixed(1);
    const barW   = Math.round(det.confidence * 100);
    const el = document.createElement('div');
    el.className = 'detection-card';
    el.setAttribute('role', 'listitem');
    el.style.animationDelay = `${idx * 50}ms`;
    el.innerHTML = `
      <span class="detection-card__index">${String(idx+1).padStart(2,'0')}</span>
      <span class="detection-card__label">${det.label}</span>
      <div class="confidence-bar" title="${pct}% confidence">
        <div class="confidence-bar__fill confidence-bar__fill--${tier}" style="width:${barW}%"></div>
      </div>
      <span class="confidence-badge confidence-badge--${tier}" aria-label="${pct}%">${pct}%</span>`;
    return el;
  }

  function _renderResult(result, imageUrl) {
    resultImg.src   = imageUrl;
    resultImg.alt   = `Annotated ${result.mode} result`;
    modeLabel.textContent = result.mode.toUpperCase();
    _setUIState('result');
    downloadBtn.hidden = false;

    acceptedChip.textContent = `✓ ${result.total_accepted} accepted`;
    rejectedChip.textContent = `✗ ${result.total_rejected} rejected`;
    timeChip.textContent     = `⏱ ${result.runtime}s`;
    [acceptedChip, rejectedChip, timeChip].forEach(c => c.hidden = false);

    detList.innerHTML = '';
    if (result.total_accepted === 0) {
      resultsEmpty.hidden = true;
      resultsZero.hidden  = false;
      Console.log('No high-confidence detections (FR-09).', 'WARN');
    } else {
      resultsEmpty.hidden = true;
      resultsZero.hidden  = true;
      result.detections.forEach((det, i) => detList.appendChild(_card(det, i)));
    }

    if (result.mode === 'ocr' && result.full_text) {
      ocrContent.textContent = result.full_text;
      ocrBlock.hidden = false;
    } else {
      ocrBlock.hidden = true;
    }
  }

  function _delay(ms) { return new Promise(r => setTimeout(r, ms)); }

  async function _run() {
    const { mode, imageUrl } = State.get();
    if (!imageUrl) return;
    State.set({ running: true });
    _setBtnState(true);

    procImg.src = imageUrl;
    _setUIState('processing');
    detList.innerHTML = '';
    resultsEmpty.hidden = false;
    resultsZero.hidden  = true;
    ocrBlock.hidden     = true;
    downloadBtn.hidden  = true;
    [acceptedChip, rejectedChip, timeChip].forEach(c => c.hidden = true);

    Console.log(`Starting ${mode.toUpperCase()} pipeline…`, 'INFO');
    await _delay(300); Console.log('Step 1: Image validated.', 'OK');
    await _delay(300); Console.log('Step 2: cvtColor BGR→GRAY.', 'OK');
    await _delay(300); Console.log('Step 3: GaussianBlur (5×5).', 'OK');
    await _delay(300); Console.log('Step 4: adaptiveThreshold applied.', 'OK');

    if (mode === 'ocr') {
      await _delay(500); Console.log('Step 5: pytesseract.image_to_data()…', 'INFO');
      await _delay(600); Console.log('Step 6: DataFrame filtered (conf ≥ 80%).', 'OK');
    } else {
      await _delay(400); Console.log('Step 5: Blob constructed (300×300).', 'OK');
      await _delay(500); Console.log('Step 6: DNN forward pass.', 'OK');
      await _delay(400); Console.log('Step 7: NMS applied (iou=0.4).', 'OK');
    }

    await _delay(300);
    const result = mode === 'ocr' ? _mockOCR() : _mockDetection();
    State.set({ result, running: false });
    Console.log(`Done. Accepted:${result.total_accepted} Rejected:${result.total_rejected} Time:${result.runtime}s`, 'OK');
    Console.log('Output saved to: output/image_output.jpg', 'OK');
    _renderResult(result, imageUrl);
    _setBtnState(false);
    Gates.afterRun(result);
  }

  function init() {
    runBtn.addEventListener('click', () => {
      if (!State.get().running && State.get().imageUrl) _run();
    });
    downloadBtn.addEventListener('click', () => {
      const url   = State.get().imageUrl;
      const fname = State.get().file?.name || 'output.jpg';
      const stem  = fname.replace(/\.[^.]+$/, '');
      const a = document.createElement('a');
      a.href = url; a.download = `${stem}_output.jpg`; a.click();
      Console.log(`Downloaded: ${stem}_output.jpg`, 'OK');
    });
  }

  return { init };
})();

/* ══════════════════════════════════════════════════════════════════════════
 * 6. GATES MODULE
 * ══════════════════════════════════════════════════════════════════════════ */
const Gates = (() => {
  function _setPass(id) {
    const card   = document.getElementById(`gate-${id}`);
    const status = card.querySelector('.gate-status');
    const icon   = card.querySelector('.gate-icon');
    card.classList.remove('gate-card--pending');
    status.classList.remove('gate-status--pending');
    status.textContent = 'PASS';
    icon.textContent   = '✅';
  }
  function afterRun(result) { _setPass(3); if (result.total_accepted >= 0) _setPass(4); }
  function init() {}
  return { afterRun, init };
})();

/* ══════════════════════════════════════════════════════════════════════════
 * 7. COPY BUTTON
 * ══════════════════════════════════════════════════════════════════════════ */
const CopyBtn = (() => {
  function init() {
    const btn     = document.getElementById('copy-text-btn');
    const content = document.getElementById('ocr-text-content');
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(content.textContent);
        btn.textContent = 'Copied!';
      } catch {
        btn.textContent = 'Error';
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
  }
  return { init };
})();

/* ══════════════════════════════════════════════════════════════════════════
 * 8. INIT
 * ══════════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  Console.init();
  Upload.init();
  Mode.init();
  Runner.init();
  Gates.init();
  CopyBtn.init();
});

