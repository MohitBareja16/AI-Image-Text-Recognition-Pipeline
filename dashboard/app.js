/* ─────────────────────────────────────────────────────────────────────────
 * app.js — DecodeLabs Project 4 Dashboard Logic (t3.chat styling)
 * ───────────────────────────────────────────────────────────────────────── */

'use strict';

document.addEventListener('DOMContentLoaded', () => {
  // ── State Management ──────────────────────────────────────────────────
  const State = {
    mode: 'ocr',               // 'ocr' | 'detection'
    file: null,                // File object if custom upload
    imageUrl: null,            // Image URL (object URL or asset path)
    imageName: null,           // Display name of selected image
    running: false,
    sessionCount: 0,
  };

  // ── DOM References ────────────────────────────────────────────────────
  const chatThread     = document.getElementById('chat-thread');
  const attachBtn      = document.getElementById('attach-btn');
  const imageUpload    = document.getElementById('image-upload');
  const runBtn         = document.getElementById('run-btn');
  const runBtnSpinner  = runBtn.querySelector('.btn-spinner');
  const inputStatusText= document.getElementById('input-status-text');
  const modeOcr        = document.getElementById('mode-ocr');
  const modeDetection  = document.getElementById('mode-detection');
  const psmSection     = document.getElementById('psm-section');
  const psmSelect      = document.getElementById('psm-select');
  const downloadBtn    = document.getElementById('download-btn');
  const newChatBtn     = document.getElementById('new-chat-btn');
  
  // Sidebar stats
  const infoLibrary    = document.getElementById('info-library');
  const infoModel      = document.getElementById('info-model');

  // Sample Buttons
  const sampleInvoice  = document.getElementById('sample-invoice');
  const sampleOffice   = document.getElementById('sample-office');

  // ── Mode Switcher ─────────────────────────────────────────────────────
  function setMode(mode) {
    if (State.running) return;
    State.mode = mode;
    
    modeOcr.classList.toggle('active', mode === 'ocr');
    modeDetection.classList.toggle('active', mode === 'detection');
    modeOcr.setAttribute('aria-pressed', String(mode === 'ocr'));
    modeDetection.setAttribute('aria-pressed', String(mode === 'detection'));
    
    psmSection.style.display = mode === 'ocr' ? 'flex' : 'none';
    
    // Update system info in sidebar
    if (mode === 'ocr') {
      infoLibrary.textContent = 'pytesseract';
      infoModel.textContent = 'Tesseract LSTM';
    } else {
      infoLibrary.textContent = 'cv2.dnn';
      infoModel.textContent = 'MobileNet-SSD';
    }
  }

  modeOcr.addEventListener('click', () => setMode('ocr'));
  modeDetection.addEventListener('click', () => setMode('detection'));

  // ── File & Sample Selection ───────────────────────────────────────────
  function selectFile(file) {
    if (State.running) return;

    // Reset previous selection
    clearSelectionStyles();
    if (State.imageUrl && State.imageUrl.startsWith('blob:')) {
      URL.revokeObjectURL(State.imageUrl);
    }

    State.file = file;
    State.imageUrl = URL.createObjectURL(file);
    State.imageName = file.name;

    inputStatusText.textContent = `Ready: ${file.name}`;
    inputStatusText.classList.add('active');
    runBtn.disabled = false;
  }

  function selectSample(type) {
    if (State.running) return;

    clearSelectionStyles();
    State.file = null;
    
    if (type === 'invoice') {
      sampleInvoice.classList.add('selected');
      State.imageUrl = 'assets/ocr_sample.png';
      State.imageName = 'ocr_sample.png';
      setMode('ocr');
    } else {
      sampleOffice.classList.add('selected');
      State.imageUrl = 'assets/det_sample.png';
      State.imageName = 'det_sample.png';
      setMode('detection');
    }

    inputStatusText.textContent = `Sample Ready: ${State.imageName}`;
    inputStatusText.classList.add('active');
    runBtn.disabled = false;
  }

  function clearSelectionStyles() {
    sampleInvoice.classList.remove('selected');
    sampleOffice.classList.remove('selected');
    inputStatusText.classList.remove('active');
  }

  attachBtn.addEventListener('click', () => imageUpload.click());
  imageUpload.addEventListener('change', (e) => {
    if (e.target.files[0]) selectFile(e.target.files[0]);
  });

  sampleInvoice.addEventListener('click', () => selectSample('invoice'));
  sampleOffice.addEventListener('click', () => selectSample('office'));

  // ── New Session Reset ─────────────────────────────────────────────────
  function resetSession() {
    if (State.running) return;
    chatThread.innerHTML = `
      <div class="message message--assistant">
        <div class="message__avatar">AI</div>
        <div class="message__content">
          <p>Welcome to the **DecodeLabs AI Recognition Pipeline**.</p>
          <p>Select one of the **Sample Inputs** in the sidebar, or attach your own image below, then click **Run Pipeline** (arrow icon) to process the image.</p>
        </div>
      </div>
    `;
    clearSelectionStyles();
    State.file = null;
    State.imageUrl = null;
    State.imageName = null;
    inputStatusText.textContent = "No image selected. Choose a sample or upload a file.";
    runBtn.disabled = true;
    downloadBtn.hidden = true;
    document.getElementById('active-session-title').textContent = "New Session";
  }

  newChatBtn.addEventListener('click', resetSession);

  // ── Mock Pipeline Data Generators ─────────────────────────────────────
  function getMockOCR() {
    const words = [
      { text: 'INVOICE',     conf: 0.98 },
      { text: 'DecodeLabs',  conf: 0.94 },
      { text: 'Date:',       conf: 0.91 },
      { text: '2026-06-25',  conf: 0.89 },
      { text: 'Total:',      conf: 0.95 },
      { text: '$150.00',     conf: 0.97 },
      { text: 'Status:',     conf: 0.85 },
      { text: 'UNPAID',      conf: 0.82 }
    ];
    return {
      mode: 'ocr',
      detections: words.map((w, i) => ({ label: w.text, confidence: w.conf, index: i + 1 })),
      full_text: words.map(w => w.text).join(' '),
      total_raw: 12,
      total_accepted: words.length,
      total_rejected: 4,
      runtime: (0.7 + Math.random() * 0.5).toFixed(2),
    };
  }

  function getMockDetection() {
    const objects = [
      { label: 'laptop',     conf: 0.96 },
      { label: 'cup',        conf: 0.92 },
      { label: 'book',       conf: 0.88 },
      { label: 'pottedplant',conf: 0.84 }
    ];
    return {
      mode: 'detection',
      detections: objects.map((o, i) => ({ label: o.label, confidence: o.conf, index: i + 1 })),
      full_text: null,
      total_raw: 8,
      total_accepted: objects.length,
      total_rejected: 4,
      runtime: (1.2 + Math.random() * 0.6).toFixed(2),
    };
  }

  // ── Execution Runner ──────────────────────────────────────────────────
  async function runPipeline() {
    if (State.running || !State.imageUrl) return;

    State.running = true;
    runBtn.disabled = true;
    runBtn.querySelector('svg').hidden = true;
    runBtnSpinner.hidden = false;

    State.sessionCount++;
    document.getElementById('active-session-title').textContent = `Analysis Session #${State.sessionCount}`;

    // 1. Append User Message (Image Card)
    const userMsg = document.createElement('div');
    userMsg.className = 'message message--user';
    userMsg.innerHTML = `
      <div class="message__avatar">U</div>
      <div class="message__content">
        <div class="image-card">
          <img src="${State.imageUrl}" class="image-card__img" alt="Uploaded source" />
          <div class="image-card__footer">
            <span class="image-card__name">${State.imageName}</span>
          </div>
        </div>
      </div>
    `;
    chatThread.appendChild(userMsg);
    chatThread.scrollTop = chatThread.scrollHeight;

    // 2. Append Assistant Message (with scanning logs & results skeleton)
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'message message--assistant';
    
    // Create inner bubble structure
    assistantMsg.innerHTML = `
      <div class="message__avatar">AI</div>
      <div class="message__content">
        <p>Initializing <strong>${State.mode.toUpperCase()}</strong> pipeline workflow...</p>
        <div class="terminal-block"></div>
        <div class="result-placeholder"></div>
      </div>
    `;
    chatThread.appendChild(assistantMsg);
    chatThread.scrollTop = chatThread.scrollHeight;

    const termBlock = assistantMsg.querySelector('.terminal-block');
    const resultPlaceholder = assistantMsg.querySelector('.result-placeholder');

    function logTerm(msg, type = 'INFO') {
      const d = new Date();
      const ts = `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
      const line = document.createElement('div');
      line.className = 'terminal-line';
      line.innerHTML = `<span class="term-ts">[${ts}]</span><span class="term-${type}">[${type}] ${msg}</span>`;
      termBlock.appendChild(line);
      termBlock.scrollTop = termBlock.scrollHeight;
    }

    const delay = (ms) => new Promise(r => setTimeout(r, ms));

    // Simulated log execution
    logTerm(`Starting ${State.mode.toUpperCase()} pipeline process...`, 'INFO');
    await delay(350);
    logTerm('Phase 1: Ingestion verified (Image shape, dimensions OK).', 'OK');
    await delay(300);
    logTerm('Phase 2: cvtColor BGR -> GRAY preprocessing conversion completed.', 'OK');
    await delay(300);
    logTerm('Phase 3: GaussianBlur noise attenuation applied (5x5 kernel).', 'OK');
    await delay(300);
    logTerm('Phase 4: adaptiveThreshold local-pixel contrast masking applied.', 'OK');
    await delay(400);

    if (State.mode === 'ocr') {
      logTerm('Phase 5: Invoking Tesseract OCR binary engine (image_to_data).', 'INFO');
      await delay(600);
      logTerm('Phase 6: Filtering dataframe elements (80% confidence threshold constraint).', 'OK');
      await delay(300);
      logTerm('Phase 7: Structural multi-line text reconstruction completed.', 'OK');
    } else {
      logTerm('Phase 5: MobileNet-SSD caffe forward blob construction (300x300).', 'INFO');
      await delay(500);
      logTerm('Phase 6: Running DNN inference pass.', 'OK');
      await delay(400);
      logTerm('Phase 7: Non-Maximum Suppression (IoU: 0.4, confidence target: >= 80%).', 'OK');
    }
    
    await delay(300);
    logTerm('Pipeline result annotations exported to output/image_output.jpg.', 'OK');
    logTerm('Processing completed with zero errors.', 'OK');

    // 3. Render Results in the Assistant bubble
    const resultData = State.mode === 'ocr' ? getMockOCR() : getMockDetection();

    // Render BBoxes directly on the original preview using CSS absolute borders, or just show the image
    // For visual confirmation, we display the output image and the structured cards
    const resultsWrapper = document.createElement('div');
    resultsWrapper.className = 'result-block';

    let summaryHtml = `
      <div class="stats-summary">
        <span class="stat-chip stat-chip--green">✓ ${resultData.total_accepted} accepted</span>
        <span class="stat-chip stat-chip--red">✗ ${resultData.total_rejected} rejected</span>
        <span class="stat-chip">⏱ ${resultData.runtime}s runtime</span>
      </div>
      
      <div class="result-visualizer">
        <img src="${State.imageUrl}" class="result-visualizer__img" alt="Annotated result" />
        <span class="result-visualizer__badge">${State.mode.toUpperCase()}</span>
      </div>
    `;

    if (State.mode === 'ocr') {
      summaryHtml += `
        <div class="result-ocr-box">
          <div class="result-ocr-header">
            <span class="result-ocr-title">Reconstructed Text</span>
            <button class="result-ocr-copy">Copy</button>
          </div>
          <pre class="result-ocr-text">${resultData.full_text}</pre>
        </div>
      `;
    }

    // Add list of cards
    let listHtml = `<div class="result-det-list">`;
    resultData.detections.forEach(det => {
      const isHigh = det.confidence >= 0.90;
      const pct = (det.confidence * 100).toFixed(1);
      listHtml += `
        <div class="det-card">
          <span class="det-card__idx">${det.index}</span>
          <span class="det-card__label">${det.label}</span>
          <div class="det-card__bar-bg">
            <div class="det-card__bar-fill det-card__bar-fill--${isHigh ? 'high' : 'med'}" style="width: ${det.confidence * 100}%"></div>
          </div>
          <span class="det-card__conf det-card__conf--${isHigh ? 'high' : 'med'}">${pct}%</span>
        </div>
      `;
    });
    listHtml += `</div>`;
    summaryHtml += listHtml;

    // Add Milestone Gates Pass Status list
    summaryHtml += `
      <div class="gate-checklist">
        <div class="checklist-item"><span class="check-icon">✓</span> Gate 1: Libraries</div>
        <div class="checklist-item"><span class="check-icon">✓</span> Gate 2: Preprocess</div>
        <div class="checklist-item"><span class="check-icon">✓</span> Gate 3: Threshold</div>
        <div class="checklist-item"><span class="check-icon">✓</span> Gate 4: Visuals</div>
      </div>
    `;

    resultsWrapper.innerHTML = summaryHtml;
    resultPlaceholder.appendChild(resultsWrapper);

    // Event listener for the Copy button inside the OCR block
    if (State.mode === 'ocr') {
      const copyBtn = resultsWrapper.querySelector('.result-ocr-copy');
      copyBtn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(resultData.full_text);
          copyBtn.textContent = 'Copied!';
          setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
        } catch {
          copyBtn.textContent = 'Error';
        }
      });
    }

    // Enable download button
    downloadBtn.hidden = false;
    downloadBtn.onclick = () => {
      const a = document.createElement('a');
      a.href = State.imageUrl;
      a.download = `output_${State.imageName}`;
      a.click();
    };

    // Clean up send button state
    State.running = false;
    runBtn.disabled = false;
    runBtn.querySelector('svg').hidden = false;
    runBtnSpinner.hidden = true;

    chatThread.scrollTop = chatThread.scrollHeight;
  }

  runBtn.addEventListener('click', runPipeline);
});
