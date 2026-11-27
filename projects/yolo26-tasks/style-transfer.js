// Neural Style Transfer page (main thread). The models run in style-worker.js (Web Worker,
// ONNX Runtime Web WASM backend), so the page stays responsive.
// Model input: float32 RGB 0..255, NCHW, no mean subtraction. Output is clipped to 0..255.

const ORT_PATH = new URL('./vendor/onnxruntime-web/', window.location.href).href;
const STYLE_DIR = './models/style/';

const $ = (id) => document.getElementById(id);
const el = {
    file: $('nFile'), drop: $('drop'), preview: $('nPreview'), dropText: $('nDropText'),
    quality: $('nQuality'), runAll: $('nRunAll'), strength: $('nStrength'), strengthVal: $('nStrengthVal'),
    engine: $('nEngine'), done: $('nDone'), error: $('nError'), processing: $('nProcessing'), processingText: $('nProcessingText'),
    thumbs: $('nThumbs'), gallery: $('nGallery'), credits: $('nCredits'),
};

let styles = [];                   // from models/style/styles.json
let photo = null;                  // decoded HTMLImageElement
let source = null;                 // { canvas, w, h, tensor }
const results = new Map();         // style id -> { card, stylized canvas | null }
const queue = [];                  // style ids waiting for the worker
let working = null;                // style id currently in the worker
let jobId = 0;                     // bumps when the photo/size changes -> stale results are ignored
let doneCount = 0, totalCount = 0;
let worker = null, workerReady = false;

const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function showError(html) { el.error.innerHTML = html; el.error.hidden = !html; }
function setProcessing(text) { el.processing.hidden = !text; el.processingText.textContent = text || ''; }

// ---------- worker ----------
// Multi-threaded WASM inside a worker needs nested workers, which some browsers cannot start
// (session creation then hangs). The worker therefore self-tests first; if that does not
// finish in time it is restarted with a single thread, which works everywhere.
const PROBE_TIMEOUT_MS = 15000;
let probeTimer = null, engineThreads = 0, singleThreadFallback = false;

function startWorker(forceThreads) {
    workerReady = false;
    if (worker) worker.terminate();
    try {
        worker = new Worker('./style-worker.js', { type: 'module' });
    } catch (err) {
        showError(`<strong>Unsupported browser.</strong> Web Workers are not available (${escapeHtml(err.message)}).`);
        return;
    }
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => showError(`<strong>The style engine crashed.</strong> ${escapeHtml(e.message || 'unknown error')}`);
    // ?threads=N in the URL overrides the WASM thread count (diagnostics)
    const threads = forceThreads || Number(new URLSearchParams(window.location.search).get('threads')) || 0;
    el.engine.textContent = 'starting...';
    worker.postMessage({ type: 'init', ortPath: ORT_PATH, threads });
}

const FALLBACK_KEY = 'nst-single-thread';
function rememberedFallback() { try { return localStorage.getItem(FALLBACK_KEY) === '1'; } catch (_) { return false; } }

function fallbackToSingleThread() {
    clearTimeout(probeTimer);
    singleThreadFallback = true;
    try { localStorage.setItem(FALLBACK_KEY, '1'); } catch (_) { /* storage blocked */ }
    if (working) { queue.unshift(working); working = null; }      // retry the interrupted style
    startWorker(1);
}

function onWorkerMessage(e) {
    const m = e.data;
    if (m.type === 'ready') {
        engineThreads = m.threads;
        el.engine.textContent = `WASM · ${m.threads} thread${m.threads > 1 ? 's' : ''}${singleThreadFallback ? ' (fallback)' : ''}`;
        if (m.threads > 1) {
            el.engine.textContent += ' · testing';
            const smallest = styles.find((s) => s.id === 'feathers') || styles[0];
            probeTimer = setTimeout(fallbackToSingleThread, PROBE_TIMEOUT_MS);
            worker.postMessage({ type: 'probe', url: STYLE_DIR + smallest.file });
        } else {
            workerReady = true;
            pump();
        }
        return;
    }
    if (m.type === 'probe-ok') {
        clearTimeout(probeTimer);
        workerReady = true;
        el.engine.textContent = `WASM · ${engineThreads} threads`;
        pump();
        return;
    }
    if (m.type === 'probe-failed') { fallbackToSingleThread(); return; }
    if (m.type === 'error' && m.stage === 'init') {
        el.engine.textContent = 'failed';
        showError(`<strong>Unsupported browser.</strong> ${escapeHtml(m.message)}`);
        return;
    }
    if (m.id !== jobId) return;                 // result of an older photo/size -> ignore
    const style = styles.find((s) => s.id === m.style);
    const r = results.get(m.style);
    if (m.type === 'status') {
        setProcessing(`${style.name}: ${m.text}`);
        setCardStatus(r?.card, m.text);
        return;
    }
    if (m.type === 'result') {
        const c = document.createElement('canvas');
        c.width = m.width; c.height = m.height;
        c.getContext('2d').putImageData(new ImageData(m.rgba, m.width, m.height), 0, 0);
        r.stylized = c;
        refreshCard(r);
        setCardStatus(r.card, '');
        r.card.querySelector('small').textContent = `${(m.ms / 1000).toFixed(1)} s${m.cached ? ' · model from cache' : ''}`;
        r.card.querySelector('button').disabled = false;
        markThumb(m.style, 'done');
    } else if (m.type === 'error') {
        setCardStatus(r.card, 'Failed');
        markThumb(m.style, '');
        const what = m.stage === 'load' ? 'model failed to load' : 'conversion failed';
        showError(`<strong>${escapeHtml(style.name)}: ${what}.</strong> ${escapeHtml(m.message)}`);
    }
    doneCount += 1;
    el.done.textContent = `${doneCount} / ${totalCount}`;
    working = null;
    pump();
}

function pump() {
    if (!workerReady || working || !queue.length || !source) {
        if (!working && !queue.length) { setProcessing(''); el.runAll.disabled = !source; }
        else if (!workerReady && queue.length) setProcessing('Starting the style engine...');
        return;
    }
    const id = queue.shift();
    const style = styles.find((s) => s.id === id);
    working = id;
    markThumb(id, 'busy');
    setCardStatus(results.get(id).card, 'Processing...');
    setProcessing(`Processing ${style.name}... (${doneCount + 1} of ${totalCount})`);
    const input = source.tensor.slice();       // copy - the buffer is transferred to the worker
    worker.postMessage({ type: 'run', id: jobId, style: id, url: STYLE_DIR + style.file, width: source.w, height: source.h, input }, [input.buffer]);
}

// ---------- photo ----------
function loadPhoto(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showError(`<strong>Unsupported image.</strong> &ldquo;${escapeHtml(file.name)}&rdquo; is not an image &ndash; please use JPG, PNG, WebP or GIF.`);
        return;
    }
    showError('');
    const img = new Image();
    img.onload = () => {
        if (!img.naturalWidth || !img.naturalHeight) { img.onerror(); return; }
        photo = img;
        el.preview.src = img.src; el.preview.hidden = false; el.dropText.hidden = true;
        prepareSource();
        resetResults();
    };
    img.onerror = () => showError(`<strong>Unsupported image.</strong> &ldquo;${escapeHtml(file.name)}&rdquo; could not be decoded by the browser (e.g. HEIC/RAW). Please convert it to JPG or PNG.`);
    img.src = URL.createObjectURL(file);
}

// resize so the longest side = chosen size; sides are multiples of 4 (the network halves twice)
function prepareSource() {
    if (!photo) return;
    const maxSide = Number(el.quality.value);
    const s = Math.min(1, maxSide / Math.max(photo.naturalWidth, photo.naturalHeight));
    const w = Math.max(16, Math.round((photo.naturalWidth * s) / 4) * 4);
    const h = Math.max(16, Math.round((photo.naturalHeight * s) / 4) * 4);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.imageSmoothingQuality = 'high';
    g.drawImage(photo, 0, 0, w, h);
    const px = g.getImageData(0, 0, w, h).data;
    const area = w * h, tensor = new Float32Array(3 * area);
    for (let i = 0, j = 0; i < area; i++, j += 4) { tensor[i] = px[j]; tensor[i + area] = px[j + 1]; tensor[i + 2 * area] = px[j + 2]; }
    source = { canvas: c, w, h, tensor };
}

function resetResults() {
    jobId += 1;
    queue.length = 0; working = null;
    results.clear();
    doneCount = 0; totalCount = 0;
    el.done.textContent = '0 / 0';
    document.querySelectorAll('.thumb').forEach((t) => t.classList.remove('done', 'busy'));
    setProcessing('');
    renderGallery();
    el.runAll.disabled = !source;
}

// ---------- conversion requests ----------
function enqueue(ids) {
    if (!source) { showError('<strong>No photo yet.</strong> Upload a photo first.'); return; }
    showError('');
    for (const id of ids) {
        if (queue.includes(id) || working === id) continue;
        if (!results.has(id)) { results.set(id, { card: makeCard(styles.find((s) => s.id === id)), stylized: null }); el.gallery.append(results.get(id).card); }
        const r = results.get(id);
        if (r.stylized) continue;                 // already converted at this size
        queue.push(id);
        totalCount += 1;
    }
    el.done.textContent = `${doneCount} / ${totalCount}`;
    el.runAll.disabled = true;
    pump();
}

// ---------- gallery ----------
function renderGallery() {
    el.gallery.innerHTML = '';
    if (!source) return;
    const orig = document.createElement('div');
    orig.className = 'card original';
    orig.innerHTML = `<div class="pic"></div><div class="meta"><div><b>Original</b><small>${source.w} × ${source.h} px</small></div></div>`;
    const oc = document.createElement('canvas');
    oc.width = source.w; oc.height = source.h;
    oc.getContext('2d').drawImage(source.canvas, 0, 0);
    orig.querySelector('.pic').append(oc);
    el.gallery.append(orig);
}

function makeCard(style) {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
        <div class="pic"><canvas></canvas><div class="busy"><div class="spinner"></div><span>Waiting...</span></div></div>
        <div class="meta"><div><b>${escapeHtml(style.name)}</b><small>waiting</small></div>
            <button type="button" disabled>&#x2B07; Download</button></div>`;
    card.querySelector('button').addEventListener('click', () => download(style));
    return card;
}

function setCardStatus(card, text) {
    if (!card) return;
    const busy = card.querySelector('.busy');
    busy.hidden = !text;
    busy.querySelector('span').textContent = text || '';
}

function drawBlend(target, stylized) {
    const { canvas: orig, w, h } = source;
    target.width = w; target.height = h;
    const g = target.getContext('2d');
    g.globalAlpha = 1; g.drawImage(orig, 0, 0, w, h);
    g.globalAlpha = Number(el.strength.value); g.drawImage(stylized, 0, 0, w, h);
    g.globalAlpha = 1;
}

function refreshCard(r) { if (r.stylized) drawBlend(r.card.querySelector('canvas'), r.stylized); }

function download(style) {
    const r = results.get(style.id);
    if (!r?.stylized) return;
    const c = document.createElement('canvas');
    drawBlend(c, r.stylized);
    c.toBlob((blob) => {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `style_${style.id}_${source.w}x${source.h}.png`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 3000);
    }, 'image/png');
}

// ---------- thumbnails ----------
function renderThumbs() {
    el.thumbs.innerHTML = styles.map((s) => `
        <button type="button" class="thumb" data-id="${s.id}" title="Convert with ${escapeHtml(s.name)}">
            <img src="${STYLE_DIR}thumbs/${s.id}.jpg" alt="${escapeHtml(s.name)} style preview" loading="lazy" onerror="this.style.visibility='hidden'">
            <span>${escapeHtml(s.name)}<small>${escapeHtml(s.source)}</small></span>
        </button>`).join('');
    el.thumbs.querySelectorAll('.thumb').forEach((b) => b.addEventListener('click', () => enqueue([b.dataset.id])));
}

function markThumb(id, state) {
    const t = el.thumbs.querySelector(`[data-id="${id}"]`);
    if (!t) return;
    t.classList.toggle('busy', state === 'busy');
    t.classList.toggle('done', state === 'done');
}

// ---------- wiring ----------
el.file.addEventListener('change', () => loadPhoto(el.file.files?.[0]));
['dragenter', 'dragover'].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.add('drag'); }));
['dragleave', 'drop'].forEach((ev) => el.drop.addEventListener(ev, (e) => { e.preventDefault(); el.drop.classList.remove('drag'); }));
el.drop.addEventListener('drop', (e) => loadPhoto(e.dataTransfer.files?.[0]));
window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === 'file');
    if (item) loadPhoto(item.getAsFile());
});
el.quality.addEventListener('change', () => { if (photo) { prepareSource(); resetResults(); } });
el.runAll.addEventListener('click', () => enqueue(styles.map((s) => s.id)));
el.strength.addEventListener('input', () => {
    el.strengthVal.textContent = `${Math.round(el.strength.value * 100)}%`;
    for (const r of results.values()) refreshCard(r);
});

// ---------- start ----------
(async function init() {
    if (typeof WebAssembly !== 'object' || typeof Worker === 'undefined') {
        showError('<strong>Unsupported browser.</strong> This page needs WebAssembly and Web Workers (any current Chrome, Edge, Firefox or Safari).');
        return;
    }
    try {
        const res = await fetch(`${STYLE_DIR}styles.json`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        styles = await res.json();
    } catch (err) {
        showError(`<strong>Style list could not be loaded</strong> (models/style/styles.json): ${escapeHtml(err.message)}`);
        return;
    }
    renderThumbs();
    singleThreadFallback = rememberedFallback();     // this browser needed the fallback before -> skip the test
    const zoo = styles.filter((s) => s.source === 'ONNX Model Zoo').map((s) => s.name).join(', ');
    const jj = styles.filter((s) => s.source !== 'ONNX Model Zoo').map((s) => s.name).join(', ');
    el.credits.innerHTML = `${escapeHtml(zoo)}: ONNX Model Zoo fast neural style models (<a href="https://huggingface.co/onnxmodelzoo" target="_blank" rel="noopener">huggingface.co/onnxmodelzoo</a>, "-9" versions), based on the PyTorch fast-neural-style example.`
        + (jj ? `<br>${escapeHtml(jj)}: converted from Justin Johnson's Torch models (<a href="https://github.com/jcjohnson/fast-neural-style" target="_blank" rel="noopener">github.com/jcjohnson/fast-neural-style</a>), which are free for personal or research use.` : '')
        + '<br>Thumbnails show each style applied to a public-domain (CC0) mountain photo.';
    startWorker(singleThreadFallback ? 1 : 0);
})();

// console hook for automated tests: await nstTest('./photo.jpg', 256)
window.nstTest = async (url, size = 256) => {
    el.quality.value = String(size);
    const blob = await (await fetch(url)).blob();
    loadPhoto(new File([blob], 'test.jpg', { type: blob.type || 'image/jpeg' }));
    await new Promise((r) => { const t = setInterval(() => { if (source && source.w) { clearInterval(t); r(); } }, 100); });
    enqueue(styles.map((s) => s.id));
    await new Promise((r) => { const t = setInterval(() => { if (!working && !queue.length) { clearInterval(t); r(); } }, 300); });
    return [...results.entries()].map(([id, r]) => [id, !!r.stylized, r.card.querySelector('small').textContent]);
};
