// Virtual Background: video-conference selfie view with MediaPipe Image Segmentation
// (Selfie Segmenter). Runs fully in the browser.
//
// Keys:  1-4  landscape photos       0 black   r red   w white   y yellow   g green
//        b    blur (each press = blurrier)       o original background
import { ImageSegmenter, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = '/models/selfie_segmenter.tflite';

const IMAGES = {
    1: { name: 'Mountains', src: '/static/backgrounds/1_mountains.jpg' },
    2: { name: 'Beach', src: '/static/backgrounds/2_beach.jpg' },
    3: { name: 'Forest Lake', src: '/static/backgrounds/3_forest_lake.jpg' },
    4: { name: 'Desert', src: '/static/backgrounds/4_desert.jpg' },
};
const COLORS = {
    0: { name: 'Black', color: '#000000' },
    r: { name: 'Red', color: '#e11d48' },
    w: { name: 'White', color: '#ffffff' },
    y: { name: 'Yellow', color: '#facc15' },
    g: { name: 'Green', color: '#16a34a' },
};
const MAX_BLUR_LEVEL = 10;
const BLUR_PX_PER_LEVEL = 3;   // at 640 px width, scaled with the video size

// preload the photos
for (const img of Object.values(IMAGES)) {
    img.el = new Image();
    img.el.src = img.src;
}

const $ = (id) => document.getElementById(id);
const els = {
    section: $('backgroundModeSection'),
    startBtn: $('vbStartBtn'), startLabel: $('vbStartLabel'),
    sourcePills: document.querySelectorAll('#vbSourcePills .pill-btn'),
    fileSection: $('vbFileSection'), fileInput: $('vbFileInput'), nameInput: $('vbNameInput'),
    edgeRange: $('vbEdgeRange'), edgeVal: $('vbEdgeVal'), cutRange: $('vbCutRange'), cutVal: $('vbCutVal'),
    mirror: $('vbMirrorToggle'),
    empty: $('vbEmptyState'), stage: $('vbStage'), canvas: $('vbCanvas'), status: $('vbStatus'),
    video: $('vbVideo'), modeName: $('vbModeName'), blurLevel: $('vbBlurLevel'), fps: $('vbFps'),
    badge: $('vbBadge'), nameTag: $('vbNameTag'), toast: $('vbToast'),
    keyButtons: document.querySelectorAll('#backgroundModeSection [data-key]'),
};
const ctx = els.canvas.getContext('2d');
// offscreen helpers: mask (at mask resolution) and the cut-out person layer
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');
const personCanvas = document.createElement('canvas');
const personCtx = personCanvas.getContext('2d');
let maskImageData = null;

// background state
const bg = { mode: 'original', key: 'o', blurLevel: 0 };

let source = 'webcam';
let segmenter = null;
let running = false;
let stream = null;
let rafId = null;
let lastVideoTime = -1;
let fpsFrames = 0, fpsWindowStart = 0, currentFps = 0;
let toastTimer = null;

// ---------- keyboard + click controls ----------
function applyKey(rawKey) {
    const key = String(rawKey).toLowerCase();
    if (IMAGES[key]) {
        Object.assign(bg, { mode: 'image', key, blurLevel: 0 });
        toast(`${key} – ${IMAGES[key].name}`);
    } else if (COLORS[key]) {
        Object.assign(bg, { mode: 'color', key, blurLevel: 0 });
        toast(`${key} – ${COLORS[key].name}`);
    } else if (key === 'b') {
        if (bg.mode !== 'blur') Object.assign(bg, { mode: 'blur', key: 'b', blurLevel: 1 });
        else if (bg.blurLevel < MAX_BLUR_LEVEL) bg.blurLevel += 1;
        toast(bg.blurLevel >= MAX_BLUR_LEVEL ? `Blur ${bg.blurLevel} (max)` : `Blur level ${bg.blurLevel}`);
    } else if (key === 'o') {
        Object.assign(bg, { mode: 'original', key: 'o', blurLevel: 0 });
        toast('Original background');
    } else {
        return false;
    }
    renderState();
    if (!running) drawIdlePreview();
    return true;
}

document.addEventListener('keydown', (e) => {
    if (els.section.style.display === 'none') return;           // only on this tab
    if (e.ctrlKey || e.altKey || e.metaKey || e.repeat) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;  // typing a name etc.
    if (applyKey(e.key)) e.preventDefault();
});

els.keyButtons.forEach((btn) => btn.addEventListener('click', () => applyKey(btn.dataset.key)));

function modeLabel() {
    if (bg.mode === 'image') return IMAGES[bg.key].name;
    if (bg.mode === 'color') return COLORS[bg.key].name;
    if (bg.mode === 'blur') return `Blur · level ${bg.blurLevel}`;
    return 'Original';
}

function renderState() {
    const label = modeLabel();
    els.modeName.textContent = label;
    els.badge.textContent = label;
    els.blurLevel.textContent = bg.mode === 'blur' ? bg.blurLevel : 0;
    els.keyButtons.forEach((b) => b.classList.toggle('active', b.dataset.key === bg.key));
}

function toast(text) {
    els.toast.textContent = text;
    els.toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.remove('show'), 1100);
}

// ---------- settings ----------
els.edgeRange.addEventListener('input', () => { els.edgeVal.textContent = `${els.edgeRange.value}px`; });
els.cutRange.addEventListener('input', () => { els.cutVal.textContent = `${Math.round(els.cutRange.value * 100)}%`; });
els.nameInput.addEventListener('input', () => { els.nameTag.textContent = els.nameInput.value.trim() || 'Me'; });

els.sourcePills.forEach((btn) => btn.addEventListener('click', () => {
    if (running) stop();
    els.sourcePills.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    source = btn.dataset.vsource;
    els.fileSection.style.display = source === 'file' ? 'block' : 'none';
    els.mirror.checked = source === 'webcam';
    els.startLabel.textContent = source === 'webcam' ? 'Start Camera' : 'Play Video';
}));

els.startBtn.addEventListener('click', () => (running ? stop() : start()));
document.addEventListener('modechange', (e) => { if (e.detail.mode !== 'background' && running) stop(); });

// ---------- camera / segmenter ----------
async function getSegmenter() {
    if (segmenter) return segmenter;
    els.status.textContent = 'Loading segmentation model...';
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const opts = (delegate) => ({
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
    });
    try {
        segmenter = await ImageSegmenter.createFromOptions(vision, opts('GPU'));
    } catch (err) {
        console.warn('GPU delegate failed, falling back to CPU', err);
        segmenter = await ImageSegmenter.createFromOptions(vision, opts('CPU'));
    }
    return segmenter;
}

async function start() {
    const video = els.video;
    els.startBtn.disabled = true;
    els.empty.style.display = 'none';
    els.stage.style.display = 'block';
    els.status.style.display = 'block';
    try {
        if (source === 'webcam') {
            if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not support camera access.');
            els.status.textContent = 'Waiting for camera permission...';
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
            });
            video.srcObject = stream;
            video.loop = false;
        } else {
            const file = els.fileInput.files?.[0];
            if (!file) throw new Error('Please choose a video file first.');
            video.srcObject = null;
            if (video.src) URL.revokeObjectURL(video.src);
            video.src = URL.createObjectURL(file);
            video.loop = true;
        }
        await new Promise((res, rej) => {
            if (video.readyState >= 2) return res();
            video.onloadeddata = () => res();
            video.onerror = () => rej(new Error('Video could not be loaded.'));
        });
        await getSegmenter();
        await video.play();

        running = true;
        lastVideoTime = -1;
        fpsFrames = 0; fpsWindowStart = 0; currentFps = 0;
        els.status.style.display = 'none';
        els.startLabel.textContent = source === 'webcam' ? 'Stop Camera' : 'Stop';
        loop();
    } catch (err) {
        console.error(err);
        const msg = err.name === 'NotAllowedError' ? 'Camera access was denied. Allow it in the browser and try again.'
            : err.name === 'NotFoundError' ? 'No camera found.' : err.message;
        els.status.style.display = 'block';
        els.status.textContent = msg;
        stopMedia();
    } finally {
        els.startBtn.disabled = false;
    }
}

function stopMedia() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    els.video.pause();
}

function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    stopMedia();
    els.startLabel.textContent = source === 'webcam' ? 'Start Camera' : 'Play Video';
}

function loop() {
    if (!running) return;
    const video = els.video;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const now = performance.now();
        // callback form: the mask is only valid inside the callback
        segmenter.segmentForVideo(video, now, (result) => {
            const mask = result.confidenceMasks?.[0];
            if (mask) composite(mask);
        });
        fpsFrames += 1;
        if (!fpsWindowStart) fpsWindowStart = now;
        if (now - fpsWindowStart >= 1000) {
            currentFps = Math.round((fpsFrames * 1000) / (now - fpsWindowStart));
            fpsFrames = 0; fpsWindowStart = now;
            els.fps.textContent = `${currentFps} FPS`;
        }
    }
    rafId = requestAnimationFrame(loop);
}

// ---------- rendering ----------
function drawVideo(target, w, h, mirror, blurPx = 0) {
    target.save();
    if (mirror) { target.translate(w, 0); target.scale(-1, 1); }
    if (blurPx > 0) {
        // draw slightly larger so the blurred, semi-transparent edges stay outside the frame
        const m = blurPx * 2;
        target.filter = `blur(${blurPx}px)`;
        target.drawImage(els.video, -m, -m, w + 2 * m, h + 2 * m);
    } else {
        target.drawImage(els.video, 0, 0, w, h);
    }
    target.restore();
}

function drawCover(target, img, w, h) {
    // like CSS object-fit: cover
    const s = Math.max(w / img.naturalWidth, h / img.naturalHeight);
    const dw = img.naturalWidth * s, dh = img.naturalHeight * s;
    target.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
}

function drawBackground(w, h, mirror) {
    if (bg.mode === 'color') {
        ctx.fillStyle = COLORS[bg.key].color;
        ctx.fillRect(0, 0, w, h);
    } else if (bg.mode === 'image') {
        const img = IMAGES[bg.key].el;
        if (img.complete && img.naturalWidth) drawCover(ctx, img, w, h);
        else { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h); }
    } else if (bg.mode === 'blur') {
        const blurPx = bg.blurLevel * BLUR_PX_PER_LEVEL * Math.max(1, w / 640);
        drawVideo(ctx, w, h, mirror, blurPx);
    }
}

function updateMaskCanvas(mask) {
    const mw = mask.width, mh = mask.height;
    if (maskCanvas.width !== mw || maskCanvas.height !== mh || !maskImageData) {
        maskCanvas.width = mw; maskCanvas.height = mh;
        maskImageData = maskCtx.createImageData(mw, mh);
        maskImageData.data.fill(255);   // white RGB, alpha set per pixel below
    }
    const conf = mask.getAsFloat32Array();
    const data = maskImageData.data;
    // soft threshold around the chosen cut-off -> smooth alpha transition at the edges
    const lo = Number(els.cutRange.value) - 0.15, range = 0.3;
    for (let i = 0, j = 3; i < conf.length; i++, j += 4) {
        const a = (conf[i] - lo) / range;
        data[j] = a <= 0 ? 0 : a >= 1 ? 255 : a * 255;
    }
    maskCtx.putImageData(maskImageData, 0, 0);
}

function composite(mask) {
    const video = els.video;
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return;
    for (const c of [els.canvas, personCanvas]) {
        if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    }
    const mirror = els.mirror.checked;

    if (bg.mode === 'original') {
        drawVideo(ctx, w, h, mirror);
        return;
    }

    updateMaskCanvas(mask);

    // person layer = video masked by the (softened) segmentation mask
    personCtx.save();
    personCtx.globalCompositeOperation = 'copy';
    drawVideo(personCtx, w, h, mirror);
    personCtx.globalCompositeOperation = 'destination-in';
    const edge = Number(els.edgeRange.value);
    personCtx.filter = edge > 0 ? `blur(${edge}px)` : 'none';
    if (mirror) { personCtx.translate(w, 0); personCtx.scale(-1, 1); }
    personCtx.drawImage(maskCanvas, 0, 0, w, h);
    personCtx.restore();

    drawBackground(w, h, mirror);
    ctx.drawImage(personCanvas, 0, 0);
}

// Before the camera runs, show the chosen background in the viewer as a preview
function drawIdlePreview() {
    if (els.stage.style.display === 'none') return;
    const w = els.canvas.width || 1280, h = els.canvas.height || 720;
    els.canvas.width = w; els.canvas.height = h;
    if (bg.mode === 'image' || bg.mode === 'color') drawBackground(w, h, false);
}

renderState();
