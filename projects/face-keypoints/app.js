// Face Keypoints: MediaPipe Face Detection with a geometric shape on each of the 6 keypoints.
// Both eyes = concentric circles, the other 4 keypoints = random shapes (eye-sized).
// Any key -> shuffle the shapes, Esc / q -> exit.
import { FaceDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite';

// MediaPipe face detector keypoint order
const KEYPOINTS = [
    { name: 'Right eye', color: '#22d3ee', eye: true },
    { name: 'Left eye', color: '#22d3ee', eye: true },
    { name: 'Nose tip', color: '#facc15' },
    { name: 'Mouth', color: '#f43f5e' },
    { name: 'Right ear', color: '#a3e635' },
    { name: 'Left ear', color: '#c084fc' },
];

// ---------- shapes (all drawn centred at x,y with "radius" r) ----------
function polygonPath(c, x, y, r, sides, rotation = -Math.PI / 2) {
    c.beginPath();
    for (let i = 0; i < sides; i++) {
        const a = rotation + (i * 2 * Math.PI) / sides;
        c.lineTo(x + r * Math.cos(a), y + r * Math.sin(a));
    }
    c.closePath();
}

const SHAPES = {
    triangle: { label: 'Triangle', path: (c, x, y, r) => polygonPath(c, x, y + r * 0.15, r * 1.1, 3) },
    square: { label: 'Square', path: (c, x, y, r) => { c.beginPath(); c.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7); } },
    diamond: { label: 'Diamond', path: (c, x, y, r) => polygonPath(c, x, y, r * 1.1, 4) },
    pentagon: { label: 'Pentagon', path: (c, x, y, r) => polygonPath(c, x, y, r, 5) },
    hexagon: { label: 'Hexagon', path: (c, x, y, r) => polygonPath(c, x, y, r, 6, 0) },
    octagon: { label: 'Octagon', path: (c, x, y, r) => polygonPath(c, x, y, r, 8, Math.PI / 8) },
    star: {
        label: 'Star',
        path: (c, x, y, r) => {
            c.beginPath();
            for (let i = 0; i < 10; i++) {
                const a = -Math.PI / 2 + (i * Math.PI) / 5;
                const rr = i % 2 === 0 ? r * 1.15 : r * 0.48;
                c.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
            }
            c.closePath();
        },
    },
    cross: {
        label: 'Cross',
        path: (c, x, y, r) => {
            const t = r * 0.35;
            c.beginPath();
            const pts = [[-t, -r], [t, -r], [t, -t], [r, -t], [r, t], [t, t], [t, r], [-t, r], [-t, t], [-r, t], [-r, -t], [-t, -t]];
            pts.forEach(([dx, dy]) => c.lineTo(x + dx, y + dy));
            c.closePath();
        },
    },
    heart: {
        label: 'Heart',
        path: (c, x, y, r) => {
            c.beginPath();
            c.moveTo(x, y + r);
            c.bezierCurveTo(x - r * 1.5, y - r * 0.1, x - r * 0.75, y - r * 1.2, x, y - r * 0.45);
            c.bezierCurveTo(x + r * 0.75, y - r * 1.2, x + r * 1.5, y - r * 0.1, x, y + r);
            c.closePath();
        },
    },
};
const RANDOM_SHAPE_NAMES = Object.keys(SHAPES);

function drawConcentric(c, x, y, r, color, lw) {
    c.lineWidth = lw;
    c.strokeStyle = color;
    for (const f of [1, 0.66]) {
        c.beginPath(); c.arc(x, y, r * f, 0, Math.PI * 2); c.stroke();
    }
    c.beginPath(); c.arc(x, y, r * 0.3, 0, Math.PI * 2);
    c.fillStyle = color; c.fill();
}

function drawShape(c, shapeName, x, y, r, color, lw) {
    if (shapeName === 'concentric') return drawConcentric(c, x, y, r, color, lw);
    SHAPES[shapeName].path(c, x, y, r);
    c.fillStyle = color + '55';          // translucent fill
    c.fill();
    c.lineWidth = lw;
    c.strokeStyle = color;
    c.lineJoin = 'round';
    c.stroke();
}

// ---------- random assignment ----------
let assignment = [];   // shape name per keypoint index
let shuffleCount = 0;

function shuffleArray(arr) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function reassignShapes() {
    const previous = assignment.join();
    let picked;
    do {
        picked = shuffleArray(RANDOM_SHAPE_NAMES).slice(0, 4);   // 4 different shapes
    } while (['concentric', 'concentric', ...picked].join() === previous);
    assignment = ['concentric', 'concentric', ...picked];
    renderLegend();
}

// ---------- UI ----------
const $ = (id) => document.getElementById(id);
const els = {
    section: $('faceKeysModeSection'),
    startBtn: $('fkStartBtn'), startLabel: $('fkStartLabel'), shuffleBtn: $('fkShuffleBtn'),
    sourcePills: document.querySelectorAll('#fkSourcePills .pill-btn'),
    fileSection: $('fkFileSection'), fileInput: $('fkFileInput'),
    confRange: $('fkConfRange'), confVal: $('fkConfVal'), showBox: $('fkBoxToggle'), mirror: $('fkMirrorToggle'),
    empty: $('fkEmptyState'), emptyTitle: $('fkEmptyTitle'), emptyText: $('fkEmptyText'),
    stage: $('fkStage'), canvas: $('fkCanvas'), status: $('fkStatus'), toast: $('fkToast'),
    video: $('fkVideo'), faceCount: $('fkFaceCount'), shuffles: $('fkShuffleCount'), fps: $('fkFps'),
    legend: $('fkLegend'),
};
const ctx = els.canvas.getContext('2d');

function renderLegend() {
    els.legend.innerHTML = '';
    KEYPOINTS.forEach((kp, i) => {
        const li = document.createElement('li');
        li.style.setProperty('--c', kp.color);
        const icon = document.createElement('canvas');
        icon.width = 26; icon.height = 26;
        drawShape(icon.getContext('2d'), assignment[i], 13, 13, 9, kp.color, 2);
        const name = assignment[i] === 'concentric' ? 'Concentric circles' : SHAPES[assignment[i]].label;
        li.append(icon);
        li.insertAdjacentHTML('beforeend', `<span class="fk-point">${kp.name}</span><span class="fk-shape">${name}</span>`);
        els.legend.append(li);
    });
    els.shuffles.textContent = shuffleCount;
}

let toastTimer = null;
function toast(text) {
    els.toast.textContent = text;
    els.toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => els.toast.classList.remove('show'), 900);
}

function shuffleNow() {
    shuffleCount += 1;
    reassignShapes();
    toast(`Shapes shuffled (#${shuffleCount})`);
}

els.shuffleBtn.addEventListener('click', shuffleNow);
els.confRange.addEventListener('input', () => {
    els.confVal.textContent = `${Math.round(els.confRange.value * 100)}%`;
    if (detector) detector.setOptions({ minDetectionConfidence: Number(els.confRange.value) });
});

const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta', 'AltGraph', 'CapsLock', 'Tab']);
document.addEventListener('keydown', (e) => {
    if (els.section.style.display === 'none') return;   // only on this tab
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat || MODIFIER_KEYS.has(e.key)) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    e.preventDefault();
    if (e.key === 'Escape' || e.key.toLowerCase() === 'q') {
        if (running) exitProgram();
        return;
    }
    shuffleNow();
});

els.sourcePills.forEach((btn) => btn.addEventListener('click', () => {
    if (running) stop();
    els.sourcePills.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    source = btn.dataset.fksource;
    els.fileSection.style.display = source === 'file' ? 'block' : 'none';
    els.mirror.checked = source === 'webcam';
    els.startLabel.textContent = source === 'webcam' ? 'Start Camera' : 'Play Video';
}));

els.startBtn.addEventListener('click', () => (running ? stop() : start()));
document.addEventListener('modechange', (e) => { if (e.detail.mode !== 'facekeys' && running) stop(); });

// ---------- camera / detector ----------
let source = 'webcam';
let detector = null;
let running = false;
let stream = null;
let rafId = null;
let lastVideoTime = -1;
let fpsFrames = 0, fpsWindowStart = 0;

async function getDetector() {
    if (detector) return detector;
    els.status.textContent = 'Loading face detection model...';
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const opts = (delegate) => ({
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        minDetectionConfidence: Number(els.confRange.value),
    });
    try {
        detector = await FaceDetector.createFromOptions(vision, opts('GPU'));
    } catch (err) {
        console.warn('GPU delegate failed, falling back to CPU', err);
        detector = await FaceDetector.createFromOptions(vision, opts('CPU'));
    }
    return detector;
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
        await getDetector();
        await video.play();

        running = true;
        lastVideoTime = -1;
        fpsFrames = 0; fpsWindowStart = 0;
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

// Esc / q: end the program (camera off, back to the start screen)
function exitProgram() {
    stop();
    els.stage.style.display = 'none';
    els.empty.style.display = 'block';
    els.emptyTitle.textContent = 'Exited';
    els.emptyText.innerHTML = 'The program was closed with <kbd>esc</kbd> / <kbd>q</kbd>. Click &ldquo;Start&rdquo; to run it again.';
    els.faceCount.textContent = '0';
    els.fps.textContent = '0 FPS';
}

function loop() {
    if (!running) return;
    const video = els.video;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const now = performance.now();
        const result = detector.detectForVideo(video, now);
        draw(result.detections || [], video.videoWidth, video.videoHeight);
        els.faceCount.textContent = (result.detections || []).length;

        fpsFrames += 1;
        if (!fpsWindowStart) fpsWindowStart = now;
        if (now - fpsWindowStart >= 1000) {
            els.fps.textContent = `${Math.round((fpsFrames * 1000) / (now - fpsWindowStart))} FPS`;
            fpsFrames = 0; fpsWindowStart = now;
        }
    }
    rafId = requestAnimationFrame(loop);
}

function draw(detections, w, h) {
    const canvas = els.canvas;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const mirror = els.mirror.checked;
    const mx = (x) => (mirror ? w - x : x);

    ctx.save();
    if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
    ctx.drawImage(els.video, 0, 0, w, h);
    ctx.restore();

    for (const det of detections) {
        const box = det.boundingBox;
        const kps = (det.keypoints || []).map((k) => [mx(k.x * w), k.y * h]);
        if (kps.length < 6) continue;

        // eye size: ~0.22 x distance between the eye centres (fallback: face box width)
        const eyeDist = Math.hypot(kps[0][0] - kps[1][0], kps[0][1] - kps[1][1]);
        const r = Math.max(6, eyeDist ? eyeDist * 0.22 : box.width * 0.08);
        const lw = Math.max(2, r * 0.16);

        if (els.showBox.checked && box) {
            const x0 = mirror ? w - box.originX - box.width : box.originX;
            ctx.lineWidth = Math.max(2, w / 500);
            ctx.strokeStyle = 'rgba(255,255,255,0.75)';
            ctx.setLineDash([10, 7]);
            ctx.strokeRect(x0, box.originY, box.width, box.height);
            ctx.setLineDash([]);
            const score = `Face ${Math.round((det.categories?.[0]?.score || 0) * 100)}%`;
            ctx.font = `600 ${Math.round(Math.max(14, w / 60))}px Inter, sans-serif`;
            const tw = ctx.measureText(score).width + 14;
            ctx.fillStyle = 'rgba(15,23,42,0.8)';
            ctx.fillRect(x0, box.originY - 28, tw, 26);
            ctx.fillStyle = '#fff';
            ctx.fillText(score, x0 + 7, box.originY - 9);
        }

        kps.forEach(([x, y], i) => drawShape(ctx, assignment[i], x, y, r, KEYPOINTS[i].color, lw));
    }
}

reassignShapes();
