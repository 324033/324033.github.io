// Fitness Coach: live arm rep counting with MediaPipe Pose (runs fully in the browser).
// One rep = arm straight -> bent -> straight. Left/right = the person's own arms.
import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = '/models/pose_landmarker.task';

const ARMS = { left: [11, 13, 15], right: [12, 14, 16] };   // shoulder, elbow, wrist
const ARM_COLORS = { left: '#00dcff', right: '#ff8c00' };
const MIN_VISIBILITY = 0.5;
// Body connections (no face points) for the skeleton
const BODY_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28], [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
];

// ---------- Rep counting (same logic as fitness.py) ----------
function elbowAngle(s, e, w) {
    const ax = s[0] - e[0], ay = s[1] - e[1], bx = w[0] - e[0], by = w[1] - e[1];
    const na = Math.hypot(ax, ay), nb = Math.hypot(bx, by);
    if (!na || !nb) return null;
    const cos = Math.max(-1, Math.min(1, (ax * bx + ay * by) / (na * nb)));
    return Math.acos(cos) * 180 / Math.PI;
}

class ArmRepCounter {
    constructor(extended = 150, bent = 60, smoothing = 0.5) {
        this.extended = extended;
        this.bent = bent;
        this.smoothing = smoothing;
        this.reset();
    }
    reset() {
        this.reps = 0;
        this.state = 'waiting';
        this.angle = null;
        this.visible = false;
        this.feedback = 'Straighten your arm to start';
        this.minSinceExtended = 180;
        this.maxSinceBent = 0;
    }
    update(raw) {
        if (raw == null) { this.visible = false; return false; }
        this.visible = true;
        this.angle = this.angle == null ? raw : this.smoothing * this.angle + (1 - this.smoothing) * raw;
        const a = this.angle;

        if (this.state === 'waiting') {
            if (a >= this.extended) { this.enterExtended(); this.feedback = 'Ready - bend your arm'; }
            return false;
        }
        if (this.state === 'extended') {
            this.minSinceExtended = Math.min(this.minSinceExtended, a);
            if (a <= this.bent) {
                this.state = 'bent';
                this.maxSinceBent = a;
                this.feedback = 'Good - now straighten fully';
            } else if (a >= this.extended && this.minSinceExtended < this.extended - 30) {
                this.feedback = `Not counted - bend further (below ${this.bent}°)`;
                this.minSinceExtended = a;
            }
            return false;
        }
        // bent
        this.maxSinceBent = Math.max(this.maxSinceBent, a);
        if (a >= this.extended) {
            this.reps += 1;
            this.enterExtended();
            this.feedback = `Rep ${this.reps}!`;
            return true;
        }
        if (a <= this.bent && this.maxSinceBent > this.bent + 40) {
            this.feedback = `Straighten fully (above ${this.extended}°) to count`;
            this.maxSinceBent = a;
        }
        return false;
    }
    enterExtended() { this.state = 'extended'; this.minSinceExtended = this.angle; }
}

// ---------- UI ----------
const $ = (id) => document.getElementById(id);
const els = {
    startBtn: $('fitStartBtn'), startLabel: $('fitStartLabel'), resetBtn: $('fitResetBtn'),
    sourcePills: document.querySelectorAll('#fitSourcePills .pill-btn'),
    fileSection: $('fitFileSection'), fileInput: $('fitFileInput'),
    extRange: $('fitExtendedRange'), extVal: $('fitExtendedVal'),
    bentRange: $('fitBentRange'), bentVal: $('fitBentVal'),
    mirror: $('fitMirrorToggle'), sound: $('fitSoundToggle'),
    empty: $('fitEmptyState'), stage: $('fitStage'), canvas: $('fitCanvas'), status: $('fitStatus'),
    video: $('fitVideo'),
    total: $('fitTotalReps'),
};
const armEls = {};
for (const side of ['left', 'right']) {
    const cap = side[0].toUpperCase() + side.slice(1);
    armEls[side] = {
        reps: $(`fit${cap}Reps`), state: $(`fit${cap}State`), angle: $(`fit${cap}Angle`),
        bar: $(`fit${cap}Bar`), feedback: $(`fit${cap}Feedback`),
    };
}
const ctx = els.canvas.getContext('2d');

let source = 'webcam';
let landmarker = null;
let running = false;
let stream = null;
let rafId = null;
let lastVideoTime = -1;
const counters = { left: new ArmRepCounter(), right: new ArmRepCounter() };

function applyThresholds() {
    let ext = Number(els.extRange.value), bent = Number(els.bentRange.value);
    if (bent > ext - 20) { bent = ext - 20; els.bentRange.value = bent; }
    els.extVal.textContent = `${ext}°`;
    els.bentVal.textContent = `${bent}°`;
    for (const c of Object.values(counters)) { c.extended = ext; c.bent = bent; }
}
els.extRange.addEventListener('input', applyThresholds);
els.bentRange.addEventListener('input', applyThresholds);
applyThresholds();

els.sourcePills.forEach((btn) => btn.addEventListener('click', () => {
    if (running) stop();
    els.sourcePills.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    source = btn.dataset.fsource;
    els.fileSection.style.display = source === 'file' ? 'block' : 'none';
    els.mirror.checked = source === 'webcam';
    els.startLabel.textContent = source === 'webcam' ? 'Start Camera' : 'Play & Count';
}));

els.resetBtn.addEventListener('click', () => {
    for (const c of Object.values(counters)) c.reset();
    renderStats();
});

els.startBtn.addEventListener('click', () => (running ? stop() : start()));

// stop camera when switching to another tab of the app
document.addEventListener('modechange', (e) => { if (e.detail.mode !== 'fitness' && running) stop(); });

let audioCtx = null;
function beep(side) {
    if (!els.sound.checked) return;
    try {
        audioCtx = audioCtx || new AudioContext();
        const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
        osc.frequency.value = side === 'left' ? 660 : 880;
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.18);
        osc.connect(gain).connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.2);
    } catch (_) { /* audio not available */ }
}

async function getLandmarker() {
    if (landmarker) return landmarker;
    els.status.textContent = 'Loading pose model...';
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const opts = (delegate) => ({
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });
    try {
        landmarker = await PoseLandmarker.createFromOptions(vision, opts('GPU'));
    } catch (err) {
        console.warn('GPU delegate failed, falling back to CPU', err);
        landmarker = await PoseLandmarker.createFromOptions(vision, opts('CPU'));
    }
    return landmarker;
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
            video.loop = false;
            for (const c of Object.values(counters)) c.reset();
        }
        await new Promise((res, rej) => {
            if (video.readyState >= 2) return res();
            video.onloadeddata = () => res();
            video.onerror = () => rej(new Error('Video could not be loaded.'));
        });
        await getLandmarker();
        await video.play();
        video.onended = () => { stop(); els.status.style.display = 'block'; els.status.textContent = 'Video finished - final counts shown above.'; };

        running = true;
        lastVideoTime = -1;
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
    els.startLabel.textContent = source === 'webcam' ? 'Start Camera' : 'Play & Count';
}

function loop() {
    if (!running) return;
    const video = els.video;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = landmarker.detectForVideo(video, performance.now());
        const person = result.landmarks?.[0] || null;
        const w = video.videoWidth, h = video.videoHeight;

        for (const [side, [s, e, wr]] of Object.entries(ARMS)) {
            let angle = null;
            if (person) {
                const pts = [person[s], person[e], person[wr]];
                if (Math.min(...pts.map((p) => p.visibility ?? 1)) >= MIN_VISIBILITY) {
                    angle = elbowAngle(...pts.map((p) => [p.x * w, p.y * h]));
                }
            }
            if (counters[side].update(angle)) beep(side);
        }
        draw(person, w, h);
        renderStats();
    }
    rafId = requestAnimationFrame(loop);
}

function draw(person, w, h) {
    const canvas = els.canvas;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const mirror = els.mirror.checked;
    ctx.save();
    if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
    ctx.drawImage(els.video, 0, 0, w, h);
    ctx.restore();
    if (!person) return;

    const scale = Math.max(0.7, w / 960);
    const P = (i) => [(mirror ? 1 - person[i].x : person[i].x) * w, person[i].y * h];
    const vis = (i) => (person[i].visibility ?? 1) >= MIN_VISIBILITY;

    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(220,220,220,0.8)';
    ctx.lineWidth = 2 * scale;
    for (const [a, b] of BODY_CONNECTIONS) {
        if (!vis(a) || !vis(b)) continue;
        const [x1, y1] = P(a), [x2, y2] = P(b);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }

    for (const [side, ids] of Object.entries(ARMS)) {
        const c = counters[side];
        if (!c.visible) continue;
        const [s, e, wr] = ids.map(P);
        ctx.strokeStyle = ARM_COLORS[side];
        ctx.lineWidth = 6 * scale;
        ctx.beginPath(); ctx.moveTo(...s); ctx.lineTo(...e); ctx.lineTo(...wr); ctx.stroke();
        for (const [x, y] of [s, e, wr]) {
            ctx.beginPath(); ctx.arc(x, y, 6 * scale, 0, Math.PI * 2);
            ctx.fillStyle = '#fff'; ctx.fill();
            ctx.lineWidth = 2; ctx.strokeStyle = ARM_COLORS[side]; ctx.stroke();
        }
        // angle label next to the elbow
        const label = `${Math.round(c.angle)}°`;
        ctx.font = `600 ${Math.round(18 * scale)}px Inter, sans-serif`;
        const tx = e[0] + 12 * scale, ty = e[1] - 12 * scale;
        ctx.lineWidth = 4; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.strokeText(label, tx, ty);
        ctx.fillStyle = ARM_COLORS[side]; ctx.fillText(label, tx, ty);
    }
}

function renderStats() {
    let total = 0;
    for (const side of ['left', 'right']) {
        const c = counters[side], ui = armEls[side];
        total += c.reps;
        ui.reps.textContent = c.reps;
        ui.state.textContent = c.visible ? c.state : 'not visible';
        ui.state.dataset.state = c.visible ? c.state : 'hidden';
        ui.angle.textContent = c.angle == null || !c.visible ? '–' : `${Math.round(c.angle)}°`;
        ui.bar.style.width = c.angle == null ? '0%' : `${Math.min(100, (c.angle / 180) * 100)}%`;
        ui.feedback.textContent = c.feedback;
    }
    els.total.textContent = total;
}
