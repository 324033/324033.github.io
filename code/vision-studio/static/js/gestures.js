// Hand Gestures: real-time two-hand gesture recognition with the MediaPipe
// Gesture Recognizer task (runs fully in the browser).
import { GestureRecognizer, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = '/models/gesture_recognizer.task';

const HAND_COLORS = { Left: '#00dcff', Right: '#ff8c00' };
const FINGER_COLORS = {
    thumb: '#ff4d6d', index: '#ffd60a', middle: '#4ade80', ring: '#38bdf8', pinky: '#c084fc', palm: '#e2e8f0',
};
// [start, end, finger] for the 21 hand landmarks
const HAND_CONNECTIONS = [
    [0, 1, 'thumb'], [1, 2, 'thumb'], [2, 3, 'thumb'], [3, 4, 'thumb'],
    [5, 6, 'index'], [6, 7, 'index'], [7, 8, 'index'],
    [9, 10, 'middle'], [10, 11, 'middle'], [11, 12, 'middle'],
    [13, 14, 'ring'], [14, 15, 'ring'], [15, 16, 'ring'],
    [17, 18, 'pinky'], [18, 19, 'pinky'], [19, 20, 'pinky'],
    [0, 5, 'palm'], [5, 9, 'palm'], [9, 13, 'palm'], [13, 17, 'palm'], [0, 17, 'palm'],
];
const fingerOfPoint = (i) => (i === 0 ? 'palm' : i <= 4 ? 'thumb' : i <= 8 ? 'index' : i <= 12 ? 'middle' : i <= 16 ? 'ring' : 'pinky');

const GESTURES = {
    None: { emoji: '⭕', label: 'None' },
    Closed_Fist: { emoji: '✊', label: 'Closed Fist' },
    Open_Palm: { emoji: '✋', label: 'Open Palm' },
    Pointing_Up: { emoji: '☝️', label: 'Pointing Up' },
    Thumb_Down: { emoji: '👎', label: 'Thumb Down' },
    Thumb_Up: { emoji: '👍', label: 'Thumb Up' },
    Victory: { emoji: '✌️', label: 'Victory' },
    ILoveYou: { emoji: '🤟', label: 'I Love You' },
};
const gestureInfo = (name) => GESTURES[name] || { emoji: '❓', label: name || 'None' };

// Named combinations when both hands are visible (order-independent)
const COMBOS = {
    'Thumb_Up+Thumb_Up': 'Double Thumbs Up 👍👍',
    'Thumb_Down+Thumb_Down': 'Double Thumbs Down 👎👎',
    'Victory+Victory': 'Double Victory ✌️✌️',
    'Open_Palm+Open_Palm': 'Both Hands Open / Stop ✋✋',
    'Closed_Fist+Closed_Fist': 'Both Fists ✊✊',
    'Pointing_Up+Pointing_Up': 'Both Pointing Up ☝️☝️',
    'ILoveYou+ILoveYou': 'Double I Love You 🤟🤟',
    'Thumb_Down+Thumb_Up': 'Thumbs Up + Down (so-so)',
    'Closed_Fist+Open_Palm': 'Fist + Palm (salute)',
};

const $ = (id) => document.getElementById(id);
const els = {
    startBtn: $('gesStartBtn'), startLabel: $('gesStartLabel'),
    sourcePills: document.querySelectorAll('#gesSourcePills .pill-btn'),
    fileSection: $('gesFileSection'), fileInput: $('gesFileInput'),
    scoreRange: $('gesScoreRange'), scoreVal: $('gesScoreVal'), colorMode: $('gesColorMode'),
    mirror: $('gesMirrorToggle'), showLabel: $('gesLabelToggle'),
    empty: $('gesEmptyState'), stage: $('gesStage'), canvas: $('gesCanvas'), status: $('gesStatus'),
    video: $('gesVideo'), handCount: $('gesHandCount'), combo: $('gesCombo'), fps: $('gesFps'),
};
const handEls = {};
for (const side of ['Left', 'Right']) {
    handEls[side] = {
        state: $(`ges${side}State`), emoji: $(`ges${side}Emoji`), name: $(`ges${side}Name`),
        bar: $(`ges${side}Bar`), topk: $(`ges${side}Topk`),
    };
}
const ctx = els.canvas.getContext('2d');

let source = 'webcam';
let recognizer = null;
let running = false;
let stream = null;
let rafId = null;
let lastVideoTime = -1;
let fpsFrames = 0;
let fpsWindowStart = 0;
let currentFps = 0;

els.scoreRange.addEventListener('input', () => {
    els.scoreVal.textContent = `${Math.round(els.scoreRange.value * 100)}%`;
});

els.sourcePills.forEach((btn) => btn.addEventListener('click', () => {
    if (running) stop();
    els.sourcePills.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    source = btn.dataset.gsource;
    els.fileSection.style.display = source === 'file' ? 'block' : 'none';
    els.mirror.checked = source === 'webcam';
    els.startLabel.textContent = source === 'webcam' ? 'Start Camera' : 'Play & Recognize';
}));

els.startBtn.addEventListener('click', () => (running ? stop() : start()));
document.addEventListener('modechange', (e) => { if (e.detail.mode !== 'gestures' && running) stop(); });

async function getRecognizer() {
    if (recognizer) return recognizer;
    els.status.textContent = 'Loading gesture model...';
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const opts = (delegate) => ({
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        cannedGesturesClassifierOptions: { scoreThreshold: 0 },  // threshold applied in the UI slider
    });
    try {
        recognizer = await GestureRecognizer.createFromOptions(vision, opts('GPU'));
    } catch (err) {
        console.warn('GPU delegate failed, falling back to CPU', err);
        recognizer = await GestureRecognizer.createFromOptions(vision, opts('CPU'));
    }
    return recognizer;
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
        } else {
            const file = els.fileInput.files?.[0];
            if (!file) throw new Error('Please choose a video file first.');
            video.srcObject = null;
            if (video.src) URL.revokeObjectURL(video.src);
            video.src = URL.createObjectURL(file);
        }
        video.loop = false;
        await new Promise((res, rej) => {
            if (video.readyState >= 2) return res();
            video.onloadeddata = () => res();
            video.onerror = () => rej(new Error('Video could not be loaded.'));
        });
        await getRecognizer();
        await video.play();
        video.onended = () => { stop(); els.status.style.display = 'block'; els.status.textContent = 'Video finished.'; };

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
    els.startLabel.textContent = source === 'webcam' ? 'Start Camera' : 'Play & Recognize';
}

/** Turns the raw task result into one entry per hand: {side, landmarks, gestures:[{name,score}], top}. */
function parseResult(result) {
    const minScore = Number(els.scoreRange.value);
    const handedness = result.handedness || result.handednesses || [];
    const hands = [];
    (result.landmarks || []).forEach((landmarks, i) => {
        // MediaPipe assumes a mirrored (selfie) image. We feed the raw, un-mirrored
        // camera frame, so the reported label is swapped -> flip it to get the person's real hand.
        const reported = handedness[i]?.[0]?.categoryName || 'Left';
        const side = reported === 'Left' ? 'Right' : 'Left';
        const gestures = (result.gestures?.[i] || [])
            .map((c) => ({ name: c.categoryName, score: c.score }))
            .sort((a, b) => b.score - a.score);
        let top = gestures[0] || { name: 'None', score: 0 };
        if (top.score < minScore) top = { name: 'None', score: top.score };
        hands.push({ side, landmarks, gestures, top });
    });
    // If both hands got the same label (rare mix-up), keep the more confident one and relabel the other
    if (hands.length === 2 && hands[0].side === hands[1].side) {
        hands[1].side = hands[0].side === 'Left' ? 'Right' : 'Left';
    }
    return hands;
}

function loop() {
    if (!running) return;
    const video = els.video;
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const now = performance.now();
        const result = recognizer.recognizeForVideo(video, now);
        // processed frames per second, measured over ~1 s windows
        fpsFrames += 1;
        if (!fpsWindowStart) fpsWindowStart = now;
        if (now - fpsWindowStart >= 1000) {
            currentFps = Math.round((fpsFrames * 1000) / (now - fpsWindowStart));
            fpsFrames = 0;
            fpsWindowStart = now;
        }

        const hands = parseResult(result);
        draw(hands, video.videoWidth, video.videoHeight);
        renderPanel(hands);
    }
    rafId = requestAnimationFrame(loop);
}

function draw(hands, w, h) {
    const canvas = els.canvas;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const mirror = els.mirror.checked;
    ctx.save();
    if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
    ctx.drawImage(els.video, 0, 0, w, h);
    ctx.restore();

    const scale = Math.max(0.7, w / 960);
    const byFinger = els.colorMode.value === 'finger';

    for (const hand of hands) {
        const handColor = HAND_COLORS[hand.side];
        const P = (i) => [(mirror ? 1 - hand.landmarks[i].x : hand.landmarks[i].x) * w, hand.landmarks[i].y * h];

        // connections
        ctx.lineCap = 'round';
        ctx.lineWidth = 4 * scale;
        for (const [a, b, finger] of HAND_CONNECTIONS) {
            ctx.strokeStyle = byFinger ? FINGER_COLORS[finger] : handColor;
            const [x1, y1] = P(a), [x2, y2] = P(b);
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        }
        // landmark points: filled with finger color, ringed with the hand color
        for (let i = 0; i < hand.landmarks.length; i++) {
            const [x, y] = P(i);
            const r = (i === 0 || i % 4 === 0 ? 6 : 4.5) * scale;
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = byFinger ? FINGER_COLORS[fingerOfPoint(i)] : '#ffffff';
            ctx.fill();
            ctx.lineWidth = 2 * scale; ctx.strokeStyle = handColor; ctx.stroke();
        }

        // bounding box + label in the hand color
        const xs = hand.landmarks.map((_, i) => P(i)[0]), ys = hand.landmarks.map((_, i) => P(i)[1]);
        const pad = 14 * scale;
        const x0 = Math.min(...xs) - pad, y0 = Math.min(...ys) - pad;
        const bw = Math.max(...xs) - Math.min(...xs) + 2 * pad, bh = Math.max(...ys) - Math.min(...ys) + 2 * pad;
        ctx.lineWidth = 2 * scale; ctx.strokeStyle = handColor;
        ctx.setLineDash([8 * scale, 6 * scale]); ctx.strokeRect(x0, y0, bw, bh); ctx.setLineDash([]);

        if (els.showLabel.checked) {
            const info = gestureInfo(hand.top.name);
            const text = `${hand.side}: ${info.emoji} ${info.label} ${Math.round(hand.top.score * 100)}%`;
            ctx.font = `600 ${Math.round(17 * scale)}px Inter, "Segoe UI Emoji", sans-serif`;
            const tw = ctx.measureText(text).width + 16 * scale, th = 28 * scale;
            const ly = Math.max(0, y0 - th - 4);
            ctx.fillStyle = handColor; ctx.fillRect(x0, ly, tw, th);
            ctx.fillStyle = '#0f172a'; ctx.textBaseline = 'middle';
            ctx.fillText(text, x0 + 8 * scale, ly + th / 2);
            ctx.textBaseline = 'alphabetic';
        }
    }
}

// Gesture history per hand: a gesture is logged once it was stable for STABLE_FRAMES frames
// (the Gesture Recognizer only returns the single best gesture per hand, so there is no top-k).
const STABLE_FRAMES = 6;
const HISTORY_LEN = 5;
const history = {
    Left: { candidate: null, count: 0, items: [] },
    Right: { candidate: null, count: 0, items: [] },
};

function updateHistory(side, name) {
    const hst = history[side];
    if (name === hst.candidate) hst.count += 1; else { hst.candidate = name; hst.count = 1; }
    if (hst.count !== STABLE_FRAMES || name === 'None' || hst.items[0]?.name === name) return;
    hst.items.unshift({ name, time: new Date().toLocaleTimeString() });
    hst.items.length = Math.min(hst.items.length, HISTORY_LEN);
    handEls[side].topk.innerHTML = hst.items
        .map((it) => `<li><span>${gestureInfo(it.name).emoji} ${gestureInfo(it.name).label}</span><span>${it.time}</span></li>`)
        .join('');
}

function renderPanel(hands) {
    els.handCount.textContent = hands.length;
    els.fps.textContent = `${currentFps} FPS`;

    for (const side of ['Left', 'Right']) {
        const ui = handEls[side];
        const hand = hands.find((h) => h.side === side);
        if (!hand) {
            ui.state.textContent = 'not visible'; ui.state.dataset.state = 'hidden';
            ui.emoji.textContent = '–'; ui.name.textContent = 'No hand';
            ui.bar.style.width = '0%';
            history[side].candidate = null; history[side].count = 0;
            continue;
        }
        const info = gestureInfo(hand.top.name);
        ui.state.textContent = 'detected'; ui.state.dataset.state = 'detected';
        ui.emoji.textContent = info.emoji;
        ui.name.textContent = `${info.label} · ${Math.round(hand.top.score * 100)}%`;
        ui.bar.style.width = `${Math.round(hand.top.score * 100)}%`;
        updateHistory(side, hand.top.name);
    }

    if (hands.length === 2) {
        const names = hands.map((h) => h.top.name);
        if (names.includes('None')) {
            els.combo.textContent = 'Both hands – no clear gesture';
        } else {
            const key = [...names].sort().join('+');
            const left = hands.find((h) => h.side === 'Left').top.name;
            const right = hands.find((h) => h.side === 'Right').top.name;
            els.combo.textContent = COMBOS[key] || `${gestureInfo(left).label} + ${gestureInfo(right).label}`;
        }
    } else {
        els.combo.textContent = hands.length === 1 ? 'Show your second hand' : '–';
    }
}
