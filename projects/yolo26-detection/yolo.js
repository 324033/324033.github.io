// YOLO26 Live Detection - Ultralytics YOLO26n (exported to ONNX) running in the browser
// with ONNX Runtime Web. Sources: live CCTV (MJPEG), YouTube clip, webcam, video/image file.
(() => {
    'use strict';

    const MODEL_URL = '../models/yolo26n.onnx';
    const INPUT = 640;
    const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    const CCTV_MIN_INTERVAL = 250;   // ms between detections on the MJPEG stream (it only sends ~1 fps)

    const COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];

    const PRESETS = {
        all: null,
        people: ['person'],
        vehicles: ['bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'boat'],
        sports: ['person', 'sports ball', 'frisbee', 'skis', 'snowboard', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket'],
        animals: ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    };

    const classColor = (c) => `hsl(${(c * 137.508) % 360}, 85%, 58%)`;

    // ---------- DOM ----------
    const $ = (id) => document.getElementById(id);
    const els = {
        dot: $('yStatusDot'), modelStatus: $('yModelStatus'),
        pills: document.querySelectorAll('#ySourcePills .pill-btn'), panels: document.querySelectorAll('.y-source-panel'),
        cctvUrl: $('yCctvUrl'), cctvBtn: $('yCctvBtn'),
        ytUrl: $('yYtUrl'), ytDuration: $('yYtDuration'), ytBtn: $('yYtBtn'), ytSamples: $('yYtSamples'),
        camBtn: $('yCamBtn'), fileInput: $('yFileInput'),
        conf: $('yConfRange'), confVal: $('yConfVal'), preset: $('yPreset'), labels: $('yLabels'), mirror: $('yMirror'),
        pauseBtn: $('yPauseBtn'), snapBtn: $('ySnapBtn'), resetBtn: $('yResetBtn'),
        objCount: $('yObjCount'), fps: $('yFps'), infer: $('yInfer'), backend: $('yBackend'),
        viewer: $('yViewer'), empty: $('yEmpty'), stage: $('yStage'), canvas: $('yCanvas'), status: $('yStatus'),
        sourceLabel: $('ySourceLabel'), video: $('yVideo'), img: $('yImg'),
        counts: $('yCounts'), chart: $('yChart'),
    };
    const ctx = els.canvas.getContext('2d');
    const prepCanvas = document.createElement('canvas');
    prepCanvas.width = prepCanvas.height = INPUT;
    const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });

    // ---------- state ----------
    let session = null, inputName = null, backendName = '';
    let sourceType = null;          // 'cctv' | 'video' | 'webcam' | 'image'
    let sourceEl = null;            // element frames are read from
    let stream = null;
    let running = false, paused = false, busy = false;
    let lastVideoTime = -1, lastCctvRun = 0;
    let detections = [];
    let fpsCount = 0, fpsStart = 0;
    const hidden = new Set();       // classes switched off by clicking
    const peaks = {};
    const history = [];             // {t, n}
    let ytPoll = null;
    let generation = 0;             // bumped on every source change

    // ---------- model ----------
    async function loadModel() {
        ort.env.wasm.wasmPaths = ORT_DIST;
        ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
        const attempts = [];
        if (navigator.gpu) attempts.push('webgpu');
        attempts.push('wasm');
        for (const ep of attempts) {
            try {
                session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [ep], graphOptimizationLevel: 'all' });
                backendName = ep === 'webgpu' ? 'WebGPU' : 'WASM (CPU)';
                break;
            } catch (err) {
                console.warn(`ONNX Runtime: ${ep} failed`, err);
            }
        }
        if (!session) throw new Error('Could not load models/yolo26n.onnx (run export_yolo.py once).');
        inputName = session.inputNames[0];
        // warm-up run so the first real frame is fast
        await session.run({ [inputName]: new ort.Tensor('float32', new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]) });
        els.backend.textContent = backendName;
    }

    // letterbox the frame into 640x640 (grey padding, like Ultralytics) -> NCHW float32 0..1
    function preprocess(src, w, h) {
        const scale = Math.min(INPUT / w, INPUT / h);
        const nw = Math.round(w * scale), nh = Math.round(h * scale);
        const padX = Math.floor((INPUT - nw) / 2), padY = Math.floor((INPUT - nh) / 2);
        prepCtx.fillStyle = 'rgb(114,114,114)';
        prepCtx.fillRect(0, 0, INPUT, INPUT);
        prepCtx.drawImage(src, padX, padY, nw, nh);
        const px = prepCtx.getImageData(0, 0, INPUT, INPUT).data;
        const area = INPUT * INPUT;
        const data = new Float32Array(3 * area);
        for (let i = 0, j = 0; i < area; i++, j += 4) {
            data[i] = px[j] / 255;
            data[i + area] = px[j + 1] / 255;
            data[i + 2 * area] = px[j + 2] / 255;
        }
        return { tensor: new ort.Tensor('float32', data, [1, 3, INPUT, INPUT]), scale, padX, padY };
    }

    function iou(a, b) {
        const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-6);
    }

    function postprocess(output, meta, w, h) {
        const minConf = Number(els.conf.value);
        const d = output.dims, v = output.data;
        const unmap = (x, y) => [(x - meta.padX) / meta.scale, (y - meta.padY) / meta.scale];
        let boxes = [];
        if (d.length === 3 && d[2] === 6) {
            // YOLO26 end-to-end output: [1, 300, 6] = x1, y1, x2, y2, score, class (already NMS-free)
            for (let i = 0; i < d[1]; i++) {
                const o = i * 6, score = v[o + 4];
                if (score < minConf) continue;
                const [x1, y1] = unmap(v[o], v[o + 1]);
                const [x2, y2] = unmap(v[o + 2], v[o + 3]);
                boxes.push({ x1, y1, x2, y2, score, cls: Math.round(v[o + 5]) });
            }
        } else {
            // classic YOLO output [1, 84, N] (cx, cy, w, h, 80 class scores) -> needs NMS
            const n = d[2], nc = d[1] - 4;
            for (let i = 0; i < n; i++) {
                let best = 0, cls = 0;
                for (let c = 0; c < nc; c++) { const s = v[(4 + c) * n + i]; if (s > best) { best = s; cls = c; } }
                if (best < minConf) continue;
                const cx = v[i], cy = v[n + i], bw = v[2 * n + i], bh = v[3 * n + i];
                const [x1, y1] = unmap(cx - bw / 2, cy - bh / 2);
                const [x2, y2] = unmap(cx + bw / 2, cy + bh / 2);
                boxes.push({ x1, y1, x2, y2, score: best, cls });
            }
            boxes.sort((a, b) => b.score - a.score);
            const kept = [];
            for (const b of boxes) if (!kept.some((k) => k.cls === b.cls && iou(k, b) > 0.45)) kept.push(b);
            boxes = kept;
        }
        return boxes.map((b) => ({
            ...b, x1: Math.max(0, b.x1), y1: Math.max(0, b.y1), x2: Math.min(w, b.x2), y2: Math.min(h, b.y2),
        }));
    }

    // ONNX Runtime must not run two inferences on one session at the same time
    // (e.g. a video frame still running while an image is dropped) -> queue them.
    let runQueue = Promise.resolve();
    function runSession(feeds) {
        const p = runQueue.then(() => session.run(feeds));
        runQueue = p.catch(() => {});
        return p;
    }

    async function detect(src, w, h) {
        const meta = preprocess(src, w, h);
        const t0 = performance.now();
        const out = await runSession({ [inputName]: meta.tensor });
        const ms = performance.now() - t0;
        const result = postprocess(out[session.outputNames[0]], meta, w, h);
        els.infer.textContent = `${Math.round(ms)} ms`;
        return result;
    }

    // ---------- filtering & stats ----------
    function visible(det) {
        const name = COCO[det.cls];
        const preset = PRESETS[els.preset.value];
        return (!preset || preset.includes(name)) && !hidden.has(name);
    }

    function updateStats(dets) {
        const counts = {};
        for (const d of dets) {
            const name = COCO[d.cls] || `class ${d.cls}`;
            counts[name] = (counts[name] || 0) + 1;
        }
        const shown = dets.filter(visible);
        els.objCount.textContent = shown.length;
        for (const [k, n] of Object.entries(counts)) peaks[k] = Math.max(peaks[k] || 0, n);

        const now = performance.now();
        history.push({ t: now, n: shown.length });
        while (history.length && now - history[0].t > 60000) history.shift();

        renderCounts(counts);
        drawChart();
    }

    function renderCounts(counts) {
        const names = Object.keys(peaks).sort((a, b) => (counts[b] || 0) - (counts[a] || 0) || peaks[b] - peaks[a]);
        if (!names.length) { els.counts.innerHTML = '<p class="param-hint">No detections yet.</p>'; return; }
        const max = Math.max(1, ...names.map((n) => peaks[n]));
        const preset = PRESETS[els.preset.value];
        els.counts.innerHTML = names.map((name) => {
            const c = COCO.indexOf(name), now = counts[name] || 0;
            const off = hidden.has(name) || (preset && !preset.includes(name));
            return `<button type="button" class="y-count-row${off ? ' off' : ''}" data-name="${name}" style="--c:${classColor(c)}"
                     title="Click to ${hidden.has(name) ? 'show' : 'hide'} ${name}">
                <span class="sw"></span><span>${name}</span>
                <span class="bar"><i style="width:${(now / max) * 100}%"></i></span>
                <span class="num"><b>${now}</b> &middot; ${peaks[name]}</span></button>`;
        }).join('');
    }

    els.counts.addEventListener('click', (e) => {
        const row = e.target.closest('.y-count-row');
        if (!row) return;
        const name = row.dataset.name;
        if (hidden.has(name)) hidden.delete(name); else hidden.add(name);
        redrawCurrent();
        updateStats(detections);
    });

    function drawChart() {
        const c = els.chart, g = c.getContext('2d');
        const w = c.clientWidth || 300, h = c.clientHeight || 140;
        if (c.width !== w * devicePixelRatio) { c.width = w * devicePixelRatio; c.height = h * devicePixelRatio; }
        g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        g.clearRect(0, 0, w, h);
        const now = performance.now();
        const maxN = Math.max(4, ...history.map((p) => p.n));
        g.strokeStyle = 'rgba(148,163,184,0.15)'; g.lineWidth = 1;
        g.fillStyle = 'rgba(148,163,184,0.7)'; g.font = '11px Inter, sans-serif';
        for (const f of [0, 0.5, 1]) {
            const y = 8 + (h - 24) * (1 - f);
            g.beginPath(); g.moveTo(28, y); g.lineTo(w, y); g.stroke();
            g.fillText(String(Math.round(maxN * f)), 0, y + 4);
        }
        g.fillText('-60 s', 28, h - 2); g.fillText('now', w - 24, h - 2);
        if (history.length < 2) return;
        const X = (t) => 28 + (w - 28) * (1 - (now - t) / 60000);
        const Y = (n) => 8 + (h - 24) * (1 - n / maxN);
        g.beginPath();
        history.forEach((p, i) => (i ? g.lineTo(X(p.t), Y(p.n)) : g.moveTo(X(p.t), Y(p.n))));
        g.strokeStyle = '#f59e0b'; g.lineWidth = 2; g.stroke();
        g.lineTo(X(history[history.length - 1].t), Y(0)); g.lineTo(X(history[0].t), Y(0)); g.closePath();
        g.fillStyle = 'rgba(245,158,11,0.15)'; g.fill();
    }

    // ---------- drawing ----------
    function drawFrame(src, w, h) {
        if (els.canvas.width !== w || els.canvas.height !== h) { els.canvas.width = w; els.canvas.height = h; }
        const mirror = sourceType === 'webcam' && els.mirror.checked;
        ctx.save();
        if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
        ctx.drawImage(src, 0, 0, w, h);
        ctx.restore();

        const lw = Math.max(2, w / 400), fs = Math.round(Math.max(12, w / 70));
        ctx.font = `600 ${fs}px Inter, sans-serif`;
        for (const d of detections) {
            if (!visible(d)) continue;
            const x1 = mirror ? w - d.x2 : d.x1, x2 = mirror ? w - d.x1 : d.x2;
            const color = classColor(d.cls);
            ctx.lineWidth = lw; ctx.strokeStyle = color;
            ctx.strokeRect(x1, d.y1, x2 - x1, d.y2 - d.y1);
            if (els.labels.checked) {
                const text = `${COCO[d.cls] || d.cls} ${Math.round(d.score * 100)}%`;
                const tw = ctx.measureText(text).width + 10, th = fs + 8;
                const ly = d.y1 - th < 0 ? d.y1 : d.y1 - th;
                ctx.fillStyle = color; ctx.fillRect(x1 - lw / 2, ly, tw, th);
                ctx.fillStyle = '#0b1120'; ctx.fillText(text, x1 + 4, ly + fs + 1);
            }
        }
    }

    function frameSize() {
        if (sourceType === 'cctv' || sourceType === 'image') return [els.img.naturalWidth, els.img.naturalHeight];
        return [els.video.videoWidth, els.video.videoHeight];
    }

    function redrawCurrent() {
        if (!sourceEl) return;
        const [w, h] = frameSize();
        if (w && h) drawFrame(sourceEl, w, h);
    }

    // ---------- main loop ----------
    function loop() {
        if (!running) return;
        const [w, h] = frameSize();
        if (w && h && !paused) {
            let doDetect = false;
            if (sourceType === 'cctv') {
                doDetect = performance.now() - lastCctvRun >= CCTV_MIN_INTERVAL;
            } else if (els.video.currentTime !== lastVideoTime) {
                doDetect = true;
            }
            drawFrame(sourceEl, w, h);
            if (doDetect && !busy) {
                busy = true;
                lastVideoTime = els.video.currentTime;
                lastCctvRun = performance.now();
                const gen = generation;
                detect(sourceEl, w, h)
                    .then((dets) => { if (gen !== generation) return; detections = dets; updateStats(dets); countFps(); })
                    .catch((err) => { console.error(err); showStatus(`Detection error: ${err.message}`); })
                    .finally(() => { busy = false; });
            }
        }
        requestAnimationFrame(loop);
    }

    function countFps() {
        const now = performance.now();
        fpsCount += 1;
        if (!fpsStart) fpsStart = now;
        if (now - fpsStart >= 1000) {
            els.fps.textContent = (fpsCount * 1000 / (now - fpsStart)).toFixed(1);
            fpsCount = 0; fpsStart = now;
        }
    }

    // ---------- sources ----------
    function stopSource() {
        generation += 1;          // results of a detection still running for the old source are dropped
        running = false; paused = false; busy = false;
        if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
        els.video.pause();
        els.video.removeAttribute('src'); els.video.srcObject = null; els.video.load();
        els.img.onload = els.img.onerror = null;
        els.img.removeAttribute('src');          // closes the MJPEG connection
        if (ytPoll) { clearInterval(ytPoll); ytPoll = null; }
        detections = [];
        els.pauseBtn.disabled = true; els.snapBtn.disabled = true;
        els.pauseBtn.innerHTML = '&#x23F8; Pause';
    }

    function showStage(label) {
        els.empty.hidden = true; els.empty.style.display = 'none';
        els.stage.hidden = false;
        els.sourceLabel.textContent = label;
    }

    function showStatus(text) {
        els.status.hidden = !text;
        els.status.textContent = text || '';
    }

    function startLoop(type, el, label) {
        sourceType = type; sourceEl = el;
        running = true; paused = false; busy = false;
        lastVideoTime = -1; lastCctvRun = 0; fpsCount = 0; fpsStart = 0;
        els.pauseBtn.disabled = type === 'image'; els.snapBtn.disabled = false;
        showStage(label); showStatus('');
        requestAnimationFrame(loop);
    }

    async function ensureModel() {
        if (session) return;
        showStatus('Loading YOLO26n model...');
        await modelReady;
        if (!session) throw new Error('Model could not be loaded.');
    }

    async function startCctv() {
        stopSource();
        const url = els.cctvUrl.value.trim();
        if (!/^https?:\/\//i.test(url)) { alert('Please enter a valid stream URL (http/https).'); return; }
        showStage('CCTV LIVE');
        try {
            await ensureModel();
            showStatus('Connecting to the CCTV stream...');
            els.img.crossOrigin = 'anonymous';    // the stream sends CORS headers -> pixels are readable
            await new Promise((resolve, reject) => {
                const timer = setTimeout(() => reject(new Error('The camera did not answer within 20 s.')), 20000);
                els.img.onload = () => { clearTimeout(timer); resolve(); };
                els.img.onerror = () => { clearTimeout(timer); reject(new Error('The stream could not be loaded (camera offline or blocked).')); };
                els.img.src = url;
            });
            startLoop('cctv', els.img, 'CCTV LIVE');
        } catch (err) {
            showStatus(err.message);
        }
    }

    async function playVideo(src, label, { isStream = false } = {}) {
        const v = els.video;
        if (isStream) { v.srcObject = src; } else { v.src = src; }
        v.loop = !isStream;
        await new Promise((resolve, reject) => {
            if (v.readyState >= 2) return resolve();
            v.onloadeddata = () => resolve();
            v.onerror = () => reject(new Error('The video could not be played.'));
        });
        await v.play();
        startLoop(isStream ? 'webcam' : 'video', v, label);
    }

    async function startWebcam() {
        stopSource();
        showStage('WEBCAM');
        try {
            await ensureModel();
            if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not support camera access.');
            showStatus('Waiting for camera permission...');
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
            els.mirror.checked = true;
            await playVideo(stream, 'WEBCAM', { isStream: true });
        } catch (err) {
            showStatus(err.name === 'NotAllowedError' ? 'Camera access was denied.' : err.message);
        }
    }

    async function startYoutube(url) {
        stopSource();
        url = (url || els.ytUrl.value).trim();
        if (!url) { alert('Please paste a YouTube link first.'); return; }
        showStage('YOUTUBE');
        els.ytBtn.disabled = true;
        try {
            await ensureModel();
            if (location.hostname.endsWith('github.io')) throw new Error('YouTube clips only work with the local Python server. Online, please use CCTV, webcam or a video file.');
            showStatus('Requesting the video...');
            const res = await fetch('/api/yolo/youtube', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, duration: Number(els.ytDuration.value) }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Could not start the download.');
            const job = await new Promise((resolve, reject) => {
                ytPoll = setInterval(async () => {
                    try {
                        const s = await (await fetch(`/api/yolo/youtube/${data.job_id}`)).json();
                        showStatus(s.step || 'Working...');
                        if (s.status === 'completed') { clearInterval(ytPoll); ytPoll = null; resolve(s); }
                        if (s.status === 'failed') { clearInterval(ytPoll); ytPoll = null; reject(new Error(s.error || 'Download failed.')); }
                    } catch (e) { /* keep polling */ }
                }, 800);
            });
            await playVideo(job.video_url, 'YOUTUBE');
        } catch (err) {
            showStatus(err.message);
        } finally {
            els.ytBtn.disabled = false;
        }
    }

    async function startFile(file) {
        if (!file) return;
        stopSource();
        const url = URL.createObjectURL(file);
        try {
            await ensureModel();
            if (file.type.startsWith('image/')) {
                showStage('IMAGE');
                els.img.removeAttribute('crossorigin');
                await new Promise((resolve, reject) => { els.img.onload = resolve; els.img.onerror = () => reject(new Error('Image could not be loaded.')); els.img.src = url; });
                sourceType = 'image'; sourceEl = els.img;
                detections = await detect(els.img, els.img.naturalWidth, els.img.naturalHeight);
                updateStats(detections);
                redrawCurrent();
                els.snapBtn.disabled = false;
                showStatus('');
            } else {
                showStage('VIDEO FILE');
                await playVideo(url, 'VIDEO FILE');
            }
        } catch (err) {
            showStage('FILE');
            showStatus(err.message);
        }
    }

    // ---------- UI wiring ----------
    els.pills.forEach((btn) => btn.addEventListener('click', () => {
        els.pills.forEach((b) => b.classList.toggle('active', b === btn));
        els.panels.forEach((p) => { p.hidden = p.dataset.panel !== btn.dataset.ysource; });
    }));
    els.cctvBtn.addEventListener('click', startCctv);
    els.ytBtn.addEventListener('click', () => startYoutube());
    els.ytSamples.querySelectorAll('[data-url]').forEach((b) => b.addEventListener('click', () => { els.ytUrl.value = b.dataset.url; }));
    els.camBtn.addEventListener('click', startWebcam);
    els.fileInput.addEventListener('change', () => startFile(els.fileInput.files?.[0]));

    ['dragenter', 'dragover'].forEach((ev) => els.viewer.addEventListener(ev, (e) => { e.preventDefault(); els.viewer.classList.add('drag-active'); }));
    ['dragleave', 'drop'].forEach((ev) => els.viewer.addEventListener(ev, (e) => { e.preventDefault(); els.viewer.classList.remove('drag-active'); }));
    els.viewer.addEventListener('drop', (e) => startFile(e.dataTransfer.files?.[0]));

    els.conf.addEventListener('input', async () => {
        els.confVal.textContent = `${Math.round(els.conf.value * 100)}%`;
        if (sourceType === 'image' && session) {   // still image: re-run with the new threshold
            detections = await detect(els.img, els.img.naturalWidth, els.img.naturalHeight);
            updateStats(detections); redrawCurrent();
        }
    });
    els.preset.addEventListener('change', () => { redrawCurrent(); updateStats(detections); });
    els.labels.addEventListener('change', redrawCurrent);
    els.mirror.addEventListener('change', redrawCurrent);

    els.pauseBtn.addEventListener('click', () => {
        paused = !paused;
        if (sourceType !== 'cctv') { if (paused) els.video.pause(); else els.video.play(); }
        els.pauseBtn.innerHTML = paused ? '&#x25B6; Resume' : '&#x23F8; Pause';
    });

    els.snapBtn.addEventListener('click', () => {
        els.canvas.toBlob((blob) => {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `yolo26_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 2000);
        }, 'image/png');
    });

    els.resetBtn.addEventListener('click', () => {
        for (const k of Object.keys(peaks)) delete peaks[k];
        history.length = 0; hidden.clear();
        updateStats(detections);
    });

    window.addEventListener('resize', drawChart);

    // ---------- start: load the model right away ----------
    const modelReady = loadModel()
        .then(() => {
            els.modelStatus.textContent = `YOLO26n ready · ${backendName}`;
        })
        .catch((err) => {
            console.error(err);
            els.modelStatus.textContent = 'Model failed to load';
            els.dot.style.background = 'var(--danger)';
            showStatus(err.message);
        });
    drawChart();
})();
