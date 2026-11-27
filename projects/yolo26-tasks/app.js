// YOLO26 Webcam Tasks - Ultralytics YOLO26 nano models (ONNX) running in the browser with
// ONNX Runtime Web. Tasks: detect, segment, semantic, depth, classify, pose, obb.
// WASM (CPU) by default, optional WebGPU with automatic fallback. Models load lazily.
/* global ort */

// absolute URL: ONNX Runtime would otherwise resolve a relative path against its own script folder
const ORT_PATH = new URL('./vendor/onnxruntime-web/', window.location.href).href;
const MANIFEST_URL = './models/manifest.json';

const TASK_INFO = {
    detect: { label: 'Detection', hint: 'Boxes for 80 everyday object classes (COCO).' },
    segment: { label: 'Instance segmentation', hint: 'A coloured mask for every detected object (COCO).' },
    semantic: { label: 'Semantic segmentation', hint: 'Every pixel gets a class. Trained on Cityscapes (street scenes), so indoors only "person" and a few classes make sense.' },
    depth: { label: 'Depth estimation', hint: 'Estimated distance per pixel: warm = near, cool = far.' },
    classify: { label: 'Classification', hint: 'Top-5 ImageNet classes for the whole picture.' },
    pose: { label: 'Pose estimation', hint: '17 body keypoints and skeleton for every person.' },
    obb: { label: 'Oriented boxes (OBB)', hint: 'Rotated boxes, trained on aerial images (DOTA: planes, ships, vehicles...). Point the camera at a top-down photo to try it.' },
};

const COCO_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]];
const CITYSCAPES_COLORS = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
    [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]];

const hsl = (i) => `hsl(${(i * 137.508) % 360}, 85%, 58%)`;
function hslRgb(i) {
    const h = ((i * 137.508) % 360) / 360, s = 0.85, l = 0.58;
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
    const f = (t) => { t = (t + 1) % 1; return t < 1 / 6 ? p + (q - p) * 6 * t : t < 0.5 ? q : t < 2 / 3 ? p + (q - p) * (2 / 3 - t) * 6 : p; };
    return [f(h + 1 / 3), f(h), f(h - 1 / 3)].map((v) => Math.round(v * 255));
}

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const el = {
    task: $('taskSelect'), taskHint: $('taskHint'), camBtn: $('camBtn'), gpu: $('gpuToggle'), every: $('everySelect'),
    conf: $('confRange'), confVal: $('confVal'), alpha: $('alphaRange'), alphaVal: $('alphaVal'),
    mirror: $('mirrorToggle'), labels: $('labelToggle'), sysInfo: $('sysInfo'),
    view: $('view'), placeholder: $('placeholder'), loading: $('loading'), loadingText: $('loadingText'),
    resultBox: $('resultBox'), error: $('errorBox'), legend: $('legend'), video: $('video'),
    stTask: $('stTask'), stBackend: $('stBackend'), stFps: $('stFps'), stModelFps: $('stModelFps'), stInfer: $('stInfer'),
};
const ctx = el.view.getContext('2d');
const prep = document.createElement('canvas');
const prepCtx = prep.getContext('2d', { willReadFrequently: true });
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');

// ---------- state ----------
let manifest = null;
let stream = null;
let running = false;
let busy = false;
let frameNo = 0;
let currentTask = el.task.value;
let result = null;               // { task, ...decoded data, meta }
const sessions = new Map();      // task -> Promise<{session, backend}>
let backendLabel = 'WASM (CPU)';
let videoFrames = 0, modelRuns = 0, fpsStart = performance.now();

// ---------- helpers ----------
function showError(html) { el.error.innerHTML = html; el.error.hidden = !html; }
function setLoading(text) { el.loading.hidden = !text; el.loadingText.textContent = text || ''; }

function checkBrowser() {
    const problems = [];
    if (typeof WebAssembly !== 'object') problems.push('WebAssembly is not supported.');
    if (!navigator.mediaDevices?.getUserMedia) problems.push('Camera access (getUserMedia) is not available. Open the page via http://localhost or https.');
    if (typeof ort === 'undefined') problems.push('ONNX Runtime Web could not be loaded from ./vendor/.');
    return problems;
}

function updateSysInfo() {
    const threads = ort?.env?.wasm?.numThreads ?? 1;
    el.sysInfo.innerHTML = [
        `Cross-origin isolated: <b>${self.crossOriginIsolated ? 'yes' : 'no'}</b> (WASM threads: ${threads})`,
        `WebGPU available: <b>${navigator.gpu ? 'yes' : 'no'}</b>`,
        'All processing stays on this computer.',
    ].join('<br>');
}

// ---------- model loading ----------
async function loadManifest() {
    if (manifest) return manifest;
    const res = await fetch(MANIFEST_URL);
    if (!res.ok) throw new Error('models/manifest.json is missing - run export_models.py first.');
    manifest = await res.json();
    return manifest;
}

async function createSession(task) {
    const m = (await loadManifest())[task];
    if (!m) throw new Error(`No model for "${task}" in manifest.json.`);
    const res = await fetch(m.file);
    if (!res.ok) throw new Error(`${m.file} not found (HTTP ${res.status}).`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    const opts = (ep) => ({ executionProviders: [ep], graphOptimizationLevel: 'all' });
    if (el.gpu.checked) {
        if (navigator.gpu) {
            try {
                const session = await ort.InferenceSession.create(bytes, opts('webgpu'));
                return { session, backend: 'WebGPU', meta: m };
            } catch (err) {
                console.warn('WebGPU failed, falling back to WASM', err);
                const session = await ort.InferenceSession.create(bytes, opts('wasm'));
                return { session, backend: 'WASM (WebGPU failed)', meta: m };
            }
        }
        const session = await ort.InferenceSession.create(bytes, opts('wasm'));
        return { session, backend: 'WASM (no WebGPU in this browser)', meta: m };
    }
    const session = await ort.InferenceSession.create(bytes, opts('wasm'));
    return { session, backend: 'WASM (CPU)', meta: m };
}

function getSession(task) {
    if (!sessions.has(task)) {
        const p = createSession(task);
        sessions.set(task, p);
        p.catch(() => sessions.delete(task));      // allow a retry later
    }
    return sessions.get(task);
}

let selectToken = 0;
async function selectTask(task) {
    const token = ++selectToken;           // only the most recent request may update the UI
    currentTask = task;
    result = null;
    el.stTask.textContent = TASK_INFO[task].label;
    el.taskHint.textContent = TASK_INFO[task].hint;
    el.resultBox.hidden = true;
    renderLegend(null);
    showError('');
    setLoading(`Loading ${TASK_INFO[task].label} model...`);
    try {
        const s = await getSession(task);
        if (token !== selectToken) return;
        backendLabel = s.backend;
        el.stBackend.textContent = backendLabel;
        setLoading('');
        renderLegend(task);
    } catch (err) {
        console.error(err);
        if (token !== selectToken) return;
        setLoading('');
        showError(`<strong>Model failed to load.</strong> ${err.message}`);
    }
}

async function releaseSessions() {
    const all = [...sessions.values()];
    sessions.clear();
    for (const p of all) { try { (await p).session.release?.(); } catch (_) { /* ignore */ } }
}

// ---------- preprocessing ----------
function letterbox(src, w, h, size) {
    const scale = Math.min(size / w, size / h);
    const nw = Math.round(w * scale), nh = Math.round(h * scale);
    const padX = Math.floor((size - nw) / 2), padY = Math.floor((size - nh) / 2);
    if (prep.width !== size) { prep.width = size; prep.height = size; }
    prepCtx.fillStyle = 'rgb(114,114,114)';
    prepCtx.fillRect(0, 0, size, size);
    prepCtx.drawImage(src, padX, padY, nw, nh);
    return { tensor: toTensor(size), scale, padX, padY, size, nw, nh };
}

// depth models are trained with Ultralytics "scale_fill" = plain stretch, no padding
function stretch(src, w, h, size) {
    if (prep.width !== size) { prep.width = size; prep.height = size; }
    prepCtx.drawImage(src, 0, 0, size, size);
    return { tensor: toTensor(size), scale: 1, padX: 0, padY: 0, size, nw: size, nh: size, stretch: true };
}

function preprocessFor(task, src, w, h, size) {
    if (task === 'classify') return centerCrop(src, w, h, size);
    if (task === 'depth') return stretch(src, w, h, size);
    return letterbox(src, w, h, size);
}

function centerCrop(src, w, h, size) {      // classification: shortest side -> size, centre crop
    const s = Math.min(w, h);
    if (prep.width !== size) { prep.width = size; prep.height = size; }
    prepCtx.drawImage(src, (w - s) / 2, (h - s) / 2, s, s, 0, 0, size, size);
    return { tensor: toTensor(size), size };
}

function toTensor(size) {
    const px = prepCtx.getImageData(0, 0, size, size).data;
    const area = size * size, data = new Float32Array(3 * area);
    for (let i = 0, j = 0; i < area; i++, j += 4) {
        data[i] = px[j] / 255; data[i + area] = px[j + 1] / 255; data[i + 2 * area] = px[j + 2] / 255;
    }
    return new ort.Tensor('float32', data, [1, 3, size, size]);
}

// ---------- generic box helpers ----------
function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-6);
}

function nms(boxes, thr = 0.45, overlap = iou, maxDet = 100) {
    boxes.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const b of boxes) {
        if (kept.length >= maxDet) break;
        if (!kept.some((k) => k.cls === b.cls && overlap(k, b) > thr)) kept.push(b);
    }
    return kept;
}

/**
 * Reads candidate rows from a YOLO head output.
 * Raw layout  [1, C, N]: channel-major (4 box + nc scores + extra).
 * End-to-end  [1, N, C]: row-major (x1,y1,x2,y2,score,cls,extra...).
 * Returns rows {cx,cy,w,h | x1..y2, score, cls, extra[]} above the threshold.
 */
function readRows(out, nc, extraLen, minConf) {
    const d = out.dims, v = out.data, rows = [];
    const e2e = d[2] === 6 + extraLen || (d[2] < d[1] && d[2] <= 64);
    if (e2e) {
        const n = d[1], c = d[2];
        for (let i = 0; i < n; i++) {
            const o = i * c, score = v[o + 4];
            if (score < minConf) continue;
            rows.push({ x1: v[o], y1: v[o + 1], x2: v[o + 2], y2: v[o + 3], score, cls: Math.round(v[o + 5]),
                extra: Array.from(v.subarray(o + 6, o + 6 + extraLen)), e2e: true });
        }
        return { rows, e2e: true };
    }
    const n = d[2];
    for (let i = 0; i < n; i++) {
        let best = 0, cls = 0;
        for (let k = 0; k < nc; k++) { const s = v[(4 + k) * n + i]; if (s > best) { best = s; cls = k; } }
        if (best < minConf) continue;
        const extra = new Array(extraLen);
        for (let k = 0; k < extraLen; k++) extra[k] = v[(4 + nc + k) * n + i];
        rows.push({ cx: v[i], cy: v[n + i], w: v[2 * n + i], h: v[3 * n + i], score: best, cls, extra });
    }
    return { rows, e2e: false };
}

const unmapFn = (meta) => (x, y) => [(x - meta.padX) / meta.scale, (y - meta.padY) / meta.scale];

// ---------- decoders ----------
const decoders = {
    detect(outs, meta, m, minConf) {
        const nc = Object.keys(m.names).length;
        const { rows, e2e } = readRows(outs[0], nc, 0, minConf);
        const un = unmapFn(meta);
        let boxes = rows.map((r) => {
            const [x1, y1] = e2e ? un(r.x1, r.y1) : un(r.cx - r.w / 2, r.cy - r.h / 2);
            const [x2, y2] = e2e ? un(r.x2, r.y2) : un(r.cx + r.w / 2, r.cy + r.h / 2);
            return { x1, y1, x2, y2, score: r.score, cls: r.cls };
        });
        if (!e2e) boxes = nms(boxes);
        return { boxes };
    },

    segment(outs, meta, m, minConf) {
        const nc = Object.keys(m.names).length;
        const det = outs.find((o) => o.dims.length === 3), proto = outs.find((o) => o.dims.length === 4);
        const nm = proto.dims[1], mh = proto.dims[2], mw = proto.dims[3];
        const { rows, e2e } = readRows(det, nc, nm, minConf);
        let boxes = rows.map((r) => {
            const lb = e2e ? { x1: r.x1, y1: r.y1, x2: r.x2, y2: r.y2 }
                : { x1: r.cx - r.w / 2, y1: r.cy - r.h / 2, x2: r.cx + r.w / 2, y2: r.cy + r.h / 2 };
            return { ...lb, score: r.score, cls: r.cls, coef: r.extra };
        });
        if (!e2e) boxes = nms(boxes, 0.45, iou, 30);
        boxes = boxes.slice(0, 30);
        // masks at prototype resolution (covers the whole letterboxed input)
        const img = new ImageData(mw, mh), pv = proto.data, area = mw * mh;
        const sx = mw / meta.size, sy = mh / meta.size;
        boxes.forEach((b, bi) => {
            const [r, g, bl] = hslRgb(b.cls);
            const x0 = Math.max(0, Math.floor(b.x1 * sx)), x1 = Math.min(mw, Math.ceil(b.x2 * sx));
            const y0 = Math.max(0, Math.floor(b.y1 * sy)), y1 = Math.min(mh, Math.ceil(b.y2 * sy));
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    const p = y * mw + x;
                    let s = 0;
                    for (let k = 0; k < nm; k++) s += b.coef[k] * pv[k * area + p];
                    if (s > 0) {          // sigmoid(s) > 0.5
                        const o = p * 4;
                        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = bl; img.data[o + 3] = 255;
                    }
                }
            }
            b.order = bi;
        });
        const un = unmapFn(meta);
        const out = boxes.map((b) => {
            const [x1, y1] = un(b.x1, b.y1), [x2, y2] = un(b.x2, b.y2);
            return { x1, y1, x2, y2, score: b.score, cls: b.cls };
        });
        return { boxes: out, mask: img };
    },

    semantic(outs, meta, m) {
        const o = outs[0], d = o.dims, v = o.data;
        let h, w, labels;
        if (d.length === 4 && d[1] > 1) {                 // [1, C, H, W] scores -> argmax
            const c = d[1]; h = d[2]; w = d[3];
            const area = h * w; labels = new Uint8Array(area);
            for (let p = 0; p < area; p++) {
                let best = -Infinity, k = 0;
                for (let ci = 0; ci < c; ci++) { const s = v[ci * area + p]; if (s > best) { best = s; k = ci; } }
                labels[p] = k;
            }
        } else {                                           // [1, H, W] or [1, 1, H, W] class ids
            h = d[d.length - 2]; w = d[d.length - 1];
            labels = Uint8Array.from(v, (x) => Number(x));
        }
        const img = new ImageData(w, h), present = new Map();
        // only count pixels inside the real frame (not the grey letterbox padding)
        const kx = w / meta.size, ky = h / meta.size;
        const rx0 = Math.round(meta.padX * kx), rx1 = Math.round((meta.padX + meta.nw) * kx);
        const ry0 = Math.round(meta.padY * ky), ry1 = Math.round((meta.padY + meta.nh) * ky);
        let counted = 0;
        for (let p = 0; p < labels.length; p++) {
            const k = labels[p], col = CITYSCAPES_COLORS[k] || hslRgb(k), q = p * 4;
            img.data[q] = col[0]; img.data[q + 1] = col[1]; img.data[q + 2] = col[2]; img.data[q + 3] = 255;
            const x = p % w, y = (p / w) | 0;
            if (x >= rx0 && x < rx1 && y >= ry0 && y < ry1) { present.set(k, (present.get(k) || 0) + 1); counted += 1; }
        }
        const classes = [...present.entries()].sort((a, b) => b[1] - a[1])
            .map(([k, n]) => ({ k, name: m.names[k] ?? `class ${k}`, share: n / (counted || 1) }));
        return { mask: img, classes };
    },

    depth(outs, meta) {
        const o = outs[0], d = o.dims, v = o.data;
        const h = d[d.length - 2], w = d[d.length - 1];
        // robust range (2nd..98th percentile) so a few outliers do not wash out the colours
        const sample = [];
        for (let i = 0; i < v.length; i += 37) sample.push(v[i]);
        sample.sort((a, b) => a - b);
        const lo = sample[Math.floor(sample.length * 0.02)], hi = sample[Math.floor(sample.length * 0.98)];
        const img = new ImageData(w, h);
        for (let p = 0; p < w * h; p++) {
            const t = Math.min(1, Math.max(0, (v[p] - lo) / (hi - lo || 1)));
            const [r, g, b] = turbo(1 - t);              // near = warm, far = cool
            const q = p * 4;
            img.data[q] = r; img.data[q + 1] = g; img.data[q + 2] = b; img.data[q + 3] = 255;
        }
        // depth at the image centre (in model coordinates of the letterboxed input)
        const cx = Math.floor(((meta.padX + meta.nw / 2) / meta.size) * w), cy = Math.floor(((meta.padY + meta.nh / 2) / meta.size) * h);
        return { mask: img, near: lo, far: hi, center: v[cy * w + cx] };
    },

    classify(outs, meta, m) {
        const v = outs[0].data;
        let probs = Array.from(v);
        const sum = probs.reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1) > 0.05 || probs.some((p) => p < 0)) {      // logits -> softmax
            const mx = Math.max(...probs), ex = probs.map((p) => Math.exp(p - mx)), s = ex.reduce((a, b) => a + b, 0);
            probs = ex.map((e) => e / s);
        }
        const top = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 5)
            .map(({ p, i }) => ({ name: m.names[i] ?? `class ${i}`, p }));
        return { top };
    },

    pose(outs, meta, m, minConf) {
        const kpt = m.kpt_shape || [17, 3], nk = kpt[0] * kpt[1];
        const { rows, e2e } = readRows(outs[0], 1, nk, minConf);
        const un = unmapFn(meta);
        let people = rows.map((r) => {
            const lb = e2e ? [r.x1, r.y1, r.x2, r.y2] : [r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2];
            return { x1: lb[0], y1: lb[1], x2: lb[2], y2: lb[3], score: r.score, cls: 0, kp: r.extra };
        });
        if (!e2e) people = nms(people);
        return {
            people: people.map((p) => {
                const [x1, y1] = un(p.x1, p.y1), [x2, y2] = un(p.x2, p.y2);
                const pts = [];
                for (let k = 0; k < kpt[0]; k++) {
                    const [x, y] = un(p.kp[k * kpt[1]], p.kp[k * kpt[1] + 1]);
                    pts.push({ x, y, v: kpt[1] > 2 ? p.kp[k * kpt[1] + 2] : 1 });
                }
                return { x1, y1, x2, y2, score: p.score, pts };
            }),
        };
    },

    obb(outs, meta, m, minConf) {
        const nc = Object.keys(m.names).length;
        const o = outs[0], d = o.dims, v = o.data;
        const boxes = [];
        if (d[2] === 7) {                                   // end-to-end [1, N, 7] = cx,cy,w,h,score,cls,angle
            for (let i = 0; i < d[1]; i++) {
                const q = i * 7;
                if (v[q + 4] < minConf) continue;
                boxes.push({ cx: v[q], cy: v[q + 1], w: v[q + 2], h: v[q + 3], score: v[q + 4], cls: Math.round(v[q + 5]), a: v[q + 6] });
            }
        } else {                                            // raw [1, 4+nc+1, N]
            const n = d[2];
            for (let i = 0; i < n; i++) {
                let best = 0, cls = 0;
                for (let k = 0; k < nc; k++) { const s = v[(4 + k) * n + i]; if (s > best) { best = s; cls = k; } }
                if (best < minConf) continue;
                boxes.push({ cx: v[i], cy: v[n + i], w: v[2 * n + i], h: v[3 * n + i], score: best, cls, a: v[(4 + nc) * n + i] });
            }
        }
        const polys = boxes.map((b) => ({ ...b, poly: rotRect(b) }));
        const kept = d[2] === 7 ? polys : nms(polys, 0.45, (a, b) => polyIoU(a.poly, b.poly));
        const un = unmapFn(meta);
        return { obbs: kept.map((b) => ({ score: b.score, cls: b.cls, poly: b.poly.map(([x, y]) => un(x, y)) })) };
    },
};

function rotRect({ cx, cy, w, h, a }) {
    const c = Math.cos(a), s = Math.sin(a);
    return [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]].map(([x, y]) => [cx + x * c - y * s, cy + x * s + y * c]);
}

function polyArea(p) { let a = 0; for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; a += x1 * y2 - x2 * y1; } return Math.abs(a) / 2; }

function clipPoly(subject, clip) {         // Sutherland-Hodgman (both convex)
    let out = subject;
    for (let i = 0; i < clip.length && out.length; i++) {
        const [ax, ay] = clip[i], [bx, by] = clip[(i + 1) % clip.length];
        const inside = ([x, y]) => (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0;
        const inter = ([px, py], [qx, qy]) => {
            const a1 = by - ay, b1 = ax - bx, c1 = a1 * ax + b1 * ay;
            const a2 = qy - py, b2 = px - qx, c2 = a2 * px + b2 * py;
            const det = a1 * b2 - a2 * b1 || 1e-9;
            return [(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det];
        };
        const input = out; out = [];
        for (let j = 0; j < input.length; j++) {
            const cur = input[j], prev = input[(j + input.length - 1) % input.length];
            if (inside(cur)) { if (!inside(prev)) out.push(inter(prev, cur)); out.push(cur); } else if (inside(prev)) out.push(inter(prev, cur));
        }
    }
    return out;
}

function polyIoU(a, b) {
    const orient = (p) => (polyArea(p) && ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0])) < 0 ? [...p].reverse() : p);
    const pa = orient(a), pb = orient(b);
    const inter = polyArea(clipPoly(pa, pb));
    return inter / (polyArea(pa) + polyArea(pb) - inter + 1e-6);
}

function turbo(t) {   // Google "turbo" colormap approximation, t in 0..1
    const r = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
    const g = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
    const b = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66))));
    return [r, g, b].map((x) => Math.max(0, Math.min(255, Math.round(x))));
}

// ---------- inference ----------
async function runInference() {
    const task = currentTask;
    const sp = sessions.get(task);
    if (!sp) return;
    let s;
    try { s = await sp; } catch (_) { return; }
    const v = el.video, w = v.videoWidth, h = v.videoHeight;
    if (!w || !h) return;
    const m = s.meta;
    const meta = preprocessFor(task, v, w, h, m.imgsz);
    const t0 = performance.now();
    const out = await s.session.run({ [s.session.inputNames[0]]: meta.tensor });
    const dt = performance.now() - t0;
    const outs = s.session.outputNames.map((n) => out[n]);
    const decoded = decoders[task](outs, meta, m, Number(el.conf.value));
    if (task !== currentTask) return;          // user switched while this frame was running
    result = { task, meta, frameW: w, frameH: h, ...decoded };
    el.stInfer.textContent = `${Math.round(dt)} ms`;
    modelRuns += 1;
    if (task === 'classify') renderClassify(decoded.top);
    if (task === 'semantic') renderLegend('semantic', decoded.classes);
    if (task === 'depth') renderLegend('depth', decoded);
}

// ---------- drawing ----------
function drawMaskImage(img, meta, w, h, mirror) {
    // the mask covers the whole letterboxed input -> draw only the part that contains the frame
    if (maskCanvas.width !== img.width || maskCanvas.height !== img.height) { maskCanvas.width = img.width; maskCanvas.height = img.height; }
    maskCtx.putImageData(img, 0, 0);
    const kx = img.width / meta.size, ky = img.height / meta.size;
    ctx.save();
    if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
    ctx.globalAlpha = Number(el.alpha.value);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(maskCanvas, meta.padX * kx, meta.padY * ky, meta.nw * kx, meta.nh * ky, 0, 0, w, h);
    ctx.restore();
}

function drawLabel(text, x, y, color, fs) {
    ctx.font = `600 ${fs}px system-ui, sans-serif`;
    const tw = ctx.measureText(text).width + 10, th = fs + 8;
    const ly = y - th < 0 ? y : y - th;
    ctx.fillStyle = color; ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = '#0b1120'; ctx.fillText(text, x + 5, ly + fs + 1);
}

function drawResult(w, h, mirror) {
    const r = result;
    if (!r || r.task !== currentTask) return;
    const mx = (x) => (mirror ? w - x : x);
    const lw = Math.max(2, w / 400), fs = Math.round(Math.max(12, w / 60));
    const names = manifest?.[r.task]?.names || {};
    const label = el.labels.checked;

    if (r.mask && r.task !== 'classify') drawMaskImage(r.mask, r.meta, w, h, mirror);

    if (r.boxes) {
        for (const b of r.boxes) {
            const x1 = Math.min(mx(b.x1), mx(b.x2)), x2 = Math.max(mx(b.x1), mx(b.x2)), color = hsl(b.cls);
            ctx.lineWidth = lw; ctx.strokeStyle = color; ctx.strokeRect(x1, b.y1, x2 - x1, b.y2 - b.y1);
            if (label) drawLabel(`${names[b.cls] ?? b.cls} ${Math.round(b.score * 100)}%`, x1, b.y1, color, fs);
        }
    }

    if (r.people) {
        for (const p of r.people) {
            ctx.lineWidth = Math.max(3, w / 250);
            for (const [a, b] of COCO_SKELETON) {
                const pa = p.pts[a - 1], pb = p.pts[b - 1];
                if (pa.v < 0.5 || pb.v < 0.5) continue;
                ctx.strokeStyle = a <= 5 && b <= 5 ? '#f472b6' : a >= 12 || b >= 12 ? '#a78bfa' : '#38bdf8';
                ctx.beginPath(); ctx.moveTo(mx(pa.x), pa.y); ctx.lineTo(mx(pb.x), pb.y); ctx.stroke();
            }
            for (const k of p.pts) {
                if (k.v < 0.5) continue;
                ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(mx(k.x), k.y, Math.max(3, w / 220), 0, Math.PI * 2); ctx.fill();
            }
            if (label) drawLabel(`person ${Math.round(p.score * 100)}%`, Math.min(mx(p.x1), mx(p.x2)), p.y1, '#38bdf8', fs);
        }
    }

    if (r.obbs) {
        for (const b of r.obbs) {
            const color = hsl(b.cls);
            ctx.lineWidth = lw; ctx.strokeStyle = color;
            ctx.beginPath(); b.poly.forEach(([x, y], i) => (i ? ctx.lineTo(mx(x), y) : ctx.moveTo(mx(x), y))); ctx.closePath(); ctx.stroke();
            if (label) {
                const top = b.poly.reduce((a, c) => (c[1] < a[1] ? c : a));
                drawLabel(`${names[b.cls] ?? b.cls} ${Math.round(b.score * 100)}%`, mx(top[0]), top[1], color, fs);
            }
        }
    }

    if (r.task === 'depth' && r.center != null) {
        const cx = w / 2, cy = h / 2;
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy); ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12); ctx.stroke();
        drawLabel(`centre ≈ ${r.center.toFixed(2)} m`, cx + 14, cy - 6, '#f8fafc', fs);
    }
}

function renderClassify(top) {
    el.resultBox.hidden = false;
    el.resultBox.innerHTML = '<div style="font-weight:700">Top-5 classes</div>' + top.map((t) => `
        <div class="row"><span>${escapeHtml(t.name)}</span><b>${(t.p * 100).toFixed(1)}%</b><div class="bar"><i style="width:${t.p * 100}%"></i></div></div>`).join('');
}

function renderLegend(task, data) {
    if (!task) { el.legend.innerHTML = ''; return; }
    if (task === 'semantic' && data) {
        el.legend.innerHTML = data.filter((c) => c.share > 0.01).slice(0, 10).map((c) => {
            const col = CITYSCAPES_COLORS[c.k] || hslRgb(c.k);
            return `<span><i style="--c:rgb(${col.join(',')})"></i>${escapeHtml(c.name)} ${(c.share * 100).toFixed(0)}%</span>`;
        }).join('');
    } else if (task === 'depth' && data) {
        const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => `rgb(${turbo(1 - t).join(',')}) ${t * 100}%`).join(',');
        el.legend.innerHTML = `<span>near ≈ ${data.near.toFixed(1)} m</span><span><i class="grad" style="background:linear-gradient(90deg,${stops})"></i></span><span>far ≈ ${data.far.toFixed(1)} m</span>`;
    } else {
        el.legend.innerHTML = `<span>${TASK_INFO[task].hint}</span>`;
    }
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

// ---------- main loop ----------
function loop() {
    if (!running) return;
    const v = el.video, w = v.videoWidth, h = v.videoHeight;
    if (w && h) {
        if (el.view.width !== w || el.view.height !== h) { el.view.width = w; el.view.height = h; }
        const mirror = el.mirror.checked;
        ctx.save();
        if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
        ctx.drawImage(v, 0, 0, w, h);
        ctx.restore();
        drawResult(w, h, mirror);

        frameNo += 1; videoFrames += 1;
        if (!busy && frameNo % Number(el.every.value) === 0 && sessions.has(currentTask)) {
            busy = true;
            runInference()
                .catch((err) => { console.error(err); showError(`<strong>Inference failed.</strong> ${escapeHtml(err.message)}`); })
                .finally(() => { busy = false; });
        }
        const now = performance.now();
        if (now - fpsStart >= 1000) {
            el.stFps.textContent = Math.round((videoFrames * 1000) / (now - fpsStart));
            el.stModelFps.textContent = ((modelRuns * 1000) / (now - fpsStart)).toFixed(1);
            videoFrames = 0; modelRuns = 0; fpsStart = now;
        }
    }
    requestAnimationFrame(loop);
}

// ---------- camera ----------
async function startCamera() {
    showError('');
    el.camBtn.disabled = true;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
        el.video.srcObject = stream;
        await el.video.play();
        running = true;
        el.placeholder.hidden = true;
        el.placeholder.style.display = 'none';
        el.camBtn.textContent = '■ Stop webcam';
        el.camBtn.classList.add('stop');
        requestAnimationFrame(loop);
    } catch (err) {
        console.error(err);
        const msg = {
            NotAllowedError: '<strong>Camera access was denied.</strong> Allow the camera for this page (lock icon in the address bar) and try again.',
            SecurityError: '<strong>Camera access was blocked.</strong> Open the page via http://localhost (not as a file) and allow the camera.',
            NotFoundError: '<strong>No camera found.</strong> Connect a webcam and try again.',
            OverconstrainedError: '<strong>No suitable camera found.</strong>',
            NotReadableError: '<strong>The camera is busy.</strong> Close other apps that use it (Zoom, Teams, another browser tab) and try again.',
        }[err.name] || `<strong>Could not start the camera.</strong> ${escapeHtml(err.message)}`;
        showError(msg);
    } finally {
        el.camBtn.disabled = false;
    }
}

function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    el.video.srcObject = null;
    el.camBtn.textContent = '▶ Start webcam';
    el.camBtn.classList.remove('stop');
    el.placeholder.style.display = '';
    el.stFps.textContent = '0'; el.stModelFps.textContent = '0';
}

// ---------- wiring ----------
el.camBtn.addEventListener('click', () => (running ? stopCamera() : startCamera()));
el.task.addEventListener('change', () => selectTask(el.task.value));
el.conf.addEventListener('input', () => { el.confVal.textContent = `${Math.round(el.conf.value * 100)}%`; });
el.alpha.addEventListener('input', () => { el.alphaVal.textContent = `${Math.round(el.alpha.value * 100)}%`; });
el.gpu.addEventListener('change', async () => {
    const wantGpu = el.gpu.checked;
    await releaseSessions();      // re-create the models on the new backend
    if (el.gpu.checked !== wantGpu) return;   // toggled again meanwhile -> the newer event handles it
    await selectTask(currentTask);
    if (wantGpu && !navigator.gpu) {
        showError('<strong>WebGPU is not available in this browser.</strong> The app keeps using WASM (CPU). Chrome/Edge 113+ or Safari 18+ support WebGPU.');
    } else if (wantGpu && backendLabel.startsWith('WASM')) {
        showError('<strong>WebGPU could not be started on this computer</strong> &ndash; switched back to WASM (CPU) automatically.');
    }
});

// ---------- self-test hook (console): runs a still image through the same pipeline ----------
// usage:  await yoloSelfTest('detect', './some-image.jpg')
window.yoloSelfTest = async (task, url) => {
    const img = new Image();
    await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error(`cannot load ${url}`)); img.src = url; });
    const s = await getSession(task), m = s.meta, w = img.naturalWidth, h = img.naturalHeight;
    const meta = preprocessFor(task, img, w, h, m.imgsz);
    const t0 = performance.now();
    const out = await s.session.run({ [s.session.inputNames[0]]: meta.tensor });
    const ms = Math.round(performance.now() - t0);
    const d = decoders[task](s.session.outputNames.map((n) => out[n]), meta, m, Number(el.conf.value));
    const name = (c) => m.names[c] ?? c;
    const box = (b) => [Math.round(b.x1), Math.round(b.y1), Math.round(b.x2), Math.round(b.y2)];
    if (d.boxes) return { ms, backend: s.backend, boxes: d.boxes.map((b) => [name(b.cls), +b.score.toFixed(2), box(b)]),
        maskPx: d.mask ? Array.from(d.mask.data).filter((_, i) => i % 4 === 3 && d.mask.data[i]).length : undefined };
    if (d.people) return { ms, backend: s.backend, people: d.people.map((p) => [+p.score.toFixed(2), box(p), [Math.round(p.pts[0].x), Math.round(p.pts[0].y)]]) };
    if (d.obbs) return { ms, backend: s.backend, obbs: d.obbs.map((b) => [name(b.cls), +b.score.toFixed(2)]) };
    if (d.top) return { ms, backend: s.backend, top: d.top.map((t) => [t.name, +t.p.toFixed(3)]) };
    if (d.classes) return { ms, backend: s.backend, classes: d.classes.slice(0, 8).map((c) => [c.name, +(c.share * 100).toFixed(1)]) };
    return { ms, backend: s.backend, near: +d.near.toFixed(2), far: +d.far.toFixed(2), center: +d.center.toFixed(2) };
};

// ---------- start ----------
(function init() {
    const problems = checkBrowser();
    if (problems.length) {
        showError(`<strong>Unsupported browser.</strong> ${problems.join(' ')}`);
        el.camBtn.disabled = true;
        return;
    }
    ort.env.wasm.wasmPaths = ORT_PATH;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    updateSysInfo();
    el.stBackend.textContent = 'WASM (CPU)';
    selectTask(currentTask);
})();
