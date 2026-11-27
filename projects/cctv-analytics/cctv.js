// CCTV Traffic Analytics: YOLO26n (ONNX Runtime Web) on a live MJPEG traffic camera with
//   1) vehicle tracking + counting at a user-drawn line (per direction and type)
//   2) traffic heatmap
//   3) speed estimation (metre scale taken from each vehicle's own width -> perspective-aware)
(() => {
    'use strict';

    const MODEL_URL = '../models/yolo26n.onnx';
    const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    const INPUT = 640;

    // COCO class id -> vehicle info (typical real width in metres for the speed scale)
    const VEHICLES = {
        1: { name: 'bicycle', width: 0.6, color: '#a3e635' },
        2: { name: 'car', width: 1.8, color: '#38bdf8' },
        3: { name: 'motorcycle', width: 0.8, color: '#f472b6' },
        5: { name: 'bus', width: 2.5, color: '#facc15' },
        7: { name: 'truck', width: 2.4, color: '#fb923c' },
    };
    const VEHICLE_IDS = Object.keys(VEHICLES).map(Number);

    const TRACK_MAX_MISSED_FRAMES = 2;      // a track survives this many frames without a match
    const TRACK_MAX_AGE_MS = 4000;
    const MAX_PLAUSIBLE_KMH = 160;
    const HEAT_W = 96, HEAT_H = 54;

    // ---------- DOM ----------
    const $ = (id) => document.getElementById(id);
    const els = {
        dot: $('cStatusDot'), modelStatus: $('cModelStatus'),
        url: $('cUrl'), startBtn: $('cStartBtn'), stopBtn: $('cStopBtn'), streamInfo: $('cStreamInfo'),
        drawLineBtn: $('cDrawLineBtn'), resetCountBtn: $('cResetCountBtn'), dirA: $('cDirA'), dirB: $('cDirB'),
        conf: $('cConf'), confVal: $('cConfVal'),
        heatToggle: $('cHeatToggle'), heatOpacity: $('cHeatOpacity'), heatOpacityVal: $('cHeatOpacityVal'),
        heatDecay: $('cHeatDecay'), resetHeatBtn: $('cResetHeatBtn'),
        limit: $('cLimit'), limitVal: $('cLimitVal'), calib: $('cCalib'), calibVal: $('cCalibVal'), showSpeed: $('cShowSpeed'),
        total: $('cTotal'), inView: $('cInView'), avgSpeed: $('cAvgSpeed'), maxSpeed: $('cMaxSpeed'), fps: $('cFps'),
        empty: $('cEmpty'), stage: $('cStage'), canvas: $('cCanvas'), status: $('cStatus'), modeBadge: $('cModeBadge'),
        img: $('cImg'), classTable: $('cClassTable'), rateChart: $('cRateChart'), log: $('cLog'), exportBtn: $('cExportBtn'),
    };
    const ctx = els.canvas.getContext('2d');
    const prepCanvas = document.createElement('canvas');
    prepCanvas.width = prepCanvas.height = INPUT;
    const prepCtx = prepCanvas.getContext('2d', { willReadFrequently: true });
    // tiny canvas used to notice when the MJPEG stream delivers a new frame
    const probeCanvas = document.createElement('canvas');
    probeCanvas.width = 48; probeCanvas.height = 27;
    const probeCtx = probeCanvas.getContext('2d', { willReadFrequently: true });
    const heatCanvas = document.createElement('canvas');
    heatCanvas.width = HEAT_W; heatCanvas.height = HEAT_H;
    const heatCtx = heatCanvas.getContext('2d');

    // ---------- state ----------
    let session = null, inputName = null, backendName = '';
    let connected = false, busy = false, pendingFrame = false;
    let lastSignature = null, frameTimes = [], lastFrameAt = 0, probeTimer = null, watchdog = null;
    let tracks = [], nextId = 1;
    let line = { a: { x: 0.05, y: 0.7 }, b: { x: 0.95, y: 0.7 } };   // normalised 0..1
    let drawingLine = false, linePoints = [];
    let events = [];          // vehicles that crossed the line
    const heat = new Float32Array(HEAT_W * HEAT_H);
    let lastHeatTime = 0;

    // ---------- model ----------
    async function loadModel() {
        ort.env.wasm.wasmPaths = ORT_DIST;
        ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
        const eps = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
        for (const ep of eps) {
            try {
                session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [ep], graphOptimizationLevel: 'all' });
                backendName = ep === 'webgpu' ? 'WebGPU' : 'WASM';
                break;
            } catch (err) { console.warn(`ONNX Runtime ${ep} failed`, err); }
        }
        if (!session) throw new Error('Could not load models/yolo26n.onnx (run export_yolo.py once).');
        inputName = session.inputNames[0];
        await session.run({ [inputName]: new ort.Tensor('float32', new Float32Array(3 * INPUT * INPUT), [1, 3, INPUT, INPUT]) });
    }

    function preprocess(src, w, h) {
        const scale = Math.min(INPUT / w, INPUT / h);
        const nw = Math.round(w * scale), nh = Math.round(h * scale);
        const padX = Math.floor((INPUT - nw) / 2), padY = Math.floor((INPUT - nh) / 2);
        prepCtx.fillStyle = 'rgb(114,114,114)';
        prepCtx.fillRect(0, 0, INPUT, INPUT);
        prepCtx.drawImage(src, padX, padY, nw, nh);
        const px = prepCtx.getImageData(0, 0, INPUT, INPUT).data;
        const area = INPUT * INPUT, data = new Float32Array(3 * area);
        for (let i = 0, j = 0; i < area; i++, j += 4) {
            data[i] = px[j] / 255; data[i + area] = px[j + 1] / 255; data[i + 2 * area] = px[j + 2] / 255;
        }
        return { tensor: new ort.Tensor('float32', data, [1, 3, INPUT, INPUT]), scale, padX, padY };
    }

    function iou(a, b) {
        const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
        const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-6);
    }

    // returns vehicle detections only, in frame pixel coordinates
    function postprocess(out, meta, w, h) {
        const minConf = Number(els.conf.value), d = out.dims, v = out.data;
        const unmap = (x, y) => [(x - meta.padX) / meta.scale, (y - meta.padY) / meta.scale];
        let boxes = [];
        if (d.length === 3 && d[2] === 6) {          // end-to-end export [1,300,6]
            for (let i = 0; i < d[1]; i++) {
                const o = i * 6, cls = Math.round(v[o + 5]);
                if (v[o + 4] < minConf || !VEHICLES[cls]) continue;
                const [x1, y1] = unmap(v[o], v[o + 1]), [x2, y2] = unmap(v[o + 2], v[o + 3]);
                boxes.push({ x1, y1, x2, y2, score: v[o + 4], cls });
            }
        } else {                                       // classic [1,84,N] -> NMS
            const n = d[2];
            for (let i = 0; i < n; i++) {
                let best = 0, cls = -1;
                for (const c of VEHICLE_IDS) { const s = v[(4 + c) * n + i]; if (s > best) { best = s; cls = c; } }
                if (best < minConf) continue;
                const cx = v[i], cy = v[n + i], bw = v[2 * n + i], bh = v[3 * n + i];
                const [x1, y1] = unmap(cx - bw / 2, cy - bh / 2), [x2, y2] = unmap(cx + bw / 2, cy + bh / 2);
                boxes.push({ x1, y1, x2, y2, score: best, cls });
            }
            boxes.sort((a, b) => b.score - a.score);
            const kept = [];
            // class-agnostic NMS: a car and a truck box on the same vehicle should not both survive
            for (const b of boxes) if (!kept.some((k) => iou(k, b) > 0.5)) kept.push(b);
            boxes = kept;
        }
        return boxes.map((b) => ({ ...b, x1: Math.max(0, b.x1), y1: Math.max(0, b.y1), x2: Math.min(w, b.x2), y2: Math.min(h, b.y2) }));
    }

    async function detect(src, w, h) {
        const meta = preprocess(src, w, h);
        const out = await session.run({ [inputName]: meta.tensor });
        return postprocess(out[session.outputNames[0]], meta, w, h);
    }

    // ---------- tracking ----------
    const anchor = (b) => ({ x: (b.x1 + b.x2) / 2, y: b.y2 });          // bottom centre = point on the road
    const boxSize = (b) => Math.max(b.x2 - b.x1, b.y2 - b.y1);
    const isTwoWheeler = (cls) => cls === 1 || cls === 3;

    function majorityClass(votes) {
        let best = null, n = -1;
        for (const [c, k] of Object.entries(votes)) if (k > n) { best = Number(c); n = k; }
        return best;
    }

    function updateTracks(dets, t) {
        const live = tracks.filter((tr) => tr.missed <= TRACK_MAX_MISSED_FRAMES && t - tr.lastSeen <= TRACK_MAX_AGE_MS);
        // candidate pairs with a size-normalised distance to the predicted position
        const pairs = [];
        live.forEach((tr, ti) => {
            const last = tr.pts[tr.pts.length - 1];
            const dt = (t - last.t) / 1000;
            const pred = { x: last.x + tr.vx * dt, y: last.y + tr.vy * dt };
            dets.forEach((d, di) => {
                if (isTwoWheeler(d.cls) !== isTwoWheeler(tr.cls)) return;   // never link a motorbike with a car
                const p = anchor(d);
                const sizeRatio = boxSize(d) / boxSize(tr.box);
                if (sizeRatio < 0.5 || sizeRatio > 2) return;                // the same vehicle keeps a similar size
                const ref = Math.min(boxSize(d), boxSize(tr.box));
                const dist = Math.hypot(p.x - pred.x, p.y - pred.y) / ref;
                const maxDist = tr.pts.length > 1 ? 1.0 : 1.5;   // no velocity yet -> allow bigger jumps
                const overlap = iou(tr.box, d);
                if (dist < maxDist || overlap > 0.3) pairs.push({ ti, di, cost: dist - overlap });
            });
        });
        pairs.sort((a, b) => a.cost - b.cost);
        const usedT = new Set(), usedD = new Set();
        for (const p of pairs) {
            if (usedT.has(p.ti) || usedD.has(p.di)) continue;
            usedT.add(p.ti); usedD.add(p.di);
            advanceTrack(live[p.ti], dets[p.di], t);
        }
        live.forEach((tr, ti) => { if (!usedT.has(ti)) tr.missed += 1; });
        dets.forEach((d, di) => {
            if (usedD.has(di)) return;
            const p = anchor(d);
            live.push({
                id: nextId++, box: d, votes: { [d.cls]: 1 }, cls: d.cls, pts: [{ ...p, t }],
                vx: 0, vy: 0, missed: 0, lastSeen: t, counted: false, speeds: [], speed: null, width: d.x2 - d.x1,
            });
        });
        tracks = live;
    }

    function advanceTrack(tr, d, t) {
        const prev = tr.pts[tr.pts.length - 1];
        const p = anchor(d);
        const dt = (t - prev.t) / 1000;
        tr.votes[d.cls] = (tr.votes[d.cls] || 0) + 1;
        tr.cls = majorityClass(tr.votes);
        if (dt > 0) {
            tr.vx = 0.5 * tr.vx + 0.5 * (p.x - prev.x) / dt;
            tr.vy = 0.5 * tr.vy + 0.5 * (p.y - prev.y) / dt;
            estimateSpeed(tr, tr.box, d, prev, p, dt);
        }
        // counting: did the movement segment cross the line?
        if (!tr.counted) {
            const dir = crossesLine(prev, p);
            if (dir) countVehicle(tr, dir, t);
        }
        tr.box = d; tr.missed = 0; tr.lastSeen = t;
        tr.pts.push({ ...p, t });
        if (tr.pts.length > 12) tr.pts.shift();
    }

    // ---------- speed ----------
    function estimateSpeed(tr, oldBox, newBox, p0, p1, dt) {
        if (dt < 0.2) return;
        const typical = VEHICLES[tr.cls]?.width || 1.8;
        const widthPx = ((oldBox.x2 - oldBox.x1) + (newBox.x2 - newBox.x1)) / 2;
        if (widthPx < 8) return;
        const metresPerPx = typical / widthPx;              // local scale at the vehicle's position
        const distM = Math.hypot(p1.x - p0.x, p1.y - p0.y) * metresPerPx;
        let kmh = (distM / dt) * 3.6 * Number(els.calib.value);
        if (kmh > MAX_PLAUSIBLE_KMH) return;               // tracking glitch
        if (kmh < 3) kmh = 0;                               // box jitter of a standing vehicle
        tr.speeds.push(kmh);
        if (tr.speeds.length > 5) tr.speeds.shift();
        const sorted = [...tr.speeds].sort((a, b) => a - b);
        tr.speed = sorted[Math.floor(sorted.length / 2)];   // median against outliers
    }

    // ---------- counting line ----------
    function linePx() {
        const w = els.canvas.width, h = els.canvas.height;
        return { a: { x: line.a.x * w, y: line.a.y * h }, b: { x: line.b.x * w, y: line.b.y * h } };
    }

    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    // returns 'A' / 'B' when segment p0->p1 crosses the counting line, else null
    function crossesLine(p0, p1) {
        const { a, b } = linePx();
        const d1 = cross(a, b, p0), d2 = cross(a, b, p1), d3 = cross(p0, p1, a), d4 = cross(p0, p1, b);
        if ((d1 > 0) === (d2 > 0) || (d3 > 0) === (d4 > 0) || d1 === 0 || d2 === 0) return null;
        return d1 > 0 ? 'A' : 'B';
    }

    function countVehicle(tr, dir, t) {
        tr.counted = true;
        events.push({
            time: new Date(), t, id: tr.id, cls: tr.cls, name: VEHICLES[tr.cls]?.name || 'vehicle',
            dir, speed: tr.speed,
        });
        renderCounts();
    }

    // ---------- heatmap ----------
    function addHeat(dets, w, h, t) {
        const decaySec = Number(els.heatDecay.value);
        if (decaySec > 0 && lastHeatTime) {
            const f = Math.exp(-((t - lastHeatTime) / 1000) / decaySec);
            for (let i = 0; i < heat.length; i++) heat[i] *= f;
        }
        lastHeatTime = t;
        for (const d of dets) {
            const p = anchor(d);
            const cx = (p.x / w) * HEAT_W, cy = (p.y / h) * HEAT_H;
            const sigma = Math.max(1, ((d.x2 - d.x1) / w) * HEAT_W * 0.35);
            const r = Math.ceil(sigma * 2.5);
            for (let y = Math.max(0, Math.floor(cy - r)); y < Math.min(HEAT_H, cy + r); y++) {
                for (let x = Math.max(0, Math.floor(cx - r)); x < Math.min(HEAT_W, cx + r); x++) {
                    const g = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma));
                    heat[y * HEAT_W + x] += g;
                }
            }
        }
        renderHeatCanvas();
    }

    function heatColor(v) {     // 0..1 -> blue, cyan, green, yellow, red
        const stops = [[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]];
        const x = Math.min(0.9999, Math.max(0, v)) * (stops.length - 1), i = Math.floor(x), f = x - i;
        return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
    }

    function renderHeatCanvas() {
        let max = 0;
        for (let i = 0; i < heat.length; i++) if (heat[i] > max) max = heat[i];
        const img = heatCtx.createImageData(HEAT_W, HEAT_H);
        for (let i = 0; i < heat.length; i++) {
            const v = max ? Math.sqrt(heat[i] / max) : 0;
            const [r, g, b] = heatColor(v);
            img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b;
            img.data[i * 4 + 3] = v < 0.05 ? 0 : Math.round(255 * Math.min(1, v * 1.4));
        }
        heatCtx.putImageData(img, 0, 0);
    }

    // ---------- drawing ----------
    function draw() {
        const w = els.img.naturalWidth, h = els.img.naturalHeight;
        if (!w || !h) return;
        if (els.canvas.width !== w || els.canvas.height !== h) { els.canvas.width = w; els.canvas.height = h; }
        ctx.drawImage(els.img, 0, 0, w, h);

        if (els.heatToggle.checked) {
            ctx.save();
            ctx.globalAlpha = Number(els.heatOpacity.value);
            ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(heatCanvas, 0, 0, w, h);
            ctx.restore();
        }

        drawLine(w);

        const fs = Math.round(Math.max(12, w / 60)), limit = Number(els.limit.value);
        ctx.font = `600 ${fs}px Inter, sans-serif`;
        for (const tr of tracks) {
            if (tr.missed > 0) continue;
            const v = VEHICLES[tr.cls], b = tr.box;
            const fast = tr.speed != null && tr.speed > limit;
            const color = fast ? '#ef4444' : v.color;
            // trail
            ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, w / 400); ctx.globalAlpha = 0.7;
            ctx.beginPath(); tr.pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.lineWidth = Math.max(2, w / 350);
            ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
            if (els.showSpeed.checked) {
                const sp = tr.speed == null ? '...' : `${Math.round(tr.speed)} km/h`;
                const text = `#${tr.id} ${v.name} ${sp}${tr.counted ? ' ✓' : ''}`;
                const tw = ctx.measureText(text).width + 10, th = fs + 8;
                const ly = b.y1 - th < 0 ? b.y1 : b.y1 - th;
                ctx.fillStyle = color; ctx.fillRect(b.x1, ly, tw, th);
                ctx.fillStyle = '#0b1120'; ctx.fillText(text, b.x1 + 5, ly + fs + 1);
            }
        }

        if (drawingLine && linePoints.length === 1) {
            const p = linePoints[0];
            ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 6, 0, Math.PI * 2); ctx.fill();
        }
    }

    function drawLine(w) {
        const { a, b } = linePx();
        ctx.save();
        ctx.lineWidth = Math.max(3, w / 250);
        ctx.strokeStyle = '#fde047'; ctx.setLineDash([16, 10]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.setLineDash([]);
        // direction labels on both sides of the line (A = side where cross() > 0 ... moving away from it)
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
        const nx = -(b.y - a.y) / len, ny = (b.x - a.x) / len;   // normal pointing to cross() > 0 side
        const off = Math.max(22, w / 30);
        const fs = Math.round(Math.max(13, w / 55));
        ctx.font = `700 ${fs}px Inter, sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const label = (txt, x, y) => {
            const tw = ctx.measureText(txt).width + 14;
            ctx.fillStyle = 'rgba(15,23,42,0.8)'; ctx.fillRect(x - tw / 2, y - fs * 0.8, tw, fs * 1.6);
            ctx.fillStyle = '#fde047'; ctx.fillText(txt, x, y);
        };
        const countA = events.filter((e) => e.dir === 'A').length, countB = events.length - countA;
        // a vehicle counted as "A" starts on the cross()>0 side and moves to the other side
        label(`A → ${countA}`, mx - nx * off, my - ny * off);
        label(`${countB} ← B`, mx + nx * off, my + ny * off);
        ctx.restore();
    }

    // ---------- stats & UI ----------
    function renderCounts() {
        const a = events.filter((e) => e.dir === 'A').length;
        els.dirA.textContent = a; els.dirB.textContent = events.length - a;
        els.total.textContent = events.length;

        const byClass = {};
        for (const e of events) {
            byClass[e.cls] = byClass[e.cls] || { A: 0, B: 0 };
            byClass[e.cls][e.dir] += 1;
        }
        const rows = Object.entries(byClass).sort((x, y) => (y[1].A + y[1].B) - (x[1].A + x[1].B));
        els.classTable.innerHTML = rows.length
            ? `<div class="c-table-row head"><span></span><span>Type</span><span>A&rarr;</span><span>&larr;B</span><span>Total</span></div>`
              + rows.map(([c, n]) => `<div class="c-table-row" style="--c:${VEHICLES[c].color}"><span class="sw"></span>
                   <span>${VEHICLES[c].name}</span><span>${n.A}</span><span>${n.B}</span><span><b>${n.A + n.B}</b></span></div>`).join('')
            : '<p class="param-hint">No vehicles counted yet.</p>';

        const limit = Number(els.limit.value);
        els.log.innerHTML = events.length
            ? [...events].reverse().slice(0, 50).map((e) => `<div class="c-log-row">
                <span>${e.time.toLocaleTimeString()}</span><span>#${e.id} ${e.name}</span>
                <span>${e.dir === 'A' ? 'A →' : '← B'}</span>
                <span class="${e.speed != null && e.speed > limit ? 'fast' : ''}">${e.speed == null ? 'n/a' : `${Math.round(e.speed)} km/h`}</span></div>`).join('')
            : '<p class="param-hint">Waiting for the first vehicle&hellip;</p>';

        const withSpeed = events.filter((e) => e.speed != null && e.speed > 0);
        els.avgSpeed.textContent = withSpeed.length ? `${Math.round(withSpeed.reduce((s, e) => s + e.speed, 0) / withSpeed.length)} km/h` : '–';
        els.maxSpeed.textContent = withSpeed.length ? `${Math.round(Math.max(...withSpeed.map((e) => e.speed)))} km/h` : '–';
        drawRateChart();
    }

    function drawRateChart() {
        const c = els.rateChart, g = c.getContext('2d');
        const w = c.clientWidth || 300, h = c.clientHeight || 140, dpr = devicePixelRatio || 1;
        if (c.width !== w * dpr) { c.width = w * dpr; c.height = h * dpr; }
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, w, h);
        const MIN = 15, now = Date.now();
        const bins = new Array(MIN).fill(0);
        for (const e of events) {
            const ago = Math.floor((now - e.time.getTime()) / 60000);
            if (ago >= 0 && ago < MIN) bins[MIN - 1 - ago] += 1;
        }
        const max = Math.max(4, ...bins), left = 24, bottom = 18, bw = (w - left) / MIN;
        g.fillStyle = 'rgba(148,163,184,0.7)'; g.font = '11px Inter, sans-serif';
        g.fillText(String(max), 0, 12); g.fillText('0', 0, h - bottom);
        g.fillText('-15 min', left, h - 3); g.fillText('now', w - 24, h - 3);
        bins.forEach((n, i) => {
            const bh = (h - bottom - 8) * (n / max);
            g.fillStyle = i === MIN - 1 ? '#22c55e' : 'rgba(56,189,248,0.75)';
            g.fillRect(left + i * bw + 2, h - bottom - bh, bw - 4, bh);
        });
    }

    function showStatus(text) { els.status.hidden = !text; els.status.textContent = text || ''; }

    // ---------- stream handling ----------
    let lastThumb = null;
    function frameSignature() {
        probeCtx.drawImage(els.img, 0, 0, probeCanvas.width, probeCanvas.height);
        const d = probeCtx.getImageData(0, 0, probeCanvas.width, probeCanvas.height).data;
        let s = 0;
        const thumb = new Uint8Array(d.length / 4);
        for (let i = 0, k = 0; i < d.length; i += 4, k++) {
            s = (s * 31 + d[i] + d[i + 1] * 7 + d[i + 2] * 13) >>> 0;
            thumb[k] = (d[i] + d[i + 1] + d[i + 2]) / 3;
        }
        return { s, thumb };
    }

    // The Keelung camera rotates between several preset views. A big jump in the
    // thumbnail means a new view: restart tracking + heatmap (counts are kept).
    function isSceneChange(thumb) {
        if (!lastThumb) return false;
        let diff = 0;
        for (let i = 0; i < thumb.length; i++) diff += Math.abs(thumb[i] - lastThumb[i]);
        return diff / thumb.length > 28;
    }

    let sceneChanges = 0;
    function onSceneChange() {
        sceneChanges += 1;
        tracks = [];
        heat.fill(0); renderHeatCanvas(); lastHeatTime = 0;
        els.modeBadge.textContent = `Camera view changed (#${sceneChanges}) – tracking & heatmap restarted`;
        setTimeout(() => { if (!drawingLine) els.modeBadge.textContent = 'Tracking'; }, 4000);
    }

    function onProbe() {
        if (!connected || !els.img.naturalWidth) return;
        const { s: sig, thumb } = frameSignature();
        if (sig === lastSignature) return;
        lastSignature = sig;
        if (isSceneChange(thumb)) onSceneChange();
        lastThumb = thumb;
        const t = performance.now();
        frameTimes.push(t); if (frameTimes.length > 10) frameTimes.shift();
        lastFrameAt = t;
        if (frameTimes.length > 1) {
            const fps = (frameTimes.length - 1) / ((t - frameTimes[0]) / 1000);
            els.fps.textContent = `${fps.toFixed(1)} fps`;
        }
        processFrame(t);
    }

    async function processFrame(t) {
        if (busy) { pendingFrame = true; return; }
        busy = true;
        try {
            const w = els.img.naturalWidth, h = els.img.naturalHeight;
            const dets = await detect(els.img, w, h);
            updateTracks(dets, t);
            addHeat(dets, w, h, t);
            els.inView.textContent = tracks.filter((tr) => tr.missed === 0).length;
            draw();
            showStatus('');
        } catch (err) {
            console.error(err);
            showStatus(`Detection error: ${err.message}`);
        } finally {
            busy = false;
            if (pendingFrame) { pendingFrame = false; processFrame(performance.now()); }
        }
    }

    async function connect() {
        const url = els.url.value.trim();
        if (!/^https?:\/\//i.test(url)) { alert('Please enter a valid stream URL (http/https).'); return; }
        disconnect();
        els.empty.style.display = 'none'; els.stage.hidden = false;
        els.startBtn.disabled = true;
        try {
            showStatus('Loading YOLO26n...');
            await modelReady;
            if (!session) throw new Error('Model could not be loaded.');
            showStatus('Connecting to the camera...');
            els.img.crossOrigin = 'anonymous';          // the stream sends CORS headers -> pixels are readable
            await new Promise((resolve, reject) => {
                const timer = setTimeout(() => reject(new Error('The camera did not answer within 20 s.')), 20000);
                els.img.onload = () => { clearTimeout(timer); resolve(); };
                els.img.onerror = () => { clearTimeout(timer); reject(new Error('Stream could not be loaded (camera offline?).')); };
                els.img.src = url;
            });
            connected = true;
            els.stopBtn.disabled = false;
            showStatus('Waiting for the first frame...');
            lastFrameAt = performance.now();
            probeTimer = setInterval(onProbe, 120);
            // reconnect if the MJPEG stream stalls
            watchdog = setInterval(() => {
                if (connected && performance.now() - lastFrameAt > 15000) {
                    showStatus('Stream stalled – reconnecting...');
                    lastFrameAt = performance.now();
                    els.img.src = `${url}${url.includes('?') ? '&' : '?'}r=${Date.now()}`;
                }
            }, 3000);
            els.streamInfo.textContent = `Connected · ${els.img.naturalWidth}×${els.img.naturalHeight} · detection on ${backendName}`;
        } catch (err) {
            showStatus(err.message);
        } finally {
            els.startBtn.disabled = false;
        }
    }

    function disconnect() {
        connected = false;
        clearInterval(probeTimer); clearInterval(watchdog);
        probeTimer = watchdog = null;
        els.img.onload = els.img.onerror = null;
        els.img.removeAttribute('src');
        els.stopBtn.disabled = true;
        tracks = []; lastSignature = null; lastThumb = null; frameTimes = [];
        els.fps.textContent = '–';
    }

    // ---------- counting-line drawing ----------
    function canvasPoint(e) {
        // the canvas is shown with object-fit: contain -> map the click into image coordinates
        const r = els.canvas.getBoundingClientRect();
        const s = Math.min(r.width / els.canvas.width, r.height / els.canvas.height);
        const dw = els.canvas.width * s, dh = els.canvas.height * s;
        const ox = r.left + (r.width - dw) / 2, oy = r.top + (r.height - dh) / 2;
        return { x: Math.min(1, Math.max(0, (e.clientX - ox) / dw)), y: Math.min(1, Math.max(0, (e.clientY - oy) / dh)) };
    }

    els.drawLineBtn.addEventListener('click', () => {
        if (!els.canvas.width) { alert('Connect to the camera first.'); return; }
        drawingLine = true; linePoints = [];
        els.canvas.classList.add('drawing');
        els.modeBadge.textContent = 'Click 2 points for the counting line';
    });

    els.canvas.addEventListener('click', (e) => {
        if (!drawingLine) return;
        linePoints.push(canvasPoint(e));
        if (linePoints.length === 2) {
            line = { a: linePoints[0], b: linePoints[1] };
            drawingLine = false;
            els.canvas.classList.remove('drawing');
            els.modeBadge.textContent = 'Tracking';
            for (const tr of tracks) tr.counted = false;
        }
        draw();
    });

    // ---------- controls ----------
    els.startBtn.addEventListener('click', connect);
    els.stopBtn.addEventListener('click', () => { disconnect(); showStatus('Stopped.'); });
    els.resetCountBtn.addEventListener('click', () => { events = []; for (const tr of tracks) tr.counted = false; renderCounts(); draw(); });
    els.resetHeatBtn.addEventListener('click', () => { heat.fill(0); renderHeatCanvas(); draw(); });
    els.conf.addEventListener('input', () => { els.confVal.textContent = `${Math.round(els.conf.value * 100)}%`; });
    els.heatOpacity.addEventListener('input', () => { els.heatOpacityVal.textContent = `${Math.round(els.heatOpacity.value * 100)}%`; draw(); });
    els.heatToggle.addEventListener('change', draw);
    els.showSpeed.addEventListener('change', draw);
    els.limit.addEventListener('input', () => { els.limitVal.textContent = `${els.limit.value} km/h`; renderCounts(); draw(); });
    els.calib.addEventListener('input', () => { els.calibVal.textContent = `×${Number(els.calib.value).toFixed(2)}`; });

    els.exportBtn.addEventListener('click', () => {
        const rows = [['time', 'track_id', 'type', 'direction', 'speed_kmh']]
            .concat(events.map((e) => [e.time.toISOString(), e.id, e.name, e.dir, e.speed == null ? '' : e.speed.toFixed(1)]));
        const blob = new Blob([rows.map((r) => r.join(',')).join('\n')], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `cctv_vehicle_counts_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    });

    setInterval(drawRateChart, 15000);   // keep the per-minute chart moving
    window.addEventListener('resize', drawRateChart);

    const modelReady = loadModel()
        .then(() => { els.modelStatus.textContent = `YOLO26n ready · ${backendName}`; })
        .catch((err) => {
            console.error(err);
            els.modelStatus.textContent = 'Model failed to load';
            els.dot.style.background = 'var(--danger)';
        });
    drawRateChart();

    // small hook for automated tests / debugging in the console
    window.cctvDebug = {
        get tracks() { return tracks; }, get events() { return events; }, get heat() { return heat; },
        get sceneChanges() { return sceneChanges; },
    };
})();
