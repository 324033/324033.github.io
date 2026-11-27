// Web Worker for Neural Style Transfer: loads the ONNX models (cached with the Cache API
// after the first download) and runs them with ONNX Runtime Web (WASM backend), so the
// page itself never freezes.
//
// Messages in:  {type:'init', ortPath}
//               {type:'run', id, style, url, width, height, input: Float32Array (NCHW RGB 0..255)}
// Messages out: {type:'ready', threads} | {type:'status', id, style, text}
//               {type:'result', id, style, width, height, rgba: Uint8ClampedArray, ms, cached}
//               {type:'error', id, style, message, stage}
// Runs as a *module* worker: ONNX Runtime's ESM build can start its own WASM threads from
// here (with importScripts in a classic worker, multi-threaded session creation hangs).

const CACHE_NAME = 'nst-models-v1';
const sessions = new Map();     // url -> Promise<InferenceSession>
let ort = null;

self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === 'init') {
        try {
            ort = await import(`${msg.ortPath}ort.webgpu.min.mjs`);
            ort.env.wasm.wasmPaths = msg.ortPath;
            const maxThreads = self.crossOriginIsolated ? Math.min(4, self.navigator.hardwareConcurrency || 2) : 1;
            ort.env.wasm.numThreads = msg.threads ? Math.min(msg.threads, maxThreads) : maxThreads;
            self.postMessage({ type: 'ready', threads: ort.env.wasm.numThreads, isolated: !!self.crossOriginIsolated });
        } catch (err) {
            self.postMessage({ type: 'error', stage: 'init', message: `ONNX Runtime could not start: ${err.message}` });
        }
    } else if (msg.type === 'probe') {
        // quick self-test: can this browser create + run a session with the chosen thread count?
        try {
            const t0 = performance.now();
            const { session } = await getSession(msg.url, () => {});
            const n = 32 * 32;
            await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * n), [1, 3, 32, 32]) });
            self.postMessage({ type: 'probe-ok', ms: performance.now() - t0 });
        } catch (err) {
            self.postMessage({ type: 'probe-failed', message: err.message });
        }
    } else if (msg.type === 'run') {
        await run(msg);
    }
};

// download with progress, store in Cache Storage so the next visit loads instantly
async function fetchModel(url, report) {
    let cache = null;
    try { cache = await caches.open(CACHE_NAME); } catch (_) { /* Cache API unavailable -> no caching */ }
    const hit = cache && (await cache.match(url));
    if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true };

    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} not found (HTTP ${res.status})`);
    const total = Number(res.headers.get('Content-Length')) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0, lastPct = -1;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        const pct = total ? Math.floor((loaded / total) * 100) : -1;
        if (pct !== lastPct && pct % 10 === 0) { lastPct = pct; report(`Downloading model ${pct}%...`); }
    }
    const bytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { bytes.set(c, off); off += c.length; }
    if (cache) {
        try { await cache.put(url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } })); } catch (_) { /* quota */ }
    }
    return { bytes, cached: false };
}

function getSession(url, report) {
    if (!sessions.has(url)) {
        const p = (async () => {
            const { bytes, cached } = await fetchModel(url, report);
            report(cached ? 'Loading model from cache...' : 'Preparing model...');
            const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
            return { session, cached };
        })();
        sessions.set(url, p);
        p.catch(() => sessions.delete(url));
    }
    return sessions.get(url);
}

async function run({ id, style, url, width, height, input }) {
    const report = (text) => self.postMessage({ type: 'status', id, style, text });
    let s;
    try {
        s = await getSession(url, report);
    } catch (err) {
        self.postMessage({ type: 'error', id, style, stage: 'load', message: err.message });
        return;
    }
    try {
        report('Processing...');
        const t0 = performance.now();
        const tensor = new ort.Tensor('float32', input, [1, 3, height, width]);
        const out = await s.session.run({ [s.session.inputNames[0]]: tensor });
        const o = out[s.session.outputNames[0]];
        const [, , oh, ow] = o.dims, v = o.data, area = ow * oh;
        const rgba = new Uint8ClampedArray(area * 4);   // clamps to 0..255 automatically
        for (let i = 0, j = 0; i < area; i++, j += 4) {
            rgba[j] = v[i]; rgba[j + 1] = v[i + area]; rgba[j + 2] = v[i + 2 * area]; rgba[j + 3] = 255;
        }
        self.postMessage({ type: 'result', id, style, width: ow, height: oh, rgba, ms: performance.now() - t0, cached: s.cached }, [rgba.buffer]);
    } catch (err) {
        self.postMessage({ type: 'error', id, style, stage: 'run', message: err.message });
    }
}
