// Vision Studio "browser server" for the static GitHub Pages build.
//
// The local app sends images/videos to the Python (Flask) server: /api/detect, /api/facemesh/detect,
// /api/video/start and /api/video/status/<id>. On GitHub Pages there is no Python, so this module
// intercepts exactly those fetch() calls and computes the same results in the browser with
// MediaPipe Tasks for JavaScript. The page (app.js) itself stays unchanged.
import { FilesetResolver, ObjectDetector, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODELS = 'https://storage.googleapis.com/mediapipe-models/';
const OBJECT_MODEL = `${MODELS}object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite`;
const FACE_MODEL = `${MODELS}face_landmarker/face_landmarker/float16/1/face_landmarker.task`;

// same colours as app.py (COLORS: RGB, FACE_PALETTES: BGR)
const COLORS = [[59, 130, 246], [16, 185, 129], [245, 158, 11], [239, 68, 68], [139, 92, 246],
    [236, 72, 153], [20, 184, 166], [249, 115, 22], [99, 102, 241], [168, 85, 247]];
const FACE_PALETTES_BGR = {
    cyan: { mesh: [240, 220, 0], contour: [255, 255, 0], point: [255, 255, 200], iris: [50, 220, 255] },
    magenta: { mesh: [220, 0, 180], contour: [255, 50, 230], point: [255, 180, 255], iris: [0, 255, 255] },
    green: { mesh: [100, 230, 40], contour: [140, 255, 80], point: [200, 255, 180], iris: [255, 240, 0] },
    purple: { mesh: [230, 80, 150], contour: [255, 120, 200], point: [255, 200, 255], iris: [255, 200, 50] },
    gold: { mesh: [0, 180, 240], contour: [0, 215, 255], point: [180, 240, 255], iris: [255, 100, 0] },
    white: { mesh: [190, 190, 190], contour: [255, 255, 255], point: [255, 255, 255], iris: [255, 200, 120] },
};
const rgbCss = ([r, g, b]) => `rgb(${r}, ${g}, ${b})`;
const bgrCss = ([b, g, r]) => `rgb(${r}, ${g}, ${b})`;
const classColor = (label) => COLORS[[...label].reduce((s, c) => s + c.charCodeAt(0), 0) % COLORS.length];

// ---------- lazily created MediaPipe tasks ----------
let fileset = null;
const tasks = {};
async function vision() { return fileset || (fileset = await FilesetResolver.forVisionTasks(WASM)); }

async function create(Cls, options) {
    const fs = await vision();
    try { return await Cls.createFromOptions(fs, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } }); } catch (e) {
        console.warn('GPU delegate failed, using CPU', e);
        return Cls.createFromOptions(fs, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
    }
}

async function objectDetector(mode, scoreThreshold, maxResults) {
    const key = `obj-${mode}`;
    if (!tasks[key]) tasks[key] = create(ObjectDetector, { baseOptions: { modelAssetPath: OBJECT_MODEL }, runningMode: mode, scoreThreshold, maxResults });
    const d = await tasks[key];
    await d.setOptions({ scoreThreshold, maxResults });
    return d;
}

async function faceLandmarker(mode, numFaces) {
    const key = `face-${mode}-${numFaces}`;
    if (!tasks[key]) {
        tasks[key] = create(FaceLandmarker, {
            baseOptions: { modelAssetPath: FACE_MODEL }, runningMode: mode, numFaces,
            minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5,
        });
    }
    return tasks[key];
}

// ---------- drawing (mirrors annotate_image / annotate_face_mesh in app.py) ----------
function drawDetections(ctx, w, h, detections) {
    const base = Math.max(w, h);
    const lw = Math.max(2, Math.floor(base / 400));
    const fs = Math.max(11, Math.round((base / 1200) * 24));
    ctx.font = `${fs}px "Segoe UI", Arial, sans-serif`;
    ctx.textBaseline = 'alphabetic';
    for (const d of detections) {
        const b = d.bounding_box, color = classColor(d.category_name);
        ctx.lineWidth = lw; ctx.strokeStyle = rgbCss(color);
        ctx.strokeRect(b.origin_x, b.origin_y, b.width, b.height);
        const caption = `${d.category_name} ${Math.floor(d.score * 100)}%`;
        const tw = ctx.measureText(caption).width, th = fs + 8;
        const y1 = Math.max(0, b.origin_y - th);
        ctx.fillStyle = rgbCss(color); ctx.fillRect(b.origin_x, y1, tw + 12, th);
        ctx.fillStyle = '#fff'; ctx.fillText(caption, b.origin_x + 6, y1 + th - 6);
    }
}

function contoursConnections() {
    if (FaceLandmarker.FACE_LANDMARKS_CONTOURS) return FaceLandmarker.FACE_LANDMARKS_CONTOURS;
    return [FaceLandmarker.FACE_LANDMARKS_LIPS, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL].flat();
}

function drawFaceMesh(ctx, w, h, faces, opt) {
    const pal = FACE_PALETTES_BGR[opt.color_theme] || FACE_PALETTES_BGR.cyan;
    const style = opt.mesh_style, thick = Math.max(1, Number(opt.line_thickness) || 1);
    const line = (pts, conns, color, width) => {
        ctx.strokeStyle = bgrCss(color); ctx.lineWidth = width; ctx.beginPath();
        for (const c of conns) { const a = pts[c.start], b = pts[c.end]; if (!a || !b) continue; ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
        ctx.stroke();
    };
    for (const lm of faces) {
        const pts = lm.map((p) => [p.x * w, p.y * h]);
        if (style === 'both' || style === 'tesselation') line(pts, FaceLandmarker.FACE_LANDMARKS_TESSELATION, pal.mesh, 1);
        if (style === 'both' || style === 'contours' || style === 'contours_irises') line(pts, contoursConnections(), pal.contour, thick);
        if ((style === 'both' || style === 'contours_irises') && pts.length >= 478) {
            line(pts, [...FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, ...FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS], pal.iris, thick);
        }
        if (opt.draw_points || style === 'points') {
            ctx.fillStyle = bgrCss(pal.point);
            for (const [x, y] of pts.slice(0, 468)) { ctx.beginPath(); ctx.arc(x, y, thick, 0, Math.PI * 2); ctx.fill(); }
        }
    }
}

function badge(ctx, text, w) {
    const fs = Math.max(13, Math.round(w / 55));
    ctx.font = `600 ${fs}px "Segoe UI", Arial, sans-serif`;
    ctx.lineWidth = 4; ctx.strokeStyle = '#000'; ctx.strokeText(text, 18, fs + 14);
    ctx.fillStyle = '#fff'; ctx.fillText(text, 18, fs + 14);
}

// ---------- helpers ----------
const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json' } });

async function decodeImage(file) {
    if (!(file instanceof Blob) || !file.type.startsWith('image/')) throw new Error('Could not decode image');
    try { return await createImageBitmap(file); } catch (_) { throw new Error('Could not decode image'); }
}

function canvasFor(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return [c, c.getContext('2d')];
}

// ---------- /api/detect (1.4 image object detection) ----------
async function apiDetect(form) {
    const img = await decodeImage(form.get('image'));
    const threshold = Math.min(0.95, Math.max(0.05, Number(form.get('threshold')) || 0.3));
    const maxResults = Math.min(50, Math.max(1, Number(form.get('max_results')) || 25));
    const det = await objectDetector('IMAGE', threshold, maxResults);
    const t0 = performance.now();
    const res = det.detect(img);
    const ms = Math.round((performance.now() - t0) * 10) / 10;
    const w = img.width, h = img.height;
    const detections = res.detections.filter((d) => d.categories?.length).map((d) => {
        const c = d.categories[0], b = d.boundingBox;
        return {
            category_name: c.categoryName, score: Math.round(c.score * 1e4) / 1e4,
            bounding_box: { origin_x: Math.round(b.originX), origin_y: Math.round(b.originY), width: Math.round(b.width), height: Math.round(b.height) },
            color: `rgb(${classColor(c.categoryName).join(', ')})`,
        };
    });
    const [canvas, ctx] = canvasFor(w, h);
    ctx.drawImage(img, 0, 0);
    drawDetections(ctx, w, h, detections);
    const counts = {};
    for (const d of detections) counts[d.category_name] = (counts[d.category_name] || 0) + 1;
    return json({
        success: true, image_width: w, image_height: h, inference_time_ms: ms, total_detected: detections.length,
        class_counts: counts, detections, annotated_image: canvas.toDataURL('image/jpeg', 0.92),
    });
}

// ---------- /api/facemesh/detect (2.1 image face mesh) ----------
async function apiFaceMesh(form) {
    const img = await decodeImage(form.get('image'));
    const numFaces = Math.min(5, Math.max(1, Number(form.get('num_faces')) || 2));
    const lm = await faceLandmarker('IMAGE', numFaces);
    const t0 = performance.now();
    const res = lm.detect(img);
    const ms = Math.round((performance.now() - t0) * 10) / 10;
    const w = img.width, h = img.height;
    const [canvas, ctx] = canvasFor(w, h);
    if (form.get('background') === 'black') { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h); } else ctx.drawImage(img, 0, 0);
    drawFaceMesh(ctx, w, h, res.faceLandmarks || [], {
        mesh_style: form.get('mesh_style') || 'both', color_theme: form.get('color_theme') || 'cyan',
        line_thickness: form.get('line_thickness'), draw_points: form.get('draw_points') === 'true',
    });
    return json({
        success: true, image_width: w, image_height: h, inference_time_ms: ms,
        faces_detected: (res.faceLandmarks || []).length, landmarks_per_face: 468,
        annotated_image: canvas.toDataURL('image/jpeg', 0.92),
    });
}

// ---------- /api/video/start + /api/video/status (1.5 / 2.2 video analysis) ----------
const jobs = {};

function pickRecorderType() {
    for (const t of ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
}

async function apiVideoStart(form) {
    const task = form.get('video_task') || 'object';
    if (form.get('source_type') === 'youtube') {
        return json({ error: 'YouTube videos need the local Python server (the browser may not read frames from YouTube). Online, please upload a video file.' }, 400);
    }
    if (task === 'fitness') {
        return json({ error: 'Arm-rep video analysis runs on the local Python server. Online, use the "Fitness Coach" tab (it also accepts a video file).' }, 400);
    }
    const file = form.get('video');
    if (!(file instanceof Blob) || !file.size) return json({ error: 'No video file uploaded' }, 400);
    if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) {
        return json({ error: 'This browser cannot record the annotated video (MediaRecorder missing). Please use a current Chrome, Edge or Firefox.' }, 400);
    }
    const id = Math.random().toString(16).slice(2, 10);
    jobs[id] = { job_id: id, video_task: task, status: 'queued', progress: 0, step: 'Loading the AI model...', proc_fps: 0, total_detections: 0, class_counts: {}, error: null };
    runVideoJob(id, file, task, form).catch((err) => {
        console.error(err);
        Object.assign(jobs[id], { status: 'failed', error: err.message || String(err) });
    });
    return json({ success: true, job_id: id });
}

async function runVideoJob(id, file, task, form) {
    const job = jobs[id];
    const duration = Math.min(120, Math.max(3, Number(form.get('duration')) || 15));
    const sampleFps = Math.min(30, Math.max(2, Number(form.get('sample_fps')) || 12));
    const threshold = Math.min(0.95, Math.max(0.05, Number(form.get('threshold')) || 0.3));
    const maxResults = Math.min(50, Math.max(1, Number(form.get('max_results')) || 20));
    const meshOpt = {
        background: form.get('background') || 'original', mesh_style: form.get('mesh_style') || 'both',
        color_theme: form.get('color_theme') || 'cyan', line_thickness: form.get('line_thickness') || 1,
        draw_points: form.get('draw_points') === 'true',
    };
    const model = task === 'facemesh' ? await faceLandmarker('VIDEO', 3) : await objectDetector('VIDEO', threshold, maxResults);

    const video = document.createElement('video');
    video.muted = true; video.playsInline = true; video.preload = 'auto';
    video.src = URL.createObjectURL(file);
    await new Promise((res, rej) => { video.onloadeddata = res; video.onerror = () => rej(new Error('OpenCV could not open video stream.'.replace('OpenCV', 'The browser'))); });
    let w = video.videoWidth, h = video.videoHeight;
    if (w > 1280) { h = Math.round((h * 1280) / w / 2) * 2; w = 1280; }
    const [canvas, ctx] = canvasFor(w, h);
    const endTime = Math.min(duration, video.duration || duration);

    const type = pickRecorderType();
    const recorder = new MediaRecorder(canvas.captureStream(30), type ? { mimeType: type, videoBitsPerSecond: 4e6 } : undefined);
    const chunks = [];
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    const stopped = new Promise((res) => { recorder.onstop = res; });

    job.status = 'processing';
    job.step = 'Analysing the video while it plays...';
    job.progress = 15;
    let last = null, lastRun = -1, processed = 0;
    const t0 = performance.now();
    const counts = {};
    const label = task === 'facemesh' ? 'faces' : 'objects';

    recorder.start(250);
    await video.play();
    await new Promise((resolve) => {
        const step = () => {
            const t = video.currentTime;
            if (video.ended || t >= endTime) { resolve(); return; }
            if (lastRun < 0 || t - lastRun >= 1 / sampleFps - 1e-3) {
                lastRun = t;
                const now = performance.now();
                if (task === 'facemesh') {
                    const r = model.detectForVideo(video, now);
                    last = r.faceLandmarks || [];
                    job.total_detections += last.length;
                    if (last.length) counts.face = (counts.face || 0) + last.length;
                } else {
                    const r = model.detectForVideo(video, now);
                    last = r.detections.filter((d) => d.categories?.length).map((d) => {
                        const c = d.categories[0], b = d.boundingBox;
                        return { category_name: c.categoryName, score: c.score,
                            bounding_box: { origin_x: b.originX * (w / video.videoWidth), origin_y: b.originY * (h / video.videoHeight), width: b.width * (w / video.videoWidth), height: b.height * (h / video.videoHeight) } };
                    });
                    job.total_detections += last.length;
                    for (const d of last) counts[d.category_name] = (counts[d.category_name] || 0) + 1;
                }
                processed += 1;
            }
            if (task === 'facemesh' && meshOpt.background === 'black') { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, w, h); } else ctx.drawImage(video, 0, 0, w, h);
            if (last) {
                if (task === 'facemesh') drawFaceMesh(ctx, w, h, last, meshOpt); else drawDetections(ctx, w, h, last);
                badge(ctx, `${task === 'facemesh' ? 'Faces' : 'Objects'}: ${last.length} | ${t.toFixed(1)}s`, w);
            }
            const secs = Math.max(0.001, (performance.now() - t0) / 1000);
            job.proc_fps = Math.round((processed / secs) * 10) / 10;
            job.progress = Math.min(98, 15 + Math.round((t / endTime) * 83));
            job.step = `Analyzing frame ${processed} (${job.proc_fps} FPS) &bull; Found ${job.total_detections} ${label}`;
            job.class_counts = { ...counts };
            if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(step); else requestAnimationFrame(step);
        };
        if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(step); else requestAnimationFrame(step);
    });
    video.pause();
    recorder.stop();
    await stopped;
    const blob = new Blob(chunks, { type: (type || 'video/webm').split(';')[0] });
    const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
    const dl = document.getElementById('videoDownloadBtn');
    if (dl) dl.download = `annotated_video.${ext}`;
    Object.assign(job, {
        status: 'completed', progress: 100, step: 'Detection complete!', video_url: URL.createObjectURL(blob),
        processed_frames: processed, elapsed_seconds: Math.round((performance.now() - t0) / 100) / 10,
        class_counts: counts, width: w, height: h,
    });
    URL.revokeObjectURL(video.src);
}

function apiVideoStatus(id) {
    return jobs[id] ? json(jobs[id]) : json({ error: 'Job not found' }, 404);
}

// ---------- fetch interception ----------
const realFetch = window.fetch.bind(window);
window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, window.location.href);
    if (url.origin === window.location.origin && url.pathname.startsWith('/api/')) {
        try {
            const p = url.pathname;
            if (p === '/api/detect') return await apiDetect(init.body);
            if (p === '/api/facemesh/detect') return await apiFaceMesh(init.body);
            if (p === '/api/video/start') return await apiVideoStart(init.body);
            if (p.startsWith('/api/video/status/')) return apiVideoStatus(p.split('/').pop());
            return json({ error: 'This feature needs the local Python server.' }, 404);
        } catch (err) {
            console.error(err);
            return json({ error: err.message || String(err) }, 400);
        }
    }
    return realFetch(input, init);
};

// ---------- open the right tab from the link (#object, #facemesh, #video, #video-facemesh) ----------
function openFromHash() {
    const hash = window.location.hash.replace('#', '');
    const tab = { object: 'object', facemesh: 'facemesh', video: 'video', 'video-facemesh': 'video' }[hash];
    if (!tab) return;
    document.querySelector(`.nav-tab[data-mode="${tab}"]`)?.click();
    if (hash === 'video-facemesh') document.querySelector('[data-vtask="facemesh"]')?.click();
    if (hash === 'video') document.querySelector('[data-vtask="object"]')?.click();
}
// app.js wires up the tabs on DOMContentLoaded -> wait for "load" so the clicks are handled
if (document.readyState === 'complete') openFromHash();
else window.addEventListener('load', () => setTimeout(openFromHash, 0));
window.addEventListener('hashchange', openFromHash);
