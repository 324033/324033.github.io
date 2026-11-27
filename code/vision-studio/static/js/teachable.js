// Teachable Machine: object recognition with a custom Google Teachable Machine image model.
// Uses the official @teachablemachine/image library (global `tmImage`) on TensorFlow.js.
(() => {
    const CLASS_COLORS = ['#38bdf8', '#f97316', '#4ade80', '#a78bfa', '#facc15', '#f43f5e', '#2dd4bf', '#e879f9'];

    const $ = (id) => document.getElementById(id);
    const els = {
        urlInput: $('tmUrlInput'), loadBtn: $('tmLoadBtn'), modelInfo: $('tmModelInfo'),
        sourcePills: document.querySelectorAll('#tmSourcePills .pill-btn'),
        imageSection: $('tmImageSection'), imageInput: $('tmImageInput'), camButtons: $('tmCamButtons'),
        startBtn: $('tmStartBtn'), startLabel: $('tmStartLabel'),
        confRange: $('tmConfRange'), confVal: $('tmConfVal'), mirror: $('tmMirrorToggle'),
        viewer: $('tmViewer'), empty: $('tmEmptyState'), stage: $('tmStage'), canvas: $('tmCanvas'),
        status: $('tmStatus'), badge: $('tmBadge'), video: $('tmVideo'),
        topClass: $('tmTopClass'), topConf: $('tmTopConf'), fps: $('tmFps'), bars: $('tmBars'),
    };
    const ctx = els.canvas.getContext('2d');
    // the frame that is fed to the model (mirrored like Teachable Machine's own webcam samples)
    const inputCanvas = document.createElement('canvas');
    const inputCtx = inputCanvas.getContext('2d');

    let model = null;
    let loadedUrl = null;
    let labels = [];
    let source = 'webcam';
    let stream = null;
    let running = false;
    let rafId = null;
    let predicting = false;
    let predCount = 0, predWindowStart = 0;

    // ---------- model ----------
    function normalizeUrl(url) {
        url = url.trim();
        if (!url) throw new Error('Please enter a model URL.');
        if (!/^https?:\/\//i.test(url)) throw new Error('The model URL must start with https://');
        url = url.replace(/(model\.json|metadata\.json)$/i, '');
        return url.endsWith('/') ? url : `${url}/`;
    }

    async function loadModel() {
        const url = normalizeUrl(els.urlInput.value);
        if (model && url === loadedUrl) return model;
        if (typeof tmImage === 'undefined') throw new Error('Teachable Machine library could not be loaded (check your internet connection).');
        els.loadBtn.disabled = true;
        els.modelInfo.textContent = 'Loading model...';
        try {
            const m = await tmImage.load(`${url}model.json`, `${url}metadata.json`);
            model = m;
            loadedUrl = url;
            labels = m.getClassLabels();
            els.modelInfo.innerHTML = `Model loaded &check; &ndash; ${labels.length} classes: <strong>${labels.map(escapeHtml).join(', ')}</strong>`;
            buildBars();
            return m;
        } catch (err) {
            model = null;
            els.modelInfo.textContent = `Could not load the model: ${err.message}`;
            throw err;
        } finally {
            els.loadBtn.disabled = false;
        }
    }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    function buildBars() {
        els.bars.innerHTML = labels.map((label, i) => `
            <div class="tm-bar-row" data-i="${i}" style="--c:${CLASS_COLORS[i % CLASS_COLORS.length]}">
                <span class="tm-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
                <div class="tm-bar-bg"><div class="tm-bar-fill"></div></div>
                <span class="tm-pct">0%</span>
            </div>`).join('');
    }

    // ---------- results ----------
    function showPrediction(prediction) {
        const rows = els.bars.querySelectorAll('.tm-bar-row');
        let best = 0;
        prediction.forEach((p, i) => { if (p.probability > prediction[best].probability) best = i; });
        prediction.forEach((p, i) => {
            const row = rows[i];
            if (!row) return;
            const pct = Math.round(p.probability * 100);
            row.querySelector('.tm-bar-fill').style.width = `${pct}%`;
            row.querySelector('.tm-pct').textContent = `${pct}%`;
            row.classList.toggle('top', i === best);
        });
        const top = prediction[best];
        const unsure = top.probability < Number(els.confRange.value);
        const name = unsure ? `Unsure (${top.className}?)` : top.className;
        els.topClass.textContent = name;
        els.topConf.textContent = `${Math.round(top.probability * 100)}%`;
        els.badge.textContent = `${name} · ${Math.round(top.probability * 100)}%`;
        els.badge.style.background = unsure ? 'rgba(15,23,42,0.8)' : CLASS_COLORS[best % CLASS_COLORS.length];
        els.badge.style.color = unsure ? '#fff' : '#0f172a';
    }

    function countPrediction() {
        const now = performance.now();
        predCount += 1;
        if (!predWindowStart) predWindowStart = now;
        if (now - predWindowStart >= 1000) {
            els.fps.textContent = Math.round((predCount * 1000) / (now - predWindowStart));
            predCount = 0; predWindowStart = now;
        }
    }

    // ---------- webcam ----------
    async function startCamera() {
        els.startBtn.disabled = true;
        showStage('Loading model...');
        try {
            await loadModel();
            if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not support camera access.');
            els.status.textContent = 'Waiting for camera permission...';
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
            });
            const video = els.video;
            video.srcObject = stream;
            await new Promise((res) => (video.readyState >= 2 ? res() : (video.onloadeddata = res)));
            await video.play();
            running = true;
            predCount = 0; predWindowStart = 0;
            els.status.style.display = 'none';
            els.startLabel.textContent = 'Stop Camera';
            loop();
        } catch (err) {
            console.error(err);
            const msg = err.name === 'NotAllowedError' ? 'Camera access was denied. Allow it in the browser and try again.'
                : err.name === 'NotFoundError' ? 'No camera found.' : err.message;
            els.status.style.display = 'block';
            els.status.textContent = msg;
            stopCamera();
        } finally {
            els.startBtn.disabled = false;
        }
    }

    function stopCamera() {
        running = false;
        if (rafId) cancelAnimationFrame(rafId);
        rafId = null;
        if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
        els.video.pause();
        els.startLabel.textContent = 'Start Camera';
    }

    function loop() {
        if (!running) return;
        const video = els.video;
        const w = video.videoWidth, h = video.videoHeight;
        if (w && h) {
            for (const c of [els.canvas, inputCanvas]) {
                if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
            }
            inputCtx.save();
            if (els.mirror.checked) { inputCtx.translate(w, 0); inputCtx.scale(-1, 1); }
            inputCtx.drawImage(video, 0, 0, w, h);
            inputCtx.restore();

            ctx.drawImage(inputCanvas, 0, 0);
            drawCropGuide(w, h);

            if (!predicting && model) {
                predicting = true;
                // the library centre-crops the input to a square, exactly like the guide shows
                model.predict(inputCanvas)
                    .then((pred) => { if (running) { showPrediction(pred); countPrediction(); } })
                    .catch((err) => console.error(err))
                    .finally(() => { predicting = false; });
            }
        }
        rafId = requestAnimationFrame(loop);
    }

    // Teachable Machine models look at the centre square of the frame -> show that area
    function drawCropGuide(w, h) {
        const s = Math.min(w, h), x = (w - s) / 2, y = (h - s) / 2;
        ctx.fillStyle = 'rgba(0,0,0,0.45)';
        ctx.fillRect(0, 0, x, h);
        ctx.fillRect(x + s, 0, w - x - s, h);
        ctx.fillRect(x, 0, s, y);
        ctx.fillRect(x, y + s, s, h - y - s);
        ctx.strokeStyle = 'rgba(255,255,255,0.8)';
        ctx.lineWidth = Math.max(2, w / 400);
        ctx.setLineDash([12, 8]);
        ctx.strokeRect(x, y, s, s);
        ctx.setLineDash([]);
    }

    // ---------- image upload ----------
    async function classifyFile(file) {
        if (!file || !file.type.startsWith('image/')) { alert('Please choose an image file.'); return; }
        if (running) stopCamera();
        showStage('Loading model...');
        try {
            await loadModel();
            const img = new Image();
            img.src = URL.createObjectURL(file);
            await img.decode();
            const w = img.naturalWidth, h = img.naturalHeight;
            els.canvas.width = w; els.canvas.height = h;
            ctx.drawImage(img, 0, 0);
            drawCropGuide(w, h);
            els.status.textContent = 'Recognizing...';
            const pred = await model.predict(img);
            showPrediction(pred);
            els.status.style.display = 'none';
            URL.revokeObjectURL(img.src);
        } catch (err) {
            console.error(err);
            els.status.style.display = 'block';
            els.status.textContent = err.message;
        }
    }

    function showStage(statusText) {
        els.empty.style.display = 'none';
        els.stage.style.display = 'block';
        els.status.style.display = 'block';
        els.status.textContent = statusText;
    }

    // ---------- wiring ----------
    els.loadBtn.addEventListener('click', () => { loadModel().catch(() => {}); });
    els.confRange.addEventListener('input', () => { els.confVal.textContent = `${Math.round(els.confRange.value * 100)}%`; });
    els.startBtn.addEventListener('click', () => (running ? stopCamera() : startCamera()));
    els.imageInput.addEventListener('change', () => classifyFile(els.imageInput.files?.[0]));

    els.sourcePills.forEach((btn) => btn.addEventListener('click', () => {
        els.sourcePills.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        source = btn.dataset.tmsource;
        els.imageSection.style.display = source === 'image' ? 'block' : 'none';
        els.camButtons.style.display = source === 'webcam' ? 'flex' : 'none';
        if (source === 'image' && running) stopCamera();
    }));

    ['dragenter', 'dragover'].forEach((ev) => els.viewer.addEventListener(ev, (e) => {
        e.preventDefault(); els.viewer.classList.add('drag-active');
    }));
    ['dragleave', 'drop'].forEach((ev) => els.viewer.addEventListener(ev, (e) => {
        e.preventDefault(); els.viewer.classList.remove('drag-active');
    }));
    els.viewer.addEventListener('drop', (e) => classifyFile(e.dataTransfer.files?.[0]));

    document.addEventListener('modechange', (e) => {
        if (e.detail.mode === 'tm') {
            if (!model) loadModel().catch(() => {});    // load the model as soon as the tab opens
        } else if (running) {
            stopCamera();
        }
    });
})();
