document.addEventListener('DOMContentLoaded', () => {
    // Top Mode Navigation (Object vs Face Mesh vs Video)
    const navTabs = document.querySelectorAll('.nav-tab[data-mode]');   // links (e.g. YOLO26 page) are excluded
    const objectModeSection = document.getElementById('objectModeSection');
    const faceMeshModeSection = document.getElementById('faceMeshModeSection');
    const videoModeSection = document.getElementById('videoModeSection');
    const fitnessModeSection = document.getElementById('fitnessModeSection');
    const gestureModeSection = document.getElementById('gestureModeSection');
    const backgroundModeSection = document.getElementById('backgroundModeSection');
    const faceKeysModeSection = document.getElementById('faceKeysModeSection');
    const tmModeSection = document.getElementById('tmModeSection');

    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            const mode = tab.dataset.mode;
            objectModeSection.style.display = mode === 'object' ? 'grid' : 'none';
            faceMeshModeSection.style.display = mode === 'facemesh' ? 'grid' : 'none';
            videoModeSection.style.display = mode === 'video' ? 'grid' : 'none';
            fitnessModeSection.style.display = mode === 'fitness' ? 'grid' : 'none';
            gestureModeSection.style.display = mode === 'gestures' ? 'grid' : 'none';
            backgroundModeSection.style.display = mode === 'background' ? 'grid' : 'none';
            faceKeysModeSection.style.display = mode === 'facekeys' ? 'grid' : 'none';
            tmModeSection.style.display = mode === 'tm' ? 'grid' : 'none';
            // let the live coach (fitness.js) stop the camera when leaving the tab
            document.dispatchEvent(new CustomEvent('modechange', { detail: { mode } }));
        });
    });

    // ==========================================
    // 1. OBJECT DETECTION LOGIC
    // ==========================================
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const thresholdRange = document.getElementById('thresholdRange');
    const thresholdVal = document.getElementById('thresholdVal');
    const maxResultsSelect = document.getElementById('maxResultsSelect');
    const reanalyzeBtn = document.getElementById('reanalyzeBtn');
    const sampleBtns = document.querySelectorAll('.btn-sample');

    const emptyState = document.getElementById('emptyState');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const displayContainer = document.getElementById('displayContainer');
    const metricsBar = document.getElementById('metricsBar');
    const detailsCard = document.getElementById('detailsCard');
    const resultImage = document.getElementById('resultImage');
    const downloadBtn = document.getElementById('downloadBtn');

    const metricCount = document.getElementById('metricCount');
    const metricTime = document.getElementById('metricTime');
    const metricResolution = document.getElementById('metricResolution');
    const tagsContainer = document.getElementById('tagsContainer');
    const detectionsTableBody = document.getElementById('detectionsTableBody');
    const tabBtns = document.querySelectorAll('#objectModeSection .tab-btn');

    let currentOriginalDataUrl = null;
    let currentAnnotatedDataUrl = null;
    let currentFile = null;
    let currentViewTab = 'annotated';

    thresholdRange.addEventListener('input', (e) => {
        thresholdVal.textContent = `${Math.round(e.target.value * 100)}%`;
        if (currentFile) reanalyzeBtn.disabled = false;
    });

    maxResultsSelect.addEventListener('change', () => {
        if (currentFile) reanalyzeBtn.disabled = false;
    });

    reanalyzeBtn.addEventListener('click', () => {
        if (currentFile) uploadAndDetectImage(currentFile);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleNewImageFile(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-active');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleNewImageFile(dt.files[0]);
        }
    });

    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.kind === 'file') {
                const blob = item.getAsFile();
                if (blob.type.startsWith('image/')) {
                    if (faceMeshModeSection.style.display !== 'none') {
                        handleNewFaceFile(blob);
                    } else if (objectModeSection.style.display !== 'none') {
                        handleNewImageFile(blob);
                    }
                } else if (blob.type.startsWith('video/')) {
                    handleNewVideoFile(blob);
                }
                break;
            }
        }
    });

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentViewTab = btn.dataset.tab;
            updateDisplayedImage();
        });
    });

    function updateDisplayedImage() {
        if (currentViewTab === 'annotated' && currentAnnotatedDataUrl) {
            resultImage.src = currentAnnotatedDataUrl;
        } else if (currentViewTab === 'original' && currentOriginalDataUrl) {
            resultImage.src = currentOriginalDataUrl;
        }
    }

    function handleNewImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPEG, PNG, WEBP).');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            currentOriginalDataUrl = e.target.result;
            uploadAndDetectImage(file);
        };
        reader.readAsDataURL(file);
    }

    async function uploadAndDetectImage(file) {
        setImageLoading(true);
        reanalyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', file);
        formData.append('threshold', thresholdRange.value);
        formData.append('max_results', maxResultsSelect.value);

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error occurred during detection.');
            }

            const data = await response.json();
            renderImageResults(data);
        } catch (err) {
            console.error(err);
            alert(`Detection failed: ${err.message}`);
        } finally {
            setImageLoading(false);
        }
    }

    function setImageLoading(isLoading) {
        loadingOverlay.style.display = isLoading ? 'flex' : 'none';
    }

    function renderImageResults(data) {
        currentAnnotatedDataUrl = data.annotated_image;

        emptyState.style.display = 'none';
        displayContainer.style.display = 'flex';
        metricsBar.style.display = 'flex';
        detailsCard.style.display = 'block';

        metricCount.textContent = data.total_detected;
        metricTime.textContent = `${data.inference_time_ms} ms`;
        metricResolution.innerHTML = `${data.image_width} &times; ${data.image_height}`;

        updateDisplayedImage();
        downloadBtn.href = currentAnnotatedDataUrl;

        tagsContainer.innerHTML = '';
        const counts = data.class_counts || {};
        const entries = Object.entries(counts);
        if (entries.length === 0) {
            tagsContainer.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.85rem;">No objects met the current confidence threshold.</span>';
        } else {
            entries.forEach(([label, count]) => {
                const tag = document.createElement('span');
                tag.className = 'class-tag';
                tag.innerHTML = `${label} <span class="badge-count">${count}</span>`;
                tagsContainer.appendChild(tag);
            });
        }

        detectionsTableBody.innerHTML = '';
        data.detections.forEach((det, idx) => {
            const tr = document.createElement('tr');
            const pct = Math.round(det.score * 100);
            const bbox = det.bounding_box;

            tr.innerHTML = `
                <td style="color: var(--text-secondary);">${idx + 1}</td>
                <td>
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: ${det.color}; margin-right: 6px;"></span>
                    <strong>${det.category_name}</strong>
                </td>
                <td>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: ${pct}%;"></div>
                        </div>
                        <span style="font-weight: 600; width: 36px; text-align: right;">${pct}%</span>
                    </div>
                </td>
                <td style="color: var(--text-secondary); font-family: monospace;">
                    [${bbox.origin_x}, ${bbox.origin_y}, ${bbox.width}, ${bbox.height}]
                </td>
            `;
            detectionsTableBody.appendChild(tr);
        });
    }

    sampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleType = btn.dataset.sample;
            generateSampleImageBlob(sampleType).then(blob => {
                handleNewImageFile(blob);
            });
        });
    });

    async function generateSampleImageBlob(type) {
        const canvas = document.createElement('canvas');
        canvas.width = 800;
        canvas.height = 600;
        const ctx = canvas.getContext('2d');

        if (type === 'street') {
            const skyGrad = ctx.createLinearGradient(0, 0, 0, 350);
            skyGrad.addColorStop(0, '#60a5fa');
            skyGrad.addColorStop(1, '#93c5fd');
            ctx.fillStyle = skyGrad;
            ctx.fillRect(0, 0, 800, 350);

            ctx.fillStyle = '#334155';
            ctx.fillRect(0, 350, 800, 250);

            ctx.fillStyle = '#fde047';
            for (let x = 30; x < 800; x += 120) {
                ctx.fillRect(x, 470, 70, 10);
            }

            drawCar(ctx, 160, 360, '#ef4444');
            drawCar(ctx, 480, 390, '#3b82f6');
            drawPerson(ctx, 80, 280, '#fbbf24');
            drawPerson(ctx, 720, 290, '#34d399');
        } else if (type === 'workspace') {
            ctx.fillStyle = '#cbd5e1';
            ctx.fillRect(0, 0, 800, 220);
            ctx.fillStyle = '#78350f';
            ctx.fillRect(0, 220, 800, 380);

            drawLaptop(ctx, 280, 260);
            drawCup(ctx, 620, 320);
            drawBook(ctx, 100, 340);
        } else {
            ctx.fillStyle = '#f1f5f9';
            ctx.fillRect(0, 0, 800, 380);
            ctx.fillStyle = '#b45309';
            ctx.fillRect(0, 380, 800, 220);

            ctx.fillStyle = '#475569';
            ctx.fillRect(150, 260, 500, 150);
            ctx.fillStyle = '#334155';
            ctx.fillRect(120, 320, 560, 90);

            drawPet(ctx, 360, 310);
            drawPlant(ctx, 60, 260);
        }

        return new Promise(resolve => {
            canvas.toBlob(blob => {
                const file = new File([blob], `sample_${type}.jpg`, { type: 'image/jpeg' });
                resolve(file);
            }, 'image/jpeg', 0.92);
        });
    }

    function drawCar(ctx, x, y, color) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(x, y + 40, 200, 50, [10, 10, 5, 5]);
        ctx.fill();
        ctx.beginPath();
        ctx.roundRect(x + 35, y, 130, 45, [18, 18, 0, 0]);
        ctx.fill();
        ctx.fillStyle = '#bae6fd';
        ctx.fillRect(x + 45, y + 6, 50, 30);
        ctx.fillRect(x + 105, y + 6, 50, 30);
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(x + 45, y + 90, 20, 0, Math.PI * 2);
        ctx.arc(x + 155, y + 90, 20, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#94a3b8';
        ctx.beginPath();
        ctx.arc(x + 45, y + 90, 8, 0, Math.PI * 2);
        ctx.arc(x + 155, y + 90, 8, 0, Math.PI * 2);
        ctx.fill();
    }

    function drawPerson(ctx, x, y, shirtColor) {
        ctx.fillStyle = '#fed7aa';
        ctx.beginPath();
        ctx.arc(x + 15, y + 15, 14, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = shirtColor;
        ctx.fillRect(x, y + 32, 30, 55);
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(x + 2, y + 87, 11, 65);
        ctx.fillRect(x + 17, y + 87, 11, 65);
    }

    function drawLaptop(ctx, x, y) {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(x + 20, y, 200, 130);
        ctx.fillStyle = '#0284c7';
        ctx.fillRect(x + 28, y + 8, 184, 114);
        ctx.fillStyle = '#94a3b8';
        ctx.beginPath();
        ctx.roundRect(x, y + 128, 240, 16, [4, 4, 8, 8]);
        ctx.fill();
    }

    function drawCup(ctx, x, y) {
        ctx.fillStyle = '#f8fafc';
        ctx.beginPath();
        ctx.roundRect(x, y, 40, 65, [2, 2, 8, 8]);
        ctx.fill();
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.arc(x + 42, y + 30, 14, -Math.PI / 2, Math.PI / 2);
        ctx.stroke();
    }

    function drawBook(ctx, x, y) {
        ctx.fillStyle = '#dc2626';
        ctx.fillRect(x, y, 100, 60);
        ctx.fillStyle = '#fef2f2';
        ctx.fillRect(x + 5, y + 5, 90, 50);
    }

    function drawPet(ctx, x, y) {
        ctx.fillStyle = '#d97706';
        ctx.beginPath();
        ctx.ellipse(x + 50, y + 40, 55, 30, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x + 105, y + 25, 24, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(x + 115, y + 36, 10, 18, Math.PI / 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#d97706';
        ctx.lineWidth = 8;
        ctx.beginPath();
        ctx.moveTo(x, y + 35);
        ctx.quadraticCurveTo(x - 20, y + 15, x - 15, y - 5);
        ctx.stroke();
    }

    function drawPlant(ctx, x, y) {
        ctx.fillStyle = '#ea580c';
        ctx.beginPath();
        ctx.moveTo(x + 10, y + 80);
        ctx.lineTo(x + 70, y + 80);
        ctx.lineTo(x + 60, y + 150);
        ctx.lineTo(x + 20, y + 150);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = '#16a34a';
        for (let i = 0; i < 6; i++) {
            ctx.beginPath();
            ctx.ellipse(x + 40 + (i - 2.5) * 15, y + 50 - Math.abs(i - 2.5) * 10, 16, 35, (i - 2.5) * 0.25, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // ==========================================
    // 2. FACE MESH (468 3D LANDMARKS) LOGIC
    // ==========================================
    const faceDropzone = document.getElementById('faceDropzone');
    const faceFileInput = document.getElementById('faceFileInput');
    const faceDropText = document.getElementById('faceDropText');
    const sampleFaceBtns = document.querySelectorAll('.btn-sample-face');

    const segmentBtns = document.querySelectorAll('.segment-btn[data-bg]');
    const meshStyleSelect = document.getElementById('meshStyleSelect');
    const colorSwatches = document.querySelectorAll('#faceColorSwatches .color-swatch');
    const faceThicknessRange = document.getElementById('faceThicknessRange');
    const faceThicknessVal = document.getElementById('faceThicknessVal');
    const drawPointsCheck = document.getElementById('drawPointsCheck');
    const reanalyzeFaceBtn = document.getElementById('reanalyzeFaceBtn');

    const faceMetricsBar = document.getElementById('faceMetricsBar');
    const faceMetricCount = document.getElementById('faceMetricCount');
    const faceMetricTime = document.getElementById('faceMetricTime');

    const faceEmptyState = document.getElementById('faceEmptyState');
    const faceLoadingOverlay = document.getElementById('faceLoadingOverlay');
    const faceDisplayContainer = document.getElementById('faceDisplayContainer');
    const faceDetailsCard = document.getElementById('faceDetailsCard');
    const faceResultImage = document.getElementById('faceResultImage');
    const faceDownloadBtn = document.getElementById('faceDownloadBtn');
    const faceTabBtns = document.querySelectorAll('[data-facetab]');

    let currentFaceFile = null;
    let currentFaceOriginalUrl = null;
    let currentFaceAnnotatedUrl = null;
    let currentFaceViewTab = 'mesh';

    let activeBackground = 'original';
    let activeColorTheme = 'cyan';

    segmentBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            segmentBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeBackground = btn.dataset.bg;
            if (currentFaceFile) uploadAndDetectFace(currentFaceFile);
        });
    });

    colorSwatches.forEach(swatch => {
        swatch.addEventListener('click', () => {
            colorSwatches.forEach(s => s.classList.remove('active'));
            swatch.classList.add('active');
            activeColorTheme = swatch.dataset.color;
            if (currentFaceFile) uploadAndDetectFace(currentFaceFile);
        });
    });

    meshStyleSelect.addEventListener('change', () => {
        if (currentFaceFile) uploadAndDetectFace(currentFaceFile);
    });

    faceThicknessRange.addEventListener('input', (e) => {
        faceThicknessVal.textContent = `${e.target.value} px`;
        if (currentFaceFile) reanalyzeFaceBtn.disabled = false;
    });

    faceThicknessRange.addEventListener('change', () => {
        if (currentFaceFile) uploadAndDetectFace(currentFaceFile);
    });

    drawPointsCheck.addEventListener('change', () => {
        if (currentFaceFile) uploadAndDetectFace(currentFaceFile);
    });

    reanalyzeFaceBtn.addEventListener('click', () => {
        if (currentFaceFile) uploadAndDetectFace(currentFaceFile);
    });

    faceFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleNewFaceFile(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        faceDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            faceDropzone.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        faceDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            faceDropzone.classList.remove('drag-active');
        }, false);
    });

    faceDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleNewFaceFile(dt.files[0]);
        }
    });

    faceTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            faceTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFaceViewTab = btn.dataset.facetab;
            updateDisplayedFaceImage();
        });
    });

    function updateDisplayedFaceImage() {
        if (currentFaceViewTab === 'mesh' && currentFaceAnnotatedUrl) {
            faceResultImage.src = currentFaceAnnotatedUrl;
        } else if (currentFaceViewTab === 'original' && currentFaceOriginalUrl) {
            faceResultImage.src = currentFaceOriginalUrl;
        }
    }

    function handleNewFaceFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid face photo (JPEG, PNG, WEBP).');
            return;
        }
        currentFaceFile = file;
        faceDropText.innerHTML = `<strong>Loaded:</strong> ${file.name}`;
        const reader = new FileReader();
        reader.onload = (e) => {
            currentFaceOriginalUrl = e.target.result;
            uploadAndDetectFace(file);
        };
        reader.readAsDataURL(file);
    }

    async function uploadAndDetectFace(file) {
        faceLoadingOverlay.style.display = 'flex';
        reanalyzeFaceBtn.disabled = true;

        const formData = new FormData();
        formData.append('image', file);
        formData.append('background', activeBackground);
        formData.append('mesh_style', meshStyleSelect.value);
        formData.append('color_theme', activeColorTheme);
        formData.append('line_thickness', faceThicknessRange.value);
        formData.append('draw_points', drawPointsCheck.checked ? 'true' : 'false');
        formData.append('num_faces', '2');

        try {
            const response = await fetch('/api/facemesh/detect', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Face Mesh detection failed');
            }

            const data = await response.json();
            renderFaceMeshResults(data);
        } catch (err) {
            console.error(err);
            alert(`Face Mesh error: ${err.message}`);
        } finally {
            faceLoadingOverlay.style.display = 'none';
        }
    }

    function renderFaceMeshResults(data) {
        currentFaceAnnotatedUrl = data.annotated_image;

        faceEmptyState.style.display = 'none';
        faceDisplayContainer.style.display = 'flex';
        faceMetricsBar.style.display = 'flex';
        faceDetailsCard.style.display = 'block';
        reanalyzeFaceBtn.disabled = false;

        faceMetricCount.textContent = data.faces_detected;
        faceMetricTime.textContent = `${data.inference_time_ms} ms`;

        updateDisplayedFaceImage();
        faceDownloadBtn.href = currentFaceAnnotatedUrl;
    }

    sampleFaceBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleType = btn.dataset.sample;
            generateSampleFaceBlob(sampleType).then(blob => {
                handleNewFaceFile(blob);
            });
        });
    });

    async function generateSampleFaceBlob(type) {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 720;
        const ctx = canvas.getContext('2d');

        const bgGrad = ctx.createRadialGradient(320, 360, 50, 320, 360, 420);
        bgGrad.addColorStop(0, '#334155');
        bgGrad.addColorStop(1, '#0f172a');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, 640, 720);

        const cx = 320;
        const cy = 350;

        ctx.fillStyle = '#f5c6a5';
        ctx.beginPath();
        ctx.roundRect(cx - 55, cy + 120, 110, 160, [10, 10, 0, 0]);
        ctx.fill();

        ctx.fillStyle = '#2563eb';
        ctx.beginPath();
        ctx.ellipse(cx, cy + 320, 240, 120, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#fed7aa';
        ctx.beginPath();
        ctx.ellipse(cx, cy, 140, 180, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#1e1b18';
        ctx.beginPath();
        ctx.arc(cx, cy - 35, 148, Math.PI, 0, false);
        ctx.fill();
        ctx.fillRect(cx - 148, cy - 60, 36, 140);
        ctx.fillRect(cx + 112, cy - 60, 36, 140);

        ctx.fillStyle = '#292524';
        ctx.beginPath();
        ctx.ellipse(cx - 55, cy - 38, 34, 7, -0.05, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(cx + 55, cy - 38, 34, 7, 0.05, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.ellipse(cx - 55, cy - 12, 26, 14, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(cx + 55, cy - 12, 26, 14, 0, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#0284c7';
        ctx.beginPath();
        ctx.arc(cx - 55, cy - 12, 11, 0, Math.PI * 2);
        ctx.arc(cx + 55, cy - 12, 11, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(cx - 55, cy - 12, 5, 0, Math.PI * 2);
        ctx.arc(cx + 55, cy - 12, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#d4976a';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(cx, cy - 25);
        ctx.lineTo(cx, cy + 30);
        ctx.lineTo(cx + 14, cy + 32);
        ctx.stroke();

        ctx.fillStyle = '#9a3412';
        ctx.beginPath();
        ctx.arc(cx - 12, cy + 34, 3, 0, Math.PI * 2);
        ctx.arc(cx + 12, cy + 34, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#e11d48';
        if (type === 'smiling') {
            ctx.beginPath();
            ctx.ellipse(cx, cy + 85, 46, 20, 0, 0, Math.PI);
            ctx.fill();
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.ellipse(cx, cy + 85, 36, 10, 0, 0, Math.PI);
            ctx.fill();
        } else {
            ctx.beginPath();
            ctx.ellipse(cx, cy + 82, 38, 12, 0, 0, Math.PI * 2);
            ctx.fill();
        }

        return new Promise(resolve => {
            canvas.toBlob(blob => {
                const file = new File([blob], `sample_face_${type}.jpg`, { type: 'image/jpeg' });
                resolve(file);
            }, 'image/jpeg', 0.94);
        });
    }

    // ==========================================
    // 3. VIDEO FILES & YOUTUBE LOGIC
    // ==========================================
    const videoTaskPills = document.querySelectorAll('#videoTaskPills .pill-btn');
    const videoObjectParamsSection = document.getElementById('videoObjectParamsSection');
    const videoFaceMeshParamsSection = document.getElementById('videoFaceMeshParamsSection');
    const videoFitnessParamsSection = document.getElementById('videoFitnessParamsSection');
    const vExtendedRange = document.getElementById('vExtendedRange');
    const vBentRange = document.getElementById('vBentRange');
    vExtendedRange.addEventListener('input', () => {
        document.getElementById('vExtendedVal').textContent = `${vExtendedRange.value}°`;
    });
    vBentRange.addEventListener('input', () => {
        document.getElementById('vBentVal').textContent = `${vBentRange.value}°`;
    });

    const videoSourcePills = document.querySelectorAll('.source-pills:not(#videoTaskPills) .pill-btn');
    const videoUploadSection = document.getElementById('videoUploadSection');
    const youtubeInputSection = document.getElementById('youtubeInputSection');
    const ytUrlInput = document.getElementById('ytUrlInput');
    const videoFileInput = document.getElementById('videoFileInput');
    const videoDropzone = document.getElementById('videoDropzone');
    const videoDropText = document.getElementById('videoDropText');
    const sampleYtBtns = document.querySelectorAll('.btn-sample-yt');

    const vBgBtns = document.querySelectorAll('[data-vbg]');
    const vMeshStyleSelect = document.getElementById('vMeshStyleSelect');
    const vColorSwatches = document.querySelectorAll('#vColorSwatches .color-swatch');

    const videoDurationSelect = document.getElementById('videoDurationSelect');
    const durationVal = document.getElementById('durationVal');
    const videoFpsSelect = document.getElementById('videoFpsSelect');
    const fpsVal = document.getElementById('fpsVal');
    const videoThresholdRange = document.getElementById('videoThresholdRange');
    const videoThresholdVal = document.getElementById('videoThresholdVal');
    const startVideoBtn = document.getElementById('startVideoBtn');

    const videoMetricsBar = document.getElementById('videoMetricsBar');
    const videoMetricObjects = document.getElementById('videoMetricObjects');
    const videoMetricFrames = document.getElementById('videoMetricFrames');
    const videoMetricTime = document.getElementById('videoMetricTime');

    const videoEmptyState = document.getElementById('videoEmptyState');
    const videoProgressContainer = document.getElementById('videoProgressContainer');
    const videoDisplayContainer = document.getElementById('videoDisplayContainer');
    const progressTitle = document.getElementById('progressTitle');
    const progressStep = document.getElementById('progressStep');
    const progressBarFill = document.getElementById('progressBarFill');
    const progressPctText = document.getElementById('progressPctText');
    const progressFpsText = document.getElementById('progressFpsText');

    const annotatedVideoPlayer = document.getElementById('annotatedVideoPlayer');
    const videoDownloadBtn = document.getElementById('videoDownloadBtn');
    const videoDetailsCard = document.getElementById('videoDetailsCard');
    const videoTagsContainer = document.getElementById('videoTagsContainer');

    let currentVideoTask = 'object';     // 'object' or 'facemesh'
    let currentVideoSource = 'upload';    // 'upload' (default) or 'youtube'
    let currentVBg = 'original';          // 'original' or 'black'
    let currentVColor = 'cyan';
    let selectedVideoFile = null;
    let videoPollInterval = null;

    // Video Task Toggle (Objects vs Face Mesh on Video)
    videoTaskPills.forEach(btn => {
        btn.addEventListener('click', () => {
            videoTaskPills.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentVideoTask = btn.dataset.vtask;

            videoObjectParamsSection.style.display = currentVideoTask === 'object' ? 'block' : 'none';
            videoFaceMeshParamsSection.style.display = currentVideoTask === 'facemesh' ? 'block' : 'none';
            videoFitnessParamsSection.style.display = currentVideoTask === 'fitness' ? 'block' : 'none';
        });
    });

    // Video Source Toggle (Upload File vs YouTube)
    videoSourcePills.forEach(btn => {
        btn.addEventListener('click', () => {
            videoSourcePills.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentVideoSource = btn.dataset.source;

            if (currentVideoSource === 'upload') {
                videoUploadSection.style.display = 'block';
                youtubeInputSection.style.display = 'none';
            } else {
                videoUploadSection.style.display = 'none';
                youtubeInputSection.style.display = 'block';
            }
        });
    });

    // Face mesh video background toggle
    vBgBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            vBgBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentVBg = btn.dataset.vbg;
        });
    });

    // Face mesh video color swatch
    vColorSwatches.forEach(swatch => {
        swatch.addEventListener('click', () => {
            vColorSwatches.forEach(s => s.classList.remove('active'));
            swatch.classList.add('active');
            currentVColor = swatch.dataset.vcolor;
        });
    });

    sampleYtBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            ytUrlInput.value = btn.dataset.url;
        });
    });

    videoDurationSelect.addEventListener('change', (e) => {
        durationVal.textContent = `${e.target.value}s`;
    });

    videoFpsSelect.addEventListener('change', (e) => {
        fpsVal.textContent = `${e.target.value} FPS`;
    });

    videoThresholdRange.addEventListener('input', (e) => {
        videoThresholdVal.textContent = `${Math.round(e.target.value * 100)}%`;
    });

    function handleNewVideoFile(file) {
        selectedVideoFile = file;
        videoDropText.innerHTML = `<strong>Selected:</strong> ${file.name} (${Math.round(file.size / 1024 / 1024 * 10) / 10} MB)`;
    }

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleNewVideoFile(e.target.files[0]);
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        videoDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            videoDropzone.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        videoDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            videoDropzone.classList.remove('drag-active');
        }, false);
    });

    videoDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleNewVideoFile(dt.files[0]);
        }
    });

    // Start Video Processing
    startVideoBtn.addEventListener('click', async () => {
        const formData = new FormData();
        formData.append('source_type', currentVideoSource);
        formData.append('video_task', currentVideoTask);
        formData.append('duration', videoDurationSelect.value);
        formData.append('sample_fps', videoFpsSelect.value);

        if (currentVideoTask === 'facemesh') {
            formData.append('background', currentVBg);
            formData.append('mesh_style', vMeshStyleSelect.value);
            formData.append('color_theme', currentVColor);
            formData.append('line_thickness', '1');
            formData.append('draw_points', 'false');
        } else if (currentVideoTask === 'fitness') {
            if (Number(vBentRange.value) >= Number(vExtendedRange.value) - 20) {
                alert('The "bent" angle must be at least 20° below the "straight" angle.');
                return;
            }
            formData.append('extended_angle', vExtendedRange.value);
            formData.append('bent_angle', vBentRange.value);
        } else {
            formData.append('threshold', videoThresholdRange.value);
            formData.append('max_results', '20');
        }

        if (currentVideoSource === 'youtube') {
            const url = ytUrlInput.value.trim();
            if (!url) {
                alert('Please enter or select a YouTube URL first.');
                return;
            }
            formData.append('youtube_url', url);
        } else {
            if (!selectedVideoFile) {
                alert('Please select or drop a video file first.');
                return;
            }
            formData.append('video', selectedVideoFile);
        }

        // Show Progress State
        videoEmptyState.style.display = 'none';
        videoDisplayContainer.style.display = 'none';
        videoMetricsBar.style.display = 'none';
        videoDetailsCard.style.display = 'none';
        videoProgressContainer.style.display = 'flex';
        startVideoBtn.disabled = true;

        const taskName = { facemesh: '3D Face Mesh', fitness: 'Arm Rep Counting' }[currentVideoTask] || 'Object Detection';
        const sourceName = currentVideoSource === 'youtube' ? 'YouTube Video' : 'Video File';
        progressTitle.textContent = `Running ${taskName} on ${sourceName}...`;
        progressStep.textContent = 'Submitting video processing job...';
        progressBarFill.style.width = '5%';
        progressPctText.textContent = '5%';
        progressFpsText.textContent = '';

        try {
            const response = await fetch('/api/video/start', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Failed to start video processing job.');
            }

            const data = await response.json();
            const jobId = data.job_id;
            pollVideoStatus(jobId);
        } catch (err) {
            console.error(err);
            alert(`Error: ${err.message}`);
            videoProgressContainer.style.display = 'none';
            videoEmptyState.style.display = 'block';
            startVideoBtn.disabled = false;
        }
    });

    function pollVideoStatus(jobId) {
        if (videoPollInterval) clearInterval(videoPollInterval);

        videoPollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/video/status/${jobId}`);
                if (!res.ok) throw new Error('Status check failed');

                const statusData = await res.json();
                updateVideoProgressUI(statusData);

                if (statusData.status === 'completed') {
                    clearInterval(videoPollInterval);
                    videoPollInterval = null;
                    renderCompletedVideo(statusData);
                } else if (statusData.status === 'failed') {
                    clearInterval(videoPollInterval);
                    videoPollInterval = null;
                    alert(`Video processing failed: ${statusData.error || 'Unknown error'}`);
                    videoProgressContainer.style.display = 'none';
                    videoEmptyState.style.display = 'block';
                    startVideoBtn.disabled = false;
                }
            } catch (err) {
                console.error(err);
            }
        }, 750);
    }

    function updateVideoProgressUI(data) {
        const pct = data.progress || 0;
        progressBarFill.style.width = `${pct}%`;
        progressPctText.textContent = `${pct}%`;
        progressStep.innerHTML = data.step || 'Processing...';

        if (data.proc_fps > 0) {
            progressFpsText.textContent = `${data.proc_fps} FPS`;
        }
    }

    function renderCompletedVideo(data) {
        videoProgressContainer.style.display = 'none';
        videoDisplayContainer.style.display = 'flex';
        videoMetricsBar.style.display = 'flex';
        videoDetailsCard.style.display = 'block';
        startVideoBtn.disabled = false;

        videoMetricObjects.textContent = data.total_detections;
        videoMetricObjects.previousElementSibling.textContent =
            data.video_task === 'fitness' ? 'Total Reps' : 'Detections Tracked';
        document.querySelector('#videoDetailsCard h3').textContent =
            data.video_task === 'fitness' ? 'Rep Counts in Video' : 'Classes Identified in Video';
        videoMetricFrames.textContent = data.processed_frames;
        videoMetricTime.textContent = `${data.elapsed_seconds}s`;

        annotatedVideoPlayer.src = data.video_url;
        annotatedVideoPlayer.load();
        annotatedVideoPlayer.play().catch(() => {});
        videoDownloadBtn.href = data.video_url;

        videoTagsContainer.innerHTML = '';
        const counts = data.class_counts || {};
        const entries = Object.entries(counts);
        if (entries.length === 0 && data.video_task !== 'fitness') {
            videoTagsContainer.innerHTML = '<span style="color: var(--text-secondary); font-size: 0.85rem;">No detections tracked with current settings.</span>';
        } else {
            entries.forEach(([label, count]) => {
                const tag = document.createElement('span');
                tag.className = 'class-tag';
                const repLabels = { left_arm_reps: 'Left arm', right_arm_reps: 'Right arm' };
                if (repLabels[label]) {
                    tag.innerHTML = `${repLabels[label]} <span class="badge-count">${count} reps</span>`;
                } else {
                    const labelText = label === 'face' ? '3D Face Landmark Mesh' : label;
                    tag.innerHTML = `${labelText} <span class="badge-count">${count} instances</span>`;
                }
                videoTagsContainer.appendChild(tag);
            });
        }
    }
});
