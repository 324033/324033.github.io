import os
import sys

# Ensure temporary matplotlib cache directory in local folder to avoid sandbox warning
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.mpl_cache'))

# Add Scripts directory containing ffmpeg to PATH
scripts_dir = os.path.join(os.path.dirname(__file__), '.venv', 'Scripts')
if os.path.exists(scripts_dir):
    os.environ['PATH'] = scripts_dir + os.pathsep + os.environ.get('PATH', '')

import io
import re
import time
import uuid
import threading
import urllib.request
import subprocess
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.context_processor
def inject_asset_version():
    """Cache-busting: version static assets by file modification time so the
    browser always loads the current JS/CSS after code changes."""
    def asset_version(rel_path):
        try:
            return int(os.path.getmtime(os.path.join(app.static_folder, rel_path)))
        except OSError:
            return 0
    return {'asset_version': asset_version}
app.config['TEMPLATES_AUTO_RELOAD'] = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OBJECT_MODEL_FILENAME = 'efficientdet_lite0.tflite'
OBJECT_MODEL_PATH = os.path.join(MODEL_DIR, OBJECT_MODEL_FILENAME)
OBJECT_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'

FACE_MODEL_FILENAME = 'face_landmarker.task'
FACE_MODEL_PATH = os.path.join(MODEL_DIR, FACE_MODEL_FILENAME)
FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

POSE_MODEL_FILENAME = 'pose_landmarker_full.task'
POSE_MODEL_PATH = os.path.join(MODEL_DIR, POSE_MODEL_FILENAME)
POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'

GESTURE_MODEL_FILENAME = 'gesture_recognizer.task'
GESTURE_MODEL_PATH = os.path.join(MODEL_DIR, GESTURE_MODEL_FILENAME)
GESTURE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'

SEGMENT_MODEL_FILENAME = 'selfie_segmenter.tflite'
SEGMENT_MODEL_PATH = os.path.join(MODEL_DIR, SEGMENT_MODEL_FILENAME)
SEGMENT_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite'

FACEDET_MODEL_FILENAME = 'blaze_face_short_range.tflite'
FACEDET_MODEL_PATH = os.path.join(MODEL_DIR, FACEDET_MODEL_FILENAME)
FACEDET_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Find ffmpeg executable
FFMPEG_EXE = os.path.join(scripts_dir, 'ffmpeg.exe')
if not os.path.exists(FFMPEG_EXE):
    try:
        import imageio_ffmpeg
        FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_EXE = 'ffmpeg'

# Palette of distinct aesthetically pleasing colors for different object classes
COLORS = [
    (59, 130, 246),   # Blue
    (16, 185, 129),   # Emerald
    (245, 158, 11),   # Amber
    (239, 68, 68),    # Red
    (139, 92, 246),   # Purple
    (236, 72, 153),   # Pink
    (20, 184, 166),   # Teal
    (249, 115, 22),   # Orange
    (99, 102, 241),   # Indigo
    (168, 85, 247),   # Violet
]

# Color palettes for Face Mesh (BGR format for OpenCV)
FACE_PALETTES = {
    'cyan': {
        'mesh': (240, 220, 0),         # Neon Cyan
        'contour': (255, 255, 0),      # Bright Aqua
        'point': (255, 255, 200),
        'iris': (50, 220, 255)
    },
    'magenta': {
        'mesh': (220, 0, 180),        # Cyberpunk Magenta
        'contour': (255, 50, 230),
        'point': (255, 180, 255),
        'iris': (0, 255, 255)
    },
    'green': {
        'mesh': (100, 230, 40),       # Matrix Emerald
        'contour': (140, 255, 80),
        'point': (200, 255, 180),
        'iris': (255, 240, 0)
    },
    'purple': {
        'mesh': (230, 80, 150),       # Electric Violet
        'contour': (255, 120, 200),
        'point': (255, 200, 255),
        'iris': (255, 200, 50)
    },
    'gold': {
        'mesh': (0, 180, 240),        # Warm Amber / Gold
        'contour': (0, 215, 255),
        'point': (180, 240, 255),
        'iris': (255, 100, 0)
    },
    'white': {
        'mesh': (190, 190, 190),      # Minimalist White
        'contour': (255, 255, 255),
        'point': (255, 255, 255),
        'iris': (0, 240, 255)
    }
}

def ensure_models_downloaded():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(OBJECT_MODEL_PATH) or os.path.getsize(OBJECT_MODEL_PATH) == 0:
        print(f"Downloading object model {OBJECT_MODEL_FILENAME}...")
        urllib.request.urlretrieve(OBJECT_MODEL_URL, OBJECT_MODEL_PATH)
    if not os.path.exists(FACE_MODEL_PATH) or os.path.getsize(FACE_MODEL_PATH) == 0:
        print(f"Downloading face model {FACE_MODEL_FILENAME}...")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    if not os.path.exists(POSE_MODEL_PATH) or os.path.getsize(POSE_MODEL_PATH) == 0:
        print(f"Downloading pose model {POSE_MODEL_FILENAME}...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    if not os.path.exists(GESTURE_MODEL_PATH) or os.path.getsize(GESTURE_MODEL_PATH) == 0:
        print(f"Downloading gesture model {GESTURE_MODEL_FILENAME}...")
        urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
    if not os.path.exists(SEGMENT_MODEL_PATH) or os.path.getsize(SEGMENT_MODEL_PATH) == 0:
        print(f"Downloading segmentation model {SEGMENT_MODEL_FILENAME}...")
        urllib.request.urlretrieve(SEGMENT_MODEL_URL, SEGMENT_MODEL_PATH)
    if not os.path.exists(FACEDET_MODEL_PATH) or os.path.getsize(FACEDET_MODEL_PATH) == 0:
        print(f"Downloading face detection model {FACEDET_MODEL_FILENAME}...")
        urllib.request.urlretrieve(FACEDET_MODEL_URL, FACEDET_MODEL_PATH)

_detector_cache = {}
_face_detector_cache = {}


def create_pose_landmarker():
    """New PoseLandmarker per job (VIDEO mode keeps tracking state, so it must not be shared)."""
    ensure_models_downloaded()
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)

def get_detector(score_threshold=0.3, max_results=20):
    cache_key = (round(score_threshold, 2), max_results)
    if cache_key in _detector_cache:
        return _detector_cache[cache_key]

    ensure_models_downloaded()

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=OBJECT_MODEL_PATH)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=score_threshold,
        max_results=max_results
    )
    detector = vision.ObjectDetector.create_from_options(options)
    _detector_cache[cache_key] = detector
    return detector

def get_face_landmarker(num_faces=2, min_confidence=0.5):
    cache_key = (num_faces, round(min_confidence, 2))
    if cache_key in _face_detector_cache:
        return _face_detector_cache[cache_key]

    ensure_models_downloaded()

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=num_faces,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    _face_detector_cache[cache_key] = landmarker
    return landmarker

def get_class_color(label):
    hash_val = sum(ord(c) for c in label)
    return COLORS[hash_val % len(COLORS)]

def annotate_image(image_bgr, detections):
    """Draw bounding boxes and clean pill labels with confidence on the image."""
    h, w, _ = image_bgr.shape
    annotated = image_bgr.copy()

    base_dim = max(w, h)
    thickness = max(2, int(base_dim / 400))
    font_scale = max(0.45, base_dim / 1200)

    for det in detections:
        bbox = det['bounding_box']
        x1 = max(0, int(bbox['origin_x']))
        y1 = max(0, int(bbox['origin_y']))
        x2 = min(w - 1, int(bbox['origin_x'] + bbox['width']))
        y2 = min(h - 1, int(bbox['origin_y'] + bbox['height']))

        label = det['category_name']
        score = det['score']
        color_rgb = get_class_color(label)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness, cv2.LINE_AA)

        caption = f"{label} {int(score * 100)}%"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)

        pill_y1 = max(0, y1 - th - baseline - 8)
        pill_y2 = pill_y1 + th + baseline + 8
        pill_x2 = min(w, x1 + tw + 12)

        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), color_bgr, -1)
        text_color = (255, 255, 255)
        cv2.putText(
            annotated,
            caption,
            (x1 + 6, pill_y2 - baseline - 4),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA
        )

    return annotated

def annotate_face_mesh(image_bgr, face_landmarks_list, background='original',
                       mesh_style='both', color_theme='cyan', line_thickness=1, draw_points=False):
    """
    Renders 468 3D facial landmarks with customizable styles and background.
    - background: 'original' or 'black'
    - mesh_style: 'both' | 'tesselation' | 'contours' | 'points' | 'contours_irises'
    - color_theme: 'cyan' | 'magenta' | 'green' | 'purple' | 'gold' | 'white'
    """
    from mediapipe.tasks.python.vision import FaceLandmarksConnections

    h, w, _ = image_bgr.shape
    if background == 'black':
        canvas = np.zeros_like(image_bgr)
    else:
        canvas = image_bgr.copy()

    palette = FACE_PALETTES.get(color_theme, FACE_PALETTES['cyan'])
    mesh_color = palette['mesh']
    contour_color = palette['contour']
    point_color = palette['point']
    iris_color = palette['iris']

    tesselation_conns = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    contours_conns = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    left_iris_conns = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
    right_iris_conns = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS

    for face_landmarks in face_landmarks_list:
        pts = []
        for lm in face_landmarks:
            px = max(0, min(w - 1, int(round(lm.x * w))))
            py = max(0, min(h - 1, int(round(lm.y * h))))
            pts.append((px, py))

        # 1. Full 3D Tesselation Mesh (468 points, Delaunay triangles)
        if mesh_style in ('both', 'tesselation'):
            tess_thick = 1
            for conn in tesselation_conns:
                p1 = pts[conn.start]
                p2 = pts[conn.end]
                cv2.line(canvas, p1, p2, mesh_color, tess_thick, cv2.LINE_AA)

        # 2. Contours (Eyes, Eyebrows, Lips, Nose, Face Oval)
        if mesh_style in ('both', 'contours', 'contours_irises'):
            cont_thick = max(1, line_thickness)
            for conn in contours_conns:
                p1 = pts[conn.start]
                p2 = pts[conn.end]
                cv2.line(canvas, p1, p2, contour_color, cont_thick, cv2.LINE_AA)

        # 3. Irises
        if mesh_style in ('both', 'contours_irises') and len(pts) >= 478:
            for conn in list(left_iris_conns) + list(right_iris_conns):
                p1 = pts[conn.start]
                p2 = pts[conn.end]
                cv2.line(canvas, p1, p2, iris_color, max(1, line_thickness), cv2.LINE_AA)

        # 4. Landmark Points (Dots)
        if draw_points or mesh_style == 'points':
            pt_radius = max(1, line_thickness)
            for i in range(min(468, len(pts))):
                cv2.circle(canvas, pts[i], pt_radius, point_color, -1, cv2.LINE_AA)

    return canvas

# Video background job tracking
video_jobs = {}

def run_video_job(job_id, source_type, file_or_url, duration, sample_fps, threshold, max_results,
                  video_task='object', background='original', mesh_style='both',
                  color_theme='cyan', line_thickness=1, draw_points=False,
                  extended_angle=150.0, bent_angle=60.0):
    job = video_jobs[job_id]
    temp_input = None

    try:
        if source_type == 'youtube':
            job['status'] = 'downloading'
            job['step'] = 'Fetching YouTube video stream...'
            job['progress'] = 5

            import yt_dlp
            import yt_dlp.utils

            temp_template = os.path.join(TEMP_DIR, f"{job_id}_in.%(ext)s")
            ydl_opts = {
                'format': 'bv*[height<=720]+ba/b[height<=720]/b',
                'outtmpl': temp_template,
                'ffmpeg_location': scripts_dir,
                'download_ranges': yt_dlp.utils.download_range_func(None, [(0, duration)]),
                'force_keyframes_at_cuts': True,
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(file_or_url, download=True)
                downloaded_file = ydl.prepare_filename(info)
                if not os.path.exists(downloaded_file):
                    matching = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.startswith(f"{job_id}_in")]
                    if matching:
                        downloaded_file = matching[0]
                temp_input = downloaded_file
        else:
            temp_input = file_or_url

        if not temp_input or not os.path.exists(temp_input):
            raise RuntimeError("Failed to obtain input video file.")

        job['status'] = 'processing'
        job['step'] = 'Reading video streams and initializing detection...'
        job['progress'] = 15

        cap = cv2.VideoCapture(temp_input)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open video stream.")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_source_frames = int(duration * orig_fps)
        total_source_frames = min(total_vid_frames, max_source_frames) if total_vid_frames > 0 else max_source_frames

        step = max(1, int(round(orig_fps / sample_fps)))
        effective_output_fps = min(sample_fps, round(orig_fps / step, 1))

        target_w = orig_w
        target_h = orig_h
        if target_w > 1280:
            scale = 1280.0 / target_w
            target_w = 1280
            target_h = int(round(orig_h * scale))
            target_w -= (target_w % 2)
            target_h -= (target_h % 2)

        output_filename = f"{job_id}.mp4"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        ffmpeg_cmd = [
            FFMPEG_EXE,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{target_w}x{target_h}',
            '-pix_fmt', 'bgr24',
            '-r', str(effective_output_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_filepath
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        import mediapipe as mp
        if video_task == 'facemesh':
            landmarker = get_face_landmarker(num_faces=3)
        elif video_task == 'fitness':
            from fitness import ArmRepTracker, annotate_pose_reps
            pose_landmarker = create_pose_landmarker()
            rep_tracker = ArmRepTracker(extended_angle=extended_angle, bent_angle=bent_angle)
        else:
            detector = get_detector(score_threshold=threshold, max_results=max_results)

        frame_idx = 0
        processed_count = 0
        total_detections = 0
        class_counts = {}
        start_proc_time = time.time()

        while cap.isOpened() and frame_idx < total_source_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                if video_task == 'facemesh':
                    det_result = landmarker.detect(mp_img)
                    faces_found = len(det_result.face_landmarks or [])
                    total_detections += faces_found
                    if faces_found > 0:
                        class_counts['face'] = class_counts.get('face', 0) + faces_found

                    annotated_frame = annotate_face_mesh(
                        frame,
                        det_result.face_landmarks or [],
                        background=background,
                        mesh_style=mesh_style,
                        color_theme=color_theme,
                        line_thickness=line_thickness,
                        draw_points=draw_points
                    )
                    badge_text = f"Faces: {faces_found} | {round(frame_idx / orig_fps, 1)}s"
                elif video_task == 'fitness':
                    ts_ms = int(frame_idx * 1000.0 / orig_fps)
                    pose_result = pose_landmarker.detect_for_video(mp_img, ts_ms)
                    person = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                    rep_tracker.update(person, target_w, target_h)
                    counts = rep_tracker.counts
                    total_detections = counts['left'] + counts['right']
                    class_counts = {'left_arm_reps': counts['left'], 'right_arm_reps': counts['right']}
                    annotated_frame = annotate_pose_reps(frame, person, rep_tracker)
                    badge_text = f"Reps L: {counts['left']}  R: {counts['right']} | {round(frame_idx / orig_fps, 1)}s"
                else:
                    det_result = detector.detect(mp_img)
                    parsed_dets = []
                    for det in det_result.detections:
                        if not det.categories:
                            continue
                        top_cat = det.categories[0]
                        cat_name = top_cat.category_name
                        class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
                        total_detections += 1

                        parsed_dets.append({
                            'category_name': cat_name,
                            'score': float(top_cat.score),
                            'bounding_box': {
                                'origin_x': int(det.bounding_box.origin_x),
                                'origin_y': int(det.bounding_box.origin_y),
                                'width': int(det.bounding_box.width),
                                'height': int(det.bounding_box.height)
                            }
                        })

                    annotated_frame = annotate_image(frame, parsed_dets)
                    badge_text = f"Objects: {len(parsed_dets)} | {round(frame_idx / orig_fps, 1)}s"

                cv2.putText(annotated_frame, badge_text, (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, badge_text, (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

                proc.stdin.write(annotated_frame.tobytes())
                processed_count += 1

                elapsed = max(0.001, time.time() - start_proc_time)
                current_proc_fps = round(processed_count / elapsed, 1)
                pct = min(98, 15 + int((frame_idx / total_source_frames) * 83))

                item_label = {'facemesh': 'faces', 'fitness': 'reps'}.get(video_task, 'objects')
                job['progress'] = pct
                job['step'] = f"Analyzing frame {processed_count} ({current_proc_fps} FPS) &bull; Found {total_detections} {item_label}"
                job['proc_fps'] = current_proc_fps
                job['total_detections'] = total_detections
                job['class_counts'] = class_counts

            frame_idx += 1

        cap.release()
        if video_task == 'fitness':
            pose_landmarker.close()
        proc.stdin.close()
        proc.wait()

        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass

        total_elapsed = round(time.time() - start_proc_time, 1)
        job['status'] = 'completed'
        job['progress'] = 100
        job['step'] = 'Detection complete!'
        job['video_url'] = f"/outputs/{output_filename}"
        job['total_detections'] = total_detections
        job['processed_frames'] = processed_count
        job['elapsed_seconds'] = total_elapsed
        job['class_counts'] = class_counts
        job['width'] = target_w
        job['height'] = target_h
        job['video_task'] = video_task

    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        job['step'] = f"Error: {str(e)}"
        if temp_input and os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass

@app.route('/')
def index():
    return render_template('index.html')


# Gemini chat (/chat): API key stays server-side in .env, see gemini_chat.py
from gemini_chat import bp as gemini_chat_bp  # noqa: E402
app.register_blueprint(gemini_chat_bp)

# Models used by the in-browser live modes (webcam). Only these files are served.
BROWSER_MODELS = {
    'pose_landmarker.task': POSE_MODEL_FILENAME,        # Fitness Coach
    'gesture_recognizer.task': GESTURE_MODEL_FILENAME,  # Hand Gestures
    'selfie_segmenter.tflite': SEGMENT_MODEL_FILENAME,  # Virtual Background
    'blaze_face_short_range.tflite': FACEDET_MODEL_FILENAME,  # Face Keypoints
    'yolo26n.onnx': 'yolo26n.onnx',                           # YOLO26 page (created by export_yolo.py)
}

@app.route('/models/<name>')
def serve_browser_model(name):
    if name not in BROWSER_MODELS:
        return jsonify({'error': 'Unknown model'}), 404
    if name == 'yolo26n.onnx':
        if not os.path.exists(os.path.join(MODEL_DIR, 'yolo26n.onnx')):
            return jsonify({'error': 'models/yolo26n.onnx missing - run export_yolo.py once'}), 404
    else:
        ensure_models_downloaded()
    return send_from_directory(MODEL_DIR, BROWSER_MODELS[name], mimetype='application/octet-stream')


# ================= YOLO26 PAGE (browser-side detection) =================
@app.route('/yolo')
def yolo_page():
    return render_template('yolo.html')


@app.route('/cctv')
def cctv_page():
    """CCTV-only traffic analytics: vehicle counting + heatmap + speed estimation (YOLO26n in the browser)."""
    return render_template('cctv.html')


yt_clip_jobs = {}


def run_yt_clip_job(job_id, url, duration):
    """Downloads the first `duration` seconds of a YouTube video as a browser-playable
    H.264 MP4 (no audio), so the page can read its frames for detection."""
    job = yt_clip_jobs[job_id]
    try:
        import yt_dlp
        import yt_dlp.utils
        job.update(status='downloading', step='Downloading video from YouTube...')
        tmpl = os.path.join(TEMP_DIR, f"yolo_{job_id}_in.%(ext)s")
        opts = {
            # prefer H.264 so no re-encoding is needed
            'format': 'bv*[height<=720][vcodec^=avc1]/bv*[height<=720]/b[height<=720]/b',
            'outtmpl': tmpl,
            'ffmpeg_location': scripts_dir,
            'download_ranges': yt_dlp.utils.download_range_func(None, [(0, duration)]),
            'force_keyframes_at_cuts': True,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.startswith(f"yolo_{job_id}_in")]
        if not files:
            raise RuntimeError('Download failed - no video file was created.')
        src = files[0]

        job.update(status='processing', step='Preparing the clip for the browser...')
        out_name = f"yolo_{job_id}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        is_h264 = str(info.get('vcodec') or '').startswith('avc1')
        cmd = [FFMPEG_EXE, '-y', '-loglevel', 'error', '-i', src, '-t', str(duration), '-an']
        cmd += ['-c:v', 'copy'] if is_h264 else ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p']
        cmd += ['-movflags', '+faststart', out_path]
        subprocess.run(cmd, check=True)
        try:
            os.remove(src)
        except OSError:
            pass
        job.update(status='completed', step='Ready', video_url=f"/outputs/{out_name}",
                   title=info.get('title') or 'YouTube video')
    except Exception as e:
        job.update(status='failed', error=str(e).replace('\x1b[0;31m', '').replace('\x1b[0m', ''))


@app.route('/api/yolo/youtube', methods=['POST'])
def start_yt_clip():
    data = request.get_json(silent=True) or {}
    url = str(data.get('url', '')).strip()
    if not re.match(r'^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/', url):
        return jsonify({'error': 'Please enter a valid YouTube link.'}), 400
    try:
        duration = max(5, min(120, int(data.get('duration', 30))))
    except (TypeError, ValueError):
        duration = 30
    job_id = str(uuid.uuid4())[:8]
    yt_clip_jobs[job_id] = {'job_id': job_id, 'status': 'queued', 'step': 'Queued...', 'error': None}
    threading.Thread(target=run_yt_clip_job, args=(job_id, url, duration), daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/yolo/youtube/<job_id>')
def yt_clip_status(job_id):
    if job_id not in yt_clip_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(yt_clip_jobs[job_id])

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype='video/mp4')

# ================= IMAGE DETECTION ENDPOINT =================
@app.route('/api/detect', methods=['POST'])
def detect_objects():
    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        score_threshold = float(request.form.get('threshold', 0.3))
        score_threshold = max(0.05, min(1.0, score_threshold))
    except ValueError:
        score_threshold = 0.3

    try:
        max_results = int(request.form.get('max_results', 25))
        max_results = max(1, min(100, max_results))
    except ValueError:
        max_results = 25

    in_memory_file = io.BytesIO(file.read())
    file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return jsonify({'error': 'Could not decode image'}), 400

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_bgr.shape

    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detector = get_detector(score_threshold=score_threshold, max_results=max_results)
    detection_result = detector.detect(mp_image)

    parsed_detections = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        top_cat = detection.categories[0] if detection.categories else None
        if not top_cat:
            continue

        parsed_detections.append({
            'category_name': top_cat.category_name,
            'score': round(float(top_cat.score), 4),
            'bounding_box': {
                'origin_x': int(bbox.origin_x),
                'origin_y': int(bbox.origin_y),
                'width': int(bbox.width),
                'height': int(bbox.height),
            },
            'normalized_box': {
                'x': round(bbox.origin_x / w, 4),
                'y': round(bbox.origin_y / h, 4),
                'w': round(bbox.width / w, 4),
                'h': round(bbox.height / h, 4),
            },
            'color': f"rgb{get_class_color(top_cat.category_name)}"
        })

    annotated_bgr = annotate_image(image_bgr, parsed_detections)
    _, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    class_counts = {}
    for d in parsed_detections:
        c = d['category_name']
        class_counts[c] = class_counts.get(c, 0) + 1

    inference_ms = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'success': True,
        'image_width': w,
        'image_height': h,
        'inference_time_ms': inference_ms,
        'total_detected': len(parsed_detections),
        'class_counts': class_counts,
        'detections': parsed_detections,
        'annotated_image': f"data:image/jpeg;base64,{annotated_base64}"
    })

# ================= FACE MESH ENDPOINT =================
@app.route('/api/facemesh/detect', methods=['POST'])
def detect_face_mesh():
    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    background = request.form.get('background', 'original')
    mesh_style = request.form.get('mesh_style', 'both')
    color_theme = request.form.get('color_theme', 'cyan')
    try:
        line_thickness = int(request.form.get('line_thickness', 1))
        line_thickness = max(1, min(4, line_thickness))
    except ValueError:
        line_thickness = 1

    draw_points = request.form.get('draw_points', 'false').lower() == 'true'

    try:
        num_faces = int(request.form.get('num_faces', 2))
        num_faces = max(1, min(5, num_faces))
    except ValueError:
        num_faces = 2

    in_memory_file = io.BytesIO(file.read())
    file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return jsonify({'error': 'Could not decode image'}), 400

    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    landmarker = get_face_landmarker(num_faces=num_faces)
    result = landmarker.detect(mp_image)

    face_landmarks_list = result.face_landmarks or []
    faces_detected = len(face_landmarks_list)

    annotated_bgr = annotate_face_mesh(
        image_bgr,
        face_landmarks_list,
        background=background,
        mesh_style=mesh_style,
        color_theme=color_theme,
        line_thickness=line_thickness,
        draw_points=draw_points
    )

    _, buffer = cv2.imencode('.jpg', annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    import base64
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')

    inference_ms = round((time.time() - start_time) * 1000, 1)

    return jsonify({
        'success': True,
        'image_width': w,
        'image_height': h,
        'inference_time_ms': inference_ms,
        'faces_detected': faces_detected,
        'landmarks_per_face': 468,
        'annotated_image': f"data:image/jpeg;base64,{annotated_base64}"
    })

# ================= VIDEO PROCESSING ENDPOINTS =================
@app.route('/api/video/start', methods=['POST'])
def start_video_detection():
    source_type = request.form.get('source_type', 'upload')
    video_task = request.form.get('video_task', 'object')  # 'object' or 'facemesh'

    try:
        duration = float(request.form.get('duration', 15))
        duration = max(3.0, min(120.0, duration))
    except ValueError:
        duration = 15.0

    try:
        sample_fps = float(request.form.get('sample_fps', 12))
        sample_fps = max(2.0, min(30.0, sample_fps))
    except ValueError:
        sample_fps = 12.0

    try:
        threshold = float(request.form.get('threshold', 0.3))
        threshold = max(0.05, min(0.95, threshold))
    except ValueError:
        threshold = 0.3

    try:
        max_results = int(request.form.get('max_results', 20))
        max_results = max(1, min(50, max_results))
    except ValueError:
        max_results = 20

    background = request.form.get('background', 'original')
    mesh_style = request.form.get('mesh_style', 'both')
    color_theme = request.form.get('color_theme', 'cyan')
    try:
        line_thickness = int(request.form.get('line_thickness', 1))
        line_thickness = max(1, min(4, line_thickness))
    except ValueError:
        line_thickness = 1
    draw_points = request.form.get('draw_points', 'false').lower() == 'true'

    try:
        extended_angle = max(110.0, min(175.0, float(request.form.get('extended_angle', 150))))
        bent_angle = max(20.0, min(100.0, float(request.form.get('bent_angle', 60))))
    except ValueError:
        extended_angle, bent_angle = 150.0, 60.0
    if bent_angle >= extended_angle - 20:
        return jsonify({'error': 'Bent angle must be at least 20 degrees below the straight angle'}), 400

    job_id = str(uuid.uuid4())[:8]
    file_or_url = None

    if source_type == 'youtube':
        yt_url = request.form.get('youtube_url', '').strip()
        if not yt_url:
            return jsonify({'error': 'No YouTube URL provided'}), 400
        file_or_url = yt_url
    else:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        uploaded_file = request.files['video']
        if uploaded_file.filename == '':
            return jsonify({'error': 'Empty video filename'}), 400

        ext = os.path.splitext(uploaded_file.filename)[1] or '.mp4'
        save_path = os.path.join(TEMP_DIR, f"{job_id}_upload{ext}")
        uploaded_file.save(save_path)
        file_or_url = save_path

    video_jobs[job_id] = {
        'job_id': job_id,
        'source_type': source_type,
        'video_task': video_task,
        'status': 'queued',
        'progress': 0,
        'step': 'Job queued...',
        'proc_fps': 0,
        'total_detections': 0,
        'class_counts': {},
        'error': None
    }

    t = threading.Thread(
        target=run_video_job,
        args=(job_id, source_type, file_or_url, duration, sample_fps, threshold, max_results,
              video_task, background, mesh_style, color_theme, line_thickness, draw_points,
              extended_angle, bent_angle),
        daemon=True
    )
    t.start()

    return jsonify({'success': True, 'job_id': job_id})

@app.route('/api/video/status/<job_id>')
def get_video_status(job_id):
    if job_id not in video_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(video_jobs[job_id])

def _port_in_use(host, port):
    """On Windows a second server can bind the same port and the browser keeps
    talking to the OLD one (with old code). Detect that before starting."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


if __name__ == '__main__':
    HOST, PORT = '127.0.0.1', 5000
    if _port_in_use(HOST, PORT):
        print("=" * 64)
        print(f"  ERROR: Port {PORT} is already in use - another server is still running!")
        print("  Close the other 'MediaPipe' console window (or end python.exe")
        print("  in Task Manager) and start run.bat again.")
        print("=" * 64)
        sys.exit(1)
    ensure_models_downloaded()
    print(f"Starting Flask server at http://{HOST}:{PORT} ...")
    app.run(host=HOST, port=PORT, debug=False)
