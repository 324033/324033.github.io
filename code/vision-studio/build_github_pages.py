"""
Builds the course projects as static files for GitHub Pages (https://324033.github.io).

    python build_github_pages.py <path to the local clone of 324033.github.io>

Creates inside that folder:
  projects/                 browser demos that run live on GitHub Pages
    index.html              overview of all demos
    assets/                 shared CSS
    models/yolo26n.onnx     YOLO26n (640) for the YOLO + CCTV pages
    fitness-coach/          2.3  MediaPipe Pose (webcam rep counter)
    hand-gestures/          2.4  MediaPipe Gesture Recognizer
    virtual-background/     2.5  MediaPipe Image Segmentation
    face-keypoints/         2.6  MediaPipe Face Detection
    teachable-machine/      3.1  Teachable Machine classifier
    yolo26-detection/       3.2  YOLO26 (CCTV / webcam / files; YouTube needs the local server)
    yolo26-tasks/           3.8  YOLO26 webcam tasks + 3.3/3.9 Neural Style Transfer (style.html)
    cctv-analytics/         3.6/3.7 CCTV counting + heatmap + speed
    course-islands/         3.5
    wanderer/               5.9
  code/vision-studio/       source of the Python (Flask) app for 1.4, 1.5, 2.1, 2.2, 3.4
                            (runs locally only - GitHub Pages cannot run Python)

Never copied: .env (API key), .venv, models of the Python app, outputs/, temp/, test media.
The browser pages load the MediaPipe models directly from Google (storage.googleapis.com).
"""
import os
import re
import shutil
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(SRC, 'templates', 'index.html')

GOOGLE = 'https://storage.googleapis.com/mediapipe-models/'
MEDIAPIPE_MODELS = {
    "'/models/pose_landmarker.task'": f"'{GOOGLE}pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'",
    "'/models/gesture_recognizer.task'": f"'{GOOGLE}gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'",
    "'/models/selfie_segmenter.tflite'": f"'{GOOGLE}image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite'",
    "'/models/blaze_face_short_range.tflite'": f"'{GOOGLE}face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'",
}

# standalone pages cut out of the Vision Studio (one tab each)
TAB_PAGES = [
    # folder, section id, title, subtitle, js file, unit
    ('fitness-coach', 'fitnessModeSection', 'Fitness Coach', 'MediaPipe Pose - counts arm reps (left / right) live from the webcam', 'fitness.js', '2.3'),
    ('hand-gestures', 'gestureModeSection', 'Hand Gestures', 'MediaPipe Gesture Recognizer - two hands, colour-coded landmarks', 'gestures.js', '2.4'),
    ('virtual-background', 'backgroundModeSection', 'Virtual Background', 'MediaPipe Image Segmentation - keys 1-4, 0 r w y g, b = blur', 'background.js', '2.5'),
    ('face-keypoints', 'faceKeysModeSection', 'Face Keypoints', 'MediaPipe Face Detection - any key shuffles the shapes, Esc / q exits', 'facekeys.js', '2.6'),
    ('teachable-machine', 'tmModeSection', 'Teachable Machine', 'My own image model (Tiger, Phone, Cup, Blank Screen) on the webcam', 'teachable.js', '3.1'),
]


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def replace_all(text, pairs, where):
    for old, new in pairs:
        if old not in text:
            raise RuntimeError(f'{where}: expected text not found: {old[:60]!r}')
        text = text.replace(old, new)
    return text


def page_shell(title, subtitle, unit, body, head_extra='', scripts=''):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="../assets/style.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
{head_extra}</head>
<body>
    <div class="app-container">
        <header class="app-header">
            <div class="header-left">
                <div class="logo-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/></svg>
                </div>
                <div>
                    <h1>{title}</h1>
                    <p class="subtitle">Assignment {unit} &bull; {subtitle}</p>
                </div>
            </div>
            <div class="header-nav">
                <a class="nav-tab nav-tab-link" href="../">&#x2190; All projects</a>
            </div>
            <div class="header-status">
                <span class="status-dot"></span>
                <span class="status-text">Runs in your browser</span>
            </div>
        </header>
{body}
    </div>
{scripts}</body>
</html>
"""


def extract_section(html, section_id):
    start = html.index(f'<main class="main-content" id="{section_id}"')
    end = html.index('</main>', start) + len('</main>')
    section = html[start:end]
    return section.replace('style="display: none;"', 'style="display: grid;"', 1)


def build_tab_pages(projects):
    html = read(TEMPLATE)
    for folder, section_id, title, subtitle, js, unit in TAB_PAGES:
        section = extract_section(html, section_id).replace('/static/backgrounds/', './backgrounds/')
        code = read(os.path.join(SRC, 'static', 'js', js))
        for old, new in MEDIAPIPE_MODELS.items():
            code = code.replace(old, new)
        code = code.replace("'/static/backgrounds/", "'./backgrounds/")
        out = os.path.join(projects, folder)
        if js == 'teachable.js':
            scripts = ('    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.7.4/dist/tf.min.js"></script>\n'
                       '    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8.5/dist/teachablemachine-image.min.js"></script>\n'
                       '    <script src="./app.js"></script>\n'
                       "    <script>document.dispatchEvent(new CustomEvent('modechange', { detail: { mode: 'tm' } }));</script>\n")
        else:
            scripts = '    <script type="module" src="./app.js"></script>\n'
        write(os.path.join(out, 'index.html'), page_shell(title, subtitle, unit, '        ' + section, scripts=scripts))
        write(os.path.join(out, 'app.js'), code)
        if js == 'background.js':
            shutil.copytree(os.path.join(SRC, 'static', 'backgrounds'), os.path.join(out, 'backgrounds'), dirs_exist_ok=True)
        if '/models/' in code:
            raise RuntimeError(f'{js}: a local /models/ path is left')
        print('  built', folder)


def build_vision_studio(projects):
    """The complete Vision Studio (all tabs) as a static page. vision-shim.js answers the
    /api/... requests in the browser (1.4, 1.5, 2.1, 2.2), the live tabs load their models from Google."""
    out = os.path.join(projects, 'vision-studio')
    asset = re.compile(r"\?v=\{\{ asset_version\('[^']+'\) \}\}")
    page = asset.sub('', read(TEMPLATE))
    page = replace_all(page, [
        ('/static/css/style.css', '../assets/style.css'),
        ('<script src="/static/js/app.js">', '<script type="module" src="./vision-shim.js"></script>\n    <script src="./app.js">'),
        ('href="/yolo"', 'href="../yolo26-detection/"'), ('href="/cctv"', 'href="../cctv-analytics/"'),
        ('Local MediaPipe Ready', 'Runs in your browser'),
        ('<label for="ytUrlInput" class="input-label">Enter YouTube Link:</label>',
         '<p class="param-hint" style="margin-bottom:8px"><b>Online version:</b> YouTube links need the local Python server. Please use &ldquo;Upload Video File&rdquo; here.</p>\n'
         '                        <label for="ytUrlInput" class="input-label">Enter YouTube Link:</label>'),
    ], 'index.html')
    for js in ('fitness.js', 'gestures.js', 'background.js', 'facekeys.js', 'teachable.js'):
        page = replace_all(page, [(f'/static/js/{js}', f'./{js}')], 'index.html')
    page = page.replace('/static/backgrounds/', './backgrounds/')
    leftover = re.findall(r'(?:src|href)="/(?!/)[^"]*"', page)
    if leftover:
        raise RuntimeError(f'vision-studio: absolute paths left: {leftover}')
    write(os.path.join(out, 'index.html'), page)
    for js in ('app.js', 'vision-shim.js', 'fitness.js', 'gestures.js', 'background.js', 'facekeys.js', 'teachable.js'):
        code = read(os.path.join(SRC, 'static', 'js', js))
        for old, new in MEDIAPIPE_MODELS.items():
            code = code.replace(old, new)
        code = code.replace("'/static/backgrounds/", "'./backgrounds/")
        write(os.path.join(out, js), code)
    shutil.copytree(os.path.join(SRC, 'static', 'backgrounds'), os.path.join(out, 'backgrounds'), dirs_exist_ok=True)
    print('  built vision-studio')


def build_yolo_pages(projects):
    os.makedirs(os.path.join(projects, 'models'), exist_ok=True)
    shutil.copyfile(os.path.join(SRC, 'models', 'yolo26n.onnx'), os.path.join(projects, 'models', 'yolo26n.onnx'))
    for css in ('style.css', 'yolo.css', 'cctv.css'):
        shutil.copyfile(os.path.join(SRC, 'static', 'css', css), os.path.join(projects, 'assets', css))
    asset = re.compile(r"\?v=\{\{ asset_version\('[^']+'\) \}\}")

    # 3.2 YOLO26 detection
    page = asset.sub('', read(os.path.join(SRC, 'templates', 'yolo.html')))
    page = replace_all(page, [
        ('/static/css/style.css', '../assets/style.css'), ('/static/css/yolo.css', '../assets/yolo.css'),
        ('/static/js/yolo.js', './yolo.js'), ('href="/cctv"', 'href="../cctv-analytics/"'),
        ('href="/">', 'href="../">'), ('Back to Vision Studio', 'All projects'),
        ('<span class="param-hint">The beginning of the video is downloaded locally first, so the browser may analyse its frames.</span>',
         '<span class="param-hint"><b>Note:</b> YouTube clips need the local Python server (browsers may not read pixels from the YouTube player). Online, use CCTV, webcam or a video file.</span>'),
    ], 'yolo.html')
    js = read(os.path.join(SRC, 'static', 'js', 'yolo.js'))
    js = replace_all(js, [
        ("const MODEL_URL = '/models/yolo26n.onnx';", "const MODEL_URL = '../models/yolo26n.onnx';"),
        ("            showStatus('Requesting the video...');",
         "            if (location.hostname.endsWith('github.io')) throw new Error('YouTube clips only work with the local Python server. Online, please use CCTV, webcam or a video file.');\n"
         "            showStatus('Requesting the video...');"),
    ], 'yolo.js')
    write(os.path.join(projects, 'yolo26-detection', 'index.html'), page)
    write(os.path.join(projects, 'yolo26-detection', 'yolo.js'), js)
    print('  built yolo26-detection')

    # 3.6 / 3.7 CCTV analytics
    page = asset.sub('', read(os.path.join(SRC, 'templates', 'cctv.html')))
    page = replace_all(page, [
        ('/static/css/style.css', '../assets/style.css'), ('/static/css/yolo.css', '../assets/yolo.css'),
        ('/static/css/cctv.css', '../assets/cctv.css'), ('/static/js/cctv.js', './cctv.js'),
        ('href="/yolo"', 'href="../yolo26-detection/"'), ('href="/">Vision Studio', 'href="../">All projects'),
    ], 'cctv.html')
    js = replace_all(read(os.path.join(SRC, 'static', 'js', 'cctv.js')),
                     [("const MODEL_URL = '/models/yolo26n.onnx';", "const MODEL_URL = '../models/yolo26n.onnx';")], 'cctv.js')
    write(os.path.join(projects, 'cctv-analytics', 'index.html'), page)
    write(os.path.join(projects, 'cctv-analytics', 'cctv.js'), js)
    print('  built cctv-analytics')


def build_static_apps(projects):
    # 3.8 YOLO26 webcam tasks + 3.3 / 3.9 neural style transfer (already pure static)
    skip = shutil.ignore_patterns('__pycache__', '_test', '*.bak', 'run_yolo26_tasks.bat', 'run_yolo26_tasks.command')
    shutil.copytree(os.path.join(SRC, 'yolo26_tasks'), os.path.join(projects, 'yolo26-tasks'), ignore=skip, dirs_exist_ok=True)
    print('  built yolo26-tasks (+ style.html)')
    write(os.path.join(projects, 'course-islands', 'index.html'), read(os.path.join(SRC, 'course_islands', 'course_islands.html')))
    print('  built course-islands')
    write(os.path.join(projects, 'wanderer', 'index.html'), read(os.path.join(SRC, 'assignment_5_9', 'wanderer.html')))
    print('  built wanderer')


OVERVIEW = [
    ('1.4', 'Image Object Recognition (MediaPipe)', 'vision-studio/#object'),
    ('1.5', 'Video Object Recognition (MediaPipe)', 'vision-studio/#video'),
    ('2.1', 'Face Landmark Detection - 3D Face Mesh (image)', 'vision-studio/#facemesh'),
    ('2.2', 'Video Face Landmark Detection - 3D Face Mesh (video)', 'vision-studio/#video-facemesh'),
    ('2.3', 'Fitness Coach (MediaPipe Pose)', 'fitness-coach/'),
    ('2.4', 'Hand Gestures (MediaPipe Gesture Recognizer)', 'hand-gestures/'),
    ('2.5', 'Virtual Background (MediaPipe Image Segmentation)', 'virtual-background/'),
    ('2.6', 'Face Keypoints (MediaPipe Face Detection)', 'face-keypoints/'),
    ('3.1', 'Teachable Machine Object Classifier', 'teachable-machine/'),
    ('3.2', 'YOLO26 Detection (CCTV, webcam, files)', 'yolo26-detection/'),
    ('3.3 / 3.9', 'Neural Style Transfer (11 styles)', 'yolo26-tasks/style.html'),
    ('3.5', 'Pforzheim AI Course Islands', 'course-islands/'),
    ('3.6 / 3.7', 'CCTV Traffic Analytics (counting, heatmap, speed)', 'cctv-analytics/'),
    ('3.8', 'YOLO26 Webcam Tasks (7 nano models)', 'yolo26-tasks/'),
    ('5.9', 'Wanderer above the Sea of Fog', 'wanderer/'),
]


def build_overview(projects):
    rows = '\n'.join(f'      <li><span class="u">{u}</span><a href="{href}">{name}</a></li>' for u, name, href in OVERVIEW)
    write(os.path.join(projects, 'index.html'), f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Course Projects - 324033</title>
  <style>
    body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 32px 16px; }}
    main {{ max-width: 760px; margin: 0 auto; }}
    h1 {{ margin: 0 0 6px; }} p {{ color: #94a3b8; }}
    ul {{ list-style: none; padding: 0; display: grid; gap: 10px; }}
    li {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 12px 16px; display: flex; gap: 14px; align-items: center; }}
    .u {{ min-width: 72px; font-weight: 700; color: #f59e0b; }}
    a {{ color: #93c5fd; font-weight: 600; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <main>
    <h1>AI Course Projects</h1>
    <p>MMI5194 &middot; student 324033. All demos run in your browser (webcam demos ask for camera permission; nothing is uploaded).</p>
    <ul>
{rows}
    </ul>
    <p>1.4&ndash;2.2 open the complete <a href="vision-studio/">Vision Studio</a> (the online version computes everything in the browser).
      The Gemini chat (3.4) needs a private API key and runs on the local Python server only &ndash; source code:
      <a href="https://github.com/324033/324033.github.io/tree/main/code/vision-studio">code/vision-studio</a>.</p>
    <p><a href="../">&larr; Back to my homepage</a></p>
  </main>
</body>
</html>
""")
    print('  built projects/index.html')


CODE_FILES = ['app.py', 'fitness.py', 'gemini_chat.py', 'export_yolo.py', 'build_github_pages.py', 'requirements.txt',
              'run.bat', '.env.example', '.gitignore', 'test_app.py', 'test_facemesh_pipeline.py', 'test_video_pipeline.py']


def build_code(target):
    code = os.path.join(target, 'code', 'vision-studio')
    for f in CODE_FILES:
        shutil.copyfile(os.path.join(SRC, f), os.path.join(os.makedirs(code, exist_ok=True) or code, f))
    shutil.copytree(os.path.join(SRC, 'templates'), os.path.join(code, 'templates'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(SRC, 'static'), os.path.join(code, 'static'), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '_tmtest'))
    write(os.path.join(code, 'README.md'), """# MediaPipe Vision Studio (Python / Flask)

Local web app used for assignments **1.4, 1.5** (image / video object recognition),
**2.1, 2.2** (3D Face Mesh on images and videos) and **3.4** (Gemini chat).
It needs a Python server, so it cannot run on GitHub Pages - see the screenshots on my
homepage, or run it yourself:

```
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
run.bat                      (or: python app.py)
```

Then open http://127.0.0.1:5000 (Gemini chat: http://127.0.0.1:5000/chat).

- MediaPipe models are downloaded automatically on first start.
- YOLO26 pages need `models/yolo26n.onnx` (`python export_yolo.py`, needs `ultralytics`).
- Gemini chat: copy `.env.example` to `.env` and add your own `GEMINI_API_KEY`
  (the key is read by the server only and is never committed).

Live browser versions of the other projects: https://324033.github.io/projects/
""")
    for bad in ('.env',):
        if os.path.exists(os.path.join(code, bad)):
            raise RuntimeError('refusing to publish ' + bad)
    print('  built code/vision-studio')


def main():
    if len(sys.argv) != 2 or not os.path.isdir(os.path.join(sys.argv[1], '.git')):
        sys.exit('usage: python build_github_pages.py <local clone of 324033.github.io>')
    target = sys.argv[1]
    projects = os.path.join(target, 'projects')
    os.makedirs(os.path.join(projects, 'assets'), exist_ok=True)
    shutil.copyfile(os.path.join(SRC, 'static', 'css', 'style.css'), os.path.join(projects, 'assets', 'style.css'))
    build_tab_pages(projects)
    build_vision_studio(projects)
    build_yolo_pages(projects)
    build_static_apps(projects)
    build_overview(projects)
    build_code(target)
    # Jekyll would skip folders/files starting with "_" or "." - serve everything as-is
    open(os.path.join(target, '.nojekyll'), 'a').close()
    print('done')


if __name__ == '__main__':
    main()
