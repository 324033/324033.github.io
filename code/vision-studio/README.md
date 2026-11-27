# MediaPipe Vision Studio (Python / Flask)

Local web app used for assignments **1.4, 1.5** (image / video object recognition),
**2.1, 2.2** (3D Face Mesh on images and videos) and **3.4** (Gemini chat).
It needs a Python server, so it cannot run on GitHub Pages - see the screenshots on my
homepage, or run it yourself:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
run.bat                      (or: python app.py)
```

Then open http://127.0.0.1:5000 (Gemini chat: http://127.0.0.1:5000/chat).

- MediaPipe models are downloaded automatically on first start.
- YOLO26 pages need `models/yolo26n.onnx` (`python export_yolo.py`, needs `ultralytics`).
- Gemini chat: copy `.env.example` to `.env` and add your own `GEMINI_API_KEY`
  (the key is read by the server only and is never committed).

Live browser versions of the other projects: https://324033.github.io/projects/
