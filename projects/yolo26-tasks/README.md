# YOLO26 Webcam Tasks

Browser app that runs the **Ultralytics YOLO26 nano** models on your webcam with
**ONNX Runtime Web** (no TensorFlow.js). Everything is served locally; nothing leaves your computer.

| Task (dropdown) | Model | Input | Output used by the app |
|---|---|---|---|
| Detection | `yolo26n` (`nms=False`) | 320 | `[1,300,6]` end-to-end boxes (YOLO26 is NMS-free) |
| Instance segmentation | `yolo26n-seg` | 320 | `[1,116,2100]` + mask prototypes `[1,32,80,80]` |
| Semantic segmentation | `yolo26n-sem` (Cityscapes) | 320 | `[1,320,320]` class id per pixel |
| Depth estimation | `yolo26n-depth` | 320 (stretched) | `[1,1,320,320]` depth in metres |
| Classification | `yolo26n-cls` (ImageNet) | 224 (centre crop) | `[1,1000]` probabilities |
| Pose estimation | `yolo26n-pose` | 320 | `[1,56,2100]` box + 17 keypoints |
| Oriented boxes | `yolo26n-obb` (DOTA) | 320 | `[1,20,2100]` box + 15 classes + angle |

## Start

Windows (double-click `run_yolo26_tasks.bat`) or in a terminal inside this folder:

```
python serve.py
```

macOS:

```
python3 serve.py
```

Then open **http://localhost:8000** (opens automatically). Stop with `Ctrl+C`.
Another port: `python serve.py --port 8080`.

`serve.py` sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, so the page is cross-origin isolated and
WebAssembly can use several threads. Do not open `index.html` directly as a file -
the camera and the models need the local server.

## Notes

- **Backend:** WASM (CPU) by default - works in current Chrome, Edge, Firefox and Safari on Windows and Mac.
  "Use GPU (WebGPU)" is optional; if WebGPU is missing or fails, the app falls back to WASM automatically.
- Models load only when their task is selected for the first time; the webcam keeps running while you switch.
- "Run the model on every 2nd / 3rd frame" keeps the video smooth on slow laptops.
- `yolo26n-sem` is trained on Cityscapes (street scenes) and `yolo26n-obb` on DOTA (aerial images),
  so indoors their results are limited by design.
- To re-export the models: `python -m pip install ultralytics onnx onnxslim onnxruntime` then `python export_models.py`.

## Neural Style Transfer (`style.html`)

Start the server (`python serve.py`, macOS `python3 serve.py`) and open
**http://localhost:8000/style.html** (or the "Neural Style Transfer" link on the main page).
Upload / drop / paste a photo, then click **"Convert all styles"** or a single style thumbnail.
Browser-only, no Python backend.

- **11 styles**, listed in `models/style/styles.json`, thumbnails in `models/style/thumbs/`:
  - ONNX Model Zoo fast neural style `-9` models from Hugging Face (`onnxmodelzoo`):
    candy, mosaic, rain_princess, udnie, pointilism. `make_style_models.py` downloads them and makes the
    declared 224x224 input size flexible (fully convolutional; identical results at 224 px).
  - Optional extras converted from Justin Johnson's Torch models (https://github.com/jcjohnson/fast-neural-style):
    starry_night, la_muse, the_scream, the_wave, feathers, composition_vii. `convert_t7_models.py`
    rebuilds each `.t7` network in PyTorch, puts the VGG pre/post-processing (BGR, mean subtraction)
    inside the graph and exports ONNX; ONNX Runtime matches PyTorch to < 0.01 on a 0..255 scale.
- Input to every model: float32 RGB 0..255, NCHW, no mean subtraction. Output is clipped to 0..255.
- Inference runs in a **Web Worker** (`style-worker.js`, ONNX Runtime Web WASM backend) - the page never freezes.
  Multi-threaded WASM inside a worker does not work in every browser; the worker self-tests and
  falls back to 1 thread automatically (remembered for the next visit).
- Models are stored in the browser's Cache Storage after the first download.
- Default size 256 px (longest side), options up to 1024 px. "Style strength" blends with the original.

> **Licence note:** the style models are free for **personal or research use**. Ask the original
> authors before any commercial use.

ONNX Runtime Web 1.20.1 (MIT License, Microsoft) is included in `vendor/onnxruntime-web/`.
