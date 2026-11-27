"""
Export the YOLO26 nano models for all 7 Ultralytics tasks to ONNX for the browser app.

Needs `ultralytics` (pulls in PyTorch) - only for this one-time export, not to run the app:
    python -m pip install ultralytics onnx onnxslim onnxruntime
    python export_models.py

Writes models/<name>.onnx and models/manifest.json (task, input size, class names,
output names/shapes) which the web app reads to pick the right post-processing.
"""
import json
import os
import shutil

import onnxruntime as ort
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'models')
os.makedirs(OUT, exist_ok=True)

# task key -> (weights, input size). Small inputs keep it fast on slower laptops.
MODELS = {
    'detect':   ('yolo26n.pt', 320),
    'segment':  ('yolo26n-seg.pt', 320),
    'semantic': ('yolo26n-sem.pt', 320),
    'depth':    ('yolo26n-depth.pt', 320),
    'classify': ('yolo26n-cls.pt', 224),   # classification models are trained at 224
    'pose':     ('yolo26n-pose.pt', 320),
    'obb':      ('yolo26n-obb.pt', 320),
}

manifest = {}
for task, (weights, imgsz) in MODELS.items():
    print(f'\n===== {task}: {weights} @ {imgsz} =====')
    model = YOLO(weights)
    kwargs = dict(format='onnx', imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    if task == 'detect':
        kwargs['nms'] = False   # raw head output, NMS is done in the browser
    path = model.export(**kwargs)
    name = os.path.basename(path)
    target = os.path.join(OUT, name)
    shutil.copyfile(path, target)

    sess = ort.InferenceSession(target, providers=['CPUExecutionProvider'])
    inputs = [{'name': i.name, 'shape': i.shape} for i in sess.get_inputs()]
    outputs = [{'name': o.name, 'shape': o.shape} for o in sess.get_outputs()]
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    manifest[task] = {
        'file': f'models/{name}',
        'weights': weights,
        'task': model.task,
        'imgsz': imgsz,
        'size_mb': round(os.path.getsize(target) / 1e6, 2),
        'inputs': inputs,
        'outputs': outputs,
        'names': {int(k): v for k, v in names.items()},
        'kpt_shape': getattr(model.model, 'kpt_shape', None) if task == 'pose' else None,
    }
    print(json.dumps({k: manifest[task][k] for k in ('file', 'task', 'inputs', 'outputs')}, indent=1))

with open(os.path.join(OUT, 'manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=1, ensure_ascii=False)
print('\nWrote', os.path.join(OUT, 'manifest.json'))
