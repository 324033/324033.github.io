"""
One-time export of Ultralytics YOLO26n to ONNX for the browser (YOLO26 page, /yolo).

The page runs the model with onnxruntime-web, so it needs an .onnx file.
This script needs `ultralytics` (which pulls in PyTorch) - it is NOT needed to run the app:

    python -m pip install ultralytics onnx onnxslim
    python export_yolo.py

Result: models/yolo26n.onnx  (input 1x3x640x640, output 1x300x6 = x1,y1,x2,y2,score,class;
YOLO26 is end-to-end / NMS-free, so no NMS is needed in the browser).
"""
import os
import shutil

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

model = YOLO('yolo26n.pt')                          # downloaded automatically from Ultralytics
onnx_path = model.export(format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False)
target = os.path.join(MODEL_DIR, 'yolo26n.onnx')
shutil.copyfile(onnx_path, target)
print(f"Saved {target} ({os.path.getsize(target) / 1e6:.1f} MB)")
print("Classes:", model.names)
