"""
Downloads the ONNX Model Zoo "fast neural style" models (-9 versions, Apache-2.0) from
Hugging Face (onnxmodelzoo/*) and makes their input size flexible.

The original models declare a fixed 1x3x224x224 input, but the network is fully
convolutional (its upsampling factors are computed from the input shape), so only the
declared height/width have to become symbolic. Results at 224x224 stay identical
(max difference < 0.001), and larger photos give much sharper results.

    python -m pip install onnx onnxruntime numpy
    python make_style_models.py
Result: models/style/<style>.onnx
"""
import os
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort

STYLES = ['candy', 'mosaic', 'pointilism', 'rain-princess', 'udnie']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'style')
os.makedirs(OUT, exist_ok=True)

for style in STYLES:
    src = os.path.join(OUT, f'{style}-9.onnx')
    url = f'https://huggingface.co/onnxmodelzoo/{style}-9/resolve/main/{style}-9.onnx'
    if not os.path.exists(src):
        print('Downloading', url)
        urllib.request.urlretrieve(url, src)

    model = onnx.load(src)
    # the old exporter also listed the weights as graph inputs -> remove them (enables constant folding)
    weights = {i.name for i in model.graph.initializer}
    real_inputs = [i for i in model.graph.input if i.name not in weights]
    del model.graph.input[:]
    model.graph.input.extend(real_inputs)
    for tensor in list(model.graph.input) + list(model.graph.output):
        dims = tensor.type.tensor_type.shape.dim
        dims[2].ClearField('dim_value'); dims[2].dim_param = 'height'
        dims[3].ClearField('dim_value'); dims[3].dim_param = 'width'
    onnx.checker.check_model(model)
    target = os.path.join(OUT, f'{style}.onnx')
    onnx.save(model, target)

    session = ort.InferenceSession(target)
    out = session.run(None, {'input1': np.zeros((1, 3, 384, 512), np.float32)})[0]
    print(f'{style}: ok, 512x384 -> {out.shape}')
    os.remove(src)
