"""
Converts Justin Johnson's fast-neural-style Torch7 models (.t7, https://github.com/jcjohnson/fast-neural-style)
to ONNX for the browser page (style.html). The models are free for personal or research use.

Torch7/Lua cannot be loaded directly, so this script
  1. reads the .t7 file with `torchfile` (pure data reader, no Lua code is executed),
  2. rebuilds the network layer by layer in PyTorch,
  3. wraps it with the model's VGG pre/post-processing (RGB->BGR, mean subtraction and back),
     so the ONNX model takes the SAME input as the ONNX Model Zoo models: float32 RGB 0..255, NCHW,
  4. exports it with flexible height/width and checks ONNX Runtime == PyTorch.

    python -m pip install torch torchfile onnx onnxruntime numpy pillow
    python convert_t7_models.py <folder with .t7 files>
Writes models/style/<name>.onnx and adds successful styles to models/style/styles.json.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile

torchfile.xrange = range   # torchfile still uses Python 2's xrange in its 8-byte-long code path

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'models', 'style')
VGG_MEAN_BGR = [103.939, 116.779, 123.68]

NAMES = {
    'starry_night': 'Starry Night', 'la_muse': 'La Muse', 'the_scream': 'The Scream',
    'the_wave': 'The Wave', 'feathers': 'Feathers', 'composition_vii': 'Composition VII',
}


def g(obj, key, default=None):
    """attribute access on torchfile objects (keys may be bytes or str)"""
    for k in (key, key.encode()):
        try:
            v = obj[k]
            return v
        except (KeyError, TypeError):
            pass
    # note: torchfile dicts return None for missing attributes instead of raising,
    # so a missing key must fall back to the default (e.g. preprocessing -> 'vgg')
    return default


def tname(obj):
    t = getattr(obj, 'torch_typename', lambda: b'')()
    return t.decode() if isinstance(t, bytes) else str(t)


def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))


class ShaveImage(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        s = self.s
        return x[:, :, s:x.shape[2] - s, s:x.shape[3] - s]


class ConcatTable(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return [b(x) for b in self.branches]


class CAddTable(nn.Module):
    def forward(self, xs):
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out


class MulConstant(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = float(c)

    def forward(self, x):
        return x * self.c


class ReflectPad(nn.Module):
    def __init__(self, l, r, t, b):
        super().__init__()
        self.p = (l, r, t, b)

    def forward(self, x):
        return F.pad(x, self.p, mode='reflect')


def build(m):
    """Torch7 module -> PyTorch module"""
    t = tname(m).split('.')[-1]
    if t in ('Sequential',):
        return nn.Sequential(*[build(c) for c in g(m, 'modules')])
    if t == 'ConcatTable':
        return ConcatTable([build(c) for c in g(m, 'modules')])
    if t == 'CAddTable':
        return CAddTable()
    if t in ('Identity', 'TotalVariation'):          # TV loss layer is a no-op at test time
        return nn.Identity()
    if t == 'SpatialConvolution':
        w = tensor(g(m, 'weight'))
        n_out, n_in = int(g(m, 'nOutputPlane')), int(g(m, 'nInputPlane'))
        kh, kw = int(g(m, 'kH')), int(g(m, 'kW'))
        conv = nn.Conv2d(n_in, n_out, (kh, kw), stride=(int(g(m, 'dH')), int(g(m, 'dW'))),
                         padding=(int(g(m, 'padH', 0)), int(g(m, 'padW', 0))), bias=g(m, 'bias') is not None)
        conv.weight.data = w.reshape(n_out, n_in, kh, kw)
        if g(m, 'bias') is not None:
            conv.bias.data = tensor(g(m, 'bias'))
        return conv
    if t == 'SpatialFullConvolution':
        w = tensor(g(m, 'weight'))
        n_out, n_in = int(g(m, 'nOutputPlane')), int(g(m, 'nInputPlane'))
        kh, kw = int(g(m, 'kH')), int(g(m, 'kW'))
        conv = nn.ConvTranspose2d(n_in, n_out, (kh, kw), stride=(int(g(m, 'dH')), int(g(m, 'dW'))),
                                  padding=(int(g(m, 'padH', 0)), int(g(m, 'padW', 0))),
                                  output_padding=(int(g(m, 'adjH', 0)), int(g(m, 'adjW', 0))), bias=g(m, 'bias') is not None)
        conv.weight.data = w.reshape(n_in, n_out, kh, kw)
        if g(m, 'bias') is not None:
            conv.bias.data = tensor(g(m, 'bias'))
        return conv
    if t == 'SpatialReflectionPadding':
        return ReflectPad(int(g(m, 'pad_l')), int(g(m, 'pad_r')), int(g(m, 'pad_t')), int(g(m, 'pad_b')))
    if t == 'SpatialReplicationPadding':
        p = (int(g(m, 'pad_l')), int(g(m, 'pad_r')), int(g(m, 'pad_t')), int(g(m, 'pad_b')))
        return nn.ReplicationPad2d(p)
    if t == 'SpatialBatchNormalization':
        n = len(np.asarray(g(m, 'running_mean')))
        bn = nn.BatchNorm2d(n, eps=float(g(m, 'eps', 1e-5)), affine=g(m, 'weight') is not None)
        bn.running_mean.data = tensor(g(m, 'running_mean'))
        rv = g(m, 'running_var')
        if rv is None:                                 # very old Torch stored 1/std instead of the variance
            rv = 1.0 / (np.asarray(g(m, 'running_std'), dtype=np.float64) ** 2) - float(g(m, 'eps', 1e-5))
        bn.running_var.data = tensor(rv)
        if g(m, 'weight') is not None:
            bn.weight.data = tensor(g(m, 'weight')); bn.bias.data = tensor(g(m, 'bias'))
        return bn
    if t == 'InstanceNormalization':                   # fast-neural-style custom layer: per-image statistics
        w = g(m, 'weight')
        n = len(np.asarray(w)) if w is not None else int(g(m, 'nOutput'))
        inn = nn.InstanceNorm2d(n, eps=float(g(m, 'eps', 1e-5)), affine=w is not None, track_running_stats=False)
        if w is not None:
            inn.weight.data = tensor(w); inn.bias.data = tensor(g(m, 'bias'))
        return inn
    if t == 'ReLU':
        return nn.ReLU()
    if t == 'Tanh':
        return nn.Tanh()
    if t == 'MulConstant':
        return MulConstant(g(m, 'constant_scalar'))
    if t == 'ShaveImage':
        return ShaveImage(int(g(m, 'size')))
    if t == 'SpatialUpSamplingNearest':
        return nn.Upsample(scale_factor=int(g(m, 'scale_factor')), mode='nearest')
    raise NotImplementedError(f'Torch layer "{tname(m)}" is not supported')


class StyleNet(nn.Module):
    """RGB 0..255 in -> RGB 0..255 out (VGG pre/post-processing inside the graph)"""
    def __init__(self, net, preprocessing):
        super().__init__()
        self.net = net
        self.vgg = preprocessing == 'vgg'
        self.register_buffer('mean', torch.tensor(VGG_MEAN_BGR).view(1, 3, 1, 1))

    def forward(self, x):
        if self.vgg:
            y = self.net(x[:, [2, 1, 0]] - self.mean)       # RGB -> BGR, subtract mean
            return (y + self.mean)[:, [2, 1, 0]]               # add mean, BGR -> RGB
        y = self.net(x / 255.0)                               # (fallback, not used by these models)
        return y * 255.0


def convert(path):
    name = os.path.splitext(os.path.basename(path))[0]
    ckpt = torchfile.load(path, force_8bytes_long=True)
    model = g(ckpt, 'model')
    opt = g(ckpt, 'opt', {}) or {}
    pre = g(opt, 'preprocessing', None) or b'vgg'      # fast-neural-style default is VGG preprocessing
    pre = pre.decode() if isinstance(pre, bytes) else str(pre)
    if pre not in ('vgg', 'resnet'):
        raise RuntimeError(f'unknown preprocessing "{pre}"')
    net = StyleNet(build(model), pre).eval()

    with torch.no_grad():
        x = torch.rand(1, 3, 256, 320) * 255
        y = net(x)
    if y.shape != x.shape:
        raise RuntimeError(f'output shape {tuple(y.shape)} != input shape {tuple(x.shape)}')

    target = os.path.join(OUT, f'{name}.onnx')
    torch.onnx.export(net, torch.rand(1, 3, 256, 256) * 255, target, input_names=['input1'], output_names=['output1'],
                      dynamic_axes={'input1': {2: 'height', 3: 'width'}, 'output1': {2: 'height', 3: 'width'}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    sess = ort.InferenceSession(target, providers=['CPUExecutionProvider'])
    for (h, w) in [(256, 320), (384, 512)]:
        # photo-like (smooth) test image: pure white noise pushes the BatchNorm models into
        # extreme ranges where tiny float differences grow - real photos never do that
        coarse = torch.rand(1, 3, h // 16, w // 16) * 255
        xi = F.interpolate(coarse, size=(h, w), mode='bilinear', align_corners=False).numpy().astype(np.float32)
        yo = sess.run(None, {'input1': xi})[0]
        with torch.no_grad():
            yt = net(torch.from_numpy(xi)).numpy()
        diff = float(np.abs(yo - yt).max())
        if yo.shape != xi.shape or diff > 0.5:
            raise RuntimeError(f'ONNX check failed at {h}x{w}: shape {yo.shape}, max diff {diff:.3f}')
    print(f'  {name}: preprocessing={pre}, ONNX == PyTorch (max diff {diff:.4f}), {os.path.getsize(target) / 1e6:.1f} MB')
    return name


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    styles_path = os.path.join(OUT, 'styles.json')
    styles = json.load(open(styles_path, encoding='utf-8'))
    ok, failed = [], []
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.t7'):
            continue
        print('Converting', f)
        try:
            name = convert(os.path.join(folder, f))
            ok.append(name)
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            print(f'  FAILED: {e}')
            failed.append((f, str(e)))
    styles = [s for s in styles if s['id'] not in ok]
    styles += [{'id': n, 'name': NAMES.get(n, n), 'source': 'fast-neural-style (J. Johnson)', 'file': f'{n}.onnx'} for n in ok]
    json.dump(styles, open(styles_path, 'w', encoding='utf-8'), indent=2)
    print('\nConverted:', ok, '\nFailed:', failed)


if __name__ == '__main__':
    main()
