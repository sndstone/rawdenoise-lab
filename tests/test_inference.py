"""Portable inference and independently decoded CFA output contracts."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy
import torch
import tifffile

from denoise_lab.arch_registry import build_model
from denoise_lab.bayer import PHASES
from denoise_lab.dng import RawMetadata, sensor_codes, write_dng
from denoise_lab.inference import Denoiser

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(tmp_path):
    config = dict(width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    torch.manual_seed(12)
    model = build_model(config).eval()
    # Exercise the blocks instead of their zero-initialized residual identity.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(('.beta', '.gamma')):
                parameter.fill_(.1)
    path = tmp_path/'tiny.pt'
    torch.save(dict(model_config=config, model=model.state_dict(), ema=model.state_dict()), path)
    return path, model


def fixture_raw(path, phase='RGGB'):
    pattern = np.array(['RGB'.index(c) for c in phase], dtype=np.uint8).reshape(2, 2)
    meta = RawMetadata(pattern=pattern, black=np.array([[512., 520.], [528., 536.]]),
                       white=np.full((2, 2), 16383.))
    codes = np.random.default_rng(3).integers(600, 14000, (64, 80), dtype=np.uint16)
    write_dng(path, codes, meta, normalized=False)
    return codes, meta


@pytest.mark.parametrize('phase', list(PHASES))
def test_dng_roundtrip(tmp_path, phase):
    path = tmp_path/'source.dng'
    codes, meta = fixture_raw(path, phase)
    normalized = np.empty(codes.shape, np.float32)
    for y in range(2):
        for x in range(2):
            normalized[y::2, x::2] = (codes[y::2, x::2]-meta.black[y,x])/(meta.white[y,x]-meta.black[y,x])
    np.testing.assert_array_equal(sensor_codes(normalized, meta), codes)
    write_dng(path, normalized, meta)
    with rawpy.imread(str(path)) as raw:
        np.testing.assert_array_equal(raw.raw_image_visible, codes)
        assert RawMetadata.from_raw(raw, path).phase == phase
        assert raw.postprocess(output_bps=16).dtype == np.uint16
    with tifffile.TiffFile(path) as tif:
        tags = tif.pages[0].tags
        assert tags[50706].value == bytes([1,4,0,0])
        assert tags[50720].value == (80,64)
        assert len(tags[50721].value) == 18


def test_torch_onnx_parity(tmp_path, checkpoint):
    pytest.importorskip('onnxruntime')
    pytest.importorskip('onnx')
    path, model = checkpoint
    onnx_path = tmp_path/'tiny.onnx'
    with torch.inference_mode():
        torch.onnx.export(model, torch.rand(1,4,32,32), str(onnx_path), input_names=['inp_1'],
                          output_names=['output'], dynamo=False, opset_version=17,
                          dynamic_axes={'inp_1': {0:'batch',2:'height',3:'width'},
                                        'output': {0:'batch',2:'height',3:'width'}})
    inp = np.random.default_rng(1).random((4,19,23), dtype=np.float32)
    a = Denoiser.load(path, device='cpu').denoise_packed(inp, tile_size=32, overlap=8)
    onnx_model = Denoiser.load(onnx_path, device='cpu')
    b = onnx_model.denoise_packed(inp, tile_size=32, overlap=8)
    np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5)
    assert onnx_model.device == 'cpu'
    np.testing.assert_array_equal(onnx_model.denoise_packed(inp, strength=0), inp)
    assert Denoiser.load(path, device='cpu', weights='ema').backend == 'torch'


@pytest.mark.parametrize('phase', ['RGGB', 'BGGR'])
def test_cli_smoke(tmp_path, checkpoint, phase):
    path, _ = checkpoint
    source = tmp_path/'source.dng'
    codes, _ = fixture_raw(source, phase)
    out = tmp_path/'out'
    command = [sys.executable, str(ROOT/'scripts/denoise.py'), str(source), '--model', str(path),
               '--device', 'cpu', '--output-dir', str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'torch: cpu' in result.stdout
    with rawpy.imread(str(out/'source_denoised.dng')) as raw:
        assert raw.raw_image_visible.shape == codes.shape
        assert RawMetadata.from_raw(raw, out/'source_denoised.dng').phase == phase
        assert raw.postprocess().shape == (*codes.shape, 3)
    with rawpy.imread(str(source)) as raw:
        np.testing.assert_array_equal(raw.raw_image_visible, codes)
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode == 1
    assert 'Refusing to overwrite' in refused.stdout


def test_non_bayer_refused():
    raw = SimpleNamespace(raw_pattern=np.zeros((6,6)), raw_image_visible=np.zeros((64,64)))
    with pytest.raises(ValueError, match='no X-Trans/Foveon'):
        RawMetadata.from_raw(raw)
    with pytest.raises(ValueError, match='Unsupported Bayer'):
        RawMetadata(pattern=np.zeros((2,2), dtype=np.uint8)).phase


def test_conditioned_refused(tmp_path):
    path = tmp_path/'conditioned.pt'
    torch.save({'model_config': {'cond': 'noise_map'}}, path)
    with pytest.raises(ValueError, match='cond=none'):
        Denoiser.load(path)


def test_coreml_platform_error(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    with pytest.raises(RuntimeError, match='requires macOS'):
        Denoiser.load('model.mlpackage')


def test_onnx_provider_fallback(tmp_path, checkpoint, monkeypatch):
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnx')
    _, model = checkpoint
    path = tmp_path/'fallback.onnx'
    with torch.inference_mode():
        torch.onnx.export(model, torch.rand(1,4,16,16), str(path), dynamo=False, opset_version=17)
    initialize = ort.InferenceSession
    attempts = []
    def session(*args, providers, **kwargs):
        attempts.append(providers.copy())
        if providers[0] == 'CoreMLExecutionProvider':
            raise RuntimeError('simulated provider initialization failure')
        return initialize(*args, providers=providers, **kwargs)
    monkeypatch.setattr(ort, 'get_available_providers', lambda: ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    monkeypatch.setattr(ort, 'InferenceSession', session)
    with pytest.warns(RuntimeWarning, match='retrying'):
        denoiser = Denoiser.load(path)
    assert denoiser.providers == ['CPUExecutionProvider']
    assert attempts == [['CoreMLExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']]


@pytest.mark.parametrize('output_format', ['tiff', 'npy'])
def test_cli_other_formats(tmp_path, checkpoint, output_format):
    path, _ = checkpoint
    source = tmp_path/'source.dng'
    codes, _ = fixture_raw(source)
    result = subprocess.run([sys.executable, str(ROOT/'scripts/denoise.py'), str(source),
                             '--model', str(path), '--device', 'cpu', '--no-prepass',
                             '--strength', '0', '--format', output_format], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    output = tmp_path/f'source_denoised.{output_format}'
    if output_format == 'tiff':
        rgb = tifffile.imread(output)
        assert rgb.dtype == np.uint16
        assert rgb.shape == (*codes.shape, 3)
    else:
        mosaic = np.load(output)
        assert mosaic.dtype == np.float32
        assert mosaic.shape == codes.shape
