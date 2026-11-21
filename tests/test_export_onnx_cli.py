"""Exercise deployment flags through the real exporter entry point."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from denoise_lab.arch_registry import build_model

ROOT = Path(__file__).resolve().parents[1]


def export(checkpoint, output, *flags):
    return subprocess.run([sys.executable, str(ROOT/'scripts/export_onnx.py'),
                           '--checkpoint', str(checkpoint), '--output', str(output), *flags],
                          cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT/'src'),
                                         'PYTHONDONTWRITEBYTECODE': '1'},
                          capture_output=True, text=True, timeout=60)


def tiny_checkpoint(path, cond='none', ema=True):
    config = dict(width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1], cond=cond)
    torch.manual_seed(8)
    model = build_model(config).eval()
    with torch.no_grad():
        model.body.ending.bias.fill_(.3)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    saved = dict(model_config=config, model=state)
    if ema:
        with torch.no_grad():
            model.body.ending.bias.fill_(-.2)
        saved['ema_model'] = model.state_dict()
    torch.save(saved, path)
    return config, model, state


def test_real_exporter_dynamic_ema_fp32(tmp_path):
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    checkpoint, output = tmp_path/'tiny.pt', tmp_path/'tiny.onnx'
    config, ema, regular = tiny_checkpoint(checkpoint)
    result = export(checkpoint, output, '--weights', 'ema', '--precision', 'fp32', '--dynamic')
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'ONNX checker passed' in result.stdout
    assert 'CPU ONNX vs torch max abs diff:' in result.stdout
    graph = onnx.load(output)
    onnx.checker.check_model(graph)
    assert {p.key: p.value for p in graph.metadata_props} == dict(cond='none', weights='ema', precision='fp32')
    assert graph.opset_import[0].version == 17
    for value in [graph.graph.input[0], graph.graph.output[0]]:
        assert value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        dims = value.type.tensor_type.shape.dim
        assert [dims[i].dim_param for i in (0, 2, 3)] == ['batch', 'height', 'width']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), sess_options=options, providers=['CPUExecutionProvider'])
    model = build_model(config).eval()
    model.load_state_dict(regular)
    generator = torch.Generator().manual_seed(9)
    for shape in [(1, 4, 16, 32), (2, 4, 32, 48)]:
        noisy = torch.rand(shape, generator=generator)
        actual = session.run(['output'], {'inp_1': noisy.numpy()})[0]
        with torch.inference_mode():
            expected = ema(noisy).numpy()
            other = model(noisy).numpy()
        assert actual.shape == shape
        assert np.max(np.abs(expected-other)) > .1
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('problem', ['conditioned', 'missing_ema'])
def test_real_exporter_rejections(tmp_path, problem):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    checkpoint, output = tmp_path/'tiny.pt', tmp_path/'rejected.onnx'
    tiny_checkpoint(checkpoint, cond='noise_map' if problem == 'conditioned' else 'none',
                    ema=problem != 'missing_ema')
    result = export(checkpoint, output, '--weights', 'ema', '--precision', 'fp32')
    assert result.returncode != 0
    message = ('Fixed 4-channel deployment contract requires cond=none' if problem == 'conditioned' else
               'Checkpoint has no ema weights')
    assert message in result.stderr
    assert not output.exists()
