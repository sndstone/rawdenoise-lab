"""Synthetic trainer checkpoint compatibility across deployment consumers."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from denoise_lab.arch_registry import build_model
from denoise_lab.checkpoints import model_config, resolve_weights
from denoise_lab.inference import Denoiser

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trainer_checkpoint(tmp_path):
    config = dict(width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)
    model = build_model(config).eval()
    ema = {k: v.clone() for k, v in model.state_dict().items()}
    ema['body.ending.weight'].zero_()
    ema['body.ending.bias'].zero_()
    saved = dict(model=model.state_dict(), model_config=config,
                 config=dict(model=config, data=dict(kind='bootstrap')), optimizer={},
                 scheduler={}, scaler={}, step=7, torch_rng_state=torch.get_rng_state(),
                 data_manifest_sha256=None, init={}, provisional_bootstrap=True,
                 ema_model=ema, ema_updates=7)
    path = tmp_path/'trainer.pt'
    torch.save(saved, path)
    return path, saved


@pytest.mark.parametrize('weights', ['model', 'ema'])
def test_trainer_inference_and_coreml_builder(trainer_checkpoint, weights):
    path, saved = trainer_checkpoint
    expected = saved['ema_model' if weights == 'ema' else 'model']
    loaded = Denoiser.load(path, device='cpu', weights=weights)._call
    exported = script('export_coreml').model_from_checkpoint(saved, weights)
    for model in (loaded, exported):
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_ema_alias_precedence_and_fallback():
    legacy, current = {'weight': torch.ones(1)}, {'weight': torch.zeros(1)}
    assert resolve_weights(dict(ema=legacy), 'ema') is legacy
    assert resolve_weights(dict(ema=legacy, ema_model=current), 'ema') is current
    assert resolve_weights(dict(ema=legacy, ema_model=None), 'ema') is legacy
    with pytest.raises(ValueError, match='no ema weights'):
        resolve_weights(dict(model=current), 'ema')
    with pytest.raises(ValueError, match='weights must be model or ema'):
        resolve_weights({}, 'other')


@pytest.mark.parametrize('saved', [None, [], {}, dict(model_config=[]), dict(model_config=None)])
def test_invalid_config(saved):
    with pytest.raises(ValueError, match='Invalid checkpoint'):
        model_config(saved)


@pytest.mark.parametrize('state', [[], 3, {'bad': None}, {1: torch.zeros(1)}])
def test_invalid_state_mapping(state):
    with pytest.raises(ValueError, match='state mapping'):
        resolve_weights(dict(model=state))


def test_trainer_onnx_export(tmp_path, trainer_checkpoint, monkeypatch):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    path, _ = trainer_checkpoint
    output = tmp_path/'ema.onnx'
    monkeypatch.setattr(sys, 'argv', ['export_onnx.py', '--checkpoint', str(path),
                                    '--output', str(output), '--precision', 'fp32',
                                    '--dynamic', '--weights', 'ema'])
    script('export_onnx').main()
    graph = onnx.load(output)
    assert {m.key: m.value for m in graph.metadata_props}['weights'] == 'ema'
    denoiser = Denoiser.load(output, device='cpu')
    x = np.random.default_rng(7).random((2, 4, 32, 48), dtype=np.float32)
    np.testing.assert_allclose(denoiser._call(x), x, atol=1e-6)


@pytest.mark.parametrize('weights', ['model', 'ema'])
def test_evaluator_selection_and_report(tmp_path, trainer_checkpoint, monkeypatch, weights):
    module = script('evaluate_dev_v2')
    path, saved = trainer_checkpoint
    entry = dict(source_sha256='source', sensor_sha256='sensor', group='group', clean='crop.npy')
    manifest = tmp_path/'val.json'
    manifest.write_text(json.dumps([entry]))
    x = torch.rand(4, 16, 16)
    monkeypatch.setattr(module, 'BootstrapDataset', lambda *a, **kw: [dict(
        noisy=x, clean=x*.9, noise_map=torch.ones(1, 16, 16)*.01)])
    original = module.build_model
    loaded = []
    def build(config):
        model = original(config)
        loaded.append(model)
        return model
    monkeypatch.setattr(module, 'build_model', build)
    args = SimpleNamespace(root=tmp_path, camera='x1d2', output=tmp_path/'result.json',
                           device='cpu', manifest=manifest, train_manifest=None,
                           crops_per_source=1, slices=['native|1|low'], previous_evaluation=None,
                           checkpoint=[f'tiny={path}'], references='', weights=weights)
    result = module.evaluate(args)
    expected = saved['ema_model' if weights == 'ema' else 'model']
    for key, value in loaded[0].state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    report = json.loads(args.output.read_text())
    assert result['weights'] == report['checkpoints']['tiny']['weights'] == weights


def test_real_coreml_package_export(tmp_path, trainer_checkpoint):
    pytest.importorskip('coremltools')
    path, _ = trainer_checkpoint
    output = tmp_path/'tiny.mlpackage'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/export_coreml.py'),
                             '--checkpoint', str(path), '--output', str(output), '--weights', 'ema'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert (output/'Manifest.json').is_file()
    assert any(p.stat().st_size for p in output.rglob('*.mlmodel'))
