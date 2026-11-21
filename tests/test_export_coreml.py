"""Exporter construction contracts that do not require coremltools."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch
import yaml

if os.environ.get('REQUIRE_COREMLTOOLS') == '1':
    importlib.import_module('coremltools')

from denoise_lab.arch_registry import build_model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_coreml", ROOT / "scripts/export_coreml.py")
export_coreml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_coreml)


@pytest.mark.parametrize("weights", ["model", "ema"])
def test_released_config_construction(weights):
    config = yaml.safe_load((ROOT / "configs/round4/r4_w40s4_plain.yaml").read_text())["model"]
    assert config["arch"] == "nafnet" and config["width"] == 40
    assert len(config["enc_blk_nums"]) == len(config["dec_blk_nums"]) == 4
    original = build_model(config)
    state = original.state_dict()
    # Distinguish selected weights without running a full-size inference.
    key = next(iter(state))
    state[key] = torch.full_like(state[key], 0.125)
    model = export_coreml.model_from_checkpoint({"model_config": config, weights: state}, weights)
    assert not model.training
    torch.testing.assert_close(model.state_dict()[key], state[key])


def test_missing_ema_and_conditioned_checkpoint_rejected():
    config = dict(width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)
    with pytest.raises(ValueError, match="no ema weights"):
        export_coreml.model_from_checkpoint({"model_config": config}, "ema")
    with pytest.raises(ValueError, match="cond=none"):
        export_coreml.model_from_checkpoint({"model_config": {**config, "cond": "noise_map"}})


@pytest.mark.parametrize('mode', export_coreml.COMPRESSION_MODES)
def test_compression_arguments(mode):
    args = export_coreml.argument_parser().parse_args([
        '--checkpoint', 'weights.pt', '--output', 'out.mlpackage', '--compress', mode,
        '--compress-skip-op', '*intro*', '--compress-skip-op', 'ending*', '--compress-sensitive'])
    assert args.compress == mode
    assert args.compress_skip_op == ['*intro*', 'ending*']
    assert args.compress_sensitive


def test_compression_defaults_and_invalid_mode():
    parser = export_coreml.argument_parser()
    args = parser.parse_args(['--checkpoint', 'weights.pt', '--output', 'out.mlpackage'])
    assert args.compress == 'none' and not args.compress_sensitive and args.compress_skip_op == []
    with pytest.raises(SystemExit):
        parser.parse_args(['--checkpoint', 'weights.pt', '--output', 'out.mlpackage', '--compress', 'int4'])


@pytest.mark.parametrize('mode', export_coreml.COMPRESSION_MODES)
def test_compression_cli_plumbing(tmp_path, monkeypatch, mode):
    """A tiny checkpoint reaches the selected compressor with exact skip options."""
    ct = pytest.importorskip('coremltools')
    import sys
    config = dict(width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)
    model = build_model(config).eval()
    checkpoint = tmp_path/'tiny.pt'
    torch.save(dict(model_config=config, model=model.state_dict()), checkpoint)
    output = tmp_path/'tiny.mlpackage'
    calls = []
    class Converted:
        def save(self, path):
            calls.append(('save', path))
    converted = Converted()
    monkeypatch.setattr(torch.jit, 'trace', lambda m, x, **kw: m)
    monkeypatch.setattr(ct, 'convert', lambda *a, **kw: converted)
    monkeypatch.setattr(export_coreml, 'supports_grouped_palettization', lambda: True)
    def compress(m, selected, **kwargs):
        calls.append((selected, kwargs))
        assert m is converted
        return m
    monkeypatch.setattr(export_coreml, 'compress_model', compress)
    monkeypatch.setattr(sys, 'argv', ['export_coreml.py', '--checkpoint', str(checkpoint),
                                    '--output', str(output), '--compress', mode,
                                    '--compress-sensitive', '--compress-skip-op', '*sensitive*'])
    export_coreml.main()
    assert calls == [(mode, dict(skip_sensitive=False, skip_ops=['*sensitive*'])), ('save', str(output))]


@pytest.mark.parametrize('mode', export_coreml.COMPRESSION_MODES)
def test_tiny_coreml_compression(mode):
    """Optional CPU-only MIL conversion/optimization; no prediction or private data."""
    ct = pytest.importorskip('coremltools')
    if 'palettize' in mode:
        pytest.importorskip('sklearn')  # coremltools k-means palettization
    import numpy as np
    torch.manual_seed(11)
    # Keep interior weights above the default compression size threshold.
    model = build_model(dict(width=32, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)).eval()
    # Activate the initially zero residual paths so the optimizer sees their weights.
    for name, parameter in model.named_parameters():
        if name.endswith(('.beta', '.gamma')):
            torch.nn.init.constant_(parameter, .01)
    with torch.no_grad():
        traced = torch.jit.trace(model, torch.zeros(1, 4, 16, 16), check_trace=False)
    assert "aten::Int" not in str(traced.inlined_graph)
    converted = ct.convert(traced, convert_to='mlprogram', skip_model_load=True,
                           compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.macOS15 if hasattr(ct.target, 'macOS15') else ct.target.macOS13,
                           inputs=[ct.TensorType(name='inp_1', shape=(1, 4, 16, 16), dtype=np.float32)],
                           outputs=[ct.TensorType(name='output', dtype=np.float32)])
    result = export_coreml.compress_model(converted, mode)
    assert result.get_spec().WhichOneof('Type') == 'mlProgram'
    if mode == 'none':
        assert result is converted
    else:
        from coremltools.optimize.coreml import get_weights_metadata
        before = get_weights_metadata(converted, weight_threshold=0)
        after = get_weights_metadata(result, weight_threshold=0)
        protected = [name for name, info in before.items()
                     if info.val.shape in ((32, 4, 3, 3), (4, 32, 3, 3))]
        assert len(protected) == 2
        # Default protected weights remain ordinary constants with identical values.
        for name in protected:
            np.testing.assert_array_equal(after[name].val, before[name].val)
        functions = result.get_spec().mlProgram.functions.values()
        assert any(op.type.startswith('constexpr_') for function in functions
                   for block in function.block_specializations.values() for op in block.operations)


def test_sensitive_skip_uses_conv_order_not_constant_order(monkeypatch):
    """Hoisted constants must not cause the final decoder conv to replace the head."""
    import sys
    import types
    ns = types.SimpleNamespace
    def child(name, weight):
        return ns(name=name, op_type='conv', params_name_mapping={'weight': weight})
    metadata = {
        'w_intro': ns(child_ops=[child('intro', 'w_intro')]),
        'w_ending': ns(child_ops=[child('ending', 'w_ending')]),
        'w_middle': ns(child_ops=[child('middle', 'w_middle')]),
    }
    optional = types.ModuleType('coremltools.optimize.coreml')
    optional.get_weights_metadata = lambda *a, **kw: metadata
    optional.OptimizationConfig = lambda **kw: ns(**kw)
    monkeypatch.setitem(sys.modules, 'coremltools.optimize.coreml', optional)
    def op(name):
        value = ns(immediateValue=ns(tensor=ns(strings=ns(values=[name]))))
        return ns(type='conv', attributes={'name': value}, blocks=[])
    function = ns(block_specializations={'CoreML8': ns(operations=[op('intro'), op('middle'), op('ending')])})
    model = ns(get_spec=lambda: ns(mlProgram=ns(functions={'main': function})))
    config = export_coreml.compression_config(model, 'palette')
    assert config.op_name_configs == {'w_intro': None, 'w_ending': None, 'intro': None, 'ending': None}
    config = export_coreml.compression_config(model, 'palette', skip_sensitive=False, skip_ops=['mid*'])
    assert config.op_name_configs == {'w_middle': None, 'middle': None}


@pytest.fixture
def coreml_gate():
    spec = importlib.util.spec_from_file_location('coreml_gate_test', ROOT/'scripts/coreml_identity_gate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_arguments_and_production_defaults(coreml_gate):
    import inspect
    from denoise_lab.inference import Denoiser
    parser = coreml_gate.argument_parser()
    args = parser.parse_args(['--package', 'fixture=fixture.mlpackage', '--out', 'report.json',
                              '--compute-units', 'all', '--compute-units', 'cpu_and_ne'])
    assert args.compute_units == ['all', 'cpu_and_ne']
    assert not args.legacy_tiling
    defaults = inspect.signature(Denoiser.denoise_packed).parameters
    for key, value in coreml_gate.tiling_settings().items():
        assert value == defaults[key].default
    assert coreml_gate.tiling_settings(True) == dict(coreml_gate.tiling_settings(), overlap=58)
    with pytest.raises(SystemExit):
        parser.parse_args(['--package', 'fixture=model.mlpackage', '--out', 'report.json',
                           '--compute-units', 'invalid'])


@pytest.mark.parametrize('unit', ['all', 'cpu_and_ne', 'cpu_and_gpu', 'cpu_only'])
@pytest.mark.parametrize('legacy', [False, True])
def test_package_identity_synthetic(coreml_gate, monkeypatch, unit, legacy):
    import numpy as np
    from denoise_lab.inference import Denoiser
    loads, calls = [], []
    def load(path, **kwargs):
        loads.append((path, kwargs))
        model = Denoiser()
        model.backend = 'coreml'
        model.compute_units = kwargs['compute_units']
        model._call = lambda tile: tile
        return model
    original = Denoiser.denoise_packed
    def denoise(self, packed, **kwargs):
        calls.append(kwargs)
        return original(self, packed, **kwargs)
    monkeypatch.setattr(coreml_gate.Denoiser, 'load', load)
    monkeypatch.setattr(Denoiser, 'denoise_packed', denoise)
    monkeypatch.setattr(coreml_gate.ident, 'KINDS', {'ramp': (.55, 1., .7, 1.)})
    monkeypatch.setattr(coreml_gate.ident, 'LEVELS', [.01, .5])
    result = coreml_gate.package_identity('fixture.mlpackage', (17, 19), unit, legacy)
    assert loads == [('fixture.mlpackage', dict(backend='coreml', compute_units=unit))]
    assert result['passed'] and result['max_abs_error'] < 1e-7
    assert result['compute_units'] == unit
    assert result['tiling'] == coreml_gate.tiling_settings(legacy)
    assert len(result['samples_ms']) == 20 and np.isfinite(result['median_ms'])
    assert len(result['rows']) == 2
    if not legacy:
        assert calls == [dict(prepass=False, highlights='off', **coreml_gate.tiling_settings())]*2
    else:
        assert calls == []


def test_gate_multi_unit_report(coreml_gate, tmp_path, monkeypatch):
    import json
    import sys
    seen = []
    def identity(path, frame_size, compute_units, legacy):
        seen.append((path, compute_units, legacy))
        return dict(passed=True, compute_units=compute_units)
    monkeypatch.setattr(coreml_gate, 'package_identity', identity)
    out = tmp_path/'gate.json'
    monkeypatch.setattr(sys, 'argv', ['gate', '--package', 'fixture=fixture.mlpackage', '--out', str(out),
                                    '--compute-units', 'all', '--compute-units', 'cpu_and_ne'])
    coreml_gate.main()
    report = json.loads(out.read_text())
    assert report['compute_units'] == ['all', 'cpu_and_ne']
    assert set(report['models']['fixture']) == {'all', 'cpu_and_ne'}
    assert seen == [('fixture.mlpackage', 'all', False), ('fixture.mlpackage', 'cpu_and_ne', False)]


def test_compression_runtime_consistency(coreml_gate, monkeypatch, tmp_path):
    import numpy as np
    import sys
    from denoise_lab.inference import Denoiser
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import evaluate_coreml_compression as compression
    loads, tiling = [], []
    def load(path, **kwargs):
        loads.append(kwargs['compute_units'])
        model = Denoiser()
        model.compute_units = kwargs['compute_units']
        model._call = lambda tile: tile
        def predict(packed, **settings):
            tiling.append(settings)
            return packed
        model.denoise_packed = predict
        return model
    monkeypatch.setattr(Denoiser, 'load', load)
    path = tmp_path/'example.dng'
    path.write_bytes(b'fixture')
    packed = np.full((4, 17, 19), .1, np.float32)
    for unit in ('all', 'cpu_and_ne', 'cpu_and_gpu', 'cpu_only'):
        timing = compression.latency('fixture.mlpackage', 2, 1, unit)
        quality, examples, dev = compression.quality(
            'fixture.mlpackage', [(path, packed)], [], None, None, unit)
        assert unit in timing and 'error' not in timing[unit]
        assert quality['compute_units'] == unit
        assert quality['tiling'] == coreml_gate.tiling_settings()
        np.testing.assert_array_equal(examples[0], packed)
        assert dev == []
    assert loads == [unit for unit in ('all', 'cpu_and_ne', 'cpu_and_gpu', 'cpu_only') for _ in range(2)]
    assert tiling == [coreml_gate.tiling_settings()]*4
    # Avoid leaking the script-directory module into other tests.
    monkeypatch.delitem(sys.modules, 'evaluate_coreml_compression')
