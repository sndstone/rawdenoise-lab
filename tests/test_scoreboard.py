"""Fast synthetic scoreboard integration and failure/coverage contracts."""
import json

import numpy as np
import pytest
import torch
import yaml

from denoise_lab.arch_registry import build_model
from denoise_lab import scoreboard as sb
from denoise_lab.noref import gauss


def score(name, value):
    return dict(suite='dev', slice=name, metric='model_psnr', value=value, gate=True)


def test_gate_detects_hidden_slice_regression_and_missing_values():
    baseline = [score('a', 30.), score('b', 30.)]
    rows = sb.compare_scores([score('a', 29.89), score('b', 32.)], baseline)
    gate = sb.regression_gate(rows)
    assert gate['mean_delta_db'] > 0 and gate['status'] == 'FAIL'
    assert sb.regression_gate(sb.compare_scores([score('a', 31.)], baseline))['status'] == 'FAIL'
    assert sb.regression_gate(sb.compare_scores([score('a', float('nan'))], baseline))['status'] == 'FAIL'
    assert sb.regression_gate([])['status'] == 'FAIL'
    with pytest.raises(ValueError):
        sb.compare_scores(baseline+baseline, baseline)


def test_gate_strict_mean_equality_boundary_and_skip_policy():
    rows = sb.compare_scores([score('a', 29.9), score('b', 30.1)], [score('a', 30.), score('b', 30.)])
    assert sb.regression_gate(rows)['status'] == 'PASS'
    assert sb.regression_gate(rows, {'require_improvement': True})['status'] == 'FAIL'
    assert sb.regression_gate(rows, {'require_improvement': True}, identical=True)['status'] == 'PASS'
    assert sb.regression_gate(rows, skipped=['camera'])['status'] == 'FAIL'
    assert sb.regression_gate(rows, {'allow_skips': True}, skipped=['camera'])['status'] == 'PASS'
    assert sb.regression_gate(rows, {'allow_skips': True}, errors=['camera'])['status'] == 'FAIL'


@pytest.fixture
def tiny_checkpoint(tmp_path):
    config = dict(arch='nafnet', cond='none', width=4, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    model = build_model(config)
    torch.nn.init.zeros_(model.body.ending.weight)
    torch.nn.init.zeros_(model.body.ending.bias)
    path = tmp_path/'identity.pt'
    torch.save(dict(model_config=config, model=model.state_dict()), path)
    return path


def test_cli_self_comparison_and_cache_invalidation(tmp_path, tiny_checkpoint):
    mosaic = np.random.default_rng(4).uniform(.1, .4, (64, 64)).astype(np.float32)
    np.save(tmp_path/'clean.npy', mosaic)
    manifest = tmp_path/'val.json'
    manifest.write_text(json.dumps([dict(clean='clean.npy', source_sha256='synthetic', sensor_sha256='sensor', group='scene')]))
    cfg = dict(threads=2, gate=dict(require_improvement=True), suites=dict(dev=dict(root=str(tmp_path),
        crop_size=16, slices=['native|1|low', 'native|1|high'], quick_sources=1,
        datasets=[dict(name='fixture', manifest=str(manifest))])))
    config = tmp_path/'suite.yaml'
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path/'output'
    args = ['--checkpoint', f'id={tiny_checkpoint}', '--baseline', str(tiny_checkpoint),
            '--config', str(config), '--quick', '--device', 'cpu', '--output', str(output)]
    assert sb.main(args) == 0
    report = json.loads((output/'scoreboard.json').read_text())
    assert report['status'] == 'PASS'
    assert all(row['delta'] == 0 for row in report['candidates']['id']['comparisons'])
    assert 'PASS' in (output/'scoreboard.md').read_text()
    assert sb.main(args) == 0
    report = json.loads((output/'scoreboard.json').read_text())
    assert report['baseline']['suites']['dev']['cached'] is True
    # Changing the exact suite config must not reuse the old baseline.
    cfg['suites']['dev']['slices'] = ['native|1|low']
    config.write_text(yaml.safe_dump(cfg))
    assert sb.main(args) == 0
    report = json.loads((output/'scoreboard.json').read_text())
    assert report['baseline']['suites']['dev']['cached'] is False
    # Missing input invalidates the cache and is reported rather than a false PASS.
    (tmp_path/'clean.npy').unlink()
    assert sb.main(args) == 1
    report = json.loads((output/'scoreboard.json').read_text())
    assert report['baseline']['suites']['dev']['status'] == 'SKIP'


def test_all_private_suites_skip_cleanly(tmp_path, tiny_checkpoint):
    cfg = dict(suites=dict(dev=dict(datasets=[dict(name='missing', manifest=str(tmp_path/'missing.json'))]),
                           burst=dict(specs=[str(tmp_path/'burst.json')]), camera=dict(root=str(tmp_path/'camera'))))
    result = sb.run_checkpoint(tiny_checkpoint, cfg, list(cfg['suites']), tmp_path/'out', 'cpu', True, cache=True)
    assert all(s['status'] == 'SKIP' for s in result['suites'].values())
    assert result['scores'] == []


def test_camera_suite_with_synthetic_identity_and_blur(tmp_path, monkeypatch):
    from denoise_lab.inference import Denoiser
    camera = sb._script('benchmark_cameras')
    y, x = np.mgrid[:48, :48]
    inp = np.broadcast_to(.2+.03*np.sin(x/2)*np.cos(y/3), (4, 48, 48)).astype(np.float32).copy()
    inp += np.random.default_rng(3).normal(0, .01, inp.shape).astype(np.float32)
    manifest = dict(files=[dict(set='real', file='invalid.raw', camera='synthetic', band='test', sha256='invalid'),
                           dict(set='real', file='fixture.raw', camera='synthetic', band='test', sha256='fixture')])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(camera, 'load_packed', lambda path: (np.zeros_like(inp) if path.name == 'invalid.raw' else inp, np.ones(3)))
    monkeypatch.setattr(sb, '_script', lambda name: camera)
    class Model:
        def __init__(self, sigma):
            self.sigma = sigma
        def denoise_packed(self, a):
            return gauss(a, self.sigma) if self.sigma else a
    monkeypatch.setattr(Denoiser, 'load', lambda path, device: Model(float(path)))
    cfg = dict(root=str(tmp_path), roi=32, context=0, bootstrap_samples=50)
    identity, details = sb._camera('0', cfg, tmp_path, 'cpu', True)
    blur, _ = sb._camera('1', cfg, tmp_path, 'cpu', True)
    values = lambda rows: {r['metric']: r['value'] for r in rows}
    assert values(identity)['texture_retention'] == pytest.approx(1.)
    assert values(blur)['residual_noise'] < values(identity)['residual_noise']
    assert values(blur)['texture_retention'] < values(identity)['texture_retention']
    assert all(r['ci95'] is not None for r in identity)
    assert details[0]['status'] == 'SKIP'
    assert details[1]['source_sha256'] == 'fixture'


def test_speed_uses_real_synchronized_timing(tmp_path, tiny_checkpoint):
    scores, details = sb._speed(tiny_checkpoint, dict(size=16), tmp_path, 'cpu', True)
    assert scores[0]['value'] > 0
    assert details['shape'] == [1, 4, 16, 16]
    assert details['iterations'] == 3


def test_burst_suite_reuses_loo_reference_correction(tmp_path, monkeypatch):
    from denoise_lab.inference import Denoiser
    evaluator = sb._script('eval_burst_loo')
    y, x = np.mgrid[:64, :64]
    clean = np.broadcast_to(.2+.03*np.sin(x/5)*np.cos(y/8), (4, 64, 64))
    rng = np.random.default_rng(8)
    frames = {f'{i}.raw': (clean+rng.normal(0, .005, clean.shape)).astype(np.float32) for i in range(5)}
    monkeypatch.setattr(evaluator, 'load_packed', lambda p: (frames[p.name], np.ones(3)))
    monkeypatch.setattr(sb, '_script', lambda name: evaluator)
    class Identity:
        def denoise_packed(self, a):
            return a
    monkeypatch.setattr(Denoiser, 'load', lambda path, device: Identity())
    spec = tmp_path/'burst.json'
    spec.write_text(json.dumps(dict(root='.', groups=[dict(name='static', files=list(frames), test=[0, 2])])))
    scores, details = sb._burst('identity.pt', dict(specs=[str(spec)], block=8, quick_crop_size=32),
                               tmp_path/'out', 'cpu', True)
    assert all(r['value'] is not None and np.isfinite(r['value']) for r in scores)
    assert scores[0]['gate'] is True
    assert details[0]['rows'][0]['gain_db_all'] == pytest.approx(0., abs=1e-6)
    assert details[0]['rows'][0]['center_crop_packed'] == 32
    assert len(details[0]['rows']) == 1


def test_burst_pooled_sensitivity_keeps_mean_frame_gate_and_pairs_frames():
    evaluator = sb._script('eval_burst_loo')
    baseline = [dict(frame='a', mse_out_all=1., peak=1., static_samples=4),
                dict(frame='b', mse_out_all=.01, peak=1., static_samples=4)]
    candidate = [dict(frame='a', mse_out_all=1., peak=1., static_samples=4),
                 dict(frame='b', mse_out_all=.001, peak=1., static_samples=4)]
    mean_delta = 5.
    pooled_delta = evaluator.pooled_delta(candidate, baseline)
    assert pooled_delta == pytest.approx(10*np.log10(1.01/1.001))
    assert pooled_delta < .04 < mean_delta
    unequal = [dict(mse_out_all=1., peak=1., static_samples=1),
               dict(mse_out_all=.1, peak=1., static_samples=9)]
    assert evaluator.pooled_psnr(unequal) == pytest.approx(10*np.log10(10/1.9))
    assert evaluator.pooled_psnr([dict(mse_out_all=-1., peak=1.)]) is None
    assert evaluator.pooled_psnr([]) is None
    def scores(mean, pooled, frames):
        return [sb._score('burst', 'group', 'psnr_out', mean, gate=True, frame_scores=frames),
                sb._score('burst', 'group', 'pooled_corrected_mse_psnr', pooled)]
    rows = sb.compare_scores(scores(30., -100., {'b': 35., 'a': 25.}),
                             scores(25., 100., {'a': 25., 'b': 25.}))
    assert sb.regression_gate(rows)['measurements'] == 1
    assert sb.regression_gate(rows)['mean_delta_db'] == 5.
    frame_rows = next(r['per_frame_deltas'] for r in rows if r['metric'] == 'psnr_out')
    assert [r['delta_db'] for r in frame_rows] == [0., 10.]


def test_weights_threaded_to_dev_burst_camera_and_speed(tmp_path, tiny_checkpoint, monkeypatch):
    from types import SimpleNamespace
    from denoise_lab.inference import Denoiser
    # Use fake evaluator boundaries to inspect the selection without RAW IO.
    received = {}
    def dev(args):
        received['dev'] = args.weights
        return dict(sources=1, crops=1, summary=[])
    monkeypatch.setattr(sb, '_script', lambda name: SimpleNamespace(evaluate=dev))
    sb._dev(tiny_checkpoint, dict(datasets=[dict(name='fixture', manifest='fixture.json')]),
            tmp_path, 'cpu', True, weights='ema')
    assert received['dev'] == 'ema'
    def burst(args):
        received['burst'] = args.weights
        return []
    monkeypatch.setattr(sb, '_script', lambda name: SimpleNamespace(evaluate=burst))
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps(dict(root='.', groups=[dict(files=['a', 'b', 'c'], test=[0])])))
    sb._burst(tiny_checkpoint, dict(specs=[str(spec)]), tmp_path, 'cpu', True, weights='ema')
    assert received['burst'] == 'ema'
    def load(path, device, weights):
        received['camera'] = weights
    monkeypatch.setattr(Denoiser, 'load', load)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files=[])))
    sb._camera(tiny_checkpoint, dict(root=str(tmp_path)), tmp_path, 'cpu', True, weights='ema')
    assert received['camera'] == 'ema'
    saved = torch.load(tiny_checkpoint, weights_only=True)
    saved['ema_model'] = {k: v.clone() for k, v in saved['model'].items()}
    saved['ema_model']['body.ending.bias'].fill_(.125)
    torch.save(saved, tiny_checkpoint)
    def benchmark(model, *args, **kwargs):
        received['speed'] = float(model.body.ending.bias[0].detach())
        return dict(median_ms=1.)
    monkeypatch.setattr(sb, '_script', lambda name: SimpleNamespace(benchmark_tile=benchmark))
    sb._speed(tiny_checkpoint, {}, tmp_path, 'cpu', True, weights='ema')
    assert received['speed'] == .125


def test_default_round9_retains_historical_config_and_adds_coverage():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    default = yaml.safe_load((root/'configs/scoreboard/default.yaml').read_text())
    round9 = yaml.safe_load((root/'configs/scoreboard/round9.yaml').read_text())
    assert round9['gate'] == default['gate']
    assert round9['suites']['dev']['datasets'][:2] == default['suites']['dev']['datasets']
    assert round9['suites']['dev']['datasets'][2]['camera'] == 'a7m3'
    for name in ('burst', 'camera', 'speed'):
        assert round9['suites'][name] == default['suites'][name]
    assert round9['suites']['measured']['gate'] is False
