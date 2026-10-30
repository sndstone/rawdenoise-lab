"""Bounded CPU tests for deterministic defect injection and correction."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from denoise_lab.arch_registry import build_model
from denoise_lab.bootstrap import add_bootstrap_noise
from denoise_lab.data import BootstrapDataset
from denoise_lab.defects import inject_defects, correct_defects, _neighbours
from denoise_lab.flat_augment import FlatAugmentedBootstrapDataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('kind', ['hot', 'warm', 'dead'])
def test_injection_kinds_density_determinism_and_copy(kind):
    a = np.full((1000, 1000), .1, np.float32)
    options = dict(density_range=(.003, .003), kind_probs={kind: 1.}, pair_probability=0,
                   hot_range=(.7, .8), warm_sigma_range=(10, 20), sigma=.01)
    r1, r2 = np.random.default_rng(8), np.random.default_rng(8)
    b, mask = inject_defects(a, r1, **options)
    b2, mask2 = inject_defects(a, r2, **options)
    assert 2700 < mask.sum() < 3300
    np.testing.assert_array_equal(b, b2)
    np.testing.assert_array_equal(mask, mask2)
    assert r1.bit_generator.state == r2.bit_generator.state
    assert np.all(a == np.float32(.1)) and np.all(b[~mask] == a[~mask])
    if kind == 'hot': assert np.all((b[mask] >= .7) & (b[mask] <= .8))
    elif kind == 'dead': assert np.max(np.abs(b[mask])) <= .002
    else: assert np.min(b[mask]) >= .2 and np.max(b[mask]) > .25


def test_pairs_and_robust_fallback():
    rng = np.random.default_rng(6)
    a = rng.normal(.1, .002, (512, 512)).astype(np.float32)
    paired, mask = inject_defects(a, rng, density_range=(.001, .001), pair_probability=1,
                                  kind_probs={'warm': 1.})
    near = np.zeros_like(mask)
    near[1:] |= mask[:-1]; near[:-1] |= mask[1:]
    near[:, 1:] |= mask[:, :-1]; near[:, :-1] |= mask[:, 1:]
    assert np.all(near[mask])
    assert 400 < mask.sum() < 650
    assert np.median(paired[mask]-a[mask]) > .04


@pytest.mark.parametrize('level,K,read', [(.05, 2e-5, .0002), (.18, .0002, .001), (.35, .001, .004)])
def test_noise_false_positives_and_recall(level, K, read):
    # A spatially varying signal exercises the intensity bins, not just one flat.
    clean = np.broadcast_to(np.linspace(level*.8, level*1.2, 384), (768, 384)).astype(np.float32)
    noisy, _ = add_bootstrap_noise(clean, np.random.default_rng(41),
        fixed_parameters=dict(K=K, read_std=read, row_std=0.))
    from denoise_lab.bayer import pack
    x = torch.from_numpy(pack(noisy))[None]
    _, false = correct_defects(x, chunk_rows=37)
    rate = float(false.float().mean())
    print(f'level={level} false_positive_rate={rate:.9g}')
    assert rate < 1e-5
    corrupt = x.clone()
    sites = torch.zeros_like(x, dtype=torch.bool)
    for c in range(4):
        for j, y in enumerate(range(6, x.shape[-2]-6, 13)):
            for i, u in enumerate(range(6, x.shape[-1]-6, 13)):
                corrupt[0, c, y, u] = 1. if (i+j+c)%2 else 0.
                sites[0, c, y, u] = True
    fixed, found = correct_defects(corrupt, chunk_rows=51)
    recall = float(found[sites].float().mean())
    print(f'level={level} recall={recall:.9g}')
    assert recall >= .99
    assert (fixed[sites]-x[sites]).abs().mean() < .04
    assert torch.equal(corrupt[~found], fixed[~found])


def test_guard_point_light_static_and_chunk_equality():
    x = torch.full((2, 4, 39, 47), .1)
    x += torch.randn(x.shape, generator=torch.Generator().manual_seed(9))*.001
    # One sensor 2x2 point covers all four packed planes.
    x[:, :, 18, 20] = .9
    x[0, 0, 7, 9] = .9
    x[1, 2, 22, 31] = -.5
    x[0, 1, 0, 0] = 1.
    x[1, 3, -1, -1] = -1.
    original = x.clone()
    rng = torch.get_rng_state()
    full, mask = correct_defects(x, chunk_rows=1000)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(x, original)
    assert not mask[:, :, 18, 20].any()
    assert mask[0, 0, 7, 9] and mask[1, 2, 22, 31]
    assert mask[0, 1, 0, 0] and mask[1, 3, -1, -1]
    for chunk in (1, 7, 19):
        p, m = correct_defects(x, chunk_rows=chunk)
        assert torch.equal(p, full) and torch.equal(m, mask)
    _, no_guard = correct_defects(x, cross_channel_guard=False)
    assert no_guard[:, :, 18, 20].all()
    static = torch.zeros_like(mask)
    static[:, :, 18, 20] = True
    fixed, m = correct_defects(x, static_mask=static)
    assert m[static].all() and (fixed[static] < .11).all()


def test_eight_neighbour_median_and_tiny_borders():
    x = torch.tensor([[1., 2., 3.], [4., 100., 6.], [7., 8., 9.]])
    lo, med, hi = _neighbours(x, torch.tensor([1]), torch.tensor([1]))
    assert (lo.item(), med.item(), hi.item()) == (1., 5., 9.)
    for h, w in [(1, 1), (1, 3), (3, 1)]:
        a = torch.full((1, 4, h, w), .2)
        p, mask = correct_defects(a)
        assert torch.equal(p, a) and not mask.any()


@pytest.mark.parametrize('augment', [True, False])
@pytest.mark.parametrize('deterministic', [True, False])
def test_disabled_dataset_outputs_and_rng(tmp_path, augment, deterministic, monkeypatch):
    np.save(tmp_path/'a.npy', np.random.default_rng(4).uniform(.01, .8, (48, 52)).astype(np.float32))
    options = dict(manifest=[dict(clean='a.npy')], root=tmp_path, crop_size=12,
        deterministic=deterministic, augment=augment, seed=43, identity_probability=.3,
        bright_exposure_probability=.3, exposure_sampling='log', impulse_range=(1e-4, 1e-3))
    parent = BootstrapDataset(**options)
    flat = FlatAugmentedBootstrapDataset(**options, defect_probability=0)
    # Capture the numpy stream immediately after each item's last normal draw.
    import denoise_lab.flat_augment as module
    original = np.random.default_rng
    generators = []
    def capture(*args, **kwargs):
        rng = original(*args, **kwargs); generators.append(rng); return rng
    monkeypatch.setattr(module.np.random, 'default_rng', capture)
    for seed in range(5):
        torch.manual_seed(seed)
        expected = parent[0]; torch_state = torch.get_rng_state(); np_state = generators[-1].bit_generator.state
        torch.manual_seed(seed)
        actual = flat[0]
        assert torch.equal(torch_state, torch.get_rng_state())
        assert np_state == generators[-1].bit_generator.state
        for key in expected:
            if isinstance(expected[key], torch.Tensor): assert torch.equal(expected[key], actual[key])
            else: assert expected[key] == actual[key]


def test_enabled_dataset_injects_after_noise_and_keeps_target(tmp_path):
    np.save(tmp_path/'a.npy', np.full((64, 64), .1, np.float32))
    opts = dict(manifest=[dict(clean='a.npy')], root=tmp_path, crop_size=16,
                deterministic=True, seed=6, augment=True, identity_probability=1)
    clean = FlatAugmentedBootstrapDataset(**opts)[0]
    ds = FlatAugmentedBootstrapDataset(**opts, defect_probability=1,
        defect_options=dict(density_range=(.1, .1), kind_probs={'hot': 1.}, hot_range=(.9, .9)))
    item = ds[0]
    assert torch.equal(item['clean'], clean['clean'])
    assert (item['noisy'] == .9).any()
    assert torch.count_nonzero(item['noise_map']) == 0
    opts['augment'] = False
    a = BootstrapDataset(**opts)[0]
    b = FlatAugmentedBootstrapDataset(**opts, defect_probability=1)[0]
    assert torch.equal(a['noisy'], b['noisy'])
    spec = importlib.util.spec_from_file_location('defects_train', ROOT/'scripts/train_arch_study.py')
    train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
    assert isinstance(train.dataset(dict(kind='bootstrap', **opts, defect_probability=1))[0], FlatAugmentedBootstrapDataset)


def test_evaluate_defects_tiny_cpu(tmp_path):
    np.save(tmp_path/'clean.npy', np.full((520, 520), .18, np.float32))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps([dict(clean='clean.npy', source_sha256='fixture')]))
    cfg = dict(arch='nafnet', cond='none', width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    model = build_model(cfg)
    with torch.no_grad():
        model.body.ending.weight.zero_(); model.body.ending.bias.zero_()
    checkpoint = tmp_path/'tiny.pt'
    torch.save(dict(model_config=cfg, model=model.state_dict()), checkpoint)
    out = tmp_path/'report.json'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_defects.py'),
        '--checkpoint', f'tiny={checkpoint}', '--checkpoint', f'copy={checkpoint}',
        '--manifest', str(manifest), '--root', str(tmp_path), '--baseline', 'tiny',
        '--device', 'cpu', '--slices', 'native|1|low', 'binned|1|high', '--limit', '1', '--out', str(out)],
        env={**os.environ, 'PYTHONPATH': str(ROOT/'src')}, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report['crops'] == 1 and len(report['rows']) == 16
    assert len(report['comparisons']) == 8
    assert all(c['mean_delta_db'] == 0 for c in report['comparisons'])
    for r in report['rows']:
        assert r['defect_sites'] > 0 and np.isfinite(r['model_psnr'])
        assert 0 <= r['defect_survival'] <= 1
    clean = next(r for r in report['rows'] if r['input'] == 'clean')
    assert clean['false_positive_per_million'] < 10
    injected = [r for r in report['rows'] if r['input'] == 'defect' and r['slice'] == 'native|1|low' and r['model'] == 'tiny']
    assert injected[1]['defect_mae'] < injected[0]['defect_mae']
    assert 'Clean FP / million' in out.with_suffix('.md').read_text()


def test_low_contrast_hot_pixel_remains_conservative():
    # A stuck-high site near a bright noisy background need not exceed 8 sigma.
    # This detector intentionally cannot promise recall for such weak defects.
    rng = np.random.default_rng(10)
    a = torch.from_numpy(rng.normal(.84, .03, (1, 4, 64, 64)).astype(np.float32))
    a[0, 0, 30, 30] = 1.
    _, flags = correct_defects(a)
    assert not flags[0, 0, 30, 30]


def test_real_metrics_with_normalized_raw_fixture(monkeypatch):
    from types import SimpleNamespace
    from contextlib import nullcontext
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('defects_eval_real', ROOT/'scripts/evaluate_defects.py')
    evaluation = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)
    codes = np.full((64, 64), 100, dtype=np.uint16)
    codes[10, 12] = 1000
    codes[22, 24] = 0
    raw = SimpleNamespace(raw_image_visible=codes,
        raw_colors_visible=np.tile([[0, 1], [3, 2]], (32, 32)),
        black_level_per_channel=[0]*4, camera_white_level_per_channel=None,
        white_level=1000, color_desc=b'RGBG')
    monkeypatch.setitem(sys.modules, 'rawpy', SimpleNamespace(imread=lambda path: nullcontext(raw)))
    report = evaluation.real_report(Path('fixture.3FR'), 'cpu')
    assert report['flagged'] == 2 and report['hot'] == report['dead'] == 1
    assert report['channels']['R']['flagged'] == 2
    assert report['per_MP'] == 2/(64*64/1e6)
    assert {(r['y'], r['x'], r['channel']) for r in report['top20']} == {(5, 6, 'R'), (11, 12, 'R')}
    assert all(abs(r['neighbour_median']-.1) < 1e-7 for r in report['top20'])
