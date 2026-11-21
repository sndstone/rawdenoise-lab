"""Synthetic CPU coverage for registered pair masks and cross-ISO preparation."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from denoise_lab.bayer import augment_mosaic, pack, unpack
from denoise_lab.data import PairedBurstDataset, _valid_crop
from denoise_lab.metrics import evaluate_pair


@pytest.fixture(scope='module')
def scripts():
    folder = Path(__file__).resolve().parents[1]/'scripts'
    old_path = sys.path[:]
    sys.path.insert(0, str(folder))
    result = []
    try:
        for name in ('evaluate', 'prepare_cross_iso_pairs'):
            spec = importlib.util.spec_from_file_location('paired_test_'+name, folder/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result.append(module)
        yield result
    finally:
        sys.path[:] = old_path


def pair_files(tmp_path, size=32):
    clean = np.arange(size*size, dtype=np.float32).reshape(size, size)/(size*size)
    np.save(tmp_path/'noisy.npy', clean+.01)
    np.save(tmp_path/'clean.npy', clean)
    return {'noisy': 'noisy.npy', 'clean': 'clean.npy', 'iso': 1600}


def test_finite_training_crops_and_seed_reset(tmp_path):
    entry = pair_files(tmp_path, 64)
    mask = np.zeros((64, 64), bool)
    mask[16:56, 12:52] = True
    np.save(tmp_path/'valid.npy', mask)
    entry['valid_mask'] = 'valid.npy'
    clean = np.load(tmp_path/'clean.npy')
    clean[:4] = np.nan
    np.save(tmp_path/'clean.npy', clean)
    dataset = PairedBurstDataset([entry], tmp_path, crop_size=8, seed=9)
    torch.manual_seed(12)
    first = [dataset[0] for _ in range(4)]
    torch.manual_seed(12)
    second = [dataset[0] for _ in range(4)]
    for a, b in zip(first, second):
        for key in ('noisy', 'clean', 'valid_mask'):
            assert torch.equal(a[key], b[key])
        assert a['valid_mask'].all()
        assert torch.isfinite(a['noisy']).all() and torch.isfinite(a['clean']).all()
        torch.testing.assert_close(a['noisy']-a['clean'], torch.full_like(a['clean'], .01))
    assert any(not torch.equal(first[0]['clean'], a['clean']) for a in first[1:])


def _worker_clean_array(sample):
    return sample['clean'].numpy()


def test_seeded_worker_streams_repeat(tmp_path):
    entry = pair_files(tmp_path, 64)
    dataset = PairedBurstDataset([entry]*4, tmp_path, crop_size=8, seed=9)
    def draw():
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2,
                                             collate_fn=_worker_clean_array,
                                             generator=torch.Generator().manual_seed(42))
        return list(loader)
    first, second = draw(), draw()
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not np.array_equal(first[0], first[1])


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=3)))
def test_mask_flip_transpose_geometry(tmp_path, monkeypatch, flags):
    entry = pair_files(tmp_path)
    mask = np.ones((32, 32), bool)
    mask[4:10, 8:16] = False
    np.save(tmp_path/'valid.npy', mask)
    entry['valid_mask'] = 'valid.npy'
    class FixedRng:
        def integers(self, *args):
            return np.array(flags)
    monkeypatch.setattr('denoise_lab.data.np.random.default_rng', lambda *a: FixedRng())
    sample = PairedBurstDataset([entry], tmp_path, crop_size=None)[0]
    expected = pack(augment_mosaic(mask, *flags))
    np.testing.assert_array_equal(sample['valid_mask'], expected)
    np.testing.assert_array_equal(sample['clean'], pack(augment_mosaic(np.load(tmp_path/'clean.npy'), *flags)))


def test_full_frame_nan_fill_and_support(tmp_path):
    entry = pair_files(tmp_path)
    noisy = np.load(tmp_path/'noisy.npy')
    noisy[0, 0] = np.nan
    noisy[2, 2] = np.inf
    np.save(tmp_path/'noisy.npy', noisy)
    counts = np.full((4, 16, 16), 4.)
    counts[1, 2, 3] = 1
    confidence = np.ones((16, 16))
    confidence[5, 6] = 0
    np.savez(tmp_path/'support.npz', effective_count=counts, confidence=confidence,
             static=np.ones((16, 16), bool), variance=np.zeros_like(counts))
    entry['support'] = 'support.npz'
    sample = PairedBurstDataset([entry], tmp_path, crop_size=None, augment=False)[0]
    assert torch.isfinite(sample['noisy']).all() and torch.isfinite(sample['clean']).all()
    for y, x in ((0, 0), (1, 1), (2, 3), (5, 6)):
        assert not sample['valid_mask'][:, y, x].any()
    assert sample['valid_mask'][:, 10, 10].all()


@pytest.mark.parametrize('shape', [(32, 32), (16, 16), (4, 16, 16)])
def test_npy_support_counts(tmp_path, shape):
    entry = pair_files(tmp_path)
    support = np.full(shape, 3.)
    support[..., :4, :4] = 1
    np.save(tmp_path/'support.npy', support)
    entry['support'] = 'support.npy'
    sample = PairedBurstDataset([entry], tmp_path, crop_size=None, augment=False)[0]
    assert not sample['valid_mask'][:, 0, 0].any()
    assert sample['valid_mask'][:, -1, -1].all()


@pytest.mark.parametrize('field', ['valid_mask', 'support'])
def test_missing_named_files_fail(tmp_path, field):
    entry = pair_files(tmp_path)
    entry[field] = 'missing.npy'
    with pytest.raises(FileNotFoundError):
        PairedBurstDataset([entry], tmp_path, crop_size=4)[0]


def test_no_valid_window_fails_clearly(tmp_path):
    entry = pair_files(tmp_path)
    mask = np.indices((32, 32)).sum(0) % 4 != 0
    np.save(tmp_path/'valid.npy', mask)
    entry['valid_mask'] = 'valid.npy'
    with pytest.raises(ValueError, match='no valid finite supported samples'):
        PairedBurstDataset([entry], tmp_path, crop_size=4)[0]
    mask[:] = False
    mask[4:8, 4:8] = True
    np.save(tmp_path/'valid.npy', mask)
    with pytest.raises(ValueError, match='No fully valid finite 4x4 packed training window'):
        PairedBurstDataset([entry], tmp_path, crop_size=4, augment=False)[0]


def test_rare_valid_window_exhaustive_fallback():
    pair = np.zeros((3, 40, 40))
    pair[2, 30:38, 28:36] = 1
    class MissRng:
        def integers(self, n):
            return 0
    out = _valid_crop(pair, 4, MissRng())
    assert out.shape == (3, 8, 8) and out[2].all()


@pytest.mark.parametrize('field', ['valid_mask', 'support'])
def test_bad_mask_geometry_fails(tmp_path, field):
    entry = pair_files(tmp_path)
    np.save(tmp_path/'wrong.npy', np.ones((7, 9)))
    entry[field] = 'wrong.npy'
    with pytest.raises(ValueError, match='geometry'):
        PairedBurstDataset([entry], tmp_path, crop_size=None)[0]


def test_masked_metrics_ignore_excluded_errors(scripts):
    evaluate, _ = scripts
    clean = torch.full((1, 4, 32, 32), .25)
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[..., :8, :8] = False
    bad = clean.clone()
    bad[~mask] = float('nan')
    metrics = evaluate.evaluate_masked_pair(bad, clean, mask)
    assert metrics['linear_psnr'] == float('inf')
    assert metrics['srgb_psnr'] == float('inf')
    assert metrics['linear_ssim'] == pytest.approx(1.)
    assert metrics['srgb_ssim'] == pytest.approx(1.)
    assert metrics['valid_fraction'] == 15/16
    bad[mask] += .1
    assert evaluate.evaluate_masked_pair(bad, clean, mask)['linear_psnr'] == pytest.approx(20., abs=1e-5)


def test_masked_metrics_preserve_legacy_all_valid(scripts):
    evaluate, _ = scripts
    torch.manual_seed(8)
    x = torch.rand((1, 4, 16, 16))
    y = x+.01*torch.randn_like(x)
    legacy = evaluate_pair(x, y)
    masked = evaluate.evaluate_masked_pair(x, y, torch.ones_like(x, dtype=torch.bool))
    for key, value in legacy.items():
        assert masked[key] == pytest.approx(value, abs=1e-10)


def test_ssim_requires_complete_valid_windows(scripts):
    evaluate, _ = scripts
    x = torch.zeros((1, 4, 16, 16))
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., ::2, ::2] = True
    row = evaluate.evaluate_masked_pair(x, x, mask)
    assert row['linear_ssim'] is None and row['linear_ssim_valid_fraction'] == 0
    assert row['srgb_ssim'] is None
    assert row['linear_psnr'] == float('inf')
    x[mask] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        evaluate.evaluate_masked_pair(x, x, mask)


def capture_files(tmp_path):
    for iso, shutter in ((100, '1/2'), (3200, '1/60')):
        for i in range(4):
            path = tmp_path/f'capture_{iso}_{i}.dng'
            path.write_bytes(b'fixture')
            Path(str(path)+'.json').write_text(json.dumps({
                'camera': 'Synthetic body', 'iso': iso, 'shutter': shutter, 'aperture': 4,
                'timestamp': f'2026:11:08 12:{int(iso != 100):02d}:{i:02d}'}))


def test_cross_iso_metadata_and_dry_run(scripts, tmp_path, monkeypatch, capsys):
    _, prepare = scripts
    capture_files(tmp_path)
    groups = prepare.discover_groups(tmp_path)
    report = prepare.group_report(groups)
    assert [(g['iso'], g['count'], g['shutter']) for g in report] == [(100, 4, .5), (3200, 4, 1/60)]
    low, high = prepare.select_groups(groups)
    assert low[0]['iso'] == 100 and high[0]['iso'] == 3200
    monkeypatch.setattr(prepare, 'read_mosaic', lambda *a: pytest.fail('dry-run decoded pixels'))
    monkeypatch.setattr(sys, 'argv', ['prepare', str(tmp_path), '--dry-run'])
    before = set(tmp_path.iterdir())
    prepare.main()
    assert set(tmp_path.iterdir()) == before
    assert len(json.loads(capsys.readouterr().out)['groups']) == 2
    with pytest.raises(ValueError, match='exactly two'):
        prepare.select_groups(groups+groups)
    with pytest.raises(ValueError, match='strictly lower'):
        prepare.select_groups(groups, 1, 0)


def synthetic_bursts():
    rng = np.random.default_rng(4)
    size = 128
    f = np.fft.fft2(rng.normal(size=(size, size)))
    freq = np.fft.fftfreq(size)
    f *= np.exp(-(freq[:, None]**2+freq**2)/.08**2)
    texture = np.fft.ifft2(f).real
    base = .25+.04*texture/texture.std()
    clean = np.stack([base*.8, base, base*.6, base*1.01]).astype(np.float32)
    low = np.stack([clean+rng.normal(0, .0005, clean.shape) for _ in range(4)]).astype(np.float32)
    high = np.stack([clean*1.2+.01+rng.normal(0, .002, clean.shape) for _ in range(4)]).astype(np.float32)
    return low, high


def test_cross_iso_preparation_roundtrip(scripts, tmp_path):
    _, prepare = scripts
    low, high = synthetic_bursts()
    aligned, target, valid, support, qa = prepare.prepare_arrays(low, high, block=8)
    assert qa['static_fraction'] > .5
    np.testing.assert_allclose(qa['gain_per_channel'], 1.2, atol=.02)
    assert qa['half_burst_frame_counts'] == [2, 2]
    assert 0 < qa['half_burst_target_difference']['rmse'] < .002
    assert qa['clipping_fraction']['clean'] == 0
    records = [{'iso': 3200, 'path': f'capture_{i}.dng'} for i in range(4)]
    prepare.write_pairs(tmp_path, aligned, target, valid, support, records, qa)
    dataset = PairedBurstDataset(tmp_path/'pairs.json', crop_size=None, augment=False)
    sample = dataset[0]
    assert len(dataset) == 4 and torch.isfinite(sample['noisy']).all()
    np.testing.assert_array_equal(sample['valid_mask'], np.broadcast_to(valid, (4, *valid.shape)))
    assert json.loads((tmp_path/'qa.json').read_text())['purpose'] == 'development_evaluation'
    # No high-ISO samples enter the independent clean merge.
    assert qa['target_independence'] == 'target uses only the lower-ISO burst'
    train = PairedBurstDataset(tmp_path/'pairs.json', crop_size=16, augment=False)[0]
    assert train['valid_mask'].all()


def test_cross_iso_rejects_nonfinite_sources(scripts):
    _, prepare = scripts
    low, high = synthetic_bursts()
    low[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite before alignment'):
        prepare.prepare_arrays(low, high, block=8)


def test_cross_iso_cli_output_and_evaluator(scripts, tmp_path, monkeypatch, capsys):
    evaluate, prepare = scripts
    capture = tmp_path/'capture'
    capture.mkdir()
    capture_files(capture)
    low, high = synthetic_bursts()
    def read_fixture(path):
        _, iso, index = Path(path).stem.split('_')
        return unpack((low if iso == '100' else high)[int(index)])
    monkeypatch.setattr(prepare, 'read_mosaic', read_fixture)
    output = tmp_path/'prepared'
    monkeypatch.setattr(sys, 'argv', ['prepare', str(capture), '--output', str(output), '--block', '8'])
    prepare.main()
    assert len(json.loads((output/'pairs.json').read_text())) == 4
    capsys.readouterr()
    model = torch.nn.Identity()
    model.cond = 'none'
    monkeypatch.setattr(evaluate, 'build_model', lambda config: model)
    monkeypatch.setattr(evaluate.torch, 'load', lambda *a, **k: {'model_config': {}, 'model': {}})
    monkeypatch.setattr(evaluate, 'tiled_inference', lambda model, noisy, **kw: model(noisy))
    metrics_path = tmp_path/'metrics.json'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoint', 'fixture.pt',
                                    '--manifest', str(output/'pairs.json'), '--output', str(metrics_path)])
    evaluate.main()
    metrics = json.loads(metrics_path.read_text())
    assert len(metrics['pairs']) == 4
    assert metrics['mean_metrics']['input'] == metrics['mean_metrics']['ours']
    assert .5 < metrics['mean_metrics']['ours']['valid_fraction'] <= 1
    assert 'NaN' not in metrics_path.read_text() and 'Infinity' not in metrics_path.read_text()


@pytest.mark.parametrize('crop_size', [0, -1, True, 1.5])
def test_invalid_crop_setting(crop_size):
    with pytest.raises(ValueError, match='crop_size'):
        PairedBurstDataset([], crop_size=crop_size)


@pytest.mark.parametrize('min_support', [0, 1, float('nan')])
def test_invalid_support_setting(min_support):
    with pytest.raises(ValueError, match='min_support'):
        PairedBurstDataset([], min_support=min_support)
