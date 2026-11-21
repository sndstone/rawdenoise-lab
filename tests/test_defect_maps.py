"""Fast synthetic calibration, geometry and CLI integration checks."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from denoise_lab.bayer import pack
from denoise_lab.defects import (HOT, WARM, STUCK, DEAD, HIGH_VARIANCE,
    dark_statistics, detect_dark_defects, make_defect_map, save_defect_map,
    load_defect_map, defect_map_mask, find_defect_map, correct_mapped_defects)
from denoise_lab.inference import Denoiser


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def darks():
    frames = np.random.default_rng(33).normal(0, .0001, (12, 64, 80)).astype(np.float32)
    frames[:, 12, 14] = .2
    frames[:, 18, 20] = .003
    frames[:, 24, 26] = 1.
    frames[:, 30, 32] = -.02
    frames[:, 36, 38] = np.tile([0, .1], 6)
    return frames


def metadata(phase='RGGB'):
    return dict(shape=[64, 80], phase=phase, model='Fixture', serial='123')


def condition(iso=100, exposure=1.):
    return dict(iso_min=iso, iso_max=iso, exposure_min=exposure, exposure_max=exposure, shutter='mechanical')


def calibration(phase='RGGB'):
    return make_defect_map([(condition(), iter(darks()))], metadata(phase))


def test_planted_kinds_streaming_and_confidence(tmp_path):
    frames = darks()
    original = frames.copy()
    stats = dark_statistics(iter(frames))
    kind, confidence = detect_dark_defects(stats)
    assert kind[12, 14] & HOT
    assert kind[18, 20] & WARM
    assert kind[24, 26] & STUCK
    assert kind[30, 32] & DEAD
    assert kind[36, 38] & HIGH_VARIANCE
    assert np.count_nonzero(kind) == 5
    assert .8 < confidence[12, 14] < 1
    np.testing.assert_array_equal(original, frames)
    path = tmp_path/'map.npz'
    m = calibration()
    save_defect_map(path, m)
    loaded = load_defect_map(path)
    for key in m:
        np.testing.assert_array_equal(m[key], loaded[key])
    assert path.stat().st_size < 10000
    assert find_defect_map(tmp_path, metadata()) == path
    assert find_defect_map(tmp_path, {**metadata(), 'serial': ''}) is None
    assert find_defect_map(tmp_path, {**metadata(), 'serial': 'other'}) is None
    save_defect_map(tmp_path/'duplicate.npz', m)
    with pytest.raises(ValueError, match='Multiple'):
        find_defect_map(tmp_path, metadata())


@pytest.mark.parametrize('phase', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
def test_source_geometry_and_settings(phase):
    m = calibration(phase)
    mask = defect_map_mask(m, metadata(phase), iso=100, exposure=1)
    full = np.zeros((64, 80), bool)
    full[tuple(m['coordinates'].T)] = True
    np.testing.assert_array_equal(mask, pack(full, phase))
    for changes, match in [({'serial': 'other'}, 'serial'), ({'shape': [80, 64]}, 'shape'),
                           ({'phase': 'invalid'}, 'phase')]:
        with pytest.raises(ValueError, match=match):
            defect_map_mask(m, {**metadata(phase), **changes}, iso=100, exposure=1)
    with pytest.raises(ValueError, match='outside observed'):
        defect_map_mask(m, metadata(phase), iso=200, exposure=1)
    with pytest.raises(ValueError, match='requires input ISO'):
        defect_map_mask(m, metadata(phase))
    m['coordinates'][0, 0] = -1
    with pytest.raises(ValueError, match='outside sensor'):
        defect_map_mask(m, metadata(phase), iso=100, exposure=1)


def test_disjoint_iso_exposure_observations_and_shutter():
    m = make_defect_map([(condition(), darks()), (condition(800, 2), darks())], metadata())
    with pytest.raises(ValueError, match='outside observed'):
        defect_map_mask(m, metadata(), iso=400, exposure=1.5)
    second = condition(800, 2)
    second['shutter'] = 'electronic'
    m = make_defect_map([(condition(), darks()), (second, darks())], metadata())
    with pytest.raises(ValueError, match='multiple shutter'):
        defect_map_mask(m, metadata(), iso=100, exposure=1)
    assert defect_map_mask(m, metadata(), iso=100, exposure=1, shutter='mechanical').sum() == 5
    with pytest.raises(ValueError, match='at least two'):
        dark_statistics([darks()[0]])


def test_sparse_correction_neighbours_clusters_stars_and_inference():
    a = torch.full((1, 4, 32, 40), .01)
    a[0, 0, 10, 11] = .7
    a[0, 0, 10, 12] = .9  # A mapped neighbour must not contaminate correction.
    a[0, 1, 18, 19] = .8  # An isolated star, absent from the map.
    a[0, 3, 0, 0] = 1.
    mask = torch.zeros_like(a, dtype=torch.bool)
    mask[0, 0, 10, 11:13] = True
    mask[0, 3, 0, 0] = True
    fixed, flags = correct_mapped_defects(a, mask)
    assert torch.equal(flags, mask)
    assert torch.equal(fixed[~mask], a[~mask])
    assert (fixed[mask] == .01).all()
    assert a[0, 0, 10, 11] == .7
    denoiser = Denoiser()
    denoiser.device = 'cpu'
    denoiser._call = lambda tile: tile
    image, mapped = a[0].numpy(), mask[0].numpy()
    auto = denoiser.denoise_packed(image, defect_mask=mapped, tile_size=32, overlap=8)
    heuristic = denoiser.denoise_packed(image, tile_size=32, overlap=8)
    explicit = denoiser.denoise_packed(image, prepass='heuristic', tile_size=32, overlap=8)
    np.testing.assert_array_equal(heuristic, explicit)  # Default unchanged.
    assert auto[1, 18, 19] == pytest.approx(.8)
    assert heuristic[1, 18, 19] < .02
    for mode in ('map+heuristic', 'off', 'map'):
        out = denoiser.denoise_packed(image, prepass=mode, defect_mask=mapped, tile_size=32, overlap=8)
        assert out[1, 18, 19] == pytest.approx(.01 if mode == 'map+heuristic' else .8)
    with pytest.raises(ValueError, match='requires a defect map'):
        denoiser.denoise_packed(image, prepass='map')
    with pytest.raises(ValueError, match='no valid'):
        correct_mapped_defects(a, torch.ones_like(mask))


def test_cli_argument_plumbing(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext
    from denoise_lab import cli
    source = tmp_path/'source.dng'
    raw = SimpleNamespace(color_desc=b'RGBG', raw_image_visible=np.zeros((64, 80), np.uint16),
                          raw_colors_visible=np.tile([[0, 1], [3, 2]], (32, 40)),
                          black_level_per_channel=[0]*4, camera_white_level_per_channel=None,
                          white_level=1000)
    m = calibration()
    path = tmp_path/'body.npz'
    save_defect_map(path, m)
    monkeypatch.setattr(cli.rawpy, 'imread', lambda _: nullcontext(raw))
    monkeypatch.setattr(cli.RawMetadata, 'from_raw', lambda *a: SimpleNamespace(
        phase='RGGB', make='Fixture', model='Synthetic', neutral=np.ones(3)))
    monkeypatch.setattr(cli, 'raw_defect_metadata', lambda *a: (metadata(), 100, 1.))
    seen = []
    def denoise(image, **kwargs):
        seen.append(kwargs)
        return image
    fake = SimpleNamespace(description='fixture', backend='torch', device='cpu',
                           providers=[], denoise_packed=denoise)
    monkeypatch.setattr(cli, 'model_hash', lambda _: '0'*64)
    monkeypatch.setattr(cli.Denoiser, 'load', lambda *a, **kw: fake)
    for index, args in enumerate([[], ['--defect-map', str(path)],
                                  ['--defect-map-dir', str(tmp_path), '--prepass', 'map+heuristic'],
                                  ['--prepass', 'off'], ['--no-prepass']]):
        monkeypatch.setattr(sys, 'argv', ['rawdenoise', str(source), '--model', 'fixture.pt',
                            '--format', 'npy', '--output-dir', str(tmp_path/str(index)), *args])
        cli.main()
    assert [s['prepass'] for s in seen] == ['auto', 'auto', 'map+heuristic', 'off', 'off']
    assert seen[0]['defect_mask'] is None
    assert seen[1]['defect_mask'].sum() == seen[2]['defect_mask'].sum() == 5


def test_builder_and_evaluator_with_synthetic_raws(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from denoise_lab.dng import RawMetadata, write_dng
    root = Path(__file__).resolve().parents[1]
    rows = []
    for i, image in enumerate(darks()[:6]):
        path = tmp_path/f'dark{i}.dng'
        meta = RawMetadata(black=np.full((2, 2), 512.), white=np.full((2, 2), 65535.))
        write_dng(path, image, meta)
        rows.append(dict(file=path.name, iso=100, exposure='1/30', shutter='mechanical'))
    (tmp_path/'manifest.json').write_text(json.dumps(rows))
    env = {**os.environ, 'PYTHONPATH': str(root/'src')}
    build = subprocess.run([sys.executable, str(root/'scripts/build_defect_map.py'), str(tmp_path),
                            '--output-dir', str(tmp_path/'maps'), '--name', 'fixture'],
                            env=env, capture_output=True, text=True, timeout=10)
    assert build.returncode == 0, build.stdout+build.stderr
    assert load_defect_map(tmp_path/'maps/fixture.npz')['kind'].size >= 4
    evaluate = subprocess.run([sys.executable, str(root/'scripts/evaluate_defect_maps.py'),
                               '--dark-dir', f'fixture={tmp_path}', '--output-dir', str(tmp_path/'evaluation'),
                               '--crop-size', '16'], env=env, capture_output=True, text=True, timeout=10)
    assert evaluate.returncode == 0, evaluate.stdout+evaluate.stderr
    result = json.loads((tmp_path/'evaluation/results.json').read_text())
    assert len(result['rows']) == 1
    assert result['stars']['mapped_stars_changed'] == 0
    assert result['stars']['heuristic_stars_changed'] == result['stars']['stars'] == 121
    assert (tmp_path/'evaluation/REPORT.md').exists()
    split = result['splits']['fixture'][0]
    assert not set(split['build']) & set(split['test'])


def test_warm_response_is_scoped_to_observed_settings():
    low = np.random.default_rng(71).normal(0, .00005, (8, 64, 80)).astype(np.float32)
    low[:, 20, 22] = .0001  # Below calibrated elevation floor at short/low settings.
    high = low.copy()
    high[:, 20, 22] = .004
    m = make_defect_map([(condition(100, .01), low), (condition(800, .1), high)], metadata())
    assert not defect_map_mask(m, metadata(), iso=100, exposure=.01).any()
    high_mask = defect_map_mask(m, metadata(), iso=800, exposure=.1)
    assert high_mask.sum() == 1 and high_mask[0, 10, 11]
    assert m['kind'][0] == WARM
