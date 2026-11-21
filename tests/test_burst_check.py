"""Fast synthetic capture screens; no private photographs or model required."""
import json

import numpy as np
import pytest

from denoise_lab.burst_check import (analyze_group, compare_groups, group_frames,
                                     phase_shift, read_metadata, write_report)


def burst(scene=None, iso=100, timestamp=0, noise=.003, count=5, seed=0):
    rng = np.random.default_rng(seed)
    if scene is None:
        scene = np.random.default_rng(12).uniform(.1, .6, (4, 256, 256)).astype('float32')
    frames = scene+rng.normal(0, noise, (count, *scene.shape))
    records = [dict(path=f'{iso}_{i}.RAW', camera='synthetic', iso=iso,
                    shutter=1/iso, aperture=4., timestamp=timestamp+i) for i in range(count)]
    return analyze_group(frames, records), scene


def test_static_noise_and_report(tmp_path):
    clean, scene = burst()
    noisy, _ = burst(scene, iso=1600, noise=.015, seed=1, timestamp=10)
    result, mask = compare_groups(clean, noisy)
    assert result['verdict'] == 'USABLE'
    assert result['static_fraction'] > .95
    assert clean['summary']['temporal_variance'] == pytest.approx(.003**2, rel=.05)
    assert noisy['summary']['noise_excess_ratio'] < 1.2
    write_report(tmp_path, [clean, noisy], [(result, mask)])
    assert (tmp_path/'index.html').exists()
    assert (tmp_path/'pair_0_static.png').exists()
    assert json.loads((tmp_path/'results.json').read_text())['pairings'][0]['verdict'] == 'USABLE'


@pytest.mark.parametrize('width,verdict', [(32, 'MARGINAL'), (192, 'RESHOOT')])
def test_moving_region(width, verdict):
    clean, scene = burst()
    changed = scene.copy()
    changed[:, :, :width] += .15
    noisy, _ = burst(changed, iso=1600, noise=.01, seed=1)
    result, _ = compare_groups(clean, noisy, min_static=.8, marginal_static=.5)
    assert result['verdict'] == verdict
    assert any('scene moved' in r for r in result['reasons'])


def test_between_group_exposure_discrepancy_is_info():
    clean, scene = burst()
    noisy, _ = burst(scene*2**.4, iso=1600, noise=.01, seed=1)
    result, _ = compare_groups(clean, noisy)
    assert result['verdict'] == 'USABLE'
    assert result['exposure_mismatch_ev'] == pytest.approx(.4, abs=.03)
    assert result['info'][0].startswith('INFO:')
    assert not any('exposure mismatch' in r for r in result['reasons'])


def test_global_shift():
    clean, scene = burst()
    noisy, _ = burst(np.roll(scene, (3, -4), (-2, -1)), iso=1600, noise=.01, seed=1)
    shift, confident = phase_shift(clean['mean'], noisy['mean'])
    assert confident
    np.testing.assert_allclose(shift, [3, -4], atol=.1)
    result, _ = compare_groups(clean, noisy)
    assert result['verdict'] == 'RESHOOT'
    assert any('camera moved' in r for r in result['reasons'])


def test_within_burst_motion_and_light_step():
    rng = np.random.default_rng(9)
    _, scene = burst()
    frames = scene+rng.normal(0, .003, (5, *scene.shape))
    # Zero-mean motion keeps these blocks in populated brightness bins.
    frames[:2, :, :40, :40] += .15
    frames[2:4, :, :40, :40] -= .15
    records = burst()[0]['summary']['frames']
    group = analyze_group(frames, records)
    assert group['static'][:2, :2].mean() < .1
    frames[2:] *= 1.2
    group = analyze_group(frames, records)
    assert group['summary']['exposure_range_ev'] > .2
    assert group['summary']['light_or_gain_drift_percent_per_minute'] > 0
    result, _ = compare_groups(burst()[0], group)
    assert result['verdict'] == 'RESHOOT'


def test_grouping_and_missing_evidence():
    records = burst()[0]['summary']['frames']
    records += [dict(records[-1], path='later', timestamp=100),
                dict(records[-1], path='iso', timestamp=101, iso=800),
                dict(records[-1], path='body', camera='other')]
    assert sorted(map(len, group_frames(records))) == [1, 1, 1, 5]
    clean, scene = burst(count=1)
    noisy, _ = burst(scene, count=1, iso=800, seed=1)
    result, _ = compare_groups(clean, noisy)
    assert result['verdict'] != 'USABLE'
    assert any('insufficient' in r for r in result['reasons'])


def test_sidecar_metadata(tmp_path):
    path = tmp_path/'missing.RAW'
    with pytest.raises(ValueError, match='incomplete capture metadata'):
        read_metadata(path)
    path.with_suffix('.RAW.json').write_text(json.dumps(dict(camera='A5100', iso=100,
        shutter='1/20', aperture=4, timestamp='2026:11:01 12:00:00')))
    assert read_metadata(path)['shutter'] == .05


def test_clipping_and_uncertain_registration():
    clean, scene = burst()
    scene[:, :16] = 1.
    clean, _ = burst(scene)
    noisy, _ = burst(scene, iso=800, seed=1)
    result, _ = compare_groups(clean, noisy)
    assert result['verdict'] != 'USABLE'
    assert any('clipped' in r for r in result['reasons'])
    flat, _ = burst(np.full((4, 256, 256), .3), noise=0.)
    result, _ = compare_groups(flat, flat)
    assert result['verdict'] == 'MARGINAL'
    assert any('registration uncertain' in r for r in result['reasons'])


def test_tiff_camera_with_sidecar(tmp_path):
    import tifffile
    path = tmp_path/'fixture.dng'
    # Preserve a TIFF camera tag when sidecars fill absent exposure tags.
    tifffile.imwrite(path, np.zeros((16, 16), 'uint16'), extratags=[(272, 's', 0, 'Fixture body', False)])
    path.with_suffix('.dng.json').write_text(json.dumps(dict(iso=200, shutter='1/50',
        aperture='28/10', timestamp='2026:11:01 12:00:00')))
    meta = read_metadata(path)
    assert meta['camera'] == 'Fixture body'
    assert meta['aperture'] == 2.8


def test_nonamplified_high_iso_and_affine_offset():
    clean, scene = burst(noise=.001)
    gain = np.array([.24, .25, .26, .25], dtype='float32')[:, None, None]
    offset = np.array([.002, -.001, .003, .001], dtype='float32')[:, None, None]
    noisy, _ = burst(scene*gain+offset, iso=25600, noise=.004, seed=1)
    for group in (clean, noisy):
        for record in group['summary']['frames']:
            record['camera'] = 'X1D II 50C'
    result, _ = compare_groups(clean, noisy)
    assert result['verdict'] == 'USABLE'
    assert result['static_fraction'] > .95
    assert result['highest_analog_iso'] == 6400
    assert result['metadata_exposure_ratio'] == pytest.approx(1)
    assert result['analog_capped_exposure_ratio'] == pytest.approx(.25)
    assert result['nominal_exposure_mismatch_ev'] == pytest.approx(-2, abs=.05)
    assert result['exposure_mismatch_ev'] == pytest.approx(0, abs=.05)
    np.testing.assert_allclose(np.array(result['gain_offset_local_coefficients'])[:, 1],
                               offset[:, 0, 0], atol=.003)
    assert result['info'] and not any('exposure' in r for r in result['reasons'])


def test_configurable_static_thresholds_and_block_count():
    clean, scene = burst()
    changed = scene.copy()
    changed[:, :, :32] += .15
    noisy, _ = burst(changed, iso=800, noise=.01, seed=1)
    result, _ = compare_groups(clean, noisy)
    assert result['verdict'] == 'USABLE'
    assert .1 < result['static_fraction'] < .8
    result, _ = compare_groups(clean, noisy, min_static=.8, marginal_static=.5)
    assert result['verdict'] == 'MARGINAL'
    result, _ = compare_groups(clean, noisy, min_static=.8, marginal_static=.79)
    assert result['verdict'] == 'RESHOOT'
    result, _ = compare_groups(clean, noisy, min_blocks=10000)
    assert result['verdict'] == 'MARGINAL'
    assert any('insufficient static blocks' in r for r in result['reasons'])
    with pytest.raises(ValueError, match='marginal-static'):
        compare_groups(clean, noisy, min_static=.1, marginal_static=.2)


def test_analog_iso_configuration(tmp_path):
    from denoise_lab.burst_check import analog_iso_table, exposure_factor
    path = tmp_path/'darks.yaml'
    path.write_text('cameras:\n  x1d2:\n    highest_analog_iso: 3200\n')
    table = analog_iso_table(path, ['ILCE-7M3=12800'])
    record = dict(camera='X1D II 50C', iso=25600, shutter=1/25600, aperture=4)
    assert exposure_factor(record, table) == pytest.approx(.125/16)
    assert table['a7m3'] == 12800
    assert analog_iso_table(path, ['X1D II=0'])['x1d2'] == 0
    assert analog_iso_table(tmp_path/'absent.yaml')['x1d2'] == 6400
    with pytest.raises(ValueError, match='nonnegative'):
        analog_iso_table(overrides=['X1D II=-1'])


def test_explicit_evaluator_alignment():
    clean, scene = burst()
    noisy, _ = burst(np.roll(scene, (0, 1), (-2, -1)), iso=800, noise=.01, seed=1)
    result, mask = compare_groups(clean, noisy, alignment=[0, 1])
    assert result['alignment_source'] == 'explicit'
    assert result['applied_shift_packed_pixels'] == [0, 1]
    assert mask.mean() > .65
    assert mask[:, 5:10].all()
    assert not mask[:, -1].any()
    with pytest.raises(ValueError, match='integer'):
        compare_groups(clean, noisy, alignment=[0, .5])
