"""Small CPU fixtures for exact measured profiles, pairing and scoreboard policy."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from denoise_lab import scoreboard as sb
from denoise_lab.arch_registry import build_model


@pytest.fixture
def measured_fixture(tmp_path):
    model_config = dict(arch='nafnet', cond='none', width=4, enc_blk_nums=[1],
                        middle_blk_num=1, dec_blk_nums=[1])
    model = build_model(model_config)
    torch.nn.init.zeros_(model.body.ending.weight)
    torch.nn.init.zeros_(model.body.ending.bias)
    state = model.state_dict()
    ema = {k: v.clone() for k, v in state.items()}
    ema['body.ending.bias'].fill_(.01)
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(model_config=model_config, model=state, ema_model=ema), checkpoint)
    np.save(tmp_path/'clean.npy', np.random.default_rng(1).uniform(.1, .4, (40, 40)).astype(np.float32))
    entry = dict(clean='clean.npy', source_sha256='val', sensor_sha256='val_sensor', group='val_scene')
    (tmp_path/'val.json').write_text(json.dumps([entry]))
    (tmp_path/'train.json').write_text(json.dumps([{**entry, 'source_sha256': 'train',
                                                 'sensor_sha256': 'train_sensor', 'group': 'train_scene'}]))
    calibration = tmp_path/'calibration.yaml'
    calibration.write_bytes((Path(__file__).resolve().parents[1]/'calib/darks_physnoise_20261101.yaml').read_bytes())
    cfg = dict(root=str(tmp_path), calibration=str(calibration), crop_size=16, crops_per_source=3,
               datasets=[dict(name='fixture', cameras=['x1d2', 'x1d2_electronic', 'a7m3'],
                              manifest=str(tmp_path/'val.json'), train_manifest=str(tmp_path/'train.json'))])
    return cfg, checkpoint


def evaluate(tmp_path, cfg, checkpoint, **overrides):
    args = dict(config=cfg, checkpoint=[f'base={checkpoint}', f'self={checkpoint}'],
                output=tmp_path/'result.json', quick=False, device='cpu', weights='model')
    args.update(overrides)
    torch.set_num_threads(2)
    return sb._script('evaluate_measured_dev').evaluate(SimpleNamespace(**args))


def test_measured_deterministic_self_pairing_gain_cap_and_provenance(tmp_path, measured_fixture):
    cfg, checkpoint = measured_fixture
    result = evaluate(tmp_path, cfg, checkpoint)
    again = evaluate(tmp_path, cfg, checkpoint)
    assert result['rows'] == again['rows']
    assert len(result['rows']) == 11*3*3*2
    assert len(result['source_paired_deltas']['self']) == 11*3
    assert all(r['delta_db'] == 0 for r in result['source_paired_deltas']['self'])
    assert all(r['crops'] == 3 for r in result['source_scores'])
    assert result['fixed_pattern_seed'] != 20261104
    assert result['K_jitter_log'] == result['read_jitter_log'] == 0
    assert len(result['calibration_sha256']) == 64
    assert all(result['physics_options'][k] for k in sb._script('evaluate_measured_dev').PHYSICS)
    assert any(p['iso'] == 65535 and p['processed'] for p in result['profiles'])
    for row in result['rows']:
        assert row['actual_exposure'] == row['exposure']*(.25 if '/leaf/25600' in row['profile'] else 1)
        assert all(isinstance(row[k], int) for k in ('crop_seed', 'noise_seed'))
    crop = [r for r in result['rows'] if r['crop'] == 0]
    assert len({tuple(r['crop_yx']) for r in crop}) == 1


@pytest.mark.parametrize('key', ['source_sha256', 'sensor_sha256', 'group'])
def test_measured_overlap_checked_before_quick_selection(tmp_path, measured_fixture, key):
    cfg, checkpoint = measured_fixture
    val = json.loads((tmp_path/'val.json').read_text())[0]
    train = json.loads((tmp_path/'train.json').read_text())
    train[0][key] = val[key]
    (tmp_path/'train.json').write_text(json.dumps(train))
    with pytest.raises(ValueError, match='Train/validation overlap: '+key):
        evaluate(tmp_path, cfg, checkpoint, quick=True)


def test_measured_missing_exact_profile_is_unavailable_not_interpolated(tmp_path, measured_fixture):
    cfg, checkpoint = measured_fixture
    path = Path(cfg['calibration'])
    calibration = yaml.safe_load(path.read_text())
    cam = calibration['cameras']['x1d2_electronic']
    del cam['iso'][800]  # Shared YAML aliases may also update the shutter mapping.
    cam['shutters']['electronic']['iso'].pop(800, None)
    path.write_text(yaml.safe_dump(calibration))
    result = evaluate(tmp_path, cfg, checkpoint, quick=True)
    profile = next(p for p in result['profiles'] if p['profile'] == 'x1d2_electronic/electronic/800')
    assert profile['status'] == 'SKIP' and 'exact' in profile['reason']
    missing = [r for r in result['summary'] if '/electronic/800/' in r['slice']]
    assert len(missing) == 6 and all(r['model_psnr'] is None for r in missing)
    assert not any('/electronic/800' in r['profile'] for r in result['rows'])


def test_measured_quick_and_ema_selection(tmp_path, measured_fixture):
    cfg, checkpoint = measured_fixture
    model = evaluate(tmp_path, cfg, checkpoint, quick=True)
    ema = evaluate(tmp_path, cfg, checkpoint, quick=True, weights='ema')
    assert model['crops_per_source'] == 1
    assert len(model['rows']) == 11*3*2
    assert ema['weights'] == 'ema'
    assert model['summary'][0]['model_psnr'] != ema['summary'][0]['model_psnr']
    assert all(r['delta_db'] == 0 for r in ema['source_paired_deltas']['self'])
    with pytest.raises(ValueError, match='differ from training'):
        evaluate(tmp_path, {**cfg, 'fixed_pattern_seed': 20261104}, checkpoint)


def test_measured_scoreboard_non_gating_cache_and_opt_in(tmp_path, measured_fixture):
    cfg, checkpoint = measured_fixture
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(suites=dict(measured=cfg))))
    output = tmp_path/'out'
    argv = ['--checkpoint', f'self={checkpoint}', '--baseline', str(checkpoint), '--config', str(config),
            '--suites', 'measured', '--quick', '--device', 'cpu', '--output', str(output)]
    assert sb.main(argv) == 0
    report = json.loads((output/'scoreboard.json').read_text())
    assert report['status'] == 'NOT_GATED'
    assert all(not r['gate'] and r['delta'] == 0 for r in report['candidates']['self']['comparisons'])
    assert all(r['delta_db'] == 0 for r in report['candidates']['self']['suites']['measured']['details']['source_paired_deltas'])
    assert 'non-gating' in (output/'scoreboard.md').read_text()
    assert sb.main(argv) == 0
    assert json.loads((output/'scoreboard.json').read_text())['baseline']['suites']['measured']['cached']
    assert sb.main(argv+['--weights', 'ema']) == 0
    assert not json.loads((output/'scoreboard.json').read_text())['baseline']['suites']['measured']['cached']
    assert sb.main(argv+['--weights', 'ema']) == 0
    assert json.loads((output/'scoreboard.json').read_text())['baseline']['suites']['measured']['cached']
    # Calibration contents participate in cache identity even if size/mtime are preserved.
    before, _ = sb._input_fingerprint('measured', cfg)
    calibration = Path(cfg['calibration'])
    stat = calibration.stat()
    calibration.write_text(calibration.read_text().replace('normalized', 'normalizeD', 1))
    os.utime(calibration, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after, _ = sb._input_fingerprint('measured', cfg)
    assert before != after
    calibration.write_text(calibration.read_text().replace('normalizeD', 'normalized', 1))
    assert sb.main(argv+['--weights', 'model']) == 0
    cfg['gate'] = True
    config.write_text(yaml.safe_dump(dict(suites=dict(measured=cfg))))
    assert sb.main(argv) == 0
    assert json.loads((output/'scoreboard.json').read_text())['status'] == 'PASS'
    cal = yaml.safe_load(calibration.read_text())
    del cal['cameras']['a7m3']['iso'][65535]
    cal['cameras']['a7m3']['shutters']['mechanical']['iso'].pop(65535, None)
    calibration.write_text(yaml.safe_dump(cal))
    assert sb.main(argv) == 1
    report = json.loads((output/'scoreboard.json').read_text())
    assert any(r['delta'] is None and r['gate'] for r in report['candidates']['self']['comparisons'])


def test_measured_diagnostic_errors_do_not_change_existing_gate(tmp_path, measured_fixture):
    cfg, checkpoint = measured_fixture
    measured = copy.deepcopy(cfg)
    measured['calibration'] = str(tmp_path/'missing.yaml')
    dev = dict(root=cfg['root'], crop_size=16, slices=['native|1|low'], datasets=cfg['datasets'])
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(gate=dict(require_improvement=True), suites=dict(dev=dev, measured=measured))))
    output = tmp_path/'out'
    argv = ['--checkpoint', f'self={checkpoint}', '--baseline', str(checkpoint), '--config', str(config),
            '--quick', '--device', 'cpu', '--output', str(output)]
    assert sb.main(argv) == 0
    report = json.loads((output/'scoreboard.json').read_text())
    assert report['status'] == 'PASS'
    assert report['candidates']['self']['gate']['measurements'] == 1
    assert report['candidates']['self']['suites']['measured']['status'] == 'SKIP'


@pytest.mark.parametrize('missing', ['camera', 'shutter', 'physics', 'processing'])
def test_unavailable_profile_reasons(tmp_path, measured_fixture, missing):
    cfg, _ = measured_fixture
    cal = yaml.safe_load(Path(cfg['calibration']).read_text())
    if missing == 'camera':
        del cal['cameras']['x1d2']
    elif missing == 'shutter':
        cal['cameras']['x1d2']['shutter'] = 'unknown'
        del cal['cameras']['x1d2']['shutters']['leaf']
    elif missing == 'physics':
        cal['cameras']['x1d2']['shutters']['leaf']['iso'][100].pop('physics')
    else:
        cal['cameras']['x1d2']['processed_isos'] = [100]
    rows = sb._script('evaluate_measured_dev').profiles(cal)
    row = next(r for r in rows if r['profile'] == 'x1d2/leaf/100')
    assert row['status'] == 'SKIP'
    assert row['reason']


def test_overlap_across_datasets_is_rejected(tmp_path, measured_fixture):
    cfg, checkpoint = measured_fixture
    cfg = copy.deepcopy(cfg)
    entry = json.loads((tmp_path/'val.json').read_text())[0]
    # The second training manifest leaks a first-dataset validation source.
    (tmp_path/'other_train.json').write_text(json.dumps([{**entry, 'sensor_sha256': 'other_sensor', 'group': 'other_group'}]))
    other = {**entry, 'source_sha256': 'other_val', 'sensor_sha256': 'other_val_sensor', 'group': 'other_val_group'}
    (tmp_path/'other_val.json').write_text(json.dumps([other]))
    cfg['datasets'].append(dict(name='other', cameras=['a7m3'], manifest=str(tmp_path/'other_val.json'),
                                train_manifest=str(tmp_path/'other_train.json')))
    with pytest.raises(ValueError, match='Train/validation overlap: source_sha256'):
        evaluate(tmp_path, cfg, checkpoint, quick=True)
