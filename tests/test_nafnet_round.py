"""Bounded CPU checks for flat targets, EMA, and production identity evaluation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from denoise_lab.arch_registry import build_model
from denoise_lab.bayer import pack
from denoise_lab.data import BootstrapDataset
from denoise_lab.flat_augment import FlatAugmentedBootstrapDataset, synthetic_smooth_mosaic

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('round_train', ROOT/'scripts/train_arch_study.py')
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def mosaic(tmp_path):
    np.save(tmp_path/'clean.npy', np.random.default_rng(9).uniform(.01, .8, (48, 52)).astype(np.float32))
    return dict(manifest=[{'clean': 'clean.npy'}], root=tmp_path, crop_size=12, deterministic=True,
                seed=13, augment=True)


def test_smooth_mosaic_layout_range_and_rng():
    options = dict(level_range=(.2, .2), ratio_range=(.5, .5), equal_probability=0,
                   gradient_probability=0)
    a = synthetic_smooth_mosaic(20, 26, np.random.default_rng(2), **options)
    assert a.shape == (20, 26) and a.dtype == np.float32
    planes = pack(a)
    np.testing.assert_allclose(planes[:, 0, 0], [.1, .2, .1, .2])
    assert np.all(planes.std((1, 2)) < 1e-6)
    equal = synthetic_smooth_mosaic(20, 26, np.random.default_rng(2), **{**options, 'equal_probability': 1})
    assert np.all(equal == np.float32(.2))
    for gradient in (0, 1):
        kwargs = dict(gradient_probability=gradient)
        a = synthetic_smooth_mosaic(21, 27, np.random.default_rng(8), **kwargs)
        b = synthetic_smooth_mosaic(21, 27, np.random.default_rng(8), **kwargs)
        np.testing.assert_array_equal(a, b)
        assert a.min() >= 0 and a.max() <= 1
    clipped = synthetic_smooth_mosaic(20, 26, np.random.default_rng(1), level_range=(.95, .95),
                                      ratio_range=(1.6, 1.6), equal_probability=0, gradient_probability=1)
    assert clipped.max() == 1
    # Each CFA plane is planar when there is no clipping.
    ramp = pack(synthetic_smooth_mosaic(20, 26, np.random.default_rng(4), level_range=(.1, .1),
                                      gradient_probability=1))
    assert np.max(np.abs(np.diff(ramp, n=2, axis=-1))) < 1e-7
    assert np.max(np.abs(np.diff(ramp, n=2, axis=-2))) < 1e-7


@pytest.mark.parametrize('augment', [True, False])
@pytest.mark.parametrize('deterministic', [True, False])
def test_disabled_matches_parent(mosaic, augment, deterministic):
    opts = {**mosaic, 'augment': augment, 'deterministic': deterministic,
            'exposure_range': (.1, 1.), 'exposure_sampling': 'log',
            'bright_exposure_probability': .3, 'identity_probability': .4,
            'column_relative_max': .2, 'impulse_range': (1e-4, 1e-3)}
    parent = BootstrapDataset(**opts)
    flat = FlatAugmentedBootstrapDataset(**opts, flat_probability=0)
    for index in range(5):
        torch.manual_seed(91+index)
        expected = parent[0]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(91+index)
        actual = flat[0]
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                assert torch.equal(actual[key], expected[key]), key
            else:
                assert actual[key] == expected[key]


def test_flat_identity_smooth_and_exposure(mosaic):
    options = dict(level_range=(.2, .2), ratio_range=(.5, .5), equal_probability=0,
                   gradient_probability=0)
    ds = FlatAugmentedBootstrapDataset(**mosaic, flat_probability=1, flat_options=options,
                                      identity_probability=1, exposure_range=(.5, .5), exposure_scale=.5)
    sample = ds[0]
    assert torch.equal(sample['noisy'], sample['clean'])
    assert sample['clean'].std((1, 2)).max() < 1e-7
    torch.testing.assert_close(sample['clean'][:, 0, 0], torch.tensor([.025, .05, .025, .05]))
    assert sample['exposure_scale'] == .25
    assert torch.count_nonzero(sample['noise_map']) == 0
    # Augmentation disabled also disables replacement, even at probability one.
    opts = {**mosaic, 'augment': False}
    expected = BootstrapDataset(**opts)[0]
    actual = FlatAugmentedBootstrapDataset(**opts, flat_probability=1)[0]
    assert torch.equal(expected['clean'], actual['clean'])
    assert type(train.dataset({'kind': 'bootstrap', **mosaic})[0]) is BootstrapDataset
    assert isinstance(train.dataset({'kind': 'bootstrap', **mosaic, 'flat_probability': 0})[0],
                      FlatAugmentedBootstrapDataset)


def test_ema_math_and_buffers():
    model = torch.nn.BatchNorm1d(2).double()
    rng = torch.get_rng_state()
    ema = train.make_ema_state(model)
    assert torch.equal(torch.get_rng_state(), rng)
    assert ema['weight'].dtype == torch.float32
    with torch.no_grad():
        model.weight.fill_(3)
        model.running_mean.fill_(2)
        model.num_batches_tracked.fill_(7)
    train.update_ema(ema, model, .99, 0)
    torch.testing.assert_close(ema['weight'], torch.full((2,), 2.8))
    torch.testing.assert_close(ema['running_mean'], torch.full((2,), 1.8))
    assert ema['num_batches_tracked'].item() == 7
    train.update_ema(ema, model, .05, 5)
    torch.testing.assert_close(ema['weight'], torch.full((2,), 2.99))
    assert ema['weight'].data_ptr() != model.weight.data_ptr()


def run_script(script, *args, check=True):
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'src')}
    return subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str, args)],
                          cwd=ROOT, env=env, text=True, capture_output=True, check=check, timeout=120)


def tiny_config(tmp_path):
    cfg = yaml.safe_load((ROOT/'configs/arch_study/smoke_nafnet_s1.yaml').read_text())
    cfg['model'] = dict(arch='nafnet', cond='none', width=2, enc_blk_nums=[1],
                        middle_blk_num=1, dec_blk_nums=[1])
    cfg['data'].update(crop_size=8, length=8)
    cfg['training'].update(steps=3, batch_size=1, grad_accum=1, checkpoint_every=1,
                           keep_checkpoints=3, ema_decay=.99)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_ema_smoke_checkpoint_and_resume(tmp_path):
    cfg, config = tiny_config(tmp_path)
    run = tmp_path/'train'
    result = run_script('train_arch_study.py', '--config', config, '--run-dir', run)
    assert 'Completed 3 steps' in result.stdout
    torch.manual_seed(cfg['seed'])
    model = build_model(cfg['model'])
    torch.nn.init.zeros_(model.body.ending.weight)
    torch.nn.init.zeros_(model.body.ending.bias)
    expected = train.make_ema_state(model)
    for t in range(3):
        saved = load(run/f'step_{t+1:07d}.pt')
        model.load_state_dict(saved['model'])
        train.update_ema(expected, model, .99, t)
        assert saved['ema_updates'] == t+1
        for name, value in expected.items():
            assert saved['ema_model'][name].dtype == torch.float32
            torch.testing.assert_close(saved['ema_model'][name], value, rtol=0, atol=0)
    assert 'ema_model' in load(run/'last.pt')
    resumed = tmp_path/'resumed'
    run_script('train_arch_study.py', '--config', config, '--run-dir', resumed,
               '--resume', run/'step_0000002.pt')
    source = load(run/'step_0000002.pt')
    saved = load(resumed/'last.pt')
    model.load_state_dict(saved['model'])
    train.update_ema(source['ema_model'], model, .99, 2)
    assert saved['ema_updates'] == 3
    for name in expected:
        torch.testing.assert_close(saved['ema_model'][name], source['ema_model'][name], rtol=0, atol=0)


def test_ema_off_preserves_training(tmp_path):
    cfg, path = tiny_config(tmp_path)
    states = []
    for name, decay in [('absent', None), ('zero', 0)]:
        cfg['training'].pop('ema_decay', None)
        if decay is not None:
            cfg['training']['ema_decay'] = decay
        path.write_text(yaml.safe_dump(cfg))
        run_script('train_arch_study.py', '--config', path, '--run-dir', tmp_path/name, '--steps', 1)
        states.append(load(tmp_path/name/'last.pt'))
    for saved in states:
        assert 'ema_model' not in saved
    for key in states[0]['model']:
        assert torch.equal(states[0]['model'][key], states[1]['model'][key])
    assert torch.equal(states[0]['torch_rng_state'], states[1]['torch_rng_state'])


def test_identity_cli_random_tiny_and_ema_selection(tmp_path):
    cfg, _ = tiny_config(tmp_path)
    torch.manual_seed(123)
    model = build_model(cfg['model'])
    # Model weights are random, whereas the EMA fixture is exact identity.
    ema = train.make_ema_state(model)
    ema['body.ending.weight'].zero_()
    ema['body.ending.bias'].zero_()
    checkpoint = tmp_path/'tiny.pt'
    torch.save(dict(model_config=cfg['model'], model=model.state_dict(), ema_model=ema), checkpoint)
    for weights in ('model', 'ema'):
        out = tmp_path/f'{weights}.json'
        run_script('evaluate_identity.py', '--checkpoint', f'tiny={checkpoint}', '--device', 'cpu',
                   '--frame-size', 17, 23, '--weights', weights, '--out', out)
        report = json.loads(out.read_text())
        assert report['frame_size'] == [17, 23]
        assert report['checkpoints']['tiny']['weights'] == weights
        assert len(report['rows']) == 100 and len(report['summary']) == 10
        assert report['tile_size'] == 512 and report['overlap'] == 58 and report['feather_power'] == 2.2
        assert all(np.isfinite(row['p99_9_abs_error']) for row in report['rows'])
        assert all(row['passed'] == (row['max_abs_error'] <= 1e-3) for row in report['rows'])
        assert '| Max abs error | Mean bias | p99.9 abs error | Pass |' in out.with_suffix('.md').read_text()
        if weights == 'ema':
            assert report['passed']
    torch.save(dict(model_config=cfg['model'], model=model.state_dict()), checkpoint)
    result = run_script('evaluate_identity.py', '--checkpoint', f'tiny={checkpoint}', '--device', 'cpu',
                        '--weights', 'ema', '--out', tmp_path/'missing.json', check=False)
    assert result.returncode != 0 and 'requested ema weights but checkpoint has no ema_model' in result.stderr
