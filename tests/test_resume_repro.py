"""CPU continuation of the actual trainers, including worker-prefetched streams."""
import ast
import importlib
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import yaml

from denoise_lab.data import (BootstrapDataset, PairedBurstDataset, ResumableBatches,
                              SeededSamples, SyntheticDataset, mixed_sampler, resumable_loader)
from denoise_lab.clean_targets import BinnedBootstrapDataset
from denoise_lab.flat_augment import FlatAugmentedBootstrapDataset
from denoise_lab.selfsup import RealNoisyDataset

ROOT = Path(__file__).resolve().parents[1]


def _numpy_collate(samples):
    # This sandbox blocks the tensor shared-memory manager. Keep real worker
    # sampling/prefetch, transporting arrays and rebuilding tensors in the parent.
    return {key: np.stack([np.asarray(sample[key]) for sample in samples]) for key in samples[0]}


def _tensors(batch):
    return {key: value if isinstance(value, torch.Tensor) else torch.from_numpy(value) for key, value in batch.items()}


@pytest.fixture(autouse=True)
def array_transport(monkeypatch):
    from denoise_lab import data
    def loader(*args, **kwargs):
        if kwargs.get('num_workers', 0):
            kwargs['collate_fn'] = _numpy_collate
        return torch.utils.data.DataLoader(*args, **kwargs)
    monkeypatch.setattr(data, 'DataLoader', loader)


@pytest.fixture
def trainers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module('train'), importlib.import_module('train_arch_study')


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


def synthetic(**kwargs):
    return SyntheticDataset(calibration=ROOT/'calib/example.yaml', random_data=True,
                            crop_size=8, length=5, **kwargs)


def batches(dataset, sampler=None, count=6, **kwargs):
    loader = resumable_loader(dataset, sampler, seed=17, **kwargs)
    iterator = iter(loader)
    try:
        return [_tensors(batch) for batch in itertools.islice(iterator, count)]
    finally:
        if loader.num_workers:
            iterator._shutdown_workers()


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('drop_last', [False, True])
def test_stream_workers_resume_epochs_and_rng_isolation(weighted, drop_last):
    ds = synthetic()
    sampler = mixed_sampler([ds, synthetic(iso=100)], [.2, .8], num_samples=5)[1] if weighted else None
    if weighted:
        ds = mixed_sampler([ds, synthetic(iso=100)], [.2, .8], num_samples=5)[0]
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    expected = batches(ds, sampler, batch_size=2, drop_last=drop_last, count=8)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert_exact(np.random.get_state()[1].tolist(), numpy_state[1].tolist())
    assert random.getstate() == python_state
    # Cross an epoch, restart on a partial epoch, and change worker count.
    actual = batches(ds, sampler, batch_size=2, drop_last=drop_last, batches=3, num_workers=2, count=5)
    assert_exact(actual, expected[3:])
    assert expected[0]['noisy'].shape[0] == 2
    assert len(expected[2]['noisy']) == (2 if drop_last else 1)


def test_weighted_block_resume_and_source_shares():
    ds, sampler = mixed_sampler([synthetic(), synthetic(iso=100)], [0, 1], num_samples=2051)
    sampler_batches = ResumableBatches(ds, sampler, 3, 8)
    full = list(itertools.islice(sampler_batches, 687))
    resumed = list(itertools.islice(ResumableBatches(ds, sampler, 3, 8, batches=341), 346))
    assert resumed == full[341:]
    assert all(5 <= index < 10 for batch in full for index, _ in batch)
    assert full[683][-1][1] == 2050
    assert full[684][0][1] == 2051


@pytest.mark.parametrize('kind', ['paired', 'bootstrap', 'flat', 'binned', 'camera', 'selfsup'])
def test_seeded_dataset_variants(tmp_path, kind):
    rng = np.random.default_rng(9)
    np.save(tmp_path/'clean.npy', rng.uniform(.01, .8, (80, 84)).astype(np.float32))
    options = dict(manifest=[{'clean': 'clean.npy'}], root=tmp_path, crop_size=16)
    if kind == 'paired':
        ds = PairedBurstDataset([{'noisy': 'clean.npy', 'clean': 'clean.npy'}], tmp_path, crop_size=16)
    elif kind == 'selfsup':
        np.save(tmp_path/'packed.npy', rng.uniform(.01, .8, (4, 40, 42)).astype(np.float32))
        manifest = tmp_path/'real.json'
        manifest.write_text(json.dumps([dict(crop='packed.npy', iso=1600, camera='fixture', sigma=.01)]))
        ds = RealNoisyDataset(manifest, tmp_path, crop_size=32)
    elif kind == 'camera':
        from denoise_lab.camera_noise import CameraNoiseDataset
        p = dict(K=.0002, read_std=.002, row_std=0., col_std=0., row_fixed_std=0., col_fixed_std=0.,
                 tukey_lambda=.14, lag1_h=0., hot_density=0., black_offset=0., code_step=1e-7,
                 black_dn=1000., white_dn=11000.)
        calibration = tmp_path/'camera.yaml'
        calibration.write_text(yaml.safe_dump(dict(units='normalized', cameras={
            'fixture': {'iso': {100: p, 1600: {**p, 'K': .0032}}}})))
        ds = CameraNoiseDataset(**options, calibration=calibration, cameras={'fixture': 1})
    else:
        cls = {'bootstrap': BootstrapDataset, 'flat': FlatAugmentedBootstrapDataset,
               'binned': BinnedBootstrapDataset}[kind]
        ds = cls(**options)
    wrapped = SeededSamples(ds, 19, 0)
    expected = wrapped[(0, 7)]
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    assert_exact(wrapped[(0, 7)], expected)
    assert not torch.equal(expected['noisy'], wrapped[(0, 8)]['noisy'])


@pytest.mark.parametrize('research', [False, True])
@pytest.mark.parametrize('workers', [0, 1])
def test_actual_trainer_resume_exact(tmp_path, monkeypatch, trainers, research, workers):
    trainer = trainers[int(research)]
    model = dict(width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1], cond='none')
    source = dict(kind='synthetic', calibration=str(ROOT/'calib/example.yaml'), random_data=True,
                  crop_size=8, length=3, iso=1600)
    cfg = dict(seed=37, device='cpu', threads=1, model=model,
               data=dict(kind='mixed', sources=[source, {**source, 'length': 2, 'iso': 100}],
                         probabilities=[.3, .7], num_samples=5),
               training=dict(steps=4, batch_size=2, workers=workers, lr=.001, amp=False,
                             checkpoint_every=2, keep_checkpoints=2, progress=False))
    accum = 2 if research else 1
    if research:
        cfg['training'].update(grad_accum=accum, warmup_steps=1, ema_decay=.9)
        np.save(tmp_path/'real.npy', np.random.default_rng(8).uniform(.01, .8, (4, 40, 48)).astype(np.float32))
        manifest = tmp_path/'real.json'
        manifest.write_text(json.dumps([dict(crop='real.npy', iso=1600, camera='fixture', sigma=.02)]))
        cfg['selfsup'] = dict(manifest=str(manifest), root=str(tmp_path), crop_size=32,
                              batch_size=1, workers=workers, weight=.1)
    path = tmp_path/'config.yaml'
    handles = []
    real_loader = trainer.resumable_loader

    class RecordingLoader:
        def __init__(self, *args, **kwargs):
            self.loader = real_loader(*args, **kwargs)
            self.seen = []
            handles.append(self)

        def __len__(self):
            return len(self.loader)

        def __iter__(self):
            self.iterator = iter(self.loader)
            return self

        def __next__(self):
            batch = _tensors(next(self.iterator))
            self.seen.append(batch)
            return batch

    monkeypatch.setattr(trainer, 'resumable_loader', RecordingLoader)
    def run(name, checkpoint=None):
        run_dir = tmp_path/name
        cfg['training']['run_dir'] = str(run_dir)
        path.write_text(yaml.safe_dump(cfg))
        argv = [trainer.__file__, '--config', str(path)]
        if checkpoint:
            argv += ['--resume', str(checkpoint)]
        monkeypatch.setattr(sys, 'argv', argv)
        trainer.main()
        streams = []
        for handle in handles:
            following = _tensors(next(handle.iterator))
            streams.append((handle.seen[:], following))
            if handle.loader.num_workers:
                handle.iterator._shutdown_workers()
        handles.clear()
        return torch.load(run_dir/'last.pt', weights_only=True), streams

    expected, full_streams = run('full')
    actual, resumed_streams = run('resumed', tmp_path/'full/step_0000002.pt')
    for key in ('model', 'optimizer', 'scheduler', 'scaler', 'data_stream', 'torch_rng_state',
                'python_rng_state', 'numpy_rng_state', 'cuda_rng_state_all'):
        assert_exact(actual[key], expected[key])
    if research:
        assert_exact(actual['ema_model'], expected['ema_model'])
        assert actual['ema_updates'] == expected['ema_updates'] == 4
    for (full, following), (resumed, resumed_following) in zip(full_streams, resumed_streams):
        assert_exact(resumed, full[2*accum:])
        assert_exact(resumed_following, following)
    assert actual['data_stream']['batches'] == 4*accum
    assert actual['data_stream']['selfsup_batches'] == (4*accum if research else 0)


def test_rng_roundtrip_and_legacy_stream(trainers, tmp_path, capsys):
    trainer, _ = trainers
    # The new states must remain loadable through the safe checkpoint reader.
    saved = trainer.capture_rng_state()
    path = tmp_path/'rng.pt'
    torch.save(saved, path)
    expected = (random.random(), np.random.random(), torch.rand(3))
    trainer.restore_rng_state(torch.load(path, weights_only=True))
    assert_exact((random.random(), np.random.random(), torch.rand(3)), expected)
    assert trainer.training_stream({'torch_rng_state': saved['torch_rng_state']}, 1) is None
    assert 'Legacy checkpoint' in capsys.readouterr().out
    assert trainer.training_stream({'data_stream': None}, 1) is None
    with pytest.raises(ValueError, match='data_stream version'):
        trainer.training_stream({'data_stream': {'version': 99}}, 1)


def test_sample_rng_restored_on_exception(trainers):
    trainer, _ = trainers
    class RandomDataset:
        def __getitem__(self, index):
            result = (random.random(), np.random.random(), torch.rand(3))
            if index:
                raise ValueError('sample failure')
            return result
    wrapped = SeededSamples(RandomDataset(), 8, 0)
    state = trainer.capture_rng_state()
    assert_exact(wrapped[(0, 2)], wrapped[(0, 2)])
    with pytest.raises(ValueError, match='sample failure'):
        wrapped[(1, 3)]
    assert_exact(trainer.capture_rng_state(), state)


def test_cuda_rng_plumbing_without_device(trainers, monkeypatch):
    trainer, _ = trainers
    states = [torch.arange(4, dtype=torch.uint8), torch.arange(5, dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: states)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda values: restored.extend(values))
    trainer.restore_rng_state(trainer.capture_rng_state())
    assert_exact(restored, states)


@pytest.mark.parametrize('research', [False, True])
def test_actual_legacy_resume(tmp_path, monkeypatch, trainers, capsys, research):
    trainer = trainers[int(research)]
    cfg = yaml.safe_load((ROOT/'configs/smoke.yaml').read_text())
    cfg['model'].update(width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    cfg['threads'] = 1
    cfg['training'].update(steps=2, run_dir=str(tmp_path/'run'), checkpoint_every=1,
                           keep_checkpoints=2, progress=False)
    cfg['data'].update(crop_size=8, length=3)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, 'argv', [trainer.__file__, '--config', str(path)])
    trainer.main()
    state = torch.load(tmp_path/'run/step_0000001.pt', weights_only=True)
    for key in ('data_stream', 'python_rng_state', 'numpy_rng_state', 'cuda_rng_state_all'):
        state.pop(key)
    checkpoint = tmp_path/'legacy.pt'
    torch.save(state, checkpoint)
    monkeypatch.setattr(sys, 'argv', [trainer.__file__, '--config', str(path), '--resume', str(checkpoint)])
    trainer.main()
    assert 'Legacy checkpoint' in capsys.readouterr().out
    saved = torch.load(tmp_path/'run/last.pt', weights_only=True)
    assert saved['step'] == 2 and saved['data_stream'] is None


def test_resume_code_parses_on_python_311():
    for name in ('scripts/train.py', 'scripts/train_arch_study.py', 'src/denoise_lab/data.py'):
        ast.parse((ROOT/name).read_text(), feature_version=(3, 11))
