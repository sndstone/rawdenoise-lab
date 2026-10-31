"""CPU hybrid/fallback plumbing plus an explicitly gated optional MPS regression."""
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

from denoise_lab import accelerator
from denoise_lab.arch_registry import build_model
from denoise_lab.inference import Denoiser
from denoise_lab.model import gamma, inverse_gamma
from denoise_lab.bayer import PHASES
from denoise_lab.dng import RawMetadata, write_dng


@pytest.fixture
def tiny(tmp_path):
    cfg = dict(width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    torch.manual_seed(12)
    model = build_model(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(('.beta', '.gamma')):
                p.fill_(.1)
    path = tmp_path/'tiny.pt'
    torch.save(dict(model_config=cfg, model=model.state_dict()), path)
    return path, model


@pytest.fixture
def dyadic(tmp_path):
    """Known non-identity body: two center-only diagonal /8 convolutions.

    No learned/random weights or scale search. For normal-valued inputs the
    exact body correction is gamma/64; there are no attention/reduction blocks.
    This isolates real device transfer, convolution and host-tiler plumbing.
    """
    cfg = dict(width=4, enc_blk_nums=[], middle_blk_num=0, dec_blk_nums=[])
    model = build_model(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        for c in range(4):
            model.body.intro.weight[c, c, 1, 1] = 1/8
            model.body.ending.weight[c, c, 1, 1] = 1/8
    path = tmp_path/'dyadic.pt'
    torch.save(dict(model_config=cfg, model=model.state_dict()), path)
    return path, model


@pytest.fixture
def cpu_hybrid(monkeypatch):
    for key in ['PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH']:
        monkeypatch.setenv(key, '0')
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: None)
    monkeypatch.setattr(accelerator, 'make_mps_body', lambda m: accelerator.HostOuterBody(
        copy.deepcopy(m.body), m.gamma_exponent, body_device='cpu'))
    monkeypatch.setattr(accelerator, 'numeric_canary', lambda *_: [{'case': 'mocked CPU-only wiring'}])


def load_hybrid(path):
    with pytest.warns(RuntimeWarning, match='not quality-accepted'):
        return Denoiser.load(path, device='mps', experimental_mps_body=True)


def test_host_outer_cpu_body_bitwise_equivalence(tiny):
    _, model = tiny
    wrapper = accelerator.HostOuterBody(model.body, model.gamma_exponent, body_device='cpu')
    x = torch.randn(2, 4, 18, 22) * .1
    with torch.inference_mode():
        assert torch.equal(wrapper(x), model(x))
        expected = model(x)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            assert torch.equal(wrapper(x), expected)


def test_positive_hardware_fixture_has_known_nontrivial_arithmetic(dyadic):
    _, model = dyadic
    data = torch.linspace(-.001, .1, 17*19).reshape(1, 1, 17, 19).expand(1, 4, -1, -1).contiguous()
    with torch.inference_mode():
        encoded = gamma(data, model.gamma_exponent)
        assert torch.equal(model.body(encoded), encoded/64)
        assert torch.equal(model(data), inverse_gamma(encoded+encoded/64, model.gamma_exponent))
        assert float((model(data)-data).abs().max()) > .001


@pytest.mark.parametrize('failure', ['exception', 'nan', 'wrong_shape'])
@pytest.mark.parametrize('prepass', [False, True])
def test_partial_image_failure_restarts_all_cpu(tiny, cpu_hybrid, failure, prepass):
    path, _ = tiny
    cpu = Denoiser.load(path, device='cpu')
    hybrid = load_hybrid(path)
    packed = np.random.default_rng(42).random((4, 31, 60), dtype=np.float32)
    opts = dict(tile_size=16, overlap=4, tile_batch_size=2, prepass=prepass, strength=.8)
    expected = cpu.denoise_packed(packed, **opts)
    calls = []
    real_body = hybrid._call.body.forward
    def failed(tile):
        calls.append(tile.shape)
        if len(calls) == 3:
            if failure == 'exception':
                raise RuntimeError('injected third tile failure')
            if failure == 'nan':
                return torch.full_like(tile, float('nan'))
            return tile[..., :-1]
        # Deliberately corrupt completed GPU tiles: equality requires discarding
        # the entire partial merge rather than mixing GPU and CPU predictions.
        return real_body(tile) + .02
    hybrid._call.body.forward = failed
    with pytest.warns(RuntimeWarning, match='whole-image CPU fallback'):
        actual = hybrid.denoise_packed(packed, **opts)
    assert len(calls) >= 3
    np.testing.assert_array_equal(actual, expected)
    assert not hybrid._accelerator_active
    assert hybrid.provenance['producing_backend'] == 'cpu fallback'
    assert hybrid.provenance['checkpoint_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert 'cpu fallback' in hybrid.description
    # A failing backend is disabled for subsequent images too.
    np.testing.assert_array_equal(hybrid.denoise_packed(packed, **opts), expected)


def test_unavailable_mps_falls_back_at_load(tiny, cpu_hybrid, monkeypatch):
    path, _ = tiny
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.warns(RuntimeWarning) as warnings:
        actual = Denoiser.load(path, device='mps', experimental_mps_body=True)
    assert any('MPS is unavailable' in str(w.message) for w in warnings)
    data = np.zeros((4, 5, 7), np.float32)
    expected = Denoiser.load(path, device='cpu').denoise_packed(data, tile_size=16, overlap=2)
    np.testing.assert_array_equal(actual.denoise_packed(data, tile_size=16, overlap=2), expected)
    assert actual.provenance['fallback_reason']


def test_canary_failure_disables_accelerator(tiny, cpu_hybrid, monkeypatch):
    def reject(*_):
        raise accelerator.AcceleratorError('injected canary bounds failure')
    monkeypatch.setattr(accelerator, 'numeric_canary', reject)
    with pytest.warns(RuntimeWarning):
        d = Denoiser.load(tiny[0], device='mps', experimental_mps_body=True)
    assert not d._accelerator_active
    assert 'canary bounds' in d.provenance['fallback_reason']


def test_cpu_fallback_uses_loaded_snapshot_not_changed_path(tiny, cpu_hybrid):
    path, _ = tiny
    cpu = Denoiser.load(path, device='cpu');hybrid = load_hybrid(path)
    expected_sha = hybrid.provenance['checkpoint_sha256']
    data = np.zeros((4, 2, 3), np.float32)
    expected = cpu.denoise_packed(data, tile_size=16, overlap=2)
    path.write_bytes(b'checkpoint replaced after model load')
    def fail(_):raise accelerator.AcceleratorError('injected device error')
    hybrid._call = fail
    with pytest.warns(RuntimeWarning, match='whole-image CPU fallback'):
        actual = hybrid.denoise_packed(data, tile_size=16, overlap=2)
    np.testing.assert_array_equal(actual, expected)
    assert hybrid.provenance['checkpoint_sha256'] == expected_sha


def test_identity_provenance_then_success(tiny, cpu_hybrid):
    d = load_hybrid(tiny[0]);x = np.zeros((4, 2, 3), np.float32)
    np.testing.assert_array_equal(d.denoise_packed(x, strength=0), x)
    assert d.provenance['producing_backend'] == 'identity (strength=0)'
    d.denoise_packed(x, tile_size=16, overlap=2)
    assert d.provenance['producing_backend'] == 'mps-body + cpu-transforms'


@pytest.mark.parametrize('key', ['PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH'])
def test_unsafe_environment_refused_before_file_load(tiny, cpu_hybrid, monkeypatch, key):
    monkeypatch.setenv(key, '1')
    with pytest.raises(ValueError, match=key):
        Denoiser.load('missing.pt', device='mps', experimental_mps_body=True)


def test_subprocess_unsafe_environment_refused():
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1', PYTORCH_MPS_FAST_MATH='0')
    script = "from denoise_lab.inference import Denoiser; Denoiser.load('missing.pt', device='mps', experimental_mps_body=True)"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert 'requires PYTORCH_ENABLE_MPS_FALLBACK=0' in result.stderr
    assert 'FileNotFoundError' not in result.stderr


@pytest.mark.parametrize('device', ['auto', 'cpu', 'cuda'])
def test_optin_requires_explicit_mps(device):
    with pytest.raises(ValueError, match='explicit device=mps'):
        Denoiser.load('missing.pt', device=device, experimental_mps_body=True)


def test_coreml_wrong_tile_rejected_before_allocation(tiny, monkeypatch):
    d = Denoiser.load(tiny[0], device='cpu');d.backend = 'coreml'
    monkeypatch.setattr(torch, 'from_numpy', lambda _: pytest.fail('must reject before tensor allocation'))
    with pytest.raises(ValueError, match='tile_size must be 512'):
        d.denoise_packed(np.zeros((4, 1, 1), np.float32), tile_size=256)


@pytest.mark.parametrize('phase', list(PHASES))
def test_cli_cpu_body_stub_preserves_source_cfa(tiny, cpu_hybrid, tmp_path, monkeypatch, phase):
    from denoise_lab import cli
    path, _ = tiny
    pattern = np.array(['RGB'.index(c) for c in phase], dtype=np.uint8).reshape(2, 2)
    meta = RawMetadata(pattern=pattern, black=np.full((2, 2), 512.), white=np.full((2, 2), 16383.))
    codes = np.random.default_rng(3).integers(600, 14000, (32, 40), dtype=np.uint16)
    source = tmp_path/'source.dng';write_dng(source, codes, meta, normalized=False)
    # Load through the actual opt-in API but shrink tile size only in this test
    # harness, keeping the production CLI's fixed512 contract unchanged.
    original_load = Denoiser.load
    def load(*args, **kwargs):
        d = original_load(*args, **kwargs);original = d.denoise_packed
        d.denoise_packed = lambda a, **options: original(a, **dict(options, tile_size=16, overlap=4))
        return d
    monkeypatch.setattr(Denoiser, 'load', load)
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: None)
    outputs=[]
    for label, args in [('cpu', ['--device', 'cpu']), ('hybrid', ['--device', 'mps', '--experimental-mps-body'])]:
        out=tmp_path/label
        monkeypatch.setattr(sys, 'argv', ['rawdenoise', str(source), '--model', str(path), '--format', 'npy',
                                         '--no-prepass', '--tile-batch-size', '2', '--output-dir', str(out), *args])
        if label == 'hybrid':
            with pytest.warns(RuntimeWarning, match='not quality-accepted'):cli.main()
        else:cli.main()
        outputs.append(np.load(out/'source_denoised.npy'))
    np.testing.assert_array_equal(*outputs)


def test_numeric_canary_bounds():
    identity = lambda x: x
    reports = accelerator.numeric_canary(identity, identity)
    assert len(reports) == 5
    assert reports[2]['case'] == 'signed_shadow_ramp'
    assert reports[4]['batch'] == 2
    with pytest.raises(accelerator.AcceleratorError, match='failed max'):
        accelerator.numeric_canary(identity, lambda x: x + .0001)


def test_experimental_batch_shape_must_be_canary_covered(tiny, cpu_hybrid):
    d = load_hybrid(tiny[0])
    with pytest.raises(ValueError, match='1 or 2 only'):
        d.denoise_packed(np.zeros((4, 1, 1), np.float32), tile_batch_size=3)


@pytest.mark.parametrize('host_stage', ['prepass', 'tiler'])
def test_host_memory_error_propagates_without_fallback(tiny, cpu_hybrid, monkeypatch, host_stage):
    from denoise_lab import inference
    d = load_hybrid(tiny[0])
    provenance = d.provenance
    def fail(*args, **kwargs):
        raise MemoryError('injected host allocation failure')
    monkeypatch.setattr(inference, 'correct_defects' if host_stage == 'prepass' else 'tiled_inference', fail)
    with warnings.catch_warnings(record=True) as seen:
        with pytest.raises(MemoryError, match='host allocation'):
            d.denoise_packed(np.zeros((4, 3, 4), np.float32), prepass=host_stage == 'prepass',
                             tile_size=16, overlap=2)
    assert d._accelerator_active
    assert d.provenance == provenance
    assert not any('fallback' in str(w.message) for w in seen)


def test_reference_canary_host_error_is_not_masked(tiny, cpu_hybrid, monkeypatch):
    def fail(*args):raise MemoryError('CPU reference allocation')
    monkeypatch.setattr(accelerator, 'numeric_canary', fail)
    with warnings.catch_warnings(record=True) as seen:
        with pytest.raises(MemoryError, match='CPU reference'):
            Denoiser.load(tiny[0], device='mps', experimental_mps_body=True)
    assert not any('whole-image CPU fallback' in str(w.message) for w in seen)


@pytest.mark.skipif(not torch.backends.mps.is_available() or any(
    os.environ.get(k) != '0' for k in ['PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH']),
    reason='Real MPS requires available hardware and explicit startup environment controls0')
def test_real_mps_random_model_canary_rejection_or_numeric_bounds(tiny):
    """An arbitrary random network need not pass strict accelerator canaries."""
    path, _ = tiny
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        cpu = Denoiser.load(path, device='cpu')
        with warnings.catch_warnings(record=True) as seen:
            hybrid = Denoiser.load(path, device='mps', experimental_mps_body=True)
        assert any('not quality-accepted' in str(w.message) for w in seen)
        image = np.random.default_rng(91041).random((4, 600, 620), dtype=np.float32)*.1
        options = dict(prepass=False, overlap=60, tile_size=512)
        if not hybrid._accelerator_active:
            assert 'numeric canary' in hybrid.provenance['fallback_reason']
            assert any('whole-image CPU fallback' in str(w.message) for w in seen)
            expected = cpu.denoise_packed(image, tile_batch_size=2, **options)
            actual = hybrid.denoise_packed(image, tile_batch_size=2, **options)
            np.testing.assert_array_equal(actual.view('uint32'), expected.view('uint32'))
            assert hybrid.provenance['producing_backend'] == 'cpu fallback'
        else:
            reference = cpu.denoise_packed(image, **options)
            for batch in [1, 2]:
                actual = hybrid.denoise_packed(image, tile_batch_size=batch, **options)
                assert hybrid._accelerator_active, hybrid.provenance
                assert np.isfinite(actual).all()
                error = (actual.astype(np.float64)-reference.astype(np.float64))*65535
                assert np.all(np.max(np.abs(error), axis=(1, 2)) <= .25)
                assert np.all(np.sqrt(np.mean(error**2, axis=(1, 2))) <= .02)
                assert np.all(np.abs(np.mean(error, axis=(1, 2))) <= .005)
    finally:
        torch.set_num_threads(prior_threads)


@pytest.mark.skipif(not torch.backends.mps.is_available() or any(
    os.environ.get(k) != '0' for k in ['PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH']),
    reason='Real MPS requires available hardware and explicit startup environment controls0')
def test_real_mps_multitile_dyadic_repeat_batch_and_cpu_bounds(dyadic):
    """Known-arithmetic device/host test; no learned photographic acceptance."""
    path, _ = dyadic
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        cpu = Denoiser.load(path, device='cpu')
        with pytest.warns(RuntimeWarning, match='not quality-accepted'):
            hybrid = Denoiser.load(path, device='mps', experimental_mps_body=True)
        assert hybrid._accelerator_active, hybrid.provenance
        image = np.random.default_rng(91041).random((4, 600, 620), dtype=np.float32)*.1
        options = dict(prepass=False, overlap=60, tile_size=512)
        reference = cpu.denoise_packed(image, **options)
        # Known body correction=gamma/64, rather than zero/identity arithmetic.
        assert np.max(np.abs(reference-image)) > .001
        single = hybrid.denoise_packed(image, tile_batch_size=1, **options)
        batched = hybrid.denoise_packed(image, tile_batch_size=2, **options)
        repeated = hybrid.denoise_packed(image, tile_batch_size=2, **options)
        assert hybrid._accelerator_active, hybrid.provenance
        assert np.isfinite(single).all() and np.isfinite(batched).all()
        np.testing.assert_array_equal(single.view('uint32'), batched.view('uint32'))
        np.testing.assert_array_equal(batched.view('uint32'), repeated.view('uint32'))
        error = (batched.astype(np.float64)-reference.astype(np.float64))*65535
        assert np.all(np.max(np.abs(error), axis=(1, 2)) <= .25)
        assert np.all(np.sqrt(np.mean(error**2, axis=(1, 2))) <= .02)
        assert np.all(np.abs(np.mean(error, axis=(1, 2))) <= .005)
    finally:
        torch.set_num_threads(prior_threads)
