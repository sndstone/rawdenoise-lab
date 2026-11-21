"""Synthetic, independent burst fixtures; shifts are packed pixels."""
import numpy as np
import pytest

from denoise_lab.bayer import pack, unpack
from denoise_lab.burst_align import (align_burst, estimate_shift, interpolation_variance,
                                    robust_merge, warp_packed, match_cross_iso)


def scene(size=128):
    rng = np.random.default_rng(8)
    f = np.fft.fft2(rng.normal(size=(size, size)))
    freq = np.fft.fftfreq(size)
    f *= np.exp(-(freq[:, None]**2+freq**2)/.08**2)
    x = np.fft.ifft2(f).real
    x = .3+.04*x/x.std()
    return np.stack([x*.8, x, x*.6, x*1.01]).astype(np.float32)


def translated(a, shift):
    """Independent analytic Fourier fixture, not the production resampler."""
    h, w = a.shape[-2:]
    phase = np.exp(-2j*np.pi*(np.fft.fftfreq(h)[:, None]*shift[0]+
                              np.fft.fftfreq(w)*shift[1]))
    return np.fft.ifft2(np.fft.fft2(a)*phase).real.astype(np.float32)


@pytest.mark.parametrize("shift", [(0.37, -1.24), (-2.61, 3.18), (.5, .5), (0, -5)])
def test_subpixel_registration_sensor_accuracy(shift):
    a = scene()
    b = translated(a, shift)
    b += np.random.default_rng(0).normal(0, .0005, b.shape)
    estimate = estimate_shift(a, b, max_shift=8)
    # 0.025 packed pixels = 0.05 sensor pixels.
    np.testing.assert_allclose(estimate, shift, atol=.025)


def test_cfa_phase_exact_integer_and_fractional_borders():
    a = np.arange(4*32*32, dtype=np.float32).reshape(4, 32, 32)
    out = pack(unpack(warp_packed(a, (2, -3))))
    np.testing.assert_array_equal(out[:, :-2, 3:], a[:, 2:, :-3])
    assert np.isnan(out[:, -2:]).all() and np.isnan(out[:, :, :3]).all()
    colors = np.broadcast_to(np.arange(4)[:, None, None], (4, 32, 32)).astype(np.float32)
    shifted = pack(unpack(warp_packed(colors, (.35, -.7))))
    np.testing.assert_allclose(shifted[:, :-1, 1:], colors[:, :-1, 1:], atol=1e-6)
    assert np.isnan(shifted[:, -1]).all()
    assert interpolation_variance((2, -3)) == 1


def test_merge_noise_and_moving_object():
    rng = np.random.default_rng(2)
    n, sigma = 16, .01
    truth = np.full((4, 128, 128), .2, np.float32)
    frames = truth+rng.normal(0, sigma, (n, *truth.shape)).astype(np.float32)
    for i in range(4):
        frames[i, :, 48:64, 48+8*i:64+8*i] += .25
    merged = robust_merge(frames, {"K": 0., "read_std": sigma}, block=8)
    static = np.zeros((128, 128), bool)
    static[:32] = True
    assert merged.static[static].mean() > .99
    assert not merged.static[48:64, 48:88].any()
    assert np.std((merged.clean-truth)[:, static]) == pytest.approx(sigma/np.sqrt(n), rel=.03)
    assert np.sqrt(merged.variance[:, static].mean()) == pytest.approx(sigma/np.sqrt(n), rel=.01)
    assert np.median(merged.effective_count[:, static]) == n
    np.testing.assert_allclose(merged.clean[:, 48:64, 48:88].mean(), .2, atol=.001)


def test_interpolation_noise_and_missing_support():
    rng = np.random.default_rng(4)
    shift = (.3, .6)
    white = rng.normal(size=(4, 192, 192)).astype(np.float32)
    q = interpolation_variance(shift)
    assert np.nanvar(warp_packed(white, shift)) == pytest.approx(q, rel=.02)
    a = np.full((3, 4, 32, 32), .2, np.float32)
    a[:, :, :8] = np.nan
    result = robust_merge(a, {"K": 0., "read_std": .01}, block=8)
    assert not result.static[:8].any()
    assert (result.effective_count[:, :8] == 0).all()
    assert np.isnan(result.clean[:, :8]).all()


def test_alignment_and_cross_iso_motion():
    a = scene()
    frames = np.stack([a, translated(a, (.4, -.3)), translated(a, (-.2, .6))])
    aligned, shifts = align_burst(frames, max_shift=4)
    np.testing.assert_allclose(shifts, [(0, 0), (.4, -.3), (-.2, .6)], atol=.025)
    # Exact known exposure fitter isolates the NLF mask from fit estimation.
    def fit(mean, clean, mask, block):
        return clean*1.5+.01, [[1.5, .01, 1., 1.]]*4
    rng = np.random.default_rng(3)
    sigma = .002
    clean = a+rng.normal(0, sigma, (8, *a.shape)).astype(np.float32)
    target = robust_merge(clean, {"K": 0., "read_std": sigma}, block=8)
    noisy = 1.5*a+.01+rng.normal(0, sigma, (4, *a.shape)).astype(np.float32)
    noisy[:, :, 48:64, 48:64] += .03
    # Mean-zero motion would evade a noisy-group-mean test alone.
    noisy[:2, :, 80:88, 80:88] += .03
    noisy[2:, :, 80:88, 80:88] -= .03
    fitted, var, valid, _ = match_cross_iso(noisy, target, {"K": 0., "read_std": sigma}, fit, block=8)
    assert not valid[48:64, 48:64].any()
    assert not valid[80:88, 80:88].any()
    assert valid[:32].mean() > .98
    np.testing.assert_allclose(var, target.variance*1.5**2, rtol=1e-6)
    assert np.isfinite(fitted).all()


def test_shot_noise_brightness_and_temporal_nlf():
    from denoise_lab.burst_align import estimate_burst_noise
    rng = np.random.default_rng(7)
    truth = np.broadcast_to(np.linspace(.01, .8, 128), (4, 128, 128)).astype(np.float32)
    k, read = .0002, .001
    frames = (rng.poisson(truth/k, (12, *truth.shape))*k+
              rng.normal(0, read, (12, *truth.shape))).astype(np.float32)
    result = robust_merge(frames, {"K": k, "read_std": read}, block=8)
    assert result.static.mean() > .99
    fit = estimate_burst_noise(frames, block=8)
    np.testing.assert_allclose([p["K"] for p in fit], k, rtol=.12)
    assert np.std((result.clean-truth)[:, :, 96:]) == pytest.approx(
        np.sqrt((k*truth[:, :, 96:]+read**2).mean()/12), rel=.04)


def test_resampled_merge_variance():
    rng = np.random.default_rng(15)
    sigma = .01
    shifts = [(0, 0), (.2, -.3), (.6, .8), (-.4, .3)]*3
    frames = np.stack([warp_packed(.2+rng.normal(0, sigma, (4, 96, 96)).astype(np.float32), s)
                       for s in shifts])
    q = [interpolation_variance(s) for s in shifts]
    target = robust_merge(frames, {"K": 0., "read_std": sigma}, q, block=8)
    sl = (slice(None), slice(8, -8), slice(8, -8))
    expected = sigma*np.sqrt(np.sum(q))/len(q)
    assert np.std(target.clean[sl]-.2) == pytest.approx(expected, rel=.04)
    assert np.sqrt(target.variance[sl].mean()) == pytest.approx(expected, rel=.01)


def test_invalid_registration_and_merge():
    with pytest.raises(ValueError, match="textured"):
        estimate_shift(np.zeros((4, 32, 32)), np.zeros((4, 32, 32)))
    with pytest.raises(ValueError):
        robust_merge(np.zeros((1, 4, 32, 32)), {"K": 0.})


def test_target_cli_independent_inputs_and_support(tmp_path, monkeypatch):
    import json
    import runpy
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1]/"scripts/make_burst_targets.py"
    rng = np.random.default_rng(5)
    a = scene()
    paths = []
    for i in range(5):
        path = tmp_path/f"frame{i}.npy"
        frame = a+rng.normal(0, .002, a.shape).astype(np.float32)
        if i == 0:
            frame[:, 48:56, 48:56] += .1
        np.save(path, unpack(frame))
        Path(str(path)+".json").write_text(json.dumps({"iso": 100, "exposure_time": .1, "aperture": 4}))
        paths.append(str(path))
    output, manifest = tmp_path/"target.npy", tmp_path/"pairs.json"
    monkeypatch.setattr(sys, "argv", [str(script), *paths, "--output", str(output),
                                      "--manifest", str(manifest), "--align"])
    runpy.run_path(str(script), run_name="__main__")
    entry = json.loads(manifest.read_text())[0]
    assert entry["noisy"] != paths[0]
    assert np.load(entry["noisy"]).shape == np.load(output).shape
    mask = np.load(entry["valid_mask"])
    assert not mask[96:112, 96:112].any()
    support = np.load(entry["support"])
    assert support["effective_count"].max() == 4
    assert support["shifts"].shape == (5, 2)
    # Held-out bright patch never enters the independent reference.
    np.testing.assert_allclose(pack(np.load(output))[:, 48:56, 48:56].mean(),
                               a[:, 48:56, 48:56].mean(), atol=.002)


def test_eval_alignment_preparation(monkeypatch):
    from pathlib import Path
    import importlib.util
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("burst_eval_fixture", scripts/"eval_real_burst.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(12)
    a = scene(192)
    clean = a+rng.normal(0, .002, (6, *a.shape)).astype(np.float32)
    noisy = 1.3*a+.005+rng.normal(0, .004, (4, *a.shape)).astype(np.float32)
    aligned, target, var, valid, coeffs, shifts, info = module.prepare_pair(clean, noisy, (0, 0), 8, True)
    # Tiny fitted shifts invalidate edge blocks and the motion-mask apron.
    assert valid.mean() > .65 and valid[16:-16, 16:-16].mean() > .98
    np.testing.assert_allclose([c[0] for c in coeffs], 1.3, atol=.015)
    assert info["clean_noise_std"] > 0
    assert (var[:, valid] > 0).all()
    assert shifts.shape == (4, 2) and aligned.shape == noisy.shape
