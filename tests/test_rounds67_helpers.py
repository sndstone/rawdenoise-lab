"""Import script helpers without running their command-line entry points."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def helpers():
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    old_path = sys.path[:]
    sys.path.insert(0, str(scripts))
    try:
        modules = []
        for name in ("eval_real_burst", "build_dark_calibration"):
            spec = importlib.util.spec_from_file_location("test_"+name, scripts/(name+".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        yield modules
    finally:
        sys.path[:] = old_path


def test_block_view_and_upsample(helpers):
    real, _ = helpers
    a = np.arange(2*7*10).reshape(2, 7, 10)
    blocks = real.block_view(a, 3)
    assert blocks.shape == (2, 2, 3, 3, 3)
    np.testing.assert_array_equal(blocks[:, 1, :, 2, :], a[:, 3:6, 6:9])
    mask = real.upsample(np.array([[True, False], [False, True]]), 3, (7, 8))
    assert mask[:3, :3].all() and mask[3:6, 3:6].all()
    assert not mask[6:].any() and not mask[:, 6:].any()


def test_dilate_does_not_wrap_at_edges(helpers):
    real, _ = helpers
    m = np.zeros((7, 8), bool)
    m[0, 0] = True
    out = real.dilate(m, 1)
    expected = np.zeros_like(m)
    expected[:2, :2] = True
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(real.dilate(m, 0), m)
    assert m.sum() == 1


@pytest.mark.parametrize("dy,dx", [(1, 2), (-1, -2), (0, 0)])
def test_shift_preserves_phase_and_marks_borders(helpers, dy, dx):
    real, _ = helpers
    a = np.arange(4*8*10, dtype=float).reshape(4, 8, 10)
    out = real.shifted(a, dy, dx)
    for y in range(8):
        for x in range(10):
            if 0 <= y+dy < 8 and 0 <= x+dx < 10:
                np.testing.assert_array_equal(out[:, y, x], a[:, y+dy, x+dx])
            else:
                assert np.isnan(out[:, y, x]).all()


def test_fit_gain_exact_affine_and_small_image(helpers):
    real, _ = helpers
    rng = np.random.default_rng(4)
    clean = rng.uniform(.05, .4, (4, 32, 48))
    gains = np.array([.8, 1.1, 1.2, .9])[:, None, None]
    offsets = np.array([.01, .02, -.01, .005])[:, None, None]
    noisy = gains*clean+offsets
    target, coeffs = real.fit_gain(noisy, clean, np.ones((32, 48), bool), block=4)
    np.testing.assert_allclose(target, noisy, atol=1e-7)
    np.testing.assert_allclose(np.array(coeffs)[:, :2], np.c_[gains.ravel(), offsets.ravel()], atol=1e-7)
    with pytest.raises(ValueError, match="static blocks"):
        real.fit_gain(noisy, clean, np.zeros((32, 48), bool), block=4)


def test_temporal_nlf_recovers_poisson_gaussian_curve(helpers):
    real, _ = helpers
    rng = np.random.default_rng(17)
    mu = np.broadcast_to(np.linspace(.02, .7, 256), (4, 256, 256))
    K, read = .001, .008
    frames = rng.normal(mu, np.sqrt(K*mu+read**2), (12, *mu.shape))
    profiles = real.temporal_nlf(frames, block=8, bins=12)
    assert len(profiles) == 4
    for p in profiles:
        assert p["K"] == pytest.approx(K, rel=.04)
        assert p["read_std"] == pytest.approx(read, rel=.12)
        assert p["points"] == 12


def test_dark_profile_pair_difference_removes_fixed_pattern(helpers, monkeypatch):
    _, dark = helpers
    rng = np.random.default_rng(3)
    sigma, row_sigma, col_sigma = .002, .0006, .0004
    fixed_row = np.linspace(-.001, .001, 96)[None, :, None]
    frames = [.01+fixed_row+rng.normal(0, sigma, (4, 96, 112))
              +rng.normal(0, row_sigma, (4, 96, 1))
              +rng.normal(0, col_sigma, (4, 1, 112)) for _ in range(8)]
    monkeypatch.setattr(dark, "load", lambda i, *_: (frames[i], 2, 100., 10100.))
    p = dark.profile(list(range(8)))
    assert p["read_std"] == pytest.approx(sigma, rel=.04)
    assert p["row_std"] == pytest.approx(np.hypot(row_sigma, sigma/np.sqrt(112)), rel=.08)
    assert p["col_std"] == pytest.approx(np.hypot(col_sigma, sigma/np.sqrt(96)), rel=.08)
    assert p["black_offset"] == pytest.approx(.01, abs=5e-5)
    assert p["code_step"] == .0002 and p["frames"] == 8
    assert abs(p["lag1_h"]) < .04 and abs(p["lag1_v"]) < .04
    assert .08 < p["tukey_lambda"] < .22 and p["ppcc"] > .99
    assert p["hot_density"] == 0
    with pytest.raises(ValueError, match="two dark frames"):
        dark.profile([0])
    assert dark.lag1(np.zeros((8, 8))) == (0., 0.)


@pytest.mark.parametrize("legacy", [False, True])
def test_paired_cli_skips_group_without_static_blocks(helpers, tmp_path, monkeypatch, legacy):
    import json
    from denoise_lab.inference import Denoiser

    real, _ = helpers
    spec = tmp_path/"spec.json"
    spec.write_text(json.dumps({"root": str(tmp_path), "groups": [
        {"name": "clipped", "clean": ["c1", "c2"], "noisy": ["n1", "n2"], "shift": [0, 0]}]}))
    # Entire clean reference is clipped, so no brightness fit or model call is valid.
    monkeypatch.setattr(real, "load_packed", lambda path: (np.ones((4, 32, 32), np.float32), None))
    monkeypatch.setattr(Denoiser, "load", lambda *args, **kwargs: object())
    out = tmp_path/"output"
    monkeypatch.setattr(sys, "argv", ["eval_real_burst.py", "--spec", str(spec),
                                    "--model", "unused.pt", "--output", str(out)]
                        + (["--legacy-metrics"] if legacy else []))
    real.main()
    result = json.loads((out/"results.json").read_text())
    assert result["frames"] == []
    assert [(r["group"], r["static_fraction"]) for r in result["skipped"]] == [("clipped", 0.)]
    protocol = "paired-burst-v1" if legacy else "paired-burst-v2"
    assert result["metric_protocol"] == protocol
    assert json.loads((out/"alignment.json").read_text())["metric_protocol"] == protocol
    assert protocol in (out/"summary.md").read_text()


def test_masked_ssim_ignores_excluded_area(helpers):
    real, _ = helpers
    rng = np.random.default_rng(23)
    x, y = rng.uniform(0, 1, (2, 4, 24, 32))
    expected = real.ssim_np(x, y, 1.)
    for padding in (8, 32):
        pads = ((0, 0), (padding, padding), (padding, padding))
        mask = np.pad(np.ones(x.shape[-2:], bool), padding)
        # Arbitrary excluded values, including missing samples, cannot contribute.
        xp, yp = np.pad(x, pads, constant_values=np.nan), np.pad(y, pads, constant_values=.5)
        assert real.ssim_np(xp, yp, 1., mask) == pytest.approx(expected, abs=1e-14)
        assert real.ssim_np(xp, yp, 1., mask, legacy_metrics=True) > expected


def test_masked_ssim_requires_complete_windows(helpers):
    real, _ = helpers
    x = np.zeros((4, 24, 24))
    mask = np.zeros((24, 24), bool)
    mask[10:15, 10:15] = True
    assert real.ssim_np(x, x, 1., mask) is None
    mask[:] = True
    mask[12, 12] = False
    x[:, 12, 12] = np.nan
    assert real.ssim_np(x, np.zeros_like(x), 1., mask) == 1.


def test_legacy_ssim_matches_target_filled_reduction(helpers):
    import torch
    from denoise_lab.metrics import ssim

    real, _ = helpers
    x, y = np.zeros((4, 48, 48)), np.full((4, 48, 48), .5)
    mask = np.zeros((48, 48), bool)
    mask[16:32, 16:32] = True
    filled = np.where(mask, x, y).astype(np.float32)
    expected = ssim(torch.from_numpy(filled)[None], torch.from_numpy(y.astype(np.float32))[None])
    assert expected == pytest.approx(.6319990907280051, abs=1e-14)
    assert real.ssim_np(x, y, 1., mask, legacy_metrics=True) == expected
    assert real.ssim_np(x, y, 1., mask) == pytest.approx(.0001/.2501)


def test_prepare_pair_propagates_spatial_gain_and_legacy_variance(helpers, monkeypatch):
    real, _ = helpers
    clean = np.full((4, 32, 48), .2)
    frames = np.stack([clean-.01, clean+.01])
    gain = np.array([.8, .9, 1.1, 1.2])[:, None, None]
    local = np.linspace(.8, 1.2, 48)[None, None, :]
    gain_map = np.broadcast_to(gain*local, clean.shape).copy()
    target = gain_map*clean
    noisy = np.stack([target-.02, target+.02])
    coeffs = [[float(g), 0., .8, 1.2] for g in gain.ravel()]

    def known_fit(*args, return_gain=False):
        return (target, coeffs, gain_map) if return_gain else (target, coeffs)

    monkeypatch.setattr(real, "fit_gain", known_fit)
    corrected = real.prepare_pair(frames, noisy, (0, 0), block=4)
    legacy = real.prepare_pair(frames, noisy, (0, 0), block=4, legacy_metrics=True)
    assert corrected[3].all()
    np.testing.assert_allclose(corrected[2], gain_map**2*.0001, atol=1e-16)
    np.testing.assert_allclose(legacy[2], np.broadcast_to(gain**2*.0001, clean.shape), atol=1e-16)
    for i in (0, 1, 3, 4, 5):
        np.testing.assert_array_equal(corrected[i], legacy[i])


def test_fit_gain_returns_exact_local_derivative(helpers):
    real, _ = helpers
    rng = np.random.default_rng(18)
    clean = rng.uniform(.05, .4, (4, 48, 64))
    local = np.linspace(.8, 1.2, 64)[None, None, :]
    noisy = (1.1*clean+.02)*local
    static = np.ones(clean.shape[-2:], bool)
    target, coeffs, gain = real.fit_gain(noisy, clean, static, block=4, local=2, return_gain=True)
    old_target, old_coeffs = real.fit_gain(noisy, clean, static, block=4, local=2)
    np.testing.assert_array_equal(target, old_target)
    assert coeffs == old_coeffs
    global_gain = np.array([c[0] for c in coeffs])[:, None, None]
    offset = np.array([c[1] for c in coeffs])[:, None, None]
    np.testing.assert_allclose(gain, global_gain*target/(global_gain*clean+offset), atol=1e-15)
    assert np.ptp(gain[0]) > .01
