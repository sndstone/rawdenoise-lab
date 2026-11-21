"""Fast synthetic checks for opt-in dark physics and exact legacy RNG compatibility."""
import hashlib

import numpy as np
import pytest
import yaml

from denoise_lab.bayer import pack
from denoise_lab.camera_noise import CameraNoise
from denoise_lab.noise import NoiseModel
from denoise_lab.physics_noise import banding, correlate, dark_statistics, fit_physics, moment_read, quantize_dn


def test_legacy_noise_bytes():
    cfg = {"units": "normalized", "iso": {1600: {"K": .002, "read_std": .01, "row_std": .003,
           "quant_step": .0001, "read_distribution": "tukey", "tukey_lambda": .14}}}
    for options in (None, {}, {"banding": False, "quantization": False, "horizontal_correlation": False}):
        model = NoiseModel(cfg, seed=194, physics_options=options)
        actual = model(np.full((64, 96), .012, np.float32), 1600, sample=False)
        # Captured from the parent commit, before any implementation changes.
        assert hashlib.sha256(actual.tobytes()).hexdigest() == "158cdc8474633bdaecafc6bfdb29b84038f2aa2c57bc45dd4271fc4277a76e71"


def test_legacy_camera_bytes():
    for options in (None, {}, {"banding": False, "quantization": False, "horizontal_correlation": False}):
        model = CameraNoise("calib/darks_20261101.yaml", {"a7m3": 1}, physics_options=options)
        rng = np.random.default_rng(194)
        p = model.sample(rng)
        actual = model.apply(np.full((64, 96), .012, np.float32), p, rng)
        assert hashlib.sha256(actual.tobytes()).hexdigest() == "00e28a6f01d4529a4307482facc23d34255c6d326a0083e2187d7de8da184ca1"


def test_fixed_and_temporal_banding():
    rng = np.random.default_rng(31)
    p = {"row_std": .02, "col_std": .01, "row_fixed_std": .03, "col_fixed_std": .015}
    stack = np.stack([pack(banding((512, 768), p, rng, "body:leaf")) for _ in range(12)])
    stats = dark_statistics(stack)
    for key, std in (("row_power", .02), ("col_power", .01), ("row_fixed_power", .03), ("col_fixed_power", .015)):
        assert stats[key] == pytest.approx(std**2, rel=.12)
    fixed = {**p, "row_std": 0., "col_std": 0.}
    a = banding((64, 96), fixed, rng, "body:leaf")
    b = banding((64, 96), fixed, np.random.default_rng(9), "body:leaf")
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, banding((64, 96), fixed, rng, "body:electronic"))


@pytest.mark.parametrize("correlation", [-.15, .0, .2])
def test_same_colour_correlation(correlation):
    read = np.random.default_rng(11).normal(size=(4, 256, 512))
    out = correlate(read, correlation)
    actual = np.mean(out[..., 1:]*out[..., :-1])/out.var()
    assert actual == pytest.approx(correlation, abs=.008)
    assert out.var() == pytest.approx(1., rel=.015)


def test_dn_quantisation_and_bit_depth():
    p = {"black_dn": 512., "white_dn": 16383., "code_step_dn": 4.}
    signal = np.random.default_rng(3).uniform(-.01, .5, (64, 96))
    out = quantize_dn(signal, p)
    dn = out*(16383-512)+512
    np.testing.assert_allclose(dn/4, np.round(dn/4), atol=1e-10)
    assert np.gcd.reduce(np.diff(np.unique(np.rint(dn).astype(int)))) == 4
    from_bits = quantize_dn(signal, {"black_dn": 512., "white_dn": 16383., "effective_bits": 12})
    np.testing.assert_array_equal(out, from_bits)
    assert np.max(np.abs(out-signal)) <= 2/(16383-512)+1e-12
    clipped = quantize_dn(np.array([-1., 2.]), p)*(16383-512)+512
    np.testing.assert_array_equal(clipped, [0., 16380.])
    model = NoiseModel({"units": "normalized", "iso": {100: {**p, "K": .001, "read_std": .001}}},
                       seed=3, physics_options={"quantization": True})
    codes = model(np.zeros((64, 96), np.float32), 100, sample=False)*(16383-512)+512
    np.testing.assert_allclose(codes/4, np.round(codes/4), atol=2e-5)


def test_moment_read_and_fitter():
    rng = np.random.default_rng(7)
    p = {"row_std": .006, "col_std": .003, "row_fixed_std": .01, "col_fixed_std": .008}
    stack = []
    for _ in range(12):
        read = moment_read((4, 128, 192), .02**2, 6., rng)
        stack.append(read+pack(banding((256, 384), p, rng)))
    fit = fit_physics(stack, 1., 512., 65535.)
    assert fit["read_variance"] == pytest.approx(.02**2, rel=.05)
    assert fit["read_excess_kurtosis"] == pytest.approx(6., rel=.2)
    for key, value in p.items():
        assert fit[key] == pytest.approx(value, rel=.15)
    with pytest.raises(ValueError, match="at least two"):
        dark_statistics(stack[:1])


def test_shutter_profiles_and_camera_components(tmp_path):
    base = {"K": .001, "read_std": .01, "row_std": 0., "col_std": 0., "row_fixed_std": 0.,
            "col_fixed_std": 0., "tukey_lambda": .14, "lag1_h": 0., "hot_density": 0.,
            "black_offset": 0., "code_step": 4/15871, "black_dn": 512., "white_dn": 16383.}
    leaf = {**base, "physics": {"read_variance": .002**2, "read_excess_kurtosis": 0.,
            "lag1_h": .2, "row_std": .001, "col_std": .0005, "row_fixed_std": .002,
            "col_fixed_std": .001, "code_step_dn": 4.}}
    electronic = {**leaf, "physics": {**leaf["physics"], "read_variance": .001**2}}
    path = tmp_path/"cal.yaml"
    path.write_text(yaml.safe_dump({"units": "normalized", "cameras": {"body": {"iso": {100: base},
        "shutters": {"leaf": {"iso": {100: leaf}}, "electronic": {"iso": {100: electronic}}}}}}))
    options = {"banding": True, "quantization": True, "horizontal_correlation": True, "moment_read": True}
    model = CameraNoise(path, {"body": 1}, physics_options=options, shutter_modes={"body": "leaf"})
    p = model.at("body", 100)
    assert p["physics"]["read_variance"] == leaf["physics"]["read_variance"]
    other = CameraNoise(path, {"body": 1}, physics_options=options, shutter_modes={"body": "electronic"})
    assert other.at("body", 100)["physics"]["read_variance"] == electronic["physics"]["read_variance"]
    rng = np.random.default_rng(12)
    frames = [pack(model.apply(np.zeros((256, 384)), p, rng)) for _ in range(12)]
    stats = dark_statistics(frames)
    assert stats["lag1_h"] == pytest.approx(.2, abs=.02)
    assert stats["row_fixed_power"] == pytest.approx(.002**2, rel=.2)
