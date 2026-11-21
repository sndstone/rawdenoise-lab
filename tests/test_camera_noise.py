"""Synthetic calibration fixtures; read_std is a pair-difference MAD, not RMS."""
import json

import numpy as np
import pytest
import torch
import yaml

from denoise_lab.bayer import pack
from denoise_lab.camera_noise import CameraNoise, CameraNoiseDataset, _corr_kernel
from denoise_lab.rawstats import robust_std


def parameters(**updates):
    p = dict(K=.0002, read_std=.002, row_std=0., col_std=0., row_fixed_std=0.,
             col_fixed_std=0., tukey_lambda=.14, lag1_h=0., hot_density=0.,
             black_offset=0., code_step=1e-7, black_dn=1000., white_dn=11000.,
             quantization=None, processed=False)
    return {**p, **updates}


@pytest.fixture
def calibration(tmp_path):
    p = parameters()
    cameras = {name: {"iso": {100: p, 1600: {**p, "K": .0032, "read_std": .008},
                              6400: {**p, "K": .01}}, "processed_isos": [6400]}
               for name in ("a", "b")}
    path = tmp_path/"calibration.yaml"
    path.write_text(yaml.safe_dump({"units": "normalized", "cameras": cameras}))
    return path


def test_interpolation_clamping_and_seeded_sampling(calibration):
    cn = CameraNoise(calibration, {"a": 1}, K_jitter_log=0, read_jitter_log=0)
    p = cn.at("a", 400)
    assert p["K"] == pytest.approx(.0008)
    assert p["read_std"] == pytest.approx(.004)
    assert cn.at("a", 1)["K"] == pytest.approx(.0002)
    assert cn.at("a", 10000)["K"] == pytest.approx(.0032)
    assert cn.at("a", 6400, processed=True)["K"] == .01
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    assert [cn.sample(a) for _ in range(20)] == [cn.sample(b) for _ in range(20)]
    processed = CameraNoise(calibration, {"a": 1}, processed_probability=1).sample(a)
    assert processed["processed"] and processed["iso"] == 6400


def test_camera_shares_and_iso_range(calibration):
    cn = CameraNoise(calibration, {"a": 1, "b": 3}, processed_probability=0, iso_range=(100, 100))
    rng = np.random.default_rng(3)
    samples = [cn.sample(rng) for _ in range(1200)]
    assert np.mean([p["camera"] == "b" for p in samples]) == pytest.approx(.75, abs=.04)
    assert all(p["iso"] == pytest.approx(100) for p in samples)


@pytest.mark.parametrize("weights", [{"a": -1}, {"a": 0}])
def test_invalid_weights(calibration, weights):
    with pytest.raises(ValueError, match="weights"):
        CameraNoise(calibration, weights)


def test_calibration_units(calibration):
    calibration.write_text("units: DN\n")
    with pytest.raises(ValueError, match="normalized"):
        CameraNoise(calibration, {"a": 1})


@pytest.mark.parametrize("signal", [.02, .2, .6])
def test_variance_tracks_signal(calibration, signal):
    cn = CameraNoise(calibration, {"a": 1})
    p = parameters()
    clean = np.full((512, 512), signal, np.float32)
    out = cn.apply(clean, p, np.random.default_rng(4))
    assert out.shape == clean.shape and out.dtype == np.float32
    np.testing.assert_array_equal(out, cn.apply(clean, p, np.random.default_rng(4)))
    residual = out-clean
    assert residual.var() == pytest.approx(p["K"]*signal+p["read_std"]**2, rel=.035)
    assert abs(residual.mean()) < 5*np.sqrt(residual.var()/residual.size)


@pytest.mark.parametrize("correlation", [0., .2, .45])
def test_correlation_kernel(calibration, correlation):
    a = _corr_kernel(correlation)
    assert 2*a/(1+2*a*a) == pytest.approx(correlation)
    p = parameters(K=1e-12, lag1_h=correlation)
    out = pack(CameraNoise(calibration, {"a": 1}).apply(
        np.full((512, 512), .3), p, np.random.default_rng(10)))-.3
    h = np.corrcoef(out[..., 1:].ravel(), out[..., :-1].ravel())[0, 1]
    v = np.corrcoef(out[:, 1:].ravel(), out[:, :-1].ravel())[0, 1]
    assert h == pytest.approx(correlation, abs=.015)
    assert abs(v) < .015


@pytest.mark.parametrize("lam", [-.2, .14])
def test_pair_difference_scale_and_tails(calibration, lam):
    cn = CameraNoise(calibration, {"a": 1})
    p = parameters(K=1e-12, tukey_lambda=lam, read_std=.001)
    clean = np.full((512, 512), .3)
    a, b = [cn.apply(clean, p, np.random.default_rng(i))-clean for i in (21, 22)]
    assert robust_std((a-b)/np.sqrt(2)) == pytest.approx(p["read_std"], rel=.025)
    tail = np.mean(np.abs(a) > 5*p["read_std"])
    if lam < 0:
        assert tail > .004
    else:
        assert tail < .0001


def test_offsets_quantization_and_clipping(calibration):
    cn = CameraNoise(calibration, {"a": 1})
    p = parameters(K=1e-12, read_std=0., black_offset=.01, code_step=.001)
    out = cn.apply(np.full((16, 16), .2), p, np.random.default_rng(0))
    np.testing.assert_allclose(out, .21, atol=1e-7)
    out = cn.apply(np.full((16, 16), -1.), p, np.random.default_rng(0))
    np.testing.assert_allclose(out, -.1)
    out = cn.apply(np.full((16, 16), 2.), p, np.random.default_rng(0))
    np.testing.assert_allclose(out, 1.)
    p.update(quantization="sony_curve", black_dn=512., white_dn=16383.)
    out = cn.apply(np.full((16, 16), .3), p, np.random.default_rng(0))
    dn = out*(16383-512)+512
    np.testing.assert_allclose(dn/8, np.round(dn/8), atol=1e-4)
    with pytest.raises(ValueError, match="even-sized"):
        cn.apply(np.zeros((7, 8)), p, np.random.default_rng(0))


def test_dataset_binning_shapes_and_determinism(tmp_path, calibration):
    np.save(tmp_path/"clean.npy", np.full((128, 128), .2, np.float32))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps([{"clean": "clean.npy"}]))
    ds = CameraNoiseDataset(manifest, root=tmp_path, crop_size=16, augment=False,
                            bin_probability=1, bin_factors=(2,), deterministic=True,
                            calibration=calibration, cameras={"a": 1}, seed=3)
    a, b = ds[0], ds[0]
    for key, shape in (("noisy", (4, 16, 16)), ("clean", (4, 16, 16)), ("noise_map", (1, 16, 16))):
        assert a[key].shape == shape and a[key].dtype == torch.float32
        assert torch.equal(a[key], b[key])
    assert a["bin_factor"] == 2
    np.testing.assert_allclose(a["clean"], .2)


def test_row_column_offsets_and_hot_pixel_processing(calibration):
    cn = CameraNoise(calibration, {"a": 1})
    clean = np.full((512, 512), .2)
    p = parameters(K=1e-12, read_std=0., row_std=.001, row_fixed_std=.002,
                   col_std=.002, col_fixed_std=.003, black_offset=.01)
    out = cn.apply(clean, p, np.random.default_rng(31))-clean
    assert out.mean(1).std() == pytest.approx(np.hypot(.001, .002), rel=.08)
    assert out.mean(0).std() == pytest.approx(np.hypot(.002, .003), rel=.08)
    assert out.mean() == pytest.approx(.01, abs=.0007)
    p = parameters(K=1e-12, read_std=.001, hot_density=.02)
    hot = cn.apply(clean, p, np.random.default_rng(12))-clean
    processed = cn.apply(clean, {**p, "processed": True}, np.random.default_rng(12))-clean
    assert np.mean(hot > .006) > .008
    assert np.mean(processed > .006) < .0001


@pytest.mark.parametrize("sampling,expected", [("linear", .55), ("log", .9/np.log(10))])
def test_dataset_respects_exposure_sampling(tmp_path, calibration, sampling, expected):
    np.save(tmp_path/"clean.npy", np.full((32, 32), .2, np.float32))
    ds = CameraNoiseDataset([{"clean": "clean.npy"}], root=tmp_path, crop_size=8,
                            augment=True, exposure_range=(.1, 1.), exposure_sampling=sampling,
                            identity_probability=1, calibration=calibration, cameras={"a": 1})
    torch.manual_seed(23)
    samples = [ds[0] for _ in range(250)]
    assert np.mean([s["exposure_scale"] for s in samples]) == pytest.approx(expected, abs=.04)
    assert all(torch.equal(s["noisy"], s["clean"]) for s in samples)


def test_dataset_applies_configured_defects_only_to_input(tmp_path, calibration):
    np.save(tmp_path/"clean.npy", np.full((64, 64), .2, np.float32))
    kwargs = dict(root=tmp_path, crop_size=16, calibration=calibration, cameras={"a": 1},
                  deterministic=True, seed=5, identity_probability=1, exposure_range=(1, 1),
                  defect_probability=1, defect_options={"density_range": (.05, .05),
                                                       "kind_probs": {"hot": 1}, "hot_range": (.9, .9)})
    ds = CameraNoiseDataset([{"clean": "clean.npy"}], **kwargs)
    a = ds[0]
    np.testing.assert_allclose(a["clean"], .2)
    assert (a["noisy"] == .9).sum() > 10
    assert torch.equal(a["noisy"], ds[0]["noisy"])
    ds.augment = False
    np.testing.assert_allclose(ds[0]["noisy"], .2, atol=.05)


@pytest.fixture
def dn_calibration(tmp_path):
    normal = parameters(black_dn=512., white_dn=16383., code_step=1/15871)
    processed = {**normal, "black_dn": 1024., "code_step": 1/15359}
    for profile in (normal, processed):
        profile["physics"] = {"black_dn": profile["black_dn"], "white_dn": profile["white_dn"],
                              "code_step_dn": 1.}
    path = tmp_path/"dn.yaml"
    path.write_text(yaml.safe_dump({"units": "normalized", "cameras": {
        "sensor": {"iso": {100: normal, 1600: normal, 65535: processed}, "processed_isos": [65535]}}}))
    return path


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("physics", [False, True])
def test_processed_dn_metadata_floor_and_spacing(dn_calibration, sample, physics):
    cn = CameraNoise(dn_calibration, {"sensor": 1}, processed_probability=1,
                     physics_options={"quantization": True} if physics else None)
    p = cn.sample(np.random.default_rng(0)) if sample else cn.at("sensor", 65535, processed=True)
    assert p["black_dn"] == 1024
    assert p["white_dn"] == 16383
    assert p["code_step"] == 1/15359
    p.update(read_std=0., row_std=0., col_std=0., row_fixed_std=0., col_fixed_std=0.,
             black_offset=0., hot_density=0.)
    floor = cn.apply(np.full((8, 8), -1.), p, np.random.default_rng(0))
    np.testing.assert_allclose(floor, -1024/15359, atol=1e-8)
    ramp = np.arange(-64, 0).reshape(8, 8)/15359
    out = cn.apply(ramp, p, np.random.default_rng(0))
    np.testing.assert_allclose(np.diff(out.ravel()), 1/15359, atol=4e-10)


def test_normal_dn_compatibility_and_processed_boundaries(dn_calibration):
    cn = CameraNoise(dn_calibration, {"sensor": 1}, processed_probability=0)
    for p in [cn.at("sensor", iso) for iso in (1, 100, 400, 1600, 65535)]+[cn.sample(np.random.default_rng(3))]:
        assert (p["black_dn"], p["white_dn"], p["code_step"]) == (512., 16383., 1/15871)
        old = {**p, "black_dn": 512., "white_dn": 16383., "code_step": 1/15871}
        clean = np.full((16, 16), .2)
        np.testing.assert_array_equal(cn.apply(clean, p, np.random.default_rng(9)),
                                      cn.apply(clean, old, np.random.default_rng(9)))
    with pytest.raises(ValueError, match="processed mode"):
        cn.at("sensor", 1600, processed=True)


@pytest.mark.parametrize("physics", [False, True])
def test_discrete_dn_metadata_nearest_log_iso(calibration, physics):
    cal = yaml.safe_load(calibration.read_text())
    cal["cameras"]["a"]["iso"][1600].update(black_dn=2000., white_dn=12000., code_step=2e-7)
    calibration.write_text(yaml.safe_dump(cal))
    cn = CameraNoise(calibration, {"a": 1}, processed_probability=0,
                     physics_options={"quantization": True} if physics else None)
    for iso, black, step in ((1, 1000., 1e-7), (399, 1000., 1e-7), (400, 1000., 1e-7),
                             (401, 2000., 2e-7), (1600, 2000., 2e-7), (10000, 2000., 2e-7)):
        p = cn.at("a", iso)
        assert p["black_dn"] == black and p["code_step"] == step
        if physics:
            assert p["physics"]["black_dn"] == black
            assert p["physics"]["code_step_dn"] == step*(p["white_dn"]-black)
    assert cn.at("a", np.nextafter(400., 0.))["black_dn"] == 1000.
    assert cn.at("a", np.nextafter(400., np.inf))["black_dn"] == 2000.
    rng = np.random.default_rng(5)
    for _ in range(30):
        p = cn.sample(rng)
        expected = cn.at("a", p["iso"])
        assert all(p[key] == expected[key] for key in ("black_dn", "white_dn", "code_step"))


@pytest.mark.parametrize("key,value", [
    ("resite", True), ("iso_coupled", {"K_6400": .001}), ("noise_map_jitter", 9),
    ("shot_range", [1e-4, .01]), ("read_range", [1e-3, .01]), ("row_max", .01),
    ("fixed_parameters", {"K": .001}), ("column_relative_max", .5),
    ("impulse_range", [.01, .02]), ("impulse_sigma_range", [4, 40]), ("impulse_positive", .9),
])
def test_reject_unused_settings_before_manifest_io(key, value):
    with pytest.raises(ValueError, match=key):
        CameraNoiseDataset("missing.json", **{key: value})


def test_unused_defaults_accepted(calibration):
    CameraNoiseDataset([{"clean": "unused.npy"}], calibration=calibration, cameras={"a": 1},
                       resite=False, iso_coupled=None, noise_map_jitter=None,
                       shot_range=[1e-5, 1e-3], read_range=[1e-4, .004], row_max=.0005,
                       fixed_parameters=None, column_relative_max=0., impulse_range=None,
                       impulse_sigma_range=[3., 30.], impulse_positive=.6)


@pytest.mark.parametrize("defects", [0, 1])
def test_identity_map_fix_is_opt_in_and_matches_parent(tmp_path, calibration, defects):
    from denoise_lab.clean_targets import BinnedBootstrapDataset

    np.save(tmp_path/"clean.npy", np.full((32, 32), .2, np.float32))
    options = dict(root=tmp_path, crop_size=8, deterministic=True, seed=3,
                   identity_probability=1, exposure_range=(1, 1), defect_probability=defects)
    entries = [{"clean": "clean.npy"}]
    legacy = CameraNoiseDataset(entries, calibration=calibration, cameras={"a": 1}, **options)[0]
    fixed = CameraNoiseDataset(entries, calibration=calibration, cameras={"a": 1},
                               zero_identity_noise_map=True, **options)[0]
    parent = BinnedBootstrapDataset(entries, **options)[0]
    assert torch.count_nonzero(fixed["noise_map"]) == torch.count_nonzero(parent["noise_map"]) == 0
    assert torch.all(legacy["noise_map"] > 0)
    for key in ("noisy", "clean"):
        assert torch.equal(fixed[key], legacy[key])


def test_reject_unused_positional_settings_before_io():
    with pytest.raises(ValueError, match="shot_range"):
        CameraNoiseDataset("missing.json", "data", 128, True, 0, False, (.01, .1))
