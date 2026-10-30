import numpy as np
import torch
import pytest

from denoise_lab.bootstrap import add_bootstrap_noise, capture_group, screen_crop, split_for_group
from denoise_lab.data import BootstrapDataset
from denoise_lab.model import LayerNorm2d


def test_capture_day_groups_prevent_adjacent_frame_split():
    a = capture_group("ILCE-7M3", "2025:03:29 12:00:00", "folder\\a.ARW")
    b = capture_group("ILCE-7M3", "2025:03:29 12:00:01", "backup\\b.ARW")
    assert a == b
    assert split_for_group(a) == split_for_group(b)
    groups = [capture_group("ILCE-7M3", f"2025:03:{day:02d} 12:00:00", "x") for day in range(1, 29)]
    train = {g for g in groups if split_for_group(g) == "train"}
    val = {g for g in groups if split_for_group(g) == "val"}
    assert train and val and not train.intersection(val)


def test_screen_rejects_black_clipped_and_nonfinite_regions():
    assert screen_crop(np.full((64, 64), .001)) is None
    assert screen_crop(np.full((64, 64), 1.)) is None
    assert screen_crop(np.full((64, 64), np.nan)) is None
    assert screen_crop(np.full((64, 64), .2)) is not None


def test_bootstrap_noise_has_expected_added_variance():
    clean = np.full((800, 1000), .2, np.float32)
    parameters = {"K": .0002, "read_std": .001, "row_std": 0.}
    noisy, _ = add_bootstrap_noise(clean, np.random.default_rng(42), fixed_parameters=parameters)
    residual = noisy-clean
    expected = parameters["K"]*.2 + parameters["read_std"]**2
    assert residual.var() == pytest.approx(expected, rel=.03)
    assert abs(residual.mean()) < 5*np.sqrt(expected/clean.size)


@pytest.mark.parametrize("size", [128, 256])
def test_cached_mosaic_supports_bayer_augmentation_and_larger_crops(tmp_path, size):
    path = tmp_path/"source.npy"
    np.save(path, np.tile(np.array([[.1, .2], [.2, .3]], np.float16), (272, 272)))
    dataset = BootstrapDataset([{"clean": path.name}], root=tmp_path, crop_size=size, augment=True,
                               deterministic=True)
    for _ in range(3):
        item = dataset[0]
        assert item["clean"].shape == item["noisy"].shape == (4, size, size)
        assert item["noise_map"].shape == (1, size, size)
        assert torch.isfinite(item["noisy"]).all()


def test_layernorm_accumulates_variance_without_half_precision_overflow():
    x = torch.tensor([1000., -1000.], dtype=torch.float16).reshape(1, 2, 1, 1)
    out = LayerNorm2d(2)(x)
    torch.testing.assert_close(out.float(), torch.tensor([1., -1.]).reshape(1, 2, 1, 1))


def test_bootstrap_identity_examples_preserve_input(tmp_path):
    np.save(tmp_path/"source.npy", np.full((544, 544), .2, np.float16))
    dataset = BootstrapDataset([{"clean": "source.npy"}], root=tmp_path, crop_size=128,
                               identity_probability=1.)
    sample = dataset[0]
    torch.testing.assert_close(sample["noisy"], sample["clean"])
    assert torch.count_nonzero(sample["noise_map"]) == 0


def test_shadow_exposure_precedes_noise_and_preserves_signed_reference(tmp_path):
    mosaic = np.tile(np.array([[.1, -.01], [.2, .3]], np.float16), (272, 272))
    np.save(tmp_path/"source.npy", mosaic)
    common = dict(root=tmp_path, crop_size=128, augment=False, deterministic=True,
                  fixed_parameters={"K": .0002, "read_std": .001, "row_std": 0.})
    base = BootstrapDataset([{"clean": "source.npy"}], **common)[0]
    shadow = BootstrapDataset([{"clean": "source.npy"}], exposure_scale=1/64, **common)[0]
    torch.testing.assert_close(shadow["clean"], base["clean"]/64)
    assert shadow["clean"].min() < 0
    assert shadow["exposure_scale"] == 1/64
    expected = torch.sqrt(.0002*shadow["noisy"].clamp_min(0).mean(0, keepdim=True)+.001**2)
    torch.testing.assert_close(shadow["noise_map"], expected)
    # Read noise stays in native sensor units rather than being attenuated too.
    residual = shadow["noisy"]-shadow["clean"]
    assert float(residual.std()) > .0008


def test_shadow_exposure_sampling_covers_stops_and_identity_stays_exact(tmp_path):
    np.save(tmp_path/"source.npy", np.full((544, 544), .2, np.float16))
    entries = [{"clean": "source.npy"}]*64
    dataset = BootstrapDataset(entries, root=tmp_path, deterministic=True,
                               exposure_range=(1/64, 1), exposure_sampling="log",
                               identity_probability=1.)
    samples = [dataset[i] for i in range(len(dataset))]
    scales = np.array([s["exposure_scale"] for s in samples])
    assert scales.min() < 1/32 and scales.max() > .5
    assert 1/64 <= scales.min() <= scales.max() <= 1
    for sample in samples:
        torch.testing.assert_close(sample["noisy"], sample["clean"])


@pytest.mark.parametrize("settings", [dict(exposure_range=(0, 1)),
    dict(exposure_range=(1, .5)), dict(exposure_sampling="unknown"), dict(exposure_scale=2),
    dict(bright_exposure_probability=1.1)])
def test_invalid_exposure_configuration_rejected(settings):
    with pytest.raises(ValueError):
        BootstrapDataset([{"clean": "unused.npy"}], **settings)


def test_bright_exposure_mix_retains_legacy_brightness_and_dark_examples(tmp_path):
    np.save(tmp_path/"source.npy", np.full((544, 544), .2, np.float16))
    dataset = BootstrapDataset([{"clean": "source.npy"}]*128, root=tmp_path,
                               deterministic=True, exposure_range=(1/64, 1),
                               exposure_sampling="log", bright_exposure_probability=.5,
                               identity_probability=1.)
    scales = np.array([dataset[i]["exposure_scale"] for i in range(len(dataset))])
    assert np.mean(scales >= .6) > .4
    assert np.mean(scales < .1) > .1
