import json

import numpy as np
import pytest
import torch

from denoise_lab.selfsup import RealNoisyDataset, neighbor_loss, neighbor_subsample


def test_all_neighbor_pairs_share_geometry_across_planes():
    x = torch.arange(4*4*4, dtype=torch.float32).reshape(1, 4, 4, 4)
    pairs = [(0, 1), (0, 4), (1, 0), (1, 5), (4, 0), (4, 5), (5, 1), (5, 4)]
    for i, (p, q) in enumerate(pairs):
        a, b = neighbor_subsample(x, torch.full((1, 2, 2), i))
        assert a.shape == b.shape == (1, 4, 2, 2)
        assert a.dtype == b.dtype == x.dtype
        for c in range(4):
            assert a[0, c, 0, 0] == x.flatten()[16*c+p]
            assert b[0, c, 0, 0] == x.flatten()[16*c+q]
        assert torch.all((a-b).abs() == abs(p-q))


def test_identity_consistency_is_zero_and_seed_is_repeatable():
    torch.manual_seed(8)
    x = torch.randn(2, 4, 32, 32)
    sigma = torch.tensor([.1, .2])
    model = torch.nn.Identity()
    torch.manual_seed(11)
    rec = neighbor_loss(model, x, sigma, gamma=0)
    torch.manual_seed(11)
    full = neighbor_loss(model, x, sigma, gamma=3)
    assert torch.equal(rec, full)
    assert torch.isfinite(neighbor_loss(model, torch.zeros_like(x), torch.zeros(2)))
    assert neighbor_loss(model, torch.ones_like(x), sigma) == 0


def test_zero_mean_noise_constant_predictor_beats_identity_and_backpropagates():
    torch.manual_seed(5)
    sigma = .1
    x = .4+sigma*torch.randn(4, 4, 128, 128)
    model = torch.nn.Conv2d(4, 4, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(.4)
    s = torch.full((4,), sigma)
    loss = neighbor_loss(model, x, s, gamma=0)
    identity = neighbor_loss(torch.nn.Identity(), x, s, gamma=0)
    assert loss.item() == pytest.approx(sigma/2, rel=.03)
    assert identity.item() == pytest.approx(sigma, rel=.03)
    loss.backward()
    assert model.bias.grad is not None and torch.isfinite(model.bias.grad).all()


@pytest.fixture
def manifest(tmp_path):
    a = np.arange(4*64*64, dtype=np.float32).reshape(4, 64, 64)/10000
    np.save(tmp_path/"crop.npy", a.astype(np.float16))
    entries = [{"camera": c, "iso": iso, "crop": "crop.npy", "sigma": .02}
               for c, iso in [("a", 100), ("a", 800), ("a", 1600), ("b", 1600)]]
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(entries))
    return path


def test_dataset_filter_crop_dtype_and_seed(manifest):
    ds = RealNoisyDataset(manifest, root=manifest.parent, crop_size=32, min_iso=800)
    assert len(ds) == 3
    torch.manual_seed(4)
    a = ds[0]
    torch.manual_seed(4)
    b = ds[0]
    assert a["noisy"].shape == (4, 32, 32)
    assert a["noisy"].dtype == a["sigma"].dtype == torch.float32
    assert a["noisy"].is_contiguous() and torch.equal(a["noisy"], b["noisy"])
    assert a["sigma"].item() == pytest.approx(.02)


def test_sampler_uses_camera_shares_not_crop_counts(manifest):
    ds = RealNoisyDataset(manifest, crop_size=32)
    sampler = ds.sampler(num_samples=100, camera_weights={"a": .3, "b": .7})
    np.testing.assert_allclose(sampler.weights, [.1, .1, .1, .7])
    np.testing.assert_allclose(ds.sampler().weights, [1/3, 1/3, 1/3, 1])
    np.testing.assert_allclose(ds.sampler(camera_weights={"b": 1}).weights, [0, 0, 0, 1])
    torch.manual_seed(9)
    a = list(sampler)
    torch.manual_seed(9)
    assert a == list(sampler)
    assert len(a) == 100


def test_dataset_rejects_empty_selection_and_invalid_crop(manifest):
    with pytest.raises(ValueError, match="No real noisy"):
        RealNoisyDataset(manifest, min_iso=100000)
    with pytest.raises(ValueError, match="multiple of 32"):
        RealNoisyDataset(manifest, crop_size=48)
