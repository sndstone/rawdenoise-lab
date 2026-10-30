import numpy as np
import torch

from denoise_lab import rawstats as rs
from denoise_lab.bootstrap import add_bootstrap_noise
from denoise_lab.bayer import pack
from denoise_lab.model import DenoiseNet, expand_unconditioned_state
from denoise_lab.noise import noise_std

SMALL = dict(width=8, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1))


def _legacy_noise(clean, rng, shot_range=(1e-5, 1e-3), read_range=(1e-4, .004), row_max=.0005):
    K = float(np.exp(rng.uniform(*np.log(shot_range))))
    read = float(np.exp(rng.uniform(*np.log(read_range))))
    row = float(rng.uniform(0, row_max))
    positive = np.maximum(clean, 0)
    noisy = rng.poisson(positive/K)*K + clean-positive
    noisy += rng.normal(0, read, clean.shape)
    noisy += rng.normal(0, row, (clean.shape[0], 1))
    return noisy.astype(np.float32), {"K": K, "read_std": read, "row_std": row}


def test_default_noise_stream_is_unchanged():
    clean = np.random.default_rng(1).uniform(0, .5, (64, 64)).astype(np.float32)
    a, pa = add_bootstrap_noise(clean, np.random.default_rng(7))
    b, pb = _legacy_noise(clean, np.random.default_rng(7))
    assert pa == pb
    np.testing.assert_array_equal(a, b)


def test_impulses_and_columns_are_added():
    clean = np.full((512, 512), .01, np.float32)
    fixed = {"K": 1e-6, "read_std": 1e-3, "row_std": 0., "col_std": 5e-4, "impulse_p": .01}
    noisy, p = add_bootstrap_noise(clean, np.random.default_rng(0), fixed_parameters=fixed)
    z = (noisy-clean)/1e-3
    tail = np.mean(np.abs(z) > 5)
    assert .003 < tail < .02 and p["impulse_p"] == .01
    col_profile = (noisy-clean).mean(0)
    assert col_profile.std() > 3e-4  # columns dominate the 512-sample mean of white noise
    _, p2 = add_bootstrap_noise(clean, np.random.default_rng(0), column_relative_max=.15, impulse_range=(1e-4, 1e-2))
    assert 0 <= p2["col_std"] <= .15*p2["read_std"] and 1e-4 <= p2["impulse_p"] <= 1e-2


def test_noise_std_includes_columns():
    base = noise_std({"K": 0., "read_std": 3e-3})
    with_col = noise_std({"K": 0., "read_std": 3e-3, "col_std": 4e-3})
    assert np.isclose(with_col, 5e-3) and np.isclose(base, 3e-3)


def test_expanded_noise_map_model_matches_unconditioned():
    torch.manual_seed(0)
    blind = DenoiseNet(**SMALL).eval()
    for p in blind.parameters():
        torch.nn.init.normal_(p, std=.05)
    cond = DenoiseNet(cond="noise_map", cond_transform="gamma", **SMALL).eval()
    cond.load_state_dict(expand_unconditioned_state(blind.state_dict(), cond))
    x = torch.rand(2, 4, 32, 32)*.3
    with torch.inference_mode():
        y0 = blind(x)
        for m in (torch.zeros(2, 1, 32, 32), torch.rand(2, 1, 32, 32)*.05):
            torch.testing.assert_close(cond(x, m), y0, rtol=0, atol=0)
    assert cond.body.intro.weight[:, 4].abs().max() == 0


def test_blind_estimator_recovers_flat_noise():
    rng = np.random.default_rng(3)
    clean = np.tile(np.linspace(.001, .2, 1024, dtype=np.float32), (1024, 1))
    noisy, _ = add_bootstrap_noise(clean, rng, fixed_parameters={"K": 4e-4, "read_std": 2e-3, "row_std": 0.})
    est = rs.estimate_noise_parameters(pack(noisy), quantile=.5)
    assert abs(est["K"]/4e-4-1) < .25
    assert abs(est["read_std"]/2e-3-1) < .25


def test_rawstats_helpers():
    a = np.random.default_rng(0).normal(size=(2, 40, 50)).astype(np.float32)
    med = rs.median3(a)
    p = np.pad(a[0], 1, mode="edge")
    assert np.isclose(med[0, 10, 10], np.median(p[10:13, 10:13]))
    np.testing.assert_allclose(rs.box_mean(np.ones((1, 9, 9)), 5), 1.)
    white = np.random.default_rng(1).normal(size=(1, 256, 256))
    assert .7 < rs.line_structure(white, -1)["ratio"][0] < 1.4
    rows = white+np.random.default_rng(2).normal(size=(1, 256, 1))
    assert rs.line_structure(rows, -1)["ratio"][0] > 5
    mu, diff, z, valid = rs.block_mean_shift(np.abs(white)*.1, np.abs(white)*.1, block=32)
    assert np.all(diff == 0) and valid.all()
    stats = rs.dark_region_stats(np.random.default_rng(4).normal(100, 2, (64, 256)), 100, 1000)
    assert abs(stats["robust_std"]-.002) < 2e-4 and stats["frac_gt_5sigma_pos"] < 1e-3
