import numpy as np
from denoise_lab.bayer import pack
from denoise_lab.clean_targets import bin_mosaic, apply_channel_gains


def test_bin_mosaic_keeps_channel_means_and_halves_noise():
    rng = np.random.default_rng(0)
    m = rng.normal(.1, .01, (544, 544)).astype(np.float32)
    m[0::2, 0::2] += .5
    b = bin_mosaic(m, 2)
    assert b.shape == (272, 272)
    assert abs(pack(b)[0].mean() - pack(m)[0].mean()) < 1e-5
    assert abs(pack(b)[1].std()/pack(m)[1].std() - .5) < .02


def test_channel_gains_attenuate_only_and_tie_greens():
    rng = np.random.default_rng(1)
    m = np.full((64, 64), .5, np.float32)
    for _ in range(20):
        g = pack(apply_channel_gains(m, rng))
        assert g.max() <= .5 + 1e-6
        assert np.allclose(g[1], g[3])


def test_resite_restores_native_offsets_on_a_ramp():
    from denoise_lab.clean_targets import resite_binned
    # A horizontal ramp sampled on a native mosaic: value = sensor x coordinate.
    xs = np.tile(np.arange(544, dtype=np.float32), (544, 1))
    b = bin_mosaic(xs, 4)
    r = resite_binned(b, 4)
    p = pack(r)[:, 8:-8, 8:-8]
    # Native geometry: G1 (and B) sit half a packed pixel (1 binned sensor px = 4 native px) right of R.
    assert abs(float((p[1] - p[0]).mean()) - 4.) < .05
    assert abs(float((p[3] - p[0]).mean())) < .05


def test_iso_coupled_parameters_scale_with_iso():
    from denoise_lab.clean_targets import iso_coupled_parameters
    rng = np.random.default_rng(3)
    lo = [iso_coupled_parameters(rng, iso_range=(100, 100), jitter_log=0)[0]["K"] for _ in range(3)]
    hi = [iso_coupled_parameters(rng, iso_range=(6400, 6400), jitter_log=0)[0]["K"] for _ in range(3)]
    assert np.allclose(lo, 6.7e-4/64) and np.allclose(hi, 6.7e-4)
