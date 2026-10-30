import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from denoise_lab.bayer import pack, unpack, unify, augment_mosaic, random_crop, normalize_raw


def test_channel_order_and_roundtrip():
    a = np.tile([[11., 22.], [44., 33.]], (5, 6))
    p = pack(a)
    for i, value in enumerate([11, 22, 33, 44]):
        assert np.all(p[i] == value)
    np.testing.assert_array_equal(unpack(p), a)


@pytest.mark.parametrize("phase", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_unify(phase):
    values = dict(R=1, G=2, B=3)
    a = np.tile(np.array([values[c] for c in phase]).reshape(2, 2), (8, 8))
    p = pack(a, phase)
    assert np.all(p[0] == 1) and np.all(p[2] == 3)
    assert np.all(p[[1, 3]] == 2)
    assert all(n % 2 == 0 for n in unify(a, phase).shape)


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_augmentation_preserves_cfa(flags):
    a = np.tile([[1., 2.], [2., 3.]], (10, 12))
    out = pack(augment_mosaic(a, *flags))
    assert np.all(out[0] == 1) and np.all(out[2] == 3)
    assert np.all(out[[1, 3]] == 2)


def test_crop_aligned_and_raw_metadata():
    a = np.arange(1030*1032).reshape(1030, 1032)
    c = random_crop(a, 512, np.random.default_rng(0))
    y, x = divmod(c[0, 0], 1032)
    assert y % 2 == x % 2 == 0 and c.shape == (1024, 1024)
    colors = np.tile([[0, 1], [3, 2]], (3, 3))
    black = np.array([10, 20, 30, 40])
    white = np.array([110, 220, 330, 440])
    raw = SimpleNamespace(raw_image_visible=(black[colors]-.1*(white[colors]-black[colors])),
                          raw_colors_visible=colors, black_level_per_channel=black,
                          camera_white_level_per_channel=white, white_level=500, color_desc=b"RGBG")
    np.testing.assert_allclose(normalize_raw(raw), -.1, atol=1e-7)
