"""Synthetic CPU checks for backend-independent, normalized Bayer protection."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from denoise_lab.bayer import PHASES, normalize_raw, pack
from denoise_lab.defects import correct_defects
from denoise_lab.inference import Denoiser, protect_highlights
from denoise_lab.tiling import tiled_inference


def denoiser(call):
    model = Denoiser()
    model.device = 'cpu'
    model._call = call
    return model


def srgb_chroma(packed):
    # Bayer-cell RGB with fixed WB, deliberately below display clipping.
    rgb = np.stack([packed[0]*1.7, (packed[1]+packed[3])*.5, packed[2]*2.])*.6
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)
    return rgb-rgb.mean(axis=0, keepdims=True)


def test_clipped_boundary_chroma():
    y, x = np.mgrid[:64, :64]
    light = .03+1.5*np.exp(-((y-32)**2+(x-32)**2)/(2*12**2))
    clean = np.minimum(np.array([.55, 1., .4, 1.])[:, None, None]*light, 1).astype(np.float32)
    noisy = np.minimum(clean+np.random.default_rng(7).normal(0, .001, clean.shape), 1).astype(np.float32)
    # Simulate per-channel smoothing/bias at a clipped light's boundary.
    gains = torch.tensor([.9, 1., 1.1, 1.])[None, :, None, None]
    model = denoiser(lambda a: F.avg_pool2d(a, 3, stride=1, padding=1)*gains)
    off = model.denoise_packed(noisy, prepass=False, tile_size=64, overlap=0, highlights='off')
    on = model.denoise_packed(noisy, prepass=False, tile_size=64, overlap=0)
    core = torch.from_numpy((noisy >= .98).any(axis=0))[None, None].float()
    ring = (F.max_pool2d(core, 5, stride=1, padding=2) > 0)[0, 0].numpy() & ~core[0, 0].bool().numpy()
    target = srgb_chroma(clean)
    before = np.linalg.norm(srgb_chroma(off)-target, axis=0)[ring].mean()
    after = np.linalg.norm(srgb_chroma(on)-target, axis=0)[ring].mean()
    assert after < .007
    assert after < .25*before


@pytest.mark.parametrize('prepass,strength', [(False, 1.), (True, .4)])
def test_off_matches_legacy_exactly(prepass, strength):
    packed = np.random.default_rng(4).uniform(-.03, 1.03, (4, 23, 29)).astype(np.float32)
    model = denoiser(lambda a: a*.97+.02)
    original = torch.from_numpy(packed.copy())[None]
    image = correct_defects(original, k=8, abs_floor=.01, adjacency_ratio=.1)[0] if prepass else original
    legacy = tiled_inference(model._call, image, tile_size=32, overlap=8, feather_power=2.2)
    legacy = strength*legacy+(1-strength)*original
    result = model.denoise_packed(packed, prepass=prepass, strength=strength,
                                  tile_size=32, overlap=8, highlights='off')
    np.testing.assert_array_equal(result, legacy[0].numpy())


def test_unclipped_and_zero_strength_unchanged():
    packed = np.random.default_rng(3).uniform(-.1, .97, (4, 19, 21)).astype(np.float32)
    model = denoiser(lambda a: a*2)  # Even overshoot is unchanged away from clipping.
    on = model.denoise_packed(packed, tile_size=32, overlap=8)
    off = model.denoise_packed(packed, tile_size=32, overlap=8, highlights='off')
    np.testing.assert_array_equal(on, off)
    packed[:, 4:8, 4:8] = 1.01
    np.testing.assert_array_equal(model.denoise_packed(packed, strength=0), packed)
    with pytest.raises(ValueError, match='highlights'):
        model.denoise_packed(packed, highlights='invalid')


@pytest.mark.parametrize('phase', list(PHASES))
def test_per_channel_white_and_cfa_phase(phase):
    # Unequal black/white levels, including distinct green channels.
    indices = {'R': 0, 'G': 1, 'B': 2}
    pattern = np.array([indices[c] for c in phase]).reshape(2, 2)
    gy, gx = np.argwhere(pattern == 1)[1]
    pattern[gy, gx] = 3
    colors = np.tile(pattern, (24, 24))
    black = np.array([100, 200, 300, 400], dtype=np.float32)
    white = np.array([1100, 2200, 4300, 8400], dtype=np.float32)
    normalized = np.full((48, 48), .2, np.float32)
    normalized[20:26, 20:26][colors[20:26, 20:26] == 1] = 1
    raw = SimpleNamespace(raw_image_visible=normalized*(white[colors]-black[colors])+black[colors],
                          raw_colors_visible=colors, black_level_per_channel=black,
                          camera_white_level_per_channel=white, white_level=8400, color_desc=b'RGBG')
    packed = pack(normalize_raw(raw))
    original = torch.from_numpy(packed)[None]
    result = protect_highlights(original*.7, original)[0].numpy()
    core = (packed >= .98).any(axis=0)
    assert core.any()
    np.testing.assert_array_equal(result[:, core], packed[:, core])


def test_feather_footprint_and_white_ceiling():
    original = torch.full((1, 4, 15, 15), .2)
    original[0, 1, 7, 7] = 1.01
    output = torch.full_like(original, 1.2)
    result = protect_highlights(output, original)
    assert result[0, 1, 7, 7] == 1
    np.testing.assert_array_equal(result[0, [0, 2, 3], 7, 7], original[0, [0, 2, 3], 7, 7])
    assert (result[..., 3:12, 3:12] <= 1).all()
    area = result[..., 3:12, 3:12]
    assert (area[original[..., 3:12, 3:12] < 1] < 1).all()
    np.testing.assert_array_equal(result[..., :3, :], output[..., :3, :])
    # Shared weights of 1, .75, .5, .25, 0 at increasing distances.
    soft = protect_highlights(torch.zeros_like(original), original)
    np.testing.assert_allclose(soft[0, 0, 7, 8:13], [.2, .15, .1, .05, 0], atol=1e-7)


def test_protection_does_not_restore_prepass_defects():
    packed = np.full((4, 32, 32), .2, np.float32)
    packed[:, 16:24, 16:24] = 1  # A real highlight shared across adjacent sites.
    packed[0, 6, 6] = 1  # Isolated saturated defect, far from that highlight.
    packed[0, 16, 14] = 1  # Another isolated defect inside the feather footprint.
    model = denoiser(lambda a: a)
    on = model.denoise_packed(packed, tile_size=32, overlap=0)
    assert on[0, 6, 6] == packed[1, 6, 6]
    assert on[0, 16, 14] == packed[1, 16, 14]
    np.testing.assert_array_equal(on[:, 18:22, 18:22], packed[:, 18:22, 18:22])
    no_prepass = model.denoise_packed(packed, prepass=False, tile_size=32, overlap=0)
    assert no_prepass[0, 6, 6] == 1
