"""Synthetic local-model invariance and area-corrected seam diagnostics."""
import numpy as np
import pytest
import torch

from denoise_lab.inference import Denoiser
from denoise_lab.metrics import seam_energy, seam_test
from denoise_lab.tiling import feather, shift_offsets, tile_starts, tiled_inference


def test_small_conv_agrees_across_grid_offsets():
    # A strictly local model should agree up to feathered tile-edge context.
    # This bound does not apply to NAFNet's tile-wide channel attention.
    image = torch.rand(1, 4, 71, 83, generator=torch.Generator().manual_seed(14))*.2
    model = torch.nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
    with torch.no_grad():
        model.weight.fill_(1/9)
    kwargs = dict(tile_size=32, overlap=12, feather_mode='cosine', feather_power=2.2)
    baseline = tiled_inference(model, image, **kwargs)
    for offset in (1, 7, (13, 3)):
        shifted = tiled_inference(model, image, grid_offset=offset, **kwargs)
        torch.testing.assert_close(baseline, shifted, rtol=0, atol=2e-5)


def test_origins_and_steps_align_to_network_stride():
    starts = tile_starts(1301, 512, 60, 128, 16)
    assert starts[0] == -192
    assert all(start % 16 == 0 for start in starts)
    assert all(b-a == 448 for a, b in zip(starts, starts[1:]))
    assert starts[-1]+512 >= 1301+64
    assert tile_starts(1301, 512, 60, 128+448, 16) == starts
    assert tile_starts(1301, 512, 60, -320, 16) == starts
    with pytest.raises(ValueError, match='grid_offset'):
        tile_starts(100, 512, 60, 8, 16)
    assert shift_offsets(512, 60, 16, 4) == [0, 112, 224, 336]


@pytest.mark.parametrize('overlap', [0, 8, 20, 31])
@pytest.mark.parametrize('mode', ['linear', 'cosine'])
def test_windows_are_positive_and_symmetric(overlap, mode):
    window = feather(32, overlap, 2.2, 'cpu', mode)
    assert torch.all(window > 0)
    torch.testing.assert_close(window, window.flip(-1), rtol=0, atol=0)
    torch.testing.assert_close(window, window.flip(-2), rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(1, 4, 1, 3), (2, 4, 51, 73)])
def test_aligned_shifted_identity_with_high_overlap(shape):
    image = torch.randn(shape, generator=torch.Generator().manual_seed(19))
    result, weights = tiled_inference(torch.nn.Identity(), image, tile_size=32, overlap=20,
        grid_alignment=4, grid_offset=(8, 12), feather_mode='cosine', return_weights=True)
    torch.testing.assert_close(result, image, rtol=1e-6, atol=1e-6)
    assert torch.all(weights > 0)


def test_seam_energy_corrects_for_area_and_detects_boundary_concentration():
    uniform = torch.ones(1, 4, 40, 50)
    kwargs = dict(tile_size=32, overlap=8, grid_offsets=(0, 5), radius=2)
    report = seam_energy(uniform, **kwargs)
    assert 0 < report['boundary_fraction'] < 1
    assert report['boundary_energy_ratio'] == pytest.approx(1)
    assert report['boundary_energy_fraction'] == pytest.approx(report['boundary_fraction'])
    # Inject only on a known start boundary: 24-8 = 16.
    concentrated = torch.zeros_like(uniform)
    concentrated[..., 16, :] = 2
    report = seam_energy(concentrated, **kwargs)
    assert report['boundary_energy'] > 0
    assert report['interior_energy'] == 0
    assert report['boundary_energy_fraction'] == 1
    assert report['boundary_energy_ratio'] is None
    assert report['max_abs_diff'] == 2
    assert report['mean_abs_diff'] == pytest.approx(2/40)


def test_seam_energy_handles_zero_error_and_empty_regions():
    report = seam_energy(torch.zeros(1, 4, 3, 5), tile_size=32, overlap=0, grid_offsets=(0,))
    assert report['boundary_energy'] is None
    assert report['interior_energy'] == 0
    assert report['boundary_energy_ratio'] is None
    assert report['boundary_energy_fraction'] is None
    report = seam_test(torch.nn.Identity(), torch.ones(1, 4, 31, 45), reference_offset=5,
                       tile_size=32, overlap=8)
    assert report['max_abs_diff'] < 1e-6


def test_shift_averaging_matches_explicit_grids_and_strength():
    denoiser = Denoiser()
    denoiser.device = 'cpu'
    denoiser._call = lambda tile: tile*.7+tile.mean((-2, -1), keepdim=True)*.2
    packed = np.random.default_rng(14).random((4, 31, 91), dtype=np.float32)
    image = torch.from_numpy(packed)[None]
    expected = sum(tiled_inference(denoiser._call, image, tile_size=64, overlap=0,
        grid_offset=offset, grid_alignment=16, feather_mode='cosine', feather_power=2.2)
        for offset in (0, 32))/2
    result = denoiser.denoise_packed(packed, prepass=False, tile_size=64, overlap=0,
        grid_alignment=16, feather_mode='cosine', tta_shift=2, strength=.6, highlights='off')
    np.testing.assert_array_equal(result, (.6*expected+.4*image)[0].numpy())
    for count in (0, 5, True, 1.5):
        with pytest.raises(ValueError, match='tta_shift'):
            denoiser.denoise_packed(packed, tile_size=64, overlap=0,
                                   grid_alignment=16, tta_shift=count)
