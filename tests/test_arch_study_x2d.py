"""Fast synthetic contracts for the real-RAW pilot diagnostics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('arch_study_x2d', Path(__file__).resolve().parents[1]/'scripts/arch_study_x2d.py')
x2d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x2d)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_block_identity_dark_rest_and_discarded_edges():
    source = np.zeros((4, 135, 131), dtype=np.float32)
    source[:, 64:128, :128] = .1
    report = x2d.block_mean_diagnostics(source, source)
    assert report['discarded_edge_packed'] == [7, 3]
    for group in ('dark', 'rest'):
        for channel in x2d.CHANNELS:
            assert report[group][channel]['count'] == 2
            for key in ('mean_abs', 'median_abs', 'p99_abs', 'signed_mean'):
                assert report[group][channel][key] == 0


def test_block_signed_offsets_and_empty_group():
    source = np.zeros((4, 128, 128), dtype=np.float32)
    offsets = np.array([.002, -.003, .004, -.005], dtype=np.float32)
    report = x2d.block_mean_diagnostics(source, source+offsets[:, None, None])
    for i, channel in enumerate(x2d.CHANNELS):
        assert report['dark'][channel]['signed_mean'] == pytest.approx(offsets[i])
        assert report['dark'][channel]['p99_abs'] == pytest.approx(abs(offsets[i]))
        assert report['rest'][channel]['count'] == 0
        assert report['rest'][channel]['mean_abs'] is None
    with pytest.raises(ValueError):
        x2d.block_mean_diagnostics(source[:, :32], source[:, :32])


def test_shifted_grid_identity_on_irregular_rectangle():
    image = torch.randn(1, 4, 179, 213, generator=torch.Generator().manual_seed(3))
    report = x2d.shifted_grid_seam(torch.nn.Identity(), image, shift=16, radius=2,
        tile_size=64, overlap=10, feather_power=2.2, alpha=1.)
    assert report['max_abs_diff'] < 2e-6
    assert report['mean_abs_diff'] < 1e-6
    assert report['boundary_mean_abs_diff'] < 1e-6
    assert report['elsewhere_mean_abs_diff'] < 1e-6
    assert x2d.tile_count(179, 213, 64, 10) == 20


def test_boundary_mask_matches_apron_and_shift():
    mask = x2d.boundary_mask(200, 210, tile_size=64, overlap=10, shift=16, radius=2)
    # Baseline start 54-10=44; shifted start 54-10-16=28.
    assert mask[43:46].all()
    assert mask[27:30].all()
    assert mask.any() and (~mask).any()


def test_shifted_grid_detects_tile_context_bias():
    class TileMean(torch.nn.Module):
        def forward(self, image):
            return image+image.mean((-2, -1), keepdim=True)
    image = torch.arange(179*213, dtype=torch.float32).reshape(1, 1, 179, 213).expand(1, 4, -1, -1)/10000
    report = x2d.shifted_grid_seam(TileMean(), image, shift=16, radius=2,
        tile_size=64, overlap=10, feather_power=2.2, alpha=1.)
    assert report['mean_abs_diff'] > .001
    assert report['boundary_excess'] > 0


def test_residual_identity_and_outlier():
    source = np.zeros((4, 64, 64), dtype=np.float32)
    assert x2d.residual_diagnostics(source, source)['R']['std'] == 0
    prediction = source.copy()
    prediction[0, 1, 1] = 1
    report = x2d.residual_diagnostics(source, prediction)['R']
    assert report['fraction_abs_gt_5_robust_std'] == 1/4096
