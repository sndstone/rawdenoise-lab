import itertools

import pytest
import torch

from denoise_lab.ensemble import bayer_self_ensemble


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3)))
def test_sensor_transforms_restore_all_sites_and_borders(flags):
    image = torch.arange(4*16*16, dtype=torch.float32).reshape(1, 4, 16, 16)/65535
    calls = []
    def identity(tile):
        calls.append(tile.shape)
        return tile
    result = bayer_self_ensemble(identity, image, ((False, False, False), flags))
    torch.testing.assert_close(result, image, rtol=0, atol=0)
    assert calls == [image.shape, image.shape]


def test_sensor_transpose_preserves_green_site_semantics():
    # Transpose swaps G1 and G2 geometrically. A plane-wise transpose would not.
    image = torch.ones(1, 4, 16, 16)
    offsets = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1)
    result = bayer_self_ensemble(lambda x: x+offsets, image)
    expected = torch.tensor([2., 4., 4., 4.]).reshape(1, 4, 1, 1).expand_as(image)
    torch.testing.assert_close(result, expected)


def test_missing_original_refused():
    with pytest.raises(ValueError, match="original prediction"):
        bayer_self_ensemble(lambda x: x, torch.ones(1, 4, 16, 16), ((True, True, False),))


def test_flip_prediction_position_and_missing_edge_are_mapped_back():
    # A row-dependent prediction proves inverse mapping beyond identity roundtrips.
    image = torch.zeros(1, 4, 4, 4)
    def row_model(tile):
        return torch.arange(4, dtype=tile.dtype)[None, None, :, None].expand_as(tile)
    result = bayer_self_ensemble(row_model, image, ((False, False, False), (True, False, False)))
    # Vertical flip drops the original first row, leaving baseline-only coverage.
    # R/G1 transformed rows map back 2,1,0 at original rows 1,2,3.
    expected_top = torch.tensor([0., 1.5, 1.5, 1.5])
    # B/G2 lose the original final row; rows 0,1,2 receive 2,1,0.
    expected_bottom = torch.tensor([1., 1., 1., 3.])
    torch.testing.assert_close(result[0, 0, :, 0], expected_top)
    torch.testing.assert_close(result[0, 1, :, 0], expected_top)
    torch.testing.assert_close(result[0, 2, :, 0], expected_bottom)
    torch.testing.assert_close(result[0, 3, :, 0], expected_bottom)
