import torch
import pytest
from denoise_lab.tiling import tiled_inference, reflected_pad
from denoise_lab.metrics import seam_test, flat_field_identity_test, evaluate_pair


@pytest.mark.parametrize("shape", [(1, 4, 53, 71), (1, 4, 512, 512), (1, 4, 701, 943), (1, 4, 1, 3)])
@pytest.mark.parametrize("power", [1., 2.])
def test_identity_partition(shape, power):
    x = torch.randn(shape)
    out, weights = tiled_inference(torch.nn.Identity(), x, feather_power=power, return_weights=True)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)
    assert torch.all(weights > 0)
    out = tiled_inference(torch.nn.Identity(), x, alpha=.3, feather_power=power)
    torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)


def test_reflect_not_replicate():
    x = torch.arange(4).reshape(1, 1, 1, 4)
    assert reflected_pad(x, 2, 2, 0, 0).flatten().tolist() == [2, 1, 0, 1, 2, 3, 2, 1]


def test_diagnostics_and_metrics():
    model = torch.nn.Identity()
    x = torch.rand(1, 4, 70, 80)
    assert seam_test(model, x, tile_size=32, overlap=8)["boundary_excess"] < 1e-6
    assert flat_field_identity_test(model)["max_abs_error"] == 0
    m = evaluate_pair(x, x)
    assert m["linear_ssim"] == pytest.approx(1.)
    assert m["srgb_ssim"] == pytest.approx(1.)
    assert m["linear_psnr"] == float("inf")
