import pytest
import torch
from denoise_lab.model import DenoiseNet, gamma, inverse_gamma, parameter_count


def test_gamma_roundtrip_and_gradients():
    x = torch.tensor([-2., -.1, -1e-6, 0., 1e-6, .1, 2.], requires_grad=True)
    y = inverse_gamma(gamma(x))
    torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-5)
    gamma(x).sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("gate", ["simplegate", "gelu"])
def test_forward_contract(gate):
    model = DenoiseNet(width=8, enc_blk_nums=[1, 1, 1, 1], middle_blk_num=1,
                       dec_blk_nums=[1, 1, 1, 1], gate=gate).eval()
    with torch.inference_mode():
        out = model(torch.rand(1, 4, 512, 512))
    assert out.shape == (1, 4, 512, 512) and out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_default_parameter_size():
    n = parameter_count(DenoiseNet(), verbose=False)
    assert 28_000_000 < n < 31_000_000


def test_conditioned_and_odd_shape():
    net = DenoiseNet(width=8, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1, cond="noise_map")
    x = torch.randn(1, 4, 19, 21)
    assert net(x, torch.ones(1, 1, 19, 21)*.01).shape == x.shape
    with pytest.raises(ValueError):
        net(x)
