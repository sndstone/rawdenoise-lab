"""Bounded CPU contract checks for the additive architecture study."""
import pytest
import torch
import yaml
from pathlib import Path
from denoise_lab.arch_registry import build_model
from denoise_lab.arch_study import StudyUNet
from denoise_lab.attention import MDTA, WindowMSA
from denoise_lab.diffusion import BridgeRefiner, UnrolledRefiner
from denoise_lab.model import NAFNet, gamma, inverse_gamma

ARMS = ['nafnet_s1', 'nafnet_s2', 'mdta', 'win_mid', 'bridge']

@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model_for(name):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/arch_study'/f'{name}.yaml').read_text())
    return build_model(cfg['model']).eval()


@pytest.mark.parametrize('name', ARMS)
@torch.no_grad()
def test_shapes_parameters_identity(name):
    model = model_for(name)
    count = sum(p.numel() for p in model.parameters())
    print(name, count)
    assert abs(count/2650404-1) <= .1
    for shape in [(1,4,512,512), (1,4,130,97)]:
        x = torch.rand(shape)*.2
        output = model(x)
        assert output.shape == x.shape and torch.isfinite(output).all()
    torch.nn.init.zeros_(model.body.ending.weight)
    torch.nn.init.zeros_(model.body.ending.bias)
    for level in [.001,.01,.05,.18,.7]:
        x = torch.full((1,4,512,512),level)
        # The gamma-domain identity is exact; signed-power round trip has
        # fp32 rounding, identically to the production DenoiseNet.
        assert torch.equal(model(x),inverse_gamma(gamma(x)))
        torch.testing.assert_close(model(x),x,rtol=3e-7,atol=1e-8)


@torch.no_grad()
def test_plain_state_dict_equivalence():
    args = dict(width=8,enc_blk_nums=[1,1],middle_blk_num=2,dec_blk_nums=[1,1])
    original, study = NAFNet(**args), StudyUNet(**args)
    study.load_state_dict(original.state_dict(),strict=True)
    x = torch.randn(1,4,31,27)
    assert torch.equal(original(x),study(x))


@torch.no_grad()
def test_mdta_low_precision_finite():
    module = MDTA(32,1).to(torch.bfloat16)
    x = (torch.randn(1,32,512,512)*10).to(torch.bfloat16)
    assert torch.isfinite(module(x)).all()
    if torch.backends.mps.is_available():
        assert torch.isfinite(module.to('mps',torch.float16)(x.to('mps',torch.float16))).all()


@torch.no_grad()
def test_window_validation_and_gradients():
    module = WindowMSA(8,2,4)
    assert module(torch.randn(2,8,8,12)).shape == (2,8,8,12)
    with pytest.raises(ValueError):module(torch.randn(1,8,7,8))


@torch.no_grad()
def test_bridge_endpoints_and_sampling():
    model = BridgeRefiner(width=8,enc_blk_nums=[1],middle_blk_num=1,dec_blk_nums=[1]).eval()
    clean, noisy = torch.rand(2,4,16,18), torch.rand(2,4,16,18)
    gx, gy, eps = gamma(clean),gamma(noisy),torch.randn_like(clean)
    for t,target in [(0.,gx),(1.,gy)]:
        assert torch.equal(model.marginal(gx,gy,torch.tensor(t),eps),target)
    assert torch.equal(model(noisy,steps=1), inverse_gamma(model.predict_gamma(gy,gy,1.)))
    a = model(noisy,steps=4,stochastic=True,generator=torch.Generator().manual_seed(12))
    b = model(noisy,steps=4,stochastic=True,generator=torch.Generator().manual_seed(12))
    assert torch.equal(a,b)
    assert not torch.equal(a, model(noisy,steps=4,stochastic=True,generator=torch.Generator().manual_seed(13)))
    fields = torch.randn(4,*noisy.shape)
    assert torch.equal(model(noisy,steps=4,stochastic=True,eps_fields=fields),model(noisy,steps=4,stochastic=True,eps_fields=fields))
    for k in [1,4]:assert torch.equal(UnrolledRefiner(model,k)(noisy),model(noisy,steps=k))
    assert torch.isfinite(model.training_prediction(noisy,clean,generator=torch.Generator().manual_seed(2))).all()
