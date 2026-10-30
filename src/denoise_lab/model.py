"""Independent NAFNet implementation of the published block architecture.

Architecture reference: Chen et al., Simple Baselines for Image Restoration
(arXiv:2204.04676). No pretrained parameters are used.
"""
import torch
from torch import nn
from torch.nn import functional as F

GAMMA = 0.4545


class _SignedPower(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent):
        ctx.save_for_backward(x)
        ctx.exponent = exponent
        return F.relu(x).pow(exponent) - F.relu(-x).pow(exponent)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        # The forward is exact; bound the singular slope at zero for training.
        return grad * ctx.exponent * x.abs().clamp_min(1e-6).pow(ctx.exponent-1), None


def gamma(x, exponent=GAMMA):
    if torch.jit.is_tracing() or not torch.is_grad_enabled():
        return F.relu(x).pow(exponent) - F.relu(-x).pow(exponent)
    return _SignedPower.apply(x, exponent)


def inverse_gamma(x, exponent=GAMMA):
    if torch.jit.is_tracing() or not torch.is_grad_enabled():
        return F.relu(x).pow(1.0/exponent) - F.relu(-x).pow(1.0/exponent)
    return _SignedPower.apply(x, 1.0/exponent)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # Accumulate statistics in fp32 under mixed precision.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = work.mean(dim=1, keepdim=True)
        var = (work-mean).square().mean(dim=1, keepdim=True)
        return ((work-mean) * torch.rsqrt(var+self.eps) * self.weight + self.bias).to(x.dtype)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a*b


class NAFBlock(nn.Module):
    def __init__(self, channels, gate="simplegate", dw_expand=2, ffn_expand=2, dropout=0.):
        super().__init__()
        if gate not in ("simplegate", "gelu"):
            raise ValueError("gate must be simplegate or gelu")
        dw, ff = channels*dw_expand, channels*ffn_expand
        if gate == "simplegate" and (dw % 2 or ff % 2):
            raise ValueError("SimpleGate needs even expanded widths")
        d_out, f_out = (dw//2, ff//2) if gate == "simplegate" else (dw, ff)
        self.norm1, self.norm2 = LayerNorm2d(channels), LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(d_out, channels, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(d_out, d_out, 1))
        # GELU retains the expanded channels.
        self.gate = SimpleGate() if gate == "simplegate" else nn.GELU()
        self.conv4, self.conv5 = nn.Conv2d(channels, ff, 1), nn.Conv2d(f_out, channels, 1)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        z = self.gate(self.conv2(self.conv1(self.norm1(x))))
        y = x + self.beta*self.drop1(self.conv3(z*self.sca(z)))
        z = self.conv5(self.gate(self.conv4(self.norm2(y))))
        return y + self.gamma*self.drop2(z)


class NAFNet(nn.Module):
    """Body produces gamma-domain correction; wrapper owns the global skip."""
    def __init__(self, in_channels=4, out_channels=4, width=32,
                 enc_blk_nums=(2, 2, 4, 8), middle_blk_num=12,
                 dec_blk_nums=(2, 2, 2, 2), gate="simplegate", dw_expand=2,
                 ffn_expand=2, dropout=0.):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("Encoder and decoder stage counts must agree")
        self.multiple = 2**len(enc_blk_nums)
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = [nn.ModuleList() for _ in range(4)]
        def blocks(c, n):
            return nn.Sequential(*(NAFBlock(c, gate, dw_expand, ffn_expand, dropout) for _ in range(n)))
        c = width
        for n in enc_blk_nums:
            self.encoders.append(blocks(c, n))
            self.downs.append(nn.Conv2d(c, c*2, 2, stride=2))
            c *= 2
        self.middle_blks = blocks(c, middle_blk_num)
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c*2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(blocks(c, n))

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % self.multiple, 0, (-h) % self.multiple))
        x, skips = self.intro(x), []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(up(x)+skip)
        return self.ending(x)[..., :h, :w]


class DenoiseNet(nn.Module):
    def __init__(self, cond="none", gamma_exponent=GAMMA, cond_transform="linear", **body):
        super().__init__()
        if cond not in ("none", "noise_map"):
            raise ValueError("cond must be none or noise_map")
        if cond_transform not in ("linear", "gamma"):
            raise ValueError("cond_transform must be linear or gamma")
        # "gamma" feeds gamma(std map), keeping ~1e-4..1e-1 linear stds in a
        # range comparable to the gamma-domain image features.
        self.cond, self.gamma_exponent, self.cond_transform = cond, gamma_exponent, cond_transform
        self.body = NAFNet(in_channels=4+(cond == "noise_map"), out_channels=4, **body)

    def forward(self, x, noise_map=None):
        g = gamma(x, self.gamma_exponent)
        features = g
        if self.cond == "noise_map":
            if noise_map is None:
                raise ValueError("Conditioned model requires a Bx1xHxW linear std map")
            m = noise_map.expand(x.shape[0], 1, *x.shape[-2:]).to(g.dtype)
            if self.cond_transform == "gamma":
                m = gamma(m.clamp_min(0), self.gamma_exponent)
            features = torch.cat((g, m), dim=1)
        out_g = self.body(features) + g
        return inverse_gamma(out_g, self.gamma_exponent)


def expand_unconditioned_state(state, model):
    """Adapt an unconditioned checkpoint to a noise_map model.

    The extra input channel of body.intro gets zero weights, so the expanded
    model initially ignores the map and reproduces the source model exactly.
    """
    key = "body.intro.weight"
    target = model.state_dict()[key]
    source = state[key]
    if source.shape == target.shape:
        return state
    if source.shape[0] != target.shape[0] or source.shape[1]+1 != target.shape[1] or source.shape[2:] != target.shape[2:]:
        raise ValueError(f"Cannot expand {tuple(source.shape)} to {tuple(target.shape)}")
    state = dict(state)
    state[key] = torch.cat((source, torch.zeros_like(source[:, :1])), dim=1)
    return state


def parameter_count(model, verbose=True):
    n = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Parameters: {n:,} ({n*2/1e6:.2f} MB in fp16)")
    return n
