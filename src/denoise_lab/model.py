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
                 ffn_expand=2, dropout=0., stage_widths=None,
                 output_head="rggb", resample="standard"):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("Encoder and decoder stage counts must agree")
        if output_head not in ("rggb", "green_split"):
            raise ValueError("output_head must be rggb or green_split")
        if output_head == "green_split" and out_channels != 4:
            raise ValueError("green_split output_head requires out_channels == 4")
        if resample not in ("standard", "aa_shallow2"):
            raise ValueError("resample must be standard or aa_shallow2")
        n_stages = len(enc_blk_nums)
        if stage_widths is None:
            widths = [width * (2**i) for i in range(n_stages + 1)]
        else:
            widths = [int(w) for w in stage_widths]
            if len(widths) != n_stages + 1 or any(w <= 0 for w in widths):
                raise ValueError("stage_widths must have len(enc_blk_nums) + 1 positive entries")
        self.stage_widths = tuple(widths)
        self.output_head = output_head
        self.resample = resample
        self.multiple = 2**n_stages
        self.intro = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        if output_head == "green_split":
            self.ending = nn.Conv2d(widths[0], 3, 3, padding=1)
            self.ending_gdiff = nn.Conv2d(widths[0], 1, 3, padding=1, bias=False)
        else:
            self.ending = nn.Conv2d(widths[0], out_channels, 3, padding=1)
            self.ending_gdiff = None
        self.encoders, self.downs, self.ups, self.decoders = [nn.ModuleList() for _ in range(4)]
        self._deep_dtype = None
        def blocks(c, n):
            return nn.Sequential(*(NAFBlock(c, gate, dw_expand, ffn_expand, dropout) for _ in range(n)))
        for i, n in enumerate(enc_blk_nums):
            c_in, c_out = widths[i], widths[i + 1]
            self.encoders.append(blocks(c_in, n))
            if resample == "aa_shallow2" and i < 2:
                self.downs.append(nn.Sequential(nn.AvgPool2d(2, stride=2), nn.Conv2d(c_in, c_out, 3, padding=1)))
            else:
                self.downs.append(nn.Conv2d(c_in, c_out, 2, stride=2))
        self.middle_blks = blocks(widths[-1], middle_blk_num)
        for i, n in enumerate(dec_blk_nums):
            c_in, c_out = widths[n_stages - i], widths[n_stages - i - 1]
            if resample == "aa_shallow2" and i >= n_stages - 2:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(c_in, c_out, 3, padding=1),
                ))
            else:
                self.ups.append(nn.Sequential(nn.Conv2d(c_in, c_out * 4, 1, bias=False), nn.PixelShuffle(2)))
            self.decoders.append(blocks(c_out, n))

    def set_outer_fp32_deep_dtype(self, deep_dtype=None):
        """Keep full-res intro/encoders[0]/decoders[-1]/ending in fp32; cast deep stages to deep_dtype."""
        self._deep_dtype = deep_dtype
        if deep_dtype is None:
            self.float()
            return self
        self.intro.float()
        self.ending.float()
        if self.ending_gdiff is not None:
            self.ending_gdiff.float()
        self.encoders[0].float()
        self.decoders[-1].float()
        for mod in list(self.encoders[1:]) + list(self.downs) + [self.middle_blks] + list(self.ups) + list(self.decoders[:-1]):
            mod.to(deep_dtype)
        return self

    def forward(self, x):
        h, w = x.shape[-2:]
        # Traced deployment inputs must be divisible by the stage multiple.
        if h % self.multiple or w % self.multiple:
            x = F.pad(x, (0, (-w) % self.multiple, 0, (-h) % self.multiple))
        deep_dtype = self._deep_dtype
        x, skips = self.intro(x), []
        for i, (enc, down) in enumerate(zip(self.encoders, self.downs)):
            x = enc(x)
            skips.append(x)
            if deep_dtype is not None and i == 0:
                x = x.to(deep_dtype)
            x = down(x)
        x = self.middle_blks(x)
        last_dec = len(self.decoders) - 1
        for i, (up, dec, skip) in enumerate(zip(self.ups, self.decoders, reversed(skips))):
            x = up(x)
            if deep_dtype is not None and i == last_dec:
                x = x.float()
            x = dec(x + skip)
        if self.output_head == "green_split":
            base = self.ending(x)[..., :h, :w]
            gd = self.ending_gdiff(x)[..., :h, :w]
            # Approximate O(1) low-frequency trend suppression on dG_diff: subtracts a
            # bilinearly upsampled 16x16 block average (or global mean when h/w is not a
            # multiple of 16). Suppresses smooth/constant G1-G2 DC drift while keeping
            # high-frequency pixel-level G1-G2 response intact; because bilinear upsampling
            # blends adjacent 16x16 blocks, individual discontinuous blocks can retain
            # nonzero local means near step boundaries.
            if min(h, w) >= 16 and h % 16 == 0 and w % 16 == 0:
                dc = F.interpolate(F.avg_pool2d(gd, kernel_size=16, stride=16),
                                   size=(h, w), mode="bilinear", align_corners=False)
            else:
                dc = gd.mean(dim=(-2, -1), keepdim=True)
            gd = gd - dc
            r, g_sum, b = base[:, 0:1], base[:, 1:2], base[:, 2:3]
            return torch.cat((r, g_sum + gd, b, g_sum - gd), dim=1)
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
