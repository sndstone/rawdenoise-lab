"""Additive architecture pilot; plain NAF state dictionaries remain compatible."""
from __future__ import annotations

import torch
from torch import nn
from .model import LayerNorm2d, SimpleGate, NAFBlock, NAFNet, GAMMA, gamma, inverse_gamma
from .attention import MDTA, WindowMSA


class AttnBlock(nn.Module):
    def __init__(self, c, mixer="mdta", heads=None, window=8):
        super().__init__()
        heads = max(1, c//32) if heads is None else heads
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        if mixer not in ("mdta", "win"):
            raise ValueError("mixer must be mdta or win")
        self.mixer = MDTA(c, heads) if mixer == "mdta" else WindowMSA(c, heads, window)
        self.conv4, self.conv5 = nn.Conv2d(c, 2*c, 1), nn.Conv2d(c, c, 1)
        self.gate = SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = x + self.beta*self.mixer(self.norm1(x))
        return y + self.gamma*self.conv5(self.gate(self.conv4(self.norm2(y))))


class StudyUNet(NAFNet):
    def __init__(self, in_channels=4, out_channels=4, width=32,
                 enc_blk_nums=(1, 1, 2), middle_blk_num=4, dec_blk_nums=(1, 1, 1),
                 block_type="naf", middle_pattern=None, heads=None, window=8, **kwargs):
        if block_type not in ("naf", "mdta"):
            raise ValueError("block_type must be naf or mdta")
        if middle_pattern is not None and len(middle_pattern) != middle_blk_num:
            raise ValueError("middle_pattern must describe every middle block")
        super().__init__(in_channels, out_channels, width, enc_blk_nums,
                         middle_blk_num, dec_blk_nums, **kwargs)
        def blocks(c, n, level, pattern=None):
            hs = (heads[level] if isinstance(heads, (list, tuple)) else heads)
            hs = max(1, c//32) if hs is None else hs
            types = pattern if pattern is not None else [block_type]*n
            if any(t not in ("naf", "mdta", "win") for t in types):
                raise ValueError("unknown block type")
            return nn.Sequential(*(NAFBlock(c, **kwargs) if t == "naf"
                                   else AttnBlock(c, t, hs, window) for t in types))
        if block_type != "naf":
            c = width
            for i, n in enumerate(enc_blk_nums):
                self.encoders[i] = blocks(c, n, i)
                c *= 2
            for i, n in enumerate(dec_blk_nums):
                c = width*2**(len(enc_blk_nums)-i-1)
                self.decoders[i] = blocks(c, n, len(enc_blk_nums)-i-1)
        if block_type != "naf" or middle_pattern is not None:
            self.middle_blks = blocks(width*2**len(enc_blk_nums), middle_blk_num,
                                      len(enc_blk_nums), middle_pattern)
        if middle_pattern and "win" in middle_pattern:
            self.multiple *= window


class StudyDenoiseNet(nn.Module):
    def __init__(self, cond="none", gamma_exponent=GAMMA, **body):
        super().__init__()
        if cond != "none":
            raise ValueError("StudyDenoiseNet supports cond=none")
        self.cond, self.gamma_exponent = cond, gamma_exponent
        self.body = StudyUNet(in_channels=4, out_channels=4, **body)

    def forward(self, x, noise_map=None):
        g = gamma(x, self.gamma_exponent)
        return inverse_gamma(self.body(g) + g, self.gamma_exponent)
