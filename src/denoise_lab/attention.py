"""Channel and non-shifted window cosine attention, without pretrained weights."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _cosine_vectors(x, scale=1.0, eps=1e-6):
    # No low-precision sum of squares over the spatial extent. RMS followed
    # by N**.25 bounds the Gram product before its N**-.5 rescale.
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rms = work.square().mean(dim=-1, keepdim=True).sqrt()
    return (work / (rms * scale + eps)).to(x.dtype)


class MDTA(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        if heads < 1 or c % heads:
            raise ValueError("heads must divide channels")
        self.heads = heads
        self.qkv = nn.Conv2d(c, 3*c, 1)
        self.dw = nn.Conv2d(3*c, 3*c, 3, padding=1, groups=3*c)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h*w
        q, k, v = [a.reshape(b, self.heads, c//self.heads, n)
                   for a in self.dw(self.qkv(x)).chunk(3, 1)]
        q = _cosine_vectors(q, n**.25)
        k = _cosine_vectors(k, n**.25)
        logits = (q @ k.transpose(-2, -1)) * n**-.5
        logits = logits * self.temperature.clamp(.1, 10)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        out = logits.softmax(-1) @ v
        return self.project(out.reshape(b, c, h, w))


class WindowMSA(nn.Module):
    """Pixel-unshuffle partitions spatial windows into channels.

    Fold batch, window grid and heads into the matmul batch: all explicit
    intermediates have rank <= 5 (including for batch sizes above one).
    No shifts, attention masks, or relative-position gathers are used.
    """
    def __init__(self, c, heads, window=8):
        super().__init__()
        if heads < 1 or c % heads or window < 1:
            raise ValueError("heads must divide channels and window must be positive")
        self.heads, self.window = heads, window
        self.qkv = nn.Conv2d(c, 3*c, 1)
        self.dw = nn.Conv2d(3*c, 3*c, 3, padding=1, groups=3*c)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.project = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        ws, heads = self.window, self.heads
        if not torch.jit.is_tracing() and (h % ws or w % ws):
            raise ValueError("spatial dimensions must be divisible by window")
        gh, gw, tokens, d = h//ws, w//ws, ws*ws, c//heads
        def partition(a):
            a = F.pixel_unshuffle(a, ws)  # B,C*T,gridH,gridW
            a = a.reshape(b, c*tokens, gh*gw).transpose(1, 2)
            return a.reshape(b*gh*gw, heads, d, tokens).transpose(-1, -2)
        q, k, v = [partition(a) for a in self.dw(self.qkv(x)).chunk(3, 1)]
        # Cosine over per-head channels rather than window positions.
        q = _cosine_vectors(q, d**.5)
        k = _cosine_vectors(k, d**.5)
        logits = (q @ k.transpose(-2, -1)) * self.temperature.clamp(.1, 10)
        logits = logits - logits.amax(-1, keepdim=True)
        out = (logits.softmax(-1) @ v).transpose(-1, -2)
        out = out.reshape(b, gh*gw, c*tokens).transpose(1, 2)
        out = F.pixel_shuffle(out.reshape(b, c*tokens, gh, gw), ws)
        return self.project(out)
