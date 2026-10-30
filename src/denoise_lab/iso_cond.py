"""Run a noise-map-conditioned model as a plain 4-plane callable, given the photo's ISO.

The map is computed inside forward() from the tile itself and our own X1D II NLF anchor
(K = 6.7e-4*ISO/6400, read = 6e-4*ISO/6400; docs/09), the same information a camera
pipeline has at inference: the ISO tag plus a per-camera noise calibration.
"""
import torch
from torch import nn


class IsoConditioned(nn.Module):
    def __init__(self, model, iso=1600., K_6400=6.7e-4, read_6400=6e-4):
        super().__init__()
        if getattr(model, "cond", None) != "noise_map":
            raise ValueError("IsoConditioned wraps a noise_map-conditioned model")
        self.model, self.K_6400, self.read_6400 = model, K_6400, read_6400
        self.set_iso(iso)

    def set_iso(self, iso):
        self.iso = float(iso)
        self.K = self.K_6400*self.iso/6400
        self.read = self.read_6400*self.iso/6400

    def forward(self, x):
        signal = x.clamp_min(0).mean(1, keepdim=True)
        std = (self.K*signal + self.read**2).sqrt()
        return self.model(x, std)
