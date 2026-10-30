"""Conditional Brownian bridge in signed-gamma space (sensor noise is observed)."""
from __future__ import annotations

import math
import torch
from torch import nn
from .model import GAMMA, gamma, inverse_gamma
from .arch_study import StudyUNet


class BridgeRefiner(nn.Module):
    def __init__(self, kappa=.02, t1_probability=.25, cond="none", gamma_exponent=GAMMA, **body):
        super().__init__()
        if cond != "none" or kappa < 0 or not 0 <= t1_probability <= 1:
            raise ValueError("invalid bridge configuration")
        self.kappa, self.t1_probability = kappa, t1_probability
        self.cond, self.gamma_exponent = cond, gamma_exponent
        self.body = StudyUNet(in_channels=9, out_channels=4, **body)

    def marginal(self, clean_g, noisy_g, t, eps):
        """Explicit marginal, including exact endpoints; tensors already gamma encoded."""
        return (1-t)*clean_g + t*noisy_g + self.kappa*(t*(1-t)).sqrt()*eps

    def predict_gamma(self, z, noisy_g, t):
        plane = torch.ones_like(z[:, :1])*t
        return self.body(torch.cat((z, noisy_g, plane), 1)) + z

    def training_prediction(self, noisy, clean, generator=None):
        shape = (noisy.shape[0], 1, 1, 1)
        t = torch.rand(shape, device=noisy.device, generator=generator)
        endpoint = torch.rand(shape, device=noisy.device, generator=generator) < self.t1_probability
        t = torch.where(endpoint, torch.ones_like(t), t)
        gy, gx = gamma(noisy, self.gamma_exponent), gamma(clean, self.gamma_exponent)
        eps = torch.randn(noisy.shape, device=noisy.device, dtype=noisy.dtype, generator=generator)
        z = self.marginal(gx, gy, t, eps)
        return inverse_gamma(self.predict_gamma(z, gy, t), self.gamma_exponent)

    def forward(self, x, noise_map=None, steps=1, stochastic=False, generator=None, eps_fields=None):
        if steps < 1 or int(steps) != steps:
            raise ValueError("steps must be a positive integer")
        if eps_fields is not None and not torch.jit.is_tracing():
            if tuple(eps_fields.shape) != (steps, *x.shape):
                raise ValueError("eps_fields must have shape (steps,B,4,H,W)")
        gy = gamma(x, self.gamma_exponent)
        z = gy
        for i in range(steps):
            t, lower = (steps-i)/steps, (steps-i-1)/steps
            estimate = self.predict_gamma(z, gy, t)
            # Special-case the endpoint: K=1 is exactly one prediction.
            z = estimate if lower == 0 else estimate + (lower/t)*(z-estimate)
            if stochastic and lower > 0:
                eps = (eps_fields[i].to(z) if eps_fields is not None else
                       torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator))
                z = z + self.kappa*math.sqrt(lower*(t-lower)/t)*eps
        return inverse_gamma(z, self.gamma_exponent)


class UnrolledRefiner(nn.Module):
    def __init__(self, refiner, steps):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.refiner, self.steps = refiner, steps

    def forward(self, x):
        return self.refiner(x, steps=self.steps)
