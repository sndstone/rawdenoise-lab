"""Losses in linear and signed-gamma domains; no output clipping."""
import torch
from torch import nn
from .model import gamma


class DenoiseLoss(nn.Module):
    def __init__(self, kind="charbonnier", linear_weight=1., gamma_weight=1., fft_weight=0., eps=1e-3):
        super().__init__()
        if kind not in ("l1", "charbonnier"):
            raise ValueError("kind must be l1 or charbonnier")
        self.kind, self.eps = kind, eps
        self.linear_weight, self.gamma_weight, self.fft_weight = linear_weight, gamma_weight, fft_weight

    def distance(self, a, b):
        d = a-b
        return d.abs().mean() if self.kind == "l1" else (d.square()+self.eps**2).sqrt().mean()

    def forward(self, prediction, target):
        # FFT and signed powers in fp32 even under autocast.
        prediction, target = prediction.float(), target.float()
        loss = self.linear_weight*self.distance(prediction, target)
        if self.gamma_weight:
            loss = loss+self.gamma_weight*self.distance(gamma(prediction), gamma(target))
        if self.fft_weight:
            a, b = [torch.fft.rfft2(t, norm="ortho") for t in (prediction, target)]
            loss = loss+self.fft_weight*(a-b).abs().mean()
        return loss
