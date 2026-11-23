"""Track B losses for PhotoSRv2: compressive Charbonnier + linear + G1/G2 + FFT + highlight guard."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..model import GAMMA
from .geometry import align_planes, luminance, to_compressive


class SRv2Loss(nn.Module):
    """Multi-term camera-linear & compressive RAW SR loss."""

    def __init__(
        self,
        *,
        domain: str = "asinh",
        domain_scale: float = 0.05,
        gamma_exp: float = GAMMA,
        compressive_weight: float = 1.0,
        linear_weight: float = 0.25,
        g12_weight: float = 0.15,
        chroma_weight: float = 0.10,
        grad_weight: float = 0.12,
        fft_weight: float = 0.03,
        overshoot_weight: float = 0.40,
        eps: float = 1e-3,
        border: int = 8,
    ):
        super().__init__()
        self.domain = domain
        self.domain_scale = float(domain_scale)
        self.gamma_exp = float(gamma_exp)
        self.compressive_weight = float(compressive_weight)
        self.linear_weight = float(linear_weight)
        self.g12_weight = float(g12_weight)
        self.chroma_weight = float(chroma_weight)
        self.grad_weight = float(grad_weight)
        self.fft_weight = float(fft_weight)
        self.overshoot_weight = float(overshoot_weight)
        self.eps = float(eps)
        self.border = int(border)

    def _charb(self, diff: torch.Tensor) -> torch.Tensor:
        return (diff.square() + self.eps * self.eps).sqrt().mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        pred = pred.float()
        target = target.float()
        b = self.border
        if b > 0 and min(pred.shape[-2:]) > 2 * b + 2:
            p = pred[..., b:-b, b:-b]
            t = target[..., b:-b, b:-b]
        else:
            p, t = pred, target

        zp = to_compressive(p, domain=self.domain, scale=self.domain_scale, gamma_exp=self.gamma_exp)
        zt = to_compressive(t, domain=self.domain, scale=self.domain_scale, gamma_exp=self.gamma_exp)

        comp_loss = self._charb(zp - zt)
        lin_loss = self._charb(p - t)
        total = self.compressive_weight * comp_loss + self.linear_weight * lin_loss

        comps = {"compressive": float(comp_loss.detach()), "linear": float(lin_loss.detach())}

        if self.g12_weight > 0.0 or self.chroma_weight > 0.0:
            zpa = align_planes(zp)
            zta = align_planes(zt)
            if self.g12_weight > 0.0:
                g12_diff = (zpa[:, 1:2] - zpa[:, 3:4]) - (zta[:, 1:2] - zta[:, 3:4])
                g12_loss = self._charb(g12_diff)
                total = total + self.g12_weight * g12_loss
                comps["g12"] = float(g12_loss.detach())
            if self.chroma_weight > 0.0:
                c_diff = (zpa - luminance(zpa)) - (zta - luminance(zta))
                chroma_loss = self._charb(c_diff)
                total = total + self.chroma_weight * chroma_loss
                comps["chroma"] = float(chroma_loss.detach())

        if self.grad_weight > 0.0:
            dx = (zp[..., 1:] - zp[..., :-1]) - (zt[..., 1:] - zt[..., :-1])
            dy = (zp[..., 1:, :] - zp[..., :-1, :]) - (zt[..., 1:, :] - zt[..., :-1, :])
            grad_loss = 0.5 * (self._charb(dx) + self._charb(dy))
            total = total + self.grad_weight * grad_loss
            comps["grad"] = float(grad_loss.detach())

        if self.fft_weight > 0.0:
            # Multi-scale high-frequency Laplacian consistency (pure conv/pool: fast & MPS-safe)
            hp3_p = zp - F.avg_pool2d(F.pad(zp, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            hp3_t = zt - F.avg_pool2d(F.pad(zt, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            hp7_p = zp - F.avg_pool2d(F.pad(zp, (3, 3, 3, 3), mode="reflect"), 7, stride=1)
            hp7_t = zt - F.avg_pool2d(F.pad(zt, (3, 3, 3, 3), mode="reflect"), 7, stride=1)
            hf_loss = 0.5 * (self._charb(hp3_p - hp3_t) + self._charb(hp7_p - hp7_t))
            total = total + self.fft_weight * hf_loss
            comps["hf"] = float(hf_loss.detach())

        if self.overshoot_weight > 0.0:
            zt_pad = F.pad(zt, (1, 1, 1, 1), mode="replicate")
            z_hi = F.max_pool2d(zt_pad, 3, stride=1) + 0.003
            z_lo = -F.max_pool2d(-zt_pad, 3, stride=1) - 0.003
            over = F.relu(zp - z_hi)
            under = F.relu(z_lo - zp)
            neg_halo = F.relu(-p) * (t > 0.0005).float()
            white_blowup = F.relu(p - 1.0) * (t <= 1.0).float()
            excursion = (over.square() + under.square() + self.eps * self.eps).sqrt() - self.eps
            ov_loss = excursion.mean() + 10.0 * neg_halo.mean() + 6.0 * white_blowup.mean()
            total = total + self.overshoot_weight * ov_loss
            comps["overshoot"] = float(ov_loss.detach())

        comps["total"] = float(total.detach())
        if return_components:
            return total, comps
        return total
