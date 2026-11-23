"""Losses in linear, signed-gamma, and WB-normalized opponent Bayer domains; no output clipping."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import gamma


def _wb_gains(target, white_balance=None, eps=1e-4, clamp_range=(0.25, 4.0)):
    """Per-sample 4-plane gains [B,4,1,1] normalizing R and B to G (G1=G2=1).

    When white_balance is None, derives detached gray-world gains from target,
    falling back to unity on near-black crops so noise is never amplified.
    """
    if white_balance is not None:
        wb = torch.as_tensor(white_balance, device=target.device, dtype=target.dtype)
        if wb.ndim == 1:
            wb = wb[None]
        g = wb[:, 1:2].clamp_min(eps)
        r = (wb[:, 0:1] / g).clamp(*clamp_range)
        b = (wb[:, 2:3] / g).clamp(*clamp_range)
        ones = torch.ones_like(r)
        return torch.cat((r, ones, b, ones), dim=1)[..., None, None]
    means = target.detach().clamp_min(0).mean(dim=(-2, -1), keepdim=True)
    green = 0.5 * (means[:, 1:2] + means[:, 3:4])
    valid = green >= eps
    r = torch.where(valid, (green / means[:, 0:1].clamp_min(eps)).clamp(*clamp_range), torch.ones_like(green))
    b = torch.where(valid, (green / means[:, 2:3].clamp_min(eps)).clamp(*clamp_range), torch.ones_like(green))
    ones = torch.ones_like(green)
    return torch.cat((r, ones, b, ones), dim=1)


def _opponent_chroma(packed):
    """Opponent (Cb, Cr) in packed R/G1/B/G2 order: Y=(R+G1+G2+B)/4, Cb=B-Y, Cr=R-Y."""
    y = 0.25 * (packed[:, 0] + packed[:, 1] + packed[:, 2] + packed[:, 3])
    return torch.stack((packed[:, 2] - y, packed[:, 0] - y), dim=1)


class DenoiseLoss(nn.Module):
    def __init__(self, kind="charbonnier", linear_weight=1., gamma_weight=1., fft_weight=0., eps=1e-3,
                 chroma_weight=0., chroma_lowpass_weight=0., green_balance_weight=0.,
                 chroma_lowpass_kernel=4, highlight_chroma_weight=0., highlight_threshold=0.75,
                 green_dc_weight=0., green_dc_kernel=16):
        super().__init__()
        if kind not in ("l1", "charbonnier"):
            raise ValueError("kind must be l1 or charbonnier")
        if any(w < 0 for w in (linear_weight, gamma_weight, fft_weight,
                               chroma_weight, chroma_lowpass_weight, green_balance_weight,
                               highlight_chroma_weight, green_dc_weight)):
            raise ValueError("Loss weights must be nonnegative")
        if int(chroma_lowpass_kernel) < 2:
            raise ValueError("chroma_lowpass_kernel must be at least 2")
        if int(green_dc_kernel) < 2:
            raise ValueError("green_dc_kernel must be at least 2")
        if not 0.0 < float(highlight_threshold) < 1.0:
            raise ValueError("highlight_threshold must be in (0, 1)")
        self.kind, self.eps = kind, eps
        self.linear_weight, self.gamma_weight, self.fft_weight = linear_weight, gamma_weight, fft_weight
        self.chroma_weight = float(chroma_weight)
        self.chroma_lowpass_weight = float(chroma_lowpass_weight)
        self.green_balance_weight = float(green_balance_weight)
        self.chroma_lowpass_kernel = int(chroma_lowpass_kernel)
        self.highlight_chroma_weight = float(highlight_chroma_weight)
        self.highlight_threshold = float(highlight_threshold)
        self.green_dc_weight = float(green_dc_weight)
        self.green_dc_kernel = int(green_dc_kernel)

    def distance(self, a, b):
        d = a-b
        return d.abs().mean() if self.kind == "l1" else (d.square()+self.eps**2).sqrt().mean()

    def _elementwise_distance(self, a, b):
        d = a - b
        return d.abs() if self.kind == "l1" else (d.square() + self.eps**2).sqrt()

    def forward(self, prediction, target, white_balance=None):
        # FFT and signed powers in fp32 even under autocast.
        prediction, target = prediction.float(), target.float()
        loss = self.linear_weight*self.distance(prediction, target)
        if self.gamma_weight:
            loss = loss+self.gamma_weight*self.distance(gamma(prediction), gamma(target))
        if self.fft_weight:
            a, b = [torch.fft.rfft2(t, norm="ortho") for t in (prediction, target)]
            loss = loss+self.fft_weight*(a-b).abs().mean()
        if self.chroma_weight or self.chroma_lowpass_weight or self.highlight_chroma_weight:
            wb = _wb_gains(target, white_balance=white_balance)
            p_chroma = _opponent_chroma(gamma(prediction * wb))
            t_chroma = _opponent_chroma(gamma(target * wb))
            if self.chroma_weight:
                loss = loss + self.chroma_weight * self.distance(p_chroma, t_chroma)
            if self.chroma_lowpass_weight:
                k = self.chroma_lowpass_kernel
                if min(p_chroma.shape[-2:]) >= k:
                    stride = max(1, k // 2)
                    lp_p = F.avg_pool2d(p_chroma, kernel_size=k, stride=stride)
                    lp_t = F.avg_pool2d(t_chroma, kernel_size=k, stride=stride)
                    loss = loss + self.chroma_lowpass_weight * self.distance(lp_p, lp_t)
                else:
                    loss = loss + self.chroma_lowpass_weight * self.distance(p_chroma, t_chroma)
            if self.highlight_chroma_weight:
                peak = target.detach().amax(dim=1, keepdim=True)
                w = ((peak - self.highlight_threshold) / max(1.0 - self.highlight_threshold, 1e-4)).clamp(0.0, 1.0)
                # Floored denominator avoids device-to-host sync and prevents a few
                # barely-bright pixels (w << 1) from amplifying highlight_chroma_weight.
                hl_loss = (self._elementwise_distance(p_chroma, t_chroma) * w).sum() / (w.sum() * 2.0).clamp_min(16.0)
                loss = loss + self.highlight_chroma_weight * hl_loss
        if self.green_balance_weight or self.green_dc_weight:
            pg = gamma(prediction)
            tg = gamma(target)
            if self.green_balance_weight:
                loss = loss + self.green_balance_weight * self.distance(pg[:, 1] - pg[:, 3], tg[:, 1] - tg[:, 3])
            if self.green_dc_weight:
                diff_g = (pg[:, 1:2] - pg[:, 3:4]) - (tg[:, 1:2] - tg[:, 3:4])
                k = self.green_dc_kernel
                if min(diff_g.shape[-2:]) >= k:
                    dc = F.avg_pool2d(diff_g, kernel_size=k, stride=max(1, k // 2))
                else:
                    dc = diff_g.mean(dim=(-2, -1), keepdim=True)
                loss = loss + self.green_dc_weight * self.distance(dc, torch.zeros_like(dc))
        return loss
