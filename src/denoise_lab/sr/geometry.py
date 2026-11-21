"""Exact Bayer CFA geometry, cross-plane alignment, and compressive transforms.

Packed tensors follow the repository convention: Bx4xHxW in R/G1/B/G2 order,
where each packed coordinate (j, i) corresponds to sensor coordinate (2j+sy, 2i+sx).
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ..bayer import SITES
from ..model import GAMMA, gamma as signed_gamma, inverse_gamma


def site_upsample(packed: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Interpolate equal-colour samples on the *sensor* pixel-centre grid.

    An output sensor coordinate o maps to (o + 0.5) / scale - 0.5 on the input.
    A packed plane coordinate j denotes sensor coordinate 2*j + site, not 2*j.
    """
    if packed.ndim != 4 or packed.shape[1] != 4 or min(packed.shape[-2:]) < 2:
        raise ValueError("Expected Bx4xHxW packed Bayer with H, W >= 2")
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError("Scale must be a positive integer")
    h, w = packed.shape[-2:]
    yy = torch.arange(h * scale, dtype=packed.dtype, device=packed.device)
    xx = torch.arange(w * scale, dtype=packed.dtype, device=packed.device)
    planes = []
    for c, (sy, sx) in enumerate(SITES):
        gy = (((2 * yy + sy + 0.5) / scale - 0.5) - sy) / 2.0
        gx = (((2 * xx + sx + 0.5) / scale - 0.5) - sx) / 2.0
        grid_y, grid_x = torch.meshgrid(
            2.0 * gy / (h - 1) - 1.0,
            2.0 * gx / (w - 1) - 1.0,
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(packed.shape[0], -1, -1, -1)
        planes.append(
            F.grid_sample(
                packed[:, c : c + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
    return torch.cat(planes, dim=1)


def align_planes(packed: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Resample staggered R/G1/B/G2 planes onto the common red-site lattice (2y, 2x).

    With inverse=True, shifts corrections back to each channel's native CFA site.
    """
    if packed.ndim != 4 or packed.shape[1] != 4 or min(packed.shape[-2:]) < 2:
        raise ValueError("Expected Bx4xHxW packed Bayer with H, W >= 2")
    b, _, h, w = packed.shape
    yy = torch.arange(h, device=packed.device, dtype=packed.dtype)
    xx = torch.arange(w, device=packed.device, dtype=packed.dtype)
    sign = 1.0 if inverse else -1.0
    result = []
    for c, (sy, sx) in enumerate(SITES):
        if sy == 0 and sx == 0:
            result.append(packed[:, c : c + 1])
            continue
        gy = yy + sign * (sy / 2.0)
        gx = xx + sign * (sx / 2.0)
        grid_y, grid_x = torch.meshgrid(
            2.0 * gy / (h - 1) - 1.0,
            2.0 * gx / (w - 1) - 1.0,
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(b, -1, -1, -1)
        result.append(
            F.grid_sample(
                packed[:, c : c + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
    return torch.cat(result, dim=1)


def luminance(aligned: torch.Tensor) -> torch.Tensor:
    """Simple camera-linear luminance estimate on aligned R/G1/B/G2 planes."""
    return 0.25 * (aligned[:, 0:1] + aligned[:, 1:2] + aligned[:, 2:3] + aligned[:, 3:4])


def edge_energy(x: torch.Tensor) -> torch.Tensor:
    """L1 finite-difference gradient magnitude (same shape as input)."""
    dx = F.pad((x[..., 1:] - x[..., :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((x[..., 1:, :] - x[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def signed_asinh(x: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    """Smooth odd compressive transform preserving signed below-black values.

    Linear for |x| << scale, logarithmic for |x| >> scale, smooth derivative at 0.
    Normalized by asinh(1/scale) so +1.0 maps to +1.0.
    """
    if scale <= 0:
        raise ValueError("asinh scale must be positive")
    norm = torch.asinh(x.new_tensor(1.0 / scale))
    return torch.asinh(x / scale) / norm


def inverse_asinh(z: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    """Exact analytic inverse of `signed_asinh`."""
    if scale <= 0:
        raise ValueError("asinh scale must be positive")
    norm = torch.asinh(z.new_tensor(1.0 / scale))
    return scale * torch.sinh(z * norm)


def to_compressive(x: torch.Tensor, domain: str = "asinh", scale: float = 0.05, gamma_exp: float = GAMMA) -> torch.Tensor:
    if domain == "asinh":
        return signed_asinh(x, scale=scale)
    if domain == "gamma":
        return signed_gamma(x, exponent=gamma_exp)
    if domain == "linear":
        return x
    raise ValueError(f"Unknown compressive domain: {domain}")


def from_compressive(z: torch.Tensor, domain: str = "asinh", scale: float = 0.05, gamma_exp: float = GAMMA) -> torch.Tensor:
    if domain == "asinh":
        return inverse_asinh(z, scale=scale)
    if domain == "gamma":
        return inverse_gamma(z, exponent=gamma_exp)
    if domain == "linear":
        return z
    raise ValueError(f"Unknown compressive domain: {domain}")
