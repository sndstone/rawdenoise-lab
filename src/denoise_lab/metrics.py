"""Packed linear and simple preview metrics (not a camera ISP/color profile)."""
import math
import torch
from torch.nn import functional as F
from .tiling import tile_starts, tiled_inference


def psnr(prediction, target, data_range=1.):
    mse = (prediction.detach().cpu().double()-target.detach().cpu().double()).square().mean().item()
    return float("inf") if mse == 0 else 10*math.log10(data_range**2/mse)


def ssim(prediction, target, data_range=1., window=11, mask=None):
    """Mean local SSIM; an optional broadcastable mask requires full window support.

    With a mask, return None when no complete window is supported. Unmasked
    calls retain the historical reduction.
    """
    # Local Gaussian SSIM with valid borders; channels are evaluated separately.
    x, y = prediction.detach().cpu().double(), target.detach().cpu().double()
    k = min(window, *x.shape[-2:])
    k = k if k % 2 else k-1
    axis = torch.arange(k, device=x.device, dtype=x.dtype)-(k-1)/2
    g = torch.exp(-axis.square()/(2*1.5**2))
    g = g/g.sum()
    kernel = (g[:, None]*g[None, :]).expand(x.shape[1], 1, k, k)
    def avg(a):
        return F.conv2d(a, kernel, groups=x.shape[1])
    mx, my = avg(x), avg(y)
    vx, vy = (avg(x*x)-mx*mx).clamp_min(0), (avg(y*y)-my*my).clamp_min(0)
    cov = avg(x*y)-mx*my
    c1, c2 = (0.01*data_range)**2, (0.03*data_range)**2
    values = ((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))
    if mask is not None:
        valid = torch.broadcast_to(mask.detach().cpu().bool(), x.shape)
        supported = F.max_pool2d((~valid).double(), k, stride=1) == 0
        return values[supported].mean().item() if supported.any() else None
    return values.mean().item()


def quick_srgb(packed):
    """Bilinear demosaic with neutral WB, no camera matrix; comparison preview."""
    b, _, h, w = packed.shape
    rgb = torch.zeros((b, 3, 2*h, 2*w), dtype=packed.dtype, device=packed.device)
    mask = torch.zeros_like(rgb)
    for c, (color, y, x) in enumerate(((0, 0, 0), (1, 0, 1), (2, 1, 1), (1, 1, 0))):
        rgb[:, color, y::2, x::2] = packed[:, c]
        mask[:, color, y::2, x::2] = 1
    kernel = packed.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).expand(3, 1, 3, 3)
    rgb = F.conv2d(rgb, kernel, padding=1, groups=3)/F.conv2d(mask, kernel, padding=1, groups=3)
    rgb = rgb.clamp(0, 1)
    return torch.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb.clamp_min(.0031308).pow(1/2.4)-.055)


def evaluate_pair(prediction, target):
    a, b = quick_srgb(prediction), quick_srgb(target)
    return {"linear_psnr": psnr(prediction, target), "linear_ssim": ssim(prediction, target),
            "srgb_psnr": psnr(a, b), "srgb_ssim": ssim(a, b)}


@torch.inference_mode()
def flat_field_identity_test(model, level=0.18, shape=(1, 4, 512, 512), device="cpu"):
    x = torch.full(shape, level, device=device)
    y = model(x)
    return {"max_abs_error": (y-x).abs().max().item(), "mean_bias": (y-x).mean().item()}


@torch.inference_mode()
def seam_energy(delta, tile_size=512, overlap=58, grid_offsets=(0, 128),
                grid_alignment=1, radius=8):
    """Squared difference energy in the union of tile start/end strips.

    Energies are per-element means, so their ratio corrects for strip area.
    The energy fraction alone is area-dependent. Undefined ratios/empty regions
    return None. Differences are in normalized linear packed units.
    """
    if delta.ndim != 4 or min(delta.shape) < 1:
        raise ValueError("Expected nonempty BxCxHxW difference")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    # Reporting in float64 on CPU also works for MPS, which lacks float64.
    work = delta.detach().to(device="cpu", dtype=torch.float64)
    h, w = delta.shape[-2:]
    mask = torch.zeros((h, w), dtype=torch.bool)
    for offset in grid_offsets:
        oy, ox = (offset, offset) if isinstance(offset, int) else offset
        for axis, (length, phase) in enumerate(((h, oy), (w, ox))):
            for start in tile_starts(length, tile_size, overlap, phase, grid_alignment):
                for edge in (start, start+tile_size):
                    if 0 < edge < length:
                        lo, hi = max(0, edge-radius), min(length, edge+radius)
                        if axis == 0:
                            mask[lo:hi, :] = True
                        else:
                            mask[:, lo:hi] = True
    absolute = work.abs().mean((0, 1))
    energy = work.square().mean((0, 1))
    near = energy[mask].mean().item() if mask.any() else None
    away = energy[~mask].mean().item() if (~mask).any() else None
    total = energy.sum().item()
    return dict(max_abs_diff=delta.abs().max().item(), mean_abs_diff=absolute.mean().item(),
        boundary_fraction=mask.double().mean().item(), boundary_energy=near, interior_energy=away,
        boundary_mean_abs_diff=absolute[mask].mean().item() if mask.any() else None,
        interior_mean_abs_diff=absolute[~mask].mean().item() if (~mask).any() else None,
        boundary_energy_ratio=near/away if near is not None and away else None,
        boundary_energy_fraction=energy[mask].sum().item()/total if total else None)


@torch.inference_mode()
def seam_test(model, image, reference_offset=None, boundary_radius=8, **tiling_kwargs):
    """Compare tiled vs full (default), or two grids with reference_offset.

    Global SCA pooling can make full-image and tiled predictions differ.
    A grid comparison avoids conflating tile phase with the full-image context.
    """
    tiled = tiled_inference(model, image, **tiling_kwargs)
    full = model(image) if reference_offset is None else tiled_inference(
        model, image, **dict(tiling_kwargs, grid_offset=reference_offset))
    delta = tiled-full
    tile, overlap = tiling_kwargs.get("tile_size", 512), tiling_kwargs.get("overlap", 58)
    offset = tiling_kwargs.get("grid_offset", 0)
    alignment = tiling_kwargs.get("grid_alignment", 1)
    report = seam_energy(delta, tile, overlap,
        (offset,) if reference_offset is None else (offset, reference_offset), alignment, boundary_radius)
    errors = []
    oy, ox = (offset, offset) if isinstance(offset, int) else offset
    for dim, phase in ((-2, oy), (-1, ox)):
        for start in tile_starts(image.shape[dim], tile, overlap, phase, alignment):
            for edge in (start, start+tile):
                if 0 < edge < image.shape[dim]:
                    errors.append((delta.select(dim, edge)-delta.select(dim, edge-1)).abs().mean())
    report["boundary_excess"] = torch.stack(errors).mean().item() if errors else 0.
    if reference_offset is None:
        report["tiled_full_mae"] = report["mean_abs_diff"]
    return report
