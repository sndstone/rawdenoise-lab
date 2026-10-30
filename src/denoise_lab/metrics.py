"""Packed linear and simple preview metrics (not a camera ISP/color profile)."""
import math
import torch
from torch.nn import functional as F
from .tiling import tiled_inference


def psnr(prediction, target, data_range=1.):
    mse = (prediction.double()-target.double()).square().mean().item()
    return float("inf") if mse == 0 else 10*math.log10(data_range**2/mse)


def ssim(prediction, target, data_range=1., window=11):
    # Local Gaussian SSIM with valid borders; channels are evaluated separately.
    x, y = prediction.double(), target.double()
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
    return (((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))).mean().item()


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
def seam_test(model, image, **tiling_kwargs):
    """Report tiled-vs-full error and excess gradients across tile boundaries.

    Global SCA pooling can make full-image and tiled predictions differ.
    """
    tiled = tiled_inference(model, image, **tiling_kwargs)
    full = model(image)
    delta = tiled-full
    tile, overlap = tiling_kwargs.get("tile_size", 512), tiling_kwargs.get("overlap", 58)
    step = tile-overlap
    errors = []
    for dim in (-2, -1):
        for i in range(step-overlap, image.shape[dim], step):
            if i > 0:
                errors.append((delta.select(dim, i)-delta.select(dim, i-1)).abs().mean())
    return {"tiled_full_mae": delta.abs().mean().item(),
            "boundary_excess": torch.stack(errors).mean().item() if errors else 0.}
