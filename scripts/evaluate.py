#!/usr/bin/env python3
"""Evaluate our checkpoint on a registered held-out paired manifest."""
import argparse
import json
import numpy as np
import torch
from torch.nn import functional as F
from denoise_lab.data import PairedBurstDataset

from denoise_lab.arch_registry import build_model
from denoise_lab.metrics import quick_srgb
from denoise_lab.tiling import tiled_inference


def masked_psnr(x, y, mask):
    if not mask.any():
        return None
    mse = (x[mask].double()-y[mask].double()).square().mean().item()
    return float("inf") if mse == 0 else float(-10*np.log10(mse))


def masked_ssim(x, y, mask, window=11):
    """Gaussian SSIM only at centres whose entire window is valid."""
    x, y = x.double(), y.double()
    k = min(window, *x.shape[-2:])
    k = k if k % 2 else k-1
    supported = F.avg_pool2d(mask.double(), k, stride=1) == 1
    fraction = supported.double().mean().item()
    if not supported.any():
        return None, fraction
    axis = torch.arange(k, dtype=x.dtype, device=x.device)-(k-1)/2
    g = torch.exp(-axis.square()/(2*1.5**2))
    g /= g.sum()
    kernel = (g[:, None]*g[None, :]).expand(x.shape[1], 1, k, k)
    def avg(a):
        return F.conv2d(a, kernel, groups=x.shape[1])
    mx, my = avg(x), avg(y)
    vx, vy = (avg(x*x)-mx*mx).clamp_min(0), (avg(y*y)-my*my).clamp_min(0)
    cov = avg(x*y)-mx*my
    score = ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return score[supported].mean().item(), fraction


def evaluate_masked_pair(prediction, target, mask):
    x, y = prediction.detach().cpu(), target.detach().cpu()
    mask = mask.detach().cpu().bool().expand_as(x)
    if not mask.any():
        raise ValueError("No valid paired metric samples")
    if not torch.isfinite(x[mask]).all() or not torch.isfinite(y[mask]).all():
        raise ValueError("Nonfinite paired metric samples inside valid_mask")
    x, y = torch.where(mask, x, 0), torch.where(mask, y, 0)
    linear_ssim, linear_fraction = masked_ssim(x, y, mask)
    # The preview interpolates a 3x3 sensor neighbourhood. Exclude all pixels
    # touching an invalid CFA tile before computing either preview metric.
    sensor = F.interpolate(mask.all(1, keepdim=True).float(), scale_factor=2, mode="nearest")
    rgb_mask = (F.max_pool2d(1-sensor, 3, stride=1, padding=1) == 0).expand(-1, 3, -1, -1)
    a, b = quick_srgb(x), quick_srgb(y)
    srgb_ssim, srgb_fraction = masked_ssim(a, b, rgb_mask)
    return {"linear_psnr": masked_psnr(x, y, mask), "linear_ssim": linear_ssim,
            "srgb_psnr": masked_psnr(a, b, rgb_mask), "srgb_ssim": srgb_ssim,
            "valid_fraction": mask.double().mean().item(),
            "linear_ssim_valid_fraction": linear_fraction,
            "srgb_valid_fraction": rgb_mask.double().mean().item(),
            "srgb_ssim_valid_fraction": srgb_fraction}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--calibration", help="Required for a noise-conditioned checkpoint")
    p.add_argument("--output", help="JSON metrics against independent paired targets")
    args = p.parse_args()
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_config"]).eval()
    model.load_state_dict(ckpt["model"])
    if model.cond == "noise_map" and not args.calibration:
        p.error("--calibration required for noise_map conditioning")
    ds = PairedBurstDataset(args.manifest, args.root, crop_size=None, augment=False, calibration=args.calibration)
    rows = []
    for i in range(len(ds)):
        batch = ds[i]
        noisy, clean = batch["noisy"][None], batch["clean"][None]
        mask = batch["valid_mask"][None]
        cond = batch["noise_map"][None] if model.cond == "noise_map" else None
        with torch.inference_mode():
            prediction = tiled_inference(model, noisy, noise_map=cond)
            row = {"index": i, "valid_fraction": mask.float().mean().item(),
                   "input": evaluate_masked_pair(noisy, clean, mask),
                   "ours": evaluate_masked_pair(prediction, clean, mask)}
        rows.append(row)
        print(json.dumps(row))
    if not rows:
        raise ValueError("Empty evaluation manifest")
    def mean_metric(name, key):
        values = [r[name][key] for r in rows if r[name][key] is not None]
        return float(np.mean(values)) if values else None
    summary = {name: {key: mean_metric(name, key) for key in rows[0][name]}
               for name in ("input", "ours")}
    print(json.dumps({"mean_metrics": summary}, indent=2))
    if args.output:
        # JSON null for perfect-match infinity, to keep the saved file strict JSON.
        def finite(obj):
            if isinstance(obj, dict):
                return {k: finite(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [finite(v) for v in obj]
            return None if isinstance(obj, float) and not np.isfinite(obj) else obj
        with open(args.output, "w") as f:
            json.dump(finite({"pairs": rows, "mean_metrics": summary}), f, indent=2)


if __name__ == "__main__":
    main()
