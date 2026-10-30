#!/usr/bin/env python3
"""Fit noise from dark/ISO/* and flat/ISO/exposure/* pairs.

All frames are normalized using their own black/white metadata. NPY/TIFF inputs
must already have the same normalized units. Flat exposure subfolders contain
an even number of consecutive same-exposure pairs. Capture at least two darks
and several unsaturated flat brightness levels per ISO. No camera data included.
"""
import argparse
from pathlib import Path
import numpy as np
import yaml
from denoise_lab.bayer import read_mosaic
from denoise_lab.noise import tukey_quantile

SUFFIXES = {".npy", ".tif", ".tiff", ".dng", ".arw", ".nef", ".cr2", ".cr3", ".raf", ".rw2", ".orf", ".3fr", ".fff"}


def files(folder):
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUFFIXES)


def ppcc_shape(samples):
    """Probability-plot correlation coefficient grid fit, with free location/scale."""
    x = np.sort(samples.ravel())
    if len(x) > 32768:
        x = x[np.linspace(0, len(x)-1, 32768).astype(int)]
    if np.std(x) < 1e-12:
        return 0.14, 1.
    u = (np.arange(len(x))+.5)/len(x)
    grid = np.linspace(-.3, .5, 161)
    scores = [np.corrcoef(x, tukey_quantile(u, s))[0, 1] for s in grid]
    i = int(np.nanargmax(scores))
    return float(grid[i]), float(scores[i])


def fit_iso(dark_folder, flat_folder, quant_step, distribution):
    darks = files(dark_folder)
    if len(darks) < 2:
        raise ValueError(f"Need >=2 dark frames in {dark_folder}")
    # Differencing removes stationary black offsets and fixed-pattern noise.
    residuals, row_variances, pixel_variances = [], [], []
    for a, b in zip(darks[::2], darks[1::2]):
        delta = (read_mosaic(a)-read_mosaic(b))/np.sqrt(2.)
        rows = delta.mean(axis=1, keepdims=True)
        pixel = delta-rows
        # Correct variance lost by estimating the row mean from finite width.
        pv = float(pixel.var(ddof=1))*delta.shape[1]/(delta.shape[1]-1)
        row_variances.append(max(0., float(rows.var(ddof=1))-pv/delta.shape[1]))
        pixel_variances.append(max(0., pv-quant_step**2/12))
        residuals.append(pixel.ravel()[::max(1, pixel.size//65536)])
    read_var, row_var = np.mean(pixel_variances), np.mean(row_variances)
    # A difference of two Tukey variables is not itself Tukey distributed.
    # For the shape fit use individual dark samples after estimating stationary
    # pattern from the temporal mean; finite-burst uncertainty remains in PPCC.
    if distribution == "tukey" and len(darks) < 8:
        raise ValueError("Use >=8 dark frames for Tukey-lambda shape calibration")
    stack = np.stack([read_mosaic(f) for f in darks])
    centered = stack-stack.mean(axis=0, keepdims=True)
    centered -= centered.mean(axis=2, keepdims=True)
    shape, ppcc = ppcc_shape(centered)
    means, variances = [], []
    groups = sorted(f for f in flat_folder.iterdir() if f.is_dir())
    if not groups:
        raise ValueError(f"Flat exposure subfolders required: {flat_folder}")
    for group in groups:
        frames = files(group)
        if len(frames) < 2 or len(frames) % 2:
            raise ValueError(f"Need an even number of flat frames: {group}")
        for a, b in zip(frames[::2], frames[1::2]):
            a, b = read_mosaic(a), read_mosaic(b)
            delta, mean = (a-b)/np.sqrt(2.), (a+b)/2
            # Reject clipped/near-black pairs; spatial structure cancels in diff.
            if np.quantile(mean, .999) >= .95 or mean.mean() <= .005:
                continue
            if abs(float(a.mean()-b.mean())) > .02*max(float(mean.mean()), 1e-6):
                raise ValueError(f"Flat exposure drift in {group}")
            means.append(float(mean.mean()))
            variances.append(float(delta.var(ddof=1)))
    if len(means) < 3 or np.ptp(means) < .02:
        raise ValueError("Need >=3 flat pairs covering at least .02 normalized signal range")
    # Intercept absorbs signal-independent read, row and quantization variance.
    K, intercept = np.polyfit(means, variances, 1)
    if K <= 0:
        raise ValueError("Nonpositive PTC gain; inspect flat captures")
    return {"K": float(K), "read_std": float(np.sqrt(read_var)), "row_std": float(np.sqrt(row_var)),
            "read_distribution": distribution, "tukey_lambda": shape, "quant_step": quant_step,
            "black_offset": 0., "diagnostics": {"ppcc": ppcc, "ptc_intercept": float(intercept),
            "flat_pair_count": len(means), "dark_pair_count": len(residuals)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dark", type=Path, required=True)
    p.add_argument("--flat", type=Path, required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--quant-step", type=float, required=True, help="1/(white-black), in normalized units")
    p.add_argument("--read-distribution", choices=("gaussian", "tukey"), default="tukey")
    args = p.parse_args()
    profiles = {}
    for folder in sorted(args.dark.iterdir()):
        if folder.is_dir() and folder.name.isdigit():
            iso = int(folder.name)
            profiles[iso] = fit_iso(folder, args.flat/folder.name, args.quant_step, args.read_distribution)
            print(f"ISO {iso}: {profiles[iso]}")
    if len(profiles) < 2:
        p.error("At least two calibrated ISOs are required for a log-linear read-noise fit")
    x = np.log([v["K"] for v in profiles.values()])
    y = np.log([max(v["read_std"], 1e-10) for v in profiles.values()])
    if np.ptp(x) < 1e-6:
        raise ValueError("Calibrated gains do not span a range")
    slope, intercept = np.polyfit(x, y, 1)
    result = {"camera": args.camera, "units": "normalized", "iso": profiles,
              "read_log_linear": {"slope": float(slope), "intercept": float(intercept),
                                  "scatter": float(np.std(y-(slope*x+intercept)))}}
    output = args.output or Path("calib")/(args.camera+".yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
