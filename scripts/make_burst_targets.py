#!/usr/bin/env python3
"""Average a pre-aligned tripod burst with robust outlier rejection.

Each input needs a JSON sidecar PATH.json containing iso, exposure_time,
aperture. Registration is verified approximately, never silently corrected.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from denoise_lab.bayer import read_mosaic, pack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frames", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=3.5)
    p.add_argument("--exposure-tolerance", type=float, default=.01)
    p.add_argument("--alignment-threshold", type=float, default=.15,
                   help="Maximum relative mean absolute difference of coarse scene luminance")
    p.add_argument("--noisy-count", type=int, default=1)
    args = p.parse_args()
    if len(args.frames) < 3:
        p.error("At least 3 aligned frames required")
    if not 1 <= args.noisy_count < len(args.frames):
        p.error("noisy-count must leave at least two independent target frames")
    meta = [json.loads(Path(str(f)+".json").read_text()) for f in args.frames]
    for key in ("iso", "exposure_time", "aperture"):
        vals = np.asarray([m[key] for m in meta], dtype=float)
        if np.any(vals <= 0) or np.max(np.abs(vals/vals[0]-1)) > args.exposure_tolerance:
            raise ValueError(f"Inconsistent capture metadata: {key}")
    images = np.stack([read_mosaic(f) for f in args.frames])
    previews = pack(images).mean(axis=1)
    # Coarse area means tolerate shot noise but catch camera/subject movement.
    h, w = previews.shape[-2:]
    block = min(32, h, w)
    if block < 2:
        raise ValueError("Burst images too small")
    coarse = previews[:, :h//block*block, :w//block*block].reshape(
        len(images), h//block, block, w//block, block).mean(axis=(2, 4))
    ref = np.median(coarse, axis=0)
    err = np.mean(np.abs(coarse-ref), axis=(1, 2))/max(float(np.mean(np.abs(ref))), 1e-5)
    if np.any(err > args.alignment_threshold):
        raise ValueError(f"Scene/exposure mismatch (coarse alignment scores {err}); register frames first")
    # Exclude chosen inputs from the target mean to avoid correlated target noise.
    targets = images[args.noisy_count:]
    center = np.median(targets, axis=0)
    mad = np.median(np.abs(targets-center), axis=0)*1.4826
    # A small floor prevents finite-precision identical pixels rejecting everything.
    keep = np.abs(targets-center) <= args.sigma*np.maximum(mad, 1e-6)
    count = keep.sum(axis=0)
    clean = np.divide((targets*keep).sum(axis=0), count, out=center.copy(), where=count > 0).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".npy":
        np.save(args.output, clean)
    elif args.output.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(args.output, clean)
    else:
        p.error("output must be .npy or .tiff")
    entries = [{"noisy": str(f.resolve()), "clean": str(args.output.resolve()), "iso": int(meta[0]["iso"])}
               for f in args.frames[:args.noisy_count]]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(entries, indent=2))
    print(f"Averaged {len(targets)} independent target frames; rejected fraction={1-keep.mean():.6f}; max coarse mismatch={err.max():.6f}")


if __name__ == "__main__":
    main()
