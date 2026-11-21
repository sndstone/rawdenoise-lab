#!/usr/bin/env python3
"""Average a pre-aligned tripod burst with robust outlier rejection.

Each input needs a JSON sidecar PATH.json containing iso, exposure_time,
aperture. Default behaviour only verifies coarse registration. --align opts in
to subpixel packed-plane registration and a noise-model merge, with support maps.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from denoise_lab.bayer import read_mosaic, pack


def make_aligned(args, images, meta):
    from denoise_lab.bayer import unpack
    from denoise_lab.burst_align import (align_burst, estimate_burst_noise,
                                        interpolation_variance, robust_merge, match_cross_iso)
    from denoise_lab.noise import NoiseModel
    aligned, shifts = align_burst(pack(images))
    targets = aligned[args.noisy_count:]
    factors = [interpolation_variance(s) for s in shifts[args.noisy_count:]]
    parameters = (NoiseModel(args.calibration).parameters(meta[0]["iso"])
                  if args.calibration else estimate_burst_noise(targets, factors))
    result = robust_merge(targets, parameters, factors, sigma=args.sigma)
    # Metadata already require matching ISO/exposure/aperture: test the held-out
    # inputs against the target without fitting away scene/exposure differences.
    def same_exposure(mean, clean, static, block):
        return clean, [[1., 0., 1., 1.]]*4
    _, _, valid, _ = match_cross_iso(aligned[:args.noisy_count], result, parameters,
                                    same_exposure,
                                    [interpolation_variance(s) for s in shifts[:args.noisy_count]])
    result.static &= valid
    result.confidence *= valid
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean = unpack(result.clean)
    if args.output.suffix.lower() == ".npy":
        np.save(args.output, clean)
    else:
        import tifffile
        tifffile.imwrite(args.output, clean)
    support = args.output.with_name(args.output.stem+"_support.npz")
    np.savez_compressed(support, effective_count=result.effective_count,
                        variance=result.variance, confidence=result.confidence,
                        static=result.static, shifts=shifts)
    mask_path = args.output.with_name(args.output.stem+"_valid.npy")
    np.save(mask_path, np.repeat(np.repeat(result.static, 2, 0), 2, 1))
    entries = []
    for i, f in enumerate(args.frames[:args.noisy_count]):
        # The manifest must point to inputs on the same grid as the target.
        path = args.output.with_name(args.output.stem+f"_noisy_{i}.npy")
        np.save(path, unpack(aligned[i]))
        entries.append({"noisy": str(path.resolve()), "source": str(f.resolve()),
                        "clean": str(args.output.resolve()), "iso": int(meta[0]["iso"]),
                        "valid_mask": str(mask_path.resolve()), "support": str(support.resolve())})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(entries, indent=2))
    print(f"Aligned {len(images)} frames; independent targets={len(targets)}; "
          f"static={result.static.mean():.2%}; NLF={parameters}; "
          "consumers must honour valid_mask (sensor coordinates)")


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
    p.add_argument("--align", action="store_true", help="Register packed planes; save motion/count/variance maps")
    p.add_argument("--calibration", type=Path, help="Normalized noise calibration for --align; otherwise fit temporal NLF")
    args = p.parse_args()
    if len(args.frames) < 3:
        p.error("At least 3 aligned frames required")
    if not 1 <= args.noisy_count < len(args.frames):
        p.error("noisy-count must leave at least two independent target frames")
    if args.align and args.noisy_count > len(args.frames)-2:
        p.error("--align needs at least two independent target frames")
    if args.output.suffix.lower() not in (".npy", ".tif", ".tiff"):
        p.error("output must be .npy or .tiff")
    meta = [json.loads(Path(str(f)+".json").read_text()) for f in args.frames]
    for key in ("iso", "exposure_time", "aperture"):
        vals = np.asarray([m[key] for m in meta], dtype=float)
        if np.any(vals <= 0) or np.max(np.abs(vals/vals[0]-1)) > args.exposure_tolerance:
            raise ValueError(f"Inconsistent capture metadata: {key}")
    images = np.stack([read_mosaic(f) for f in args.frames])
    if args.align:
        make_aligned(args, images, meta)
        return
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
