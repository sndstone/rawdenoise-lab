#!/usr/bin/env python3
"""Denoise Bayer RAWs with PyTorch, ONNX Runtime, or macOS Core ML."""
import argparse
from pathlib import Path
import time

import numpy as np
import rawpy
import torch
import tifffile

from denoise_lab.bayer import PHASES, normalize_raw, pack, unpack
from denoise_lab.dng import RawMetadata, sensor_codes, write_dng
from denoise_lab.inference import Denoiser


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: rawdenoise photo.dng --model models/r4_plain_fp32.onnx --device cpu")
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more rawpy-readable Bayer RAW files")
    parser.add_argument("--model", required=True, type=Path, help="Model file: .pt, .onnx, or .mlpackage")
    parser.add_argument("--output-dir", type=Path, help="Default: beside each input with _denoised suffix")
    parser.add_argument("--format", choices=["dng", "tiff", "npy"], default="dng", help="DNG mosaic, rendered 16-bit sRGB TIFF, or normalized float32 NPY")
    parser.add_argument("--backend", choices=["auto", "torch", "onnx", "coreml"], default="auto", help="Select runtime; auto uses the model suffix")
    parser.add_argument("--device", default="auto", help="auto or cpu; torch also accepts cuda, cuda:0, or mps (ONNX/Core ML: auto or cpu only)")
    parser.add_argument("--no-prepass", action="store_true", help="Disable hot/dead-pixel correction; recommended for isolated stars")
    parser.add_argument("--overlap", type=int, default=60, help="Tile overlap in packed pixels (twice this extent in sensor pixels), 0–511")
    parser.add_argument("--strength", type=float, default=1., help="Blend processed result with input: 0 preserves input, 1 applies full processing")
    args = parser.parse_args()
    if not 0 <= args.strength <= 1 or not 0 <= args.overlap < 512:
        parser.error("strength must be in [0,1] and overlap in [0,511]")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    start = time.monotonic()
    try:
        denoiser = Denoiser.load(args.model, backend=args.backend, device=args.device)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        parser.exit(1, f"Model load failed: {exc}\n")
    print(f"Loaded {denoiser.description} in {time.monotonic()-start:.2f}s", flush=True)
    failures = 0
    for index, source in enumerate(args.inputs, 1):
        start = time.monotonic()
        destination = (args.output_dir or source.parent) / f"{source.stem}_denoised.{args.format}"
        print(f"[{index}/{len(args.inputs)}] {source} -> {destination}", flush=True)
        try:
            if destination.resolve() == source.resolve() or destination.exists():
                raise ValueError(f"Refusing to overwrite existing file: {destination}")
            with rawpy.imread(str(source)) as raw:
                meta = RawMetadata.from_raw(raw, source)
                packed = pack(normalize_raw(raw))
                predicted = denoiser.denoise_packed(packed, prepass=not args.no_prepass,
                                                    overlap=args.overlap, strength=args.strength)
                # Preserve borders cropped by unify for non-RGGB and odd dimensions.
                colors = raw.raw_colors_visible
                black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
                whites = raw.camera_white_level_per_channel
                white = np.asarray(whites if whites is not None else [raw.white_level]*len(black), dtype=np.float32)
                white = np.where(white > black, white, raw.white_level)
                visible = (raw.raw_image_visible.astype(np.float32)-black[colors])/(white[colors]-black[colors])
                y, x = PHASES[meta.phase]
                mosaic = unpack(predicted)
                visible[y:y+mosaic.shape[0], x:x+mosaic.shape[1]] = mosaic
                destination.parent.mkdir(parents=True, exist_ok=True)
                if args.format == "dng":
                    write_dng(destination, visible, meta)
                elif args.format == "npy":
                    # Full visible sensor geometry and source CFA, signed linear normalization.
                    np.save(destination, visible, allow_pickle=False)
                else:
                    raw.raw_image_visible[:] = sensor_codes(visible, meta)
                    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                          output_color=rawpy.ColorSpace.sRGB, adjust_maximum_thr=0.)
                    tifffile.imwrite(destination, rgb, photometric="rgb", metadata=None)
            print(f"  Done in {time.monotonic()-start:.2f}s ({denoiser.description})", flush=True)
        except Exception as exc:
            failures += 1
            print(f"  FAILED: {exc}", flush=True)
    if failures:
        parser.exit(1, f"{failures} file(s) failed\n")


if __name__ == "__main__":
    main()
