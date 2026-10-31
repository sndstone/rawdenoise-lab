#!/usr/bin/env python3
"""Denoise Bayer RAWs, or uint16 RGB TIFFs with a matching experimental RGB checkpoint."""
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
from denoise_lab.rgb import COLOR_SPACES, is_rgb_tiff, read_rgb_tiff, write_rgb_tiff


def main(argv=None, *, denoiser=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: rawdenoise photo.dng --model models/r4_plain_fp32.onnx --device cpu")
    parser.add_argument("inputs", nargs="+", type=Path, help="Bayer RAW files or uint16 RGB TIFFs (require RGB checkpoint)")
    parser.add_argument("--model", required=True, type=Path, help="Model file: .pt, .onnx, or .mlpackage")
    parser.add_argument("--output-dir", type=Path, help="Default: beside each input with _denoised suffix")
    parser.add_argument("--format", choices=["dng", "tiff", "npy"], help="DNG mosaic (Bayer default), 16-bit TIFF (RGB default), or normalized float32 NPY (Bayer only)")
    parser.add_argument("--backend", choices=["auto", "torch", "onnx", "coreml"], default="auto", help="Select runtime; auto uses the model suffix")
    parser.add_argument("--device", default="auto", help="auto or cpu; torch also accepts cuda, cuda:0, or mps (ONNX/Core ML: auto or cpu only)")
    parser.add_argument("--experimental-mps-body", action="store_true",
                        help="Explicit experimental torch Bayer MPS body with CPU transforms/fallback; NOT quality-accepted. Requires --device mps and both PYTORCH_ENABLE_MPS_FALLBACK=0 / PYTORCH_MPS_FAST_MATH=0 before Python startup")
    parser.add_argument("--no-prepass", action="store_true", help="Disable hot/dead-pixel correction; recommended for isolated stars")
    parser.add_argument("--overlap", type=int, default=60, help="Tile overlap in packed Bayer pixels or RGB pixels, 0–511")
    parser.add_argument("--tile-batch-size", type=int, default=1,
                        help="Tiles per model call for CPU PyTorch Bayer inference; values above 1 use more memory")
    parser.add_argument("--cpu-threads", type=int,
                        help="PyTorch intra-op threads; default runtime setting capped at 8")
    parser.add_argument("--strength", type=float, default=1., help="Blend processed result with input: 0 preserves input, 1 applies full processing")
    parser.add_argument("--input-kind", choices=["auto", "bayer", "rgb"], default="auto", help="auto detects RGB-photometric .tif/.tiff; other inputs use Bayer")
    parser.add_argument("--rgb-color-space", choices=COLOR_SPACES, help="Declare untagged RGB encoding; must match metadata and checkpoint. No transforms")
    args = parser.parse_args(argv)
    if not 0 <= args.strength <= 1 or not 0 <= args.overlap < 512:
        parser.error("strength must be in [0,1] and overlap in [0,511]")
    if args.tile_batch_size < 1:
        parser.error("tile-batch-size must be a positive integer")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("cpu-threads must be a positive integer")
    torch.set_num_threads(args.cpu_threads if args.cpu_threads is not None else min(8, torch.get_num_threads()))
    start = time.monotonic()
    if denoiser is None:
        try:
            load_options = dict(backend=args.backend, device=args.device)
            if args.experimental_mps_body:
                load_options['experimental_mps_body'] = True
            denoiser = Denoiser.load(args.model, **load_options)
        except (ValueError, RuntimeError, ImportError, OSError) as exc:
            parser.exit(1, f"Model load failed: {exc}\n")
        print(f"Loaded {denoiser.description} in {time.monotonic()-start:.2f}s", flush=True)
    else:
        print(f"Reusing {denoiser.description}", flush=True)
    failures = 0
    for index, source in enumerate(args.inputs, 1):
        start = time.monotonic()
        try:
            kind = args.input_kind
            if kind == "auto":
                kind = "rgb" if is_rgb_tiff(source) else "bayer"
            output_format = args.format or ("tiff" if kind == "rgb" else "dng")
            destination = (args.output_dir or source.parent) / f"{source.stem}_denoised.{output_format}"
            print(f"[{index}/{len(args.inputs)}] {source} -> {destination}", flush=True)
            if kind == "rgb" and args.tile_batch_size > 1:
                raise ValueError("--tile-batch-size > 1 is unsupported for RGB inputs; use CPU PyTorch Bayer inference")
            if kind != denoiser.image_kind:
                raise ValueError(f"Input/model mismatch: {kind} input requires a {kind} checkpoint; loaded {denoiser.image_kind}. RGB never uses Bayer weights")
            if kind == "rgb" and output_format != "tiff":
                raise ValueError("RGB output must be uint16 TIFF; Bayer/DNG/NPY output is unsupported")
            if destination.resolve() == source.resolve() or destination.exists():
                raise ValueError(f"Refusing to overwrite existing file: {destination}")
            if kind == "rgb":
                image, rgb_meta = read_rgb_tiff(source, color_space=args.rgb_color_space)
                predicted = denoiser.denoise_rgb(image, color_contract=rgb_meta.contract,
                                               overlap=args.overlap, strength=args.strength)
                write_rgb_tiff(destination, predicted, rgb_meta)
                print(f"  Done in {time.monotonic()-start:.2f}s ({denoiser.description}; experimental RGB)", flush=True)
                continue
            if args.rgb_color_space is not None:
                raise ValueError("--rgb-color-space applies only to RGB TIFF inputs")
            with rawpy.imread(str(source)) as raw:
                meta = RawMetadata.from_raw(raw, source)
                packed = pack(normalize_raw(raw))
                predicted = denoiser.denoise_packed(packed, prepass=not args.no_prepass,
                                                    overlap=args.overlap, strength=args.strength,
                                                    tile_batch_size=args.tile_batch_size)
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
                if output_format == "dng":
                    write_dng(destination, visible, meta)
                elif output_format == "npy":
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
