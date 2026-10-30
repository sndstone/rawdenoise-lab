#!/usr/bin/env python3
"""Render our normalized evaluation mosaic using its original RAW metadata.

Keeps brightness, colour conversion and demosaicing fixed for comparisons.
Rendering clips to the original integer container; inference output stays signed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rawpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ev", type=int, choices=[0, 2, 4], default=4)
    parser.add_argument("--crop", action="append", default=[], metavar="NAME:X:Y:SIZE")
    parser.add_argument("--tiff", action="store_true", help="Also save full-resolution 16-bit RGB")
    args = parser.parse_args()
    mosaic = np.load(args.normalized, mmap_mode="r", allow_pickle=False)
    with rawpy.imread(str(args.source)) as raw:
        desc = raw.color_desc.decode("ascii")
        phase = "".join(desc[int(c)] for c in raw.raw_colors_visible[:2, :2].ravel())
        if phase != "RGGB" or mosaic.shape != raw.raw_image_visible.shape:
            raise ValueError("Rendering requires matching RGGB visible geometry")
        colors = raw.raw_colors_visible
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        meta = raw.camera_white_level_per_channel
        white = np.asarray(meta if meta is not None else [raw.white_level]*4, dtype=np.float32)
        white = np.where(white > black, white, raw.white_level)
        for c in range(4):
            selected = colors == c
            values = mosaic[selected]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite normalized input")
            codes = np.rint(values*(white[c]-black[c])+black[c])
            raw.raw_image_visible[selected] = np.clip(codes, 0, 65535).astype(np.uint16)
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, adjust_maximum_thr=0.,
                              output_bps=16, output_color=rawpy.ColorSpace.sRGB,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                              median_filter_passes=0, fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                              exp_shift=4. if args.ev else 1., exp_preserve_highlights=0.,
                              bright=4. if args.ev == 4 else 1., highlight_mode=rawpy.HighlightMode.Clip)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tiff:
        import tifffile
        tifffile.imwrite(args.output/"full_16bit.tif", rgb, photometric="rgb", metadata=None)
    image = Image.fromarray((rgb//257).astype(np.uint8))
    overview = image.copy()
    overview.thumbnail((1800, 1350), Image.Resampling.LANCZOS)
    overview.save(args.output/"overview.jpg", quality=96, subsampling=0)
    crops = {}
    for specification in args.crop:
        name, x, y, size = specification.split(":")
        x, y, size = int(x), int(y), int(size)
        if min(x, y) < 0 or size <= 0 or x+size > image.width or y+size > image.height:
            raise ValueError("Crop is outside the image")
        if Path(name).name != name:
            raise ValueError("Crop name must be a filename component")
        image.crop((x, y, x+size, y+size)).save(args.output/f"{name}.png")
        crops[name] = [x, y, size]
    (args.output/"render.json").write_text(json.dumps({"source": str(args.source),
        "normalized": str(args.normalized), "display_ev": args.ev, "crops": crops,
        "rendering": "Original RAW metadata, camera WB, AHD, fixed sRGB conversion, no additional sharpening or denoising."}, indent=2)+"\n")
    print("RENDERED", args.output)


if __name__ == "__main__":
    main()
