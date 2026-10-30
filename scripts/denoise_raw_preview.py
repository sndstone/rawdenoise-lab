#!/usr/bin/env python3
"""Run our checkpoint on an evaluation RAW and save normalized sensor output.

No training or proprietary reference processing is performed by this command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rawpy
import torch

from denoise_lab.bayer import normalize_raw, pack, unpack

from denoise_lab.arch_registry import build_model
from denoise_lab.tiling import tiled_inference
from denoise_lab.noise import noise_std
from denoise_lab.rawstats import estimate_noise_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--float16", action="store_true", help="Store output as float16 (diagnostic transfer size)")
    parser.add_argument("--estimate-box", help="Packed Y0:Y1:X0:X1 region for the blind noise estimate (noise_map models)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["model_config"]).to(args.device).eval()
    model.load_state_dict(saved["model"])
    with rawpy.imread(str(args.raw)) as raw:
        mosaic = normalize_raw(raw)
    start = time.monotonic()
    packed = pack(mosaic)
    image = torch.from_numpy(packed)[None].to(args.device)
    cond, estimate = None, None
    if model.cond == "noise_map":
        # Realistic path only: blind single-image estimate, never an oracle.
        y0, y1, x0, x1 = [int(v) for v in args.estimate_box.split(":")] if args.estimate_box else (0, None, 0, None)
        estimate = estimate_noise_parameters(packed[:, y0:y1, x0:x1])
        cond = torch.from_numpy(noise_std(estimate, np.maximum(packed, 0).mean(0, keepdims=True)))[None].to(args.device)
    print("INFERENCE_START", list(image.shape), estimate, flush=True)
    result = tiled_inference(model, image, tile_size=512, overlap=64,
                             feather_power=2.2, alpha=1., noise_map=cond)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite output")
    prediction = unpack(result[0].cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prediction.astype(np.float16) if args.float16 else prediction, allow_pickle=False)
    with args.raw.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = dict(source=str(args.raw), source_sha256=digest,
                    checkpoint=str(args.checkpoint), model_config=saved["model_config"],
                    shape=list(prediction.shape), seconds=time.monotonic()-start,
                    tile_size=512, overlap=64, feather_power=2.2, alpha=1.,
                    evaluation_only=True, noise_estimate=estimate, dtype="float16" if args.float16 else "float32",
                    note="Camera compatibility is not implied by successful inference.")
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("INFERENCE_COMPLETE", metadata["seconds"], flush=True)


if __name__ == "__main__":
    main()
