#!/usr/bin/env python3
"""Compare our checkpoints on independently held-out references plus added noise.

Includes dark exposure slices and a zero-output baseline. These are development
diagnostics against imperfect low-ISO frames, not real-camera quality scores.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from denoise_lab.data import BootstrapDataset
from denoise_lab.metrics import psnr
from denoise_lab.model import gamma
from denoise_lab.arch_registry import build_model

LEVELS = {"low": {"K": .00002, "read_std": .0002, "row_std": .00005},
          "medium": {"K": .0002, "read_std": .001, "row_std": .0002},
          "high": {"K": .001, "read_std": .004, "row_std": .0005}}


def color_error(prediction, target):
    difference = prediction-target
    green = (difference[:, 1]+difference[:, 3])/2
    chroma = torch.stack((difference[:, 0]-green, difference[:, 2]-green), dim=1)
    return float(chroma.square().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--camera", choices=["x1d2", "a7m3"], default="x1d2")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-sources", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = args.root/"bootstrap"/f"{args.camera}_val.json"
    train = json.loads((args.root/"bootstrap"/f"{args.camera}_train.json").read_text())
    validation = json.loads(manifest.read_text())
    for key in ["source_sha256", "sensor_sha256", "group"]:
        if {entry[key] for entry in train} & {entry[key] for entry in validation}:
            raise ValueError(f"Train/validation overlap: {key}")
    grouped = defaultdict(list)
    for entry in validation:
        grouped[entry["source_sha256"]].append(entry)
    rng = random.Random(42)
    entries = [rng.choice(values) for _, values in sorted(grouped.items())]
    rng.shuffle(entries)
    entries = entries[:args.max_sources]
    if not entries:
        raise ValueError("No validation photos")
    models, provenance = {}, {}
    for specification in args.checkpoint:
        label, path = specification.split("=", 1)
        if label in models:
            raise ValueError(f"Duplicate checkpoint name: {label}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(saved["model_config"]).to(args.device).eval()
        model.load_state_dict(saved["model"])
        models[label] = model
        with open(path, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance[label] = {"checkpoint": path, "sha256": digest, "stage_step": saved["step"],
                             "model_config": saved["model_config"],
                             "training_data": saved["config"]["data"]}
    rows = []
    for exposure in [1., 1/16, 1/64]:
        for level, parameters in LEVELS.items():
            dataset = BootstrapDataset(entries, root=args.root, crop_size=128, augment=False,
                                       deterministic=True, seed=20261026,
                                       fixed_parameters=parameters, exposure_scale=exposure)
            for i in range(len(dataset)):
                item = dataset[i]
                target = item["clean"][None].to(args.device)
                noisy = item["noisy"][None].to(args.device)
                common = {"exposure_scale": exposure, "level": level,
                          "source_sha256": entries[i]["source_sha256"],
                          "input_psnr": psnr(noisy, target),
                          "zero_psnr": psnr(torch.zeros_like(target), target),
                          "input_color_mse": color_error(noisy, target)}
                with torch.inference_mode():
                    for label, model in models.items():
                        prediction = model(noisy)
                        if not torch.isfinite(prediction).all():
                            raise ValueError(f"Nonfinite output: {label}")
                        model_psnr = psnr(prediction, target)
                        bias = (prediction-target).mean(dim=(0, 2, 3)).cpu().tolist()
                        rows.append({**common, "model": label, "model_psnr": model_psnr,
                                     "gain_db": model_psnr-common["input_psnr"],
                                     "over_zero_db": model_psnr-common["zero_psnr"],
                                     "gamma_mse": float((gamma(prediction)-gamma(target)).square().mean()),
                                     "color_mse": color_error(prediction, target),
                                     "channel_bias": bias})
            print("EVALUATED", exposure, level, len(dataset), flush=True)
    summary = []
    for label in models:
        for exposure in [1., 1/16, 1/64]:
            for level in LEVELS:
                sample = [row for row in rows if row["model"] == label and
                          row["exposure_scale"] == exposure and row["level"] == level]
                keys = ["input_psnr", "model_psnr", "gain_db", "over_zero_db", "gamma_mse", "color_mse"]
                means = {key: float(np.mean([row[key] for row in sample])) for key in keys}
                summary.append({"model": label, "exposure_scale": exposure, "level": level,
                                "photos": len(sample), **means,
                                "median_gain_db": float(np.median([row["gain_db"] for row in sample])),
                                "negative_gain_photos": sum(row["gain_db"] < 0 for row in sample)})
    result = {"scope": "Development diagnostics using added synthetic noise and held-out imperfect low-ISO references; no real-ISO quality claim.",
              "reference_attenuation": "Reference is scaled BEFORE drawing new noise; native reference noise is attenuated too.",
              "camera": args.camera, "sources": len(entries), "checkpoints": provenance,
              "noise_parameters": LEVELS, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
