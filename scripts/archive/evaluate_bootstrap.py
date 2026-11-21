#!/usr/bin/env python3
"""Measure removal of ADDED synthetic noise against held-out low-ISO frames.

These references contain native camera noise. Results do not establish real RAW
denoising quality or performance at a particular camera ISO.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from denoise_lab.data import BootstrapDataset
from denoise_lab.metrics import psnr

from denoise_lab.arch_registry import build_model

LEVELS = {"low": {"K": .00002, "read_std": .0002, "row_std": .00005},
          "medium": {"K": .0002, "read_std": .001, "row_std": .0002},
          "high": {"K": .001, "read_std": .004, "row_std": .0005}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-sources", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["model_config"]).to(args.device).eval()
    model.load_state_dict(saved["model"])
    grouped = defaultdict(list)
    for entry in json.loads(args.manifest.read_text()):
        grouped[entry["source_sha256"]].append(entry)
    train_manifest = saved["config"]["data"].get("manifest")
    split_checks = {}
    if train_manifest:
        train = json.loads(Path(train_manifest).read_text())
        val = [entry for values in grouped.values() for entry in values]
        for key in ["source_sha256", "sensor_sha256", "group"]:
            shared = {entry[key] for entry in train} & {entry[key] for entry in val}
            split_checks[f"shared_{key}"] = len(shared)
            if shared:
                raise ValueError(f"Training/validation leakage in {key}")
    rng = random.Random(42)
    entries = [rng.choice(rows) for _, rows in sorted(grouped.items())]
    rng.shuffle(entries)
    entries = entries[:args.max_sources]
    if not entries:
        raise ValueError("No held-out source images")
    rows, images = [], []
    for level, parameters in LEVELS.items():
        dataset = BootstrapDataset(entries, root=args.root, crop_size=128, augment=False,
                                   deterministic=True, seed=20261026, fixed_parameters=parameters)
        for index in range(len(dataset)):
            item = dataset[index]
            noisy, target = item["noisy"][None].to(args.device), item["clean"][None].to(args.device)
            with torch.inference_mode():
                prediction = model(noisy)
            before, after = psnr(noisy, target), psnr(prediction, target)
            half = psnr(.5*prediction + .5*noisy, target)
            rows.append({"level": level, "source_sha256": entries[index]["source_sha256"],
                         "group": entries[index]["group"], "input_psnr": before,
                         "model_psnr": after, "gain_db": after-before,
                         "half_blend_psnr": half, "half_blend_gain_db": half-before})
            if level == "high" and index < 4:
                images.append((item["clean"].numpy(), item["noisy"].numpy(), prediction[0].cpu().numpy(), before, after))
        print("EVALUATED", level, len(dataset), flush=True)
    summary = {level: {key: float(np.mean([row[key] for row in rows if row["level"] == level]))
                       for key in ["input_psnr", "model_psnr", "gain_db", "half_blend_psnr", "half_blend_gain_db"]} for level in LEVELS}
    result = {"checkpoint": str(args.checkpoint), "step": saved["step"], "noise_parameters": LEVELS,
              "reference": "held-out low-ISO single frames; not verified noiseless",
              "metric_scope": "removal of additional simulated noise only; no real-camera ISO claim",
              "source_photos": len(entries), "split_checks": split_checks, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(images), 3, figsize=(10, 3.2*len(images)), squeeze=False)
    for row, (target, noisy, output, before, after) in enumerate(images):
        reference = (target[1]+target[3])/2
        scale = max(float(np.quantile(reference, .99)), .05)
        for col, (image, label) in enumerate([(target, "Low-ISO reference"), (noisy, f"Added noise: {before:.2f} dB"), (output, f"Model: {after:.2f} dB")]):
            green = np.clip((image[1]+image[3])/2/scale, 0, 1)**(1/2.2)
            axes[row, col].imshow(green, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(label)
            axes[row, col].axis("off")
    figure.suptitle("Provisional denoising — held-out photos with added synthetic noise\nGreen-plane detail; identical display transform within each row", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(figure)
    print("BOOTSTRAP_EVALUATION", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
