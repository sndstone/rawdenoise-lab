#!/usr/bin/env python3
"""Evaluate our checkpoint on a registered held-out paired manifest."""
import argparse
import json
import numpy as np
import torch
from denoise_lab.data import PairedBurstDataset

from denoise_lab.arch_registry import build_model
from denoise_lab.metrics import evaluate_pair
from denoise_lab.tiling import tiled_inference


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
        cond = batch["noise_map"][None] if model.cond == "noise_map" else None
        with torch.inference_mode():
            prediction = tiled_inference(model, noisy, noise_map=cond)
            row = {"index": i, "input": evaluate_pair(noisy, clean), "ours": evaluate_pair(prediction, clean)}
        rows.append(row)
        print(json.dumps(row))
    if not rows:
        raise ValueError("Empty evaluation manifest")
    summary = {name: {key: float(np.mean([r[name][key] for r in rows])) for key in rows[0][name]}
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
