#!/usr/bin/env python3
"""Show provisional model outputs on real high-ISO crops, without quality scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from denoise_lab.bayer import read_mosaic, pack

from denoise_lab.arch_registry import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("runs/bootstrap/models.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/bootstrap/real_examples/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/bootstrap/real_examples.png"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    models = json.loads(args.models.read_text())
    entries = json.loads(args.manifest.read_text())
    outputs, loaded = [], {}
    for entry in entries:
        slug = "a7m3" if entry["camera"] == "ILCE-7M3" else "x1d2"
        device = "cuda:0" if slug == "a7m3" else "cuda:1"
        if slug not in loaded:
            saved = torch.load(models[slug]["checkpoint"], map_location="cpu", weights_only=True)
            model = build_model(saved["model_config"]).to(device).eval()
            model.load_state_dict(saved["model"])
            loaded[slug] = model
        inp = torch.from_numpy(pack(read_mosaic(Path("data")/entry["crop"])))[None].to(device)
        with torch.inference_mode():
            prediction = loaded[slug](inp)
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite real-RAW prediction")
        outputs.append((entry, inp[0].cpu().numpy(), prediction[0].cpu().numpy()))
    if not outputs:
        raise ValueError("No real RAW examples cached")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(outputs), 3, figsize=(12, 3.8*len(outputs)), squeeze=False)
    for row, (entry, inp, pred) in enumerate(outputs):
        green = (inp[1]+inp[3])/2
        scale = max(float(np.quantile(green, .99)), .03)
        for col, (array, label) in enumerate([(inp, "Input"), (pred, "Full model"), ((inp+pred)/2, "50% blend")]):
            display = np.clip((array[1]+array[3])/2/scale, 0, 1)**(1/2.2)
            axes[row, col].imshow(display, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            axes[row, col].set_title(f"{entry['camera']} | ISO {entry['iso']} | {label}", fontsize=9)
    figure.suptitle("Real high-ISO RAW crops — unpaired visual review\nNo clean reference or quality metric; same display exposure within each row", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .975))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print("REAL_PREVIEWS", len(outputs), args.output, flush=True)


if __name__ == "__main__":
    main()
