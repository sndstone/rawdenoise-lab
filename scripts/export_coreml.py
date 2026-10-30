#!/usr/bin/env python3
"""Export our own checkpoint at the fixed ShopPhoto 4-plane contract."""
import argparse
import numpy as np
import torch
from denoise_lab.arch_registry import build_model


def model_from_checkpoint(checkpoint, weights="model"):
    """Build and load an unconditioned model without importing Core ML tools."""
    if weights not in {"model", "ema"}:
        raise ValueError("weights must be model or ema")
    if checkpoint["model_config"].get("cond", "none") != "none":
        raise ValueError("Fixed 4-channel deployment contract requires cond=none")
    model = build_model(checkpoint["model_config"]).eval()
    if checkpoint.get(weights) is None:
        raise ValueError(f"Checkpoint has no {weights} weights")
    model.load_state_dict(checkpoint[weights])
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Destination .mlpackage")
    p.add_argument("--weights", choices=["model", "ema"], default="model",
                   help="Checkpoint weights to export (default: model)")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        model = model_from_checkpoint(checkpoint, args.weights)
    except ValueError as exc:
        p.error(str(exc))
    import coremltools as ct
    example = torch.zeros(1, 4, 512, 512, dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    converted = ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                           inputs=[ct.TensorType(name="inp_1", shape=example.shape, dtype=np.float32)],
                           outputs=[ct.TensorType(name="output", dtype=np.float32)])
    converted.save(args.output)
    print(f"Saved {args.output}; float32 I/O, fp16 compute")


if __name__ == "__main__":
    main()
