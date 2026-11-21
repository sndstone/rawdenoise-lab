#!/usr/bin/env python3
"""Blend compatible checkpoint weights without training or optimizer state."""
import argparse
from collections.abc import Mapping
from decimal import Decimal
import math
from pathlib import Path

import torch

from denoise_lab.checkpoints import resolve_weights
from denoise_lab.models_registry import model_hash


def interpolate_checkpoints(a, b, alphas, output_dir, weights="model"):
    """Write loader-compatible weights; alpha=0 selects A, alpha=1 selects B."""
    if weights not in {"model", "ema"}:
        raise ValueError("weights must be model or ema")
    alphas = [float(alpha) for alpha in alphas]
    if not alphas or any(not math.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in alphas):
        raise ValueError("alpha must be finite and between 0 and 1")
    output_dir = Path(output_dir)
    names = [format(Decimal(str(alpha)).normalize(), "f").replace(".", "") for alpha in alphas]
    outputs = [output_dir/f"mix_a{name}.pt" for name in names]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Duplicate alpha output names")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    parents, saved, states = {}, [], []
    for label, source in (("a", a), ("b", b)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Parent {label} must be a checkpoint file: {source}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_config"), Mapping):
            raise ValueError(f"Parent {label} has no model_config mapping")
        try:  # accepts the trainer's ema_model key as well as legacy ema
            states.append(resolve_weights(checkpoint, weights))
        except ValueError as exc:
            raise ValueError(f"Parent {label}: {exc}") from exc
        parents[label] = {"path": str(source), "sha256": model_hash(path)}
        saved.append(checkpoint)
    first, second = saved
    if first["model_config"] != second["model_config"]:
        raise ValueError("Parents have different model_config")
    left, right = states
    if left.keys() != right.keys():
        raise ValueError("Parents have different tensor keys")
    for key, tensor in left.items():
        other = right[key]
        if not isinstance(tensor, torch.Tensor) or not isinstance(other, torch.Tensor):
            raise ValueError(f"State entry {key} must be a tensor")
        if tensor.shape != other.shape:
            raise ValueError(f"Parents have different tensor shapes for {key}")
        if tensor.dtype != other.dtype:
            raise ValueError(f"Parents have different tensor dtypes for {key}")
        if not tensor.is_floating_point() and not torch.equal(tensor, other):
            raise ValueError(f"Parents have unequal non-floating tensor {key}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for alpha, path in zip(alphas, outputs):
        state = {}
        for key, tensor in left.items():
            if alpha == 0 or not tensor.is_floating_point():
                state[key] = tensor
            elif alpha == 1:
                state[key] = right[key]
            else:
                state[key] = (1-alpha)*tensor + alpha*right[key]
        artifact = {"model_config": first["model_config"], "model": state,
                    "provenance": {"parents": parents, "weights": weights, "alpha": alpha}}
        with path.open("xb") as stream:
            torch.save(artifact, stream)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--alpha", type=float, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=["model", "ema"], default="model")
    args = parser.parse_args()
    try:
        outputs = interpolate_checkpoints(args.a, args.b, args.alpha, args.output_dir, args.weights)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
