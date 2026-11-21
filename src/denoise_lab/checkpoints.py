"""Shared checkpoint validation and explicit weight selection."""
from collections.abc import Mapping
import pickle


def load_checkpoint(path):
    """Deserialize tensor-only checkpoints without exposing pickle diagnostics."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError, ValueError, IndexError) as exc:
        raise ValueError("Invalid checkpoint: could not deserialize model file") from exc


def model_config(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Invalid checkpoint: expected a mapping")
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("Invalid checkpoint: model_config must be a mapping")
    return config


def resolve_weights(checkpoint, weights="model"):
    """Prefer trainer EMA weights, falling back to the legacy ema key."""
    if weights not in {"model", "ema"}:
        raise ValueError("weights must be model or ema")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Invalid checkpoint: expected a mapping")
    import torch
    keys = ("ema_model", "ema") if weights == "ema" else ("model",)
    for key in keys:
        state = checkpoint.get(key)
        if state is not None:
            if not isinstance(state, Mapping) or not all(
                    isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
                raise ValueError(f"Invalid checkpoint: {key} weights must be a state mapping")
            return state
    raise ValueError(f"Checkpoint has no {weights} weights")
