"""Weight selection shared by inference, export and experiment tools."""


def checkpoint_weights(checkpoint, weights="model"):
    """Resolve current training EMA keys and legacy portable checkpoints.

    Training checkpoints use ``ema_model``; some portable artifacts use ``ema``.
    Prefer the training key when both exist and never silently fall back to model.
    """
    if weights not in {"model", "ema"}:
        raise ValueError("weights must be model or ema")
    keys = ("ema_model", "ema") if weights == "ema" else ("model",)
    for key in keys:
        if checkpoint.get(key) is not None:
            return checkpoint[key]
    raise ValueError(f"Checkpoint has no {weights} weights")
