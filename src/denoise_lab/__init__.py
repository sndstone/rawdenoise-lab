"""Training from scratch for linear packed Bayer data; no bundled weights."""
__all__ = ["DenoiseNet"]


def __getattr__(name):
    if name == "DenoiseNet":
        from .model import DenoiseNet
        return DenoiseNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
