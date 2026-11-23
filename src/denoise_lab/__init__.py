"""Training from scratch for linear packed Bayer data; no bundled weights."""
__version__ = "0.2.0"
__all__ = ["DenoiseNet", "__version__"]


def __getattr__(name):
    if name == "DenoiseNet":
        from .model import DenoiseNet
        return DenoiseNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
