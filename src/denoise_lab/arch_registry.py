"""Model factory shared by architecture study training and measurements."""
from __future__ import annotations

from .model import DenoiseNet
from .arch_study import StudyDenoiseNet
from .diffusion import BridgeRefiner


def build_model(model_cfg):
    cfg = dict(model_cfg)
    arch = cfg.pop("arch", "nafnet")
    builders = {"nafnet": DenoiseNet, "study": StudyDenoiseNet, "bridge_refiner": BridgeRefiner}
    if arch not in builders:
        raise ValueError("Unknown architecture: %s" % arch)
    return builders[arch](**cfg)
