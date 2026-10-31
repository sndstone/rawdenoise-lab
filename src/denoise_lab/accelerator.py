"""Explicit experimental MPS body, retaining sensor transforms on the CPU.

Numeric canaries are runtime diagnostics, not photographic/render acceptance.
Environment controls must be exported before starting Python; this module never
changes them and cannot retrospectively establish when an API caller set them.
"""
import os
import copy
from contextlib import nullcontext

import torch
from torch import nn

from .model import gamma, inverse_gamma


class AcceleratorError(RuntimeError):
    """Recoverable accelerator operation/output failure, not a host error."""


def require_mps_environment():
    for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH"):
        if os.environ.get(key) != "0":
            raise ValueError(f"Experimental MPS body requires {key}=0 before Python startup")


class HostOuterBody(nn.Module):
    """CPU FP32 signed gamma/skip/inverse, explicit body device and synchronization."""
    def __init__(self, body, gamma_exponent, *, body_device="mps", synchronize=None):
        super().__init__()
        self.body = body
        self.gamma_exponent = gamma_exponent
        self.body_device = body_device
        self.synchronize = synchronize or (lambda: None)

    def forward(self, tile):
        if tile.device.type != "cpu" or tile.dtype != torch.float32:
            raise ValueError("Experimental body expects CPU float32 tiles")
        mps_context = torch.autocast('mps', enabled=False) if self.body_device == 'mps' else nullcontext()
        with torch.autocast('cpu', enabled=False), mps_context:
            g = gamma(tile, self.gamma_exponent)
            try:
                correction = self.body(g.to(self.body_device)).to("cpu")
                self.synchronize()
            except Exception as exc:
                raise AcceleratorError(f"MPS body/copy/synchronization failed: {type(exc).__name__}: {exc}") from exc
            if correction.shape != g.shape or correction.dtype != torch.float32:
                raise AcceleratorError("Experimental body returned wrong shape or dtype")
            out = inverse_gamma(correction + g, self.gamma_exponent)
        if not torch.isfinite(out).all():
            raise AcceleratorError("Experimental body produced nonfinite output")
        return out


def make_mps_body(model):
    # Copying keeps the canary's CPU reference independent of device transfers.
    body = copy.deepcopy(model.body)
    try:
        body = body.to(device="mps", dtype=torch.float32).eval()
    except Exception as exc:
        raise AcceleratorError(f"MPS body initialization failed: {type(exc).__name__}: {exc}") from exc
    return HostOuterBody(body, model.gamma_exponent, synchronize=torch.mps.synchronize).eval()


@torch.inference_mode()
def numeric_canary(reference, candidate):
    """Three fixed tiles; conservative engineering bounds in DN65535, not camera GT."""
    y, x = torch.meshgrid(torch.arange(512), torch.arange(512), indexing="ij")
    pattern = (.08 + .025 * torch.sin(x.float()*1.71) * torch.cos(y.float()*1.39)
               + .04 * (x > 256)).expand(1, 4, -1, -1).contiguous()
    reports = []
    flat = torch.full((1, 4, 512, 512), .9)
    shadow = torch.linspace(-3e-4, 2e-3, 512).reshape(1, 1, 1, 512).expand(1, 4, 512, -1).contiguous()
    # Kernel selection can depend on batch shape. Cover both supported call
    # shapes, including a partial final group in a batch2 image.
    for name, tile in [("flat_0.9", flat), ("texture_edge", pattern), ("signed_shadow_ramp", shadow),
                       ("flat_and_texture_batch2", torch.cat((flat, pattern))),
                       ("shadow_and_flat_batch2", torch.cat((shadow, flat)))]:
        with torch.autocast('cpu', enabled=False):
            expected = reference(tile)
            if not isinstance(expected, torch.Tensor) or expected.shape != tile.shape or expected.dtype != torch.float32 or not torch.isfinite(expected).all():
                raise ValueError(f"CPU numeric canary reference {name}: invalid output")
            observed = candidate(tile)
        if observed.shape != expected.shape or observed.dtype != torch.float32 or not torch.isfinite(observed).all():
            raise AcceleratorError(f"MPS numeric canary {name}: invalid output")
        error = (observed.double() - expected.double()) * 65535
        axes = (2, 3)  # Gate each sample/CFA independently, without cancellation.
        maximum = error.abs().amax(axes)
        rms = error.square().mean(axes).sqrt()
        bias = error.mean(axes).abs()
        reports.append(dict(case=name, batch=tile.shape[0], max_DN=maximum.tolist(), rms_DN=rms.tolist(),
                            abs_bias_DN=bias.tolist()))
        if not (torch.all(maximum <= .25) and torch.all(rms <= .02) and torch.all(bias <= .005)):
            raise AcceleratorError(f"MPS numeric canary {name} failed max0.25/RMS0.02/bias0.005 DN")
    return reports
