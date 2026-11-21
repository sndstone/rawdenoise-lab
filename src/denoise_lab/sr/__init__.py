"""Isolated 2x RAW-to-RAW Super-Resolution package (Track A + Track B).

Kept strictly separate from the frozen production photo denoiser (`r4_plain`).
All packed Bayer tensors use canonical R/G1/B/G2 channel order (Bx4xHxW).
"""
from .geometry import (
    SITES,
    align_planes,
    edge_energy,
    inverse_asinh,
    luminance,
    signed_asinh,
    site_upsample,
)
from .inference_v1 import (
    LocalAvgPool2d,
    PhotoSRv1,
    apply_tlc,
    context_tiled_sr,
    load_v1_checkpoint,
)
from .postprocess import (
    RefineConfig,
    cfa_chroma_guard,
    compressive_range_guard,
    highlight_blend_sr,
    refine_sr_prediction,
)
from .model_v2 import (
    LocalNAFBlock,
    LocalNAFNet,
    LocalSCA,
    PhotoSRv2,
    PhotoSRv2Config,
    infer_photosr_v2,
)
from .degradation_v2 import (
    DegradationParams,
    anisotropic_psf_kernel,
    forward_degrade_v2,
    inject_specular_highlights,
    sample_degradation_params,
    simulate_lr_observation,
    synthesize_batch_on_device,
)
from .losses_v2 import SRv2Loss

__all__ = [
    "SITES",
    "site_upsample",
    "align_planes",
    "luminance",
    "edge_energy",
    "signed_asinh",
    "inverse_asinh",
    "PhotoSRv1",
    "LocalAvgPool2d",
    "apply_tlc",
    "context_tiled_sr",
    "load_v1_checkpoint",
    "RefineConfig",
    "highlight_blend_sr",
    "compressive_range_guard",
    "cfa_chroma_guard",
    "refine_sr_prediction",
    "LocalSCA",
    "LocalNAFBlock",
    "LocalNAFNet",
    "PhotoSRv2Config",
    "PhotoSRv2",
    "infer_photosr_v2",
    "DegradationParams",
    "anisotropic_psf_kernel",
    "forward_degrade_v2",
    "sample_degradation_params",
    "simulate_lr_observation",
    "SRv2Loss",
]
