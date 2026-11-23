"""Track A inference-side refinement guards for frozen linear-space 2x SR models.

All operations preserve signed float32 R/G1/B/G2 camera-linear contracts:
- Zero alteration of unclipped midtone/shadow regions when no saturation/ringing occurs.
- Shared Bayer-cell highlight protection around clipped photosites (>= 0.98).
- Smooth C1 compressive-domain range guard preventing negative undershoot halos on
  dark-to-bright specular steps without flat-top hard clipping.
- CFA-aligned chroma and G1/G2 residual damping on high-contrast edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .geometry import (
    align_planes,
    edge_energy,
    inverse_asinh,
    luminance,
    signed_asinh,
    site_upsample,
)


@dataclass(frozen=True)
class RefineConfig:
    # Highlight protection near clipped sensor photosites (>= threshold)
    use_highlight_blend: bool = True
    highlight_threshold: float = 0.98
    highlight_core_dilate_lr: int = 1
    highlight_feather_steps_lr: int = 5
    highlight_cap_white: bool = True

    # CFA-aligned chroma / G1-G2 residual damping at strong edges
    use_chroma_guard: bool = True
    chroma_edge_threshold: float = 0.02
    chroma_reduction: float = 0.65
    green_reduction: float = 0.75

    # Compressive-domain smooth local-envelope range guard
    use_range_guard: bool = True
    range_radius_lr: int = 1
    range_domain_scale: float = 0.02
    range_slack_abs: float = 0.0015
    range_slack_rel: float = 0.025
    range_positive_floor_frac: float = 0.35
    range_soft_knee: float = 0.45
    range_cap_unclipped_ceiling: bool = True
    range_preserve_point_sources: bool = True


def _soft_upper_bound(x: torch.Tensor, hi: torch.Tensor, knee: torch.Tensor) -> torch.Tensor:
    """C1-smooth identity below (hi - knee), asymptotically bounded strictly by hi."""
    k = knee.clamp_min(1e-7)
    start = hi - k
    excess = (x - start).clamp_min(0.0)
    return torch.where(x > start, start + k * torch.tanh(excess / k), x)


def _soft_lower_bound(x: torch.Tensor, lo: torch.Tensor, knee: torch.Tensor) -> torch.Tensor:
    """C1-smooth identity above (lo + knee), asymptotically bounded strictly by lo."""
    k = knee.clamp_min(1e-7)
    start = lo + k
    deficit = (start - x).clamp_min(0.0)
    return torch.where(x < start, start - k * torch.tanh(deficit / k), x)


def highlight_blend_sr(
    pred: torch.Tensor,
    guide_lr: torch.Tensor,
    original_lr: torch.Tensor,
    *,
    threshold: float = 0.98,
    core_dilate_lr: int = 1,
    feather_steps_lr: int = 5,
    cap_white: bool = True,
) -> torch.Tensor:
    """Blend SR back toward CFA-upsampled guide inside & around clipped photosites.

    Any photosite >= threshold in `original_lr` (or `guide_lr`) seeds a shared
    Bayer-cell mask so all four CFA planes blend with identical weight, preserving
    white-balance colour ratios. Unclipped scenes (e.g. room, mushroom) return `pred`
    unchanged.
    """
    sat = ((original_lr >= threshold) | (guide_lr >= threshold)).any(dim=1, keepdim=True).to(pred.dtype)
    if not sat.any():
        return pred

    mask = sat
    for _ in range(max(0, int(core_dilate_lr))):
        mask = F.max_pool2d(mask, 3, stride=1, padding=1)

    if feather_steps_lr > 0:
        weight_lr = mask.clone()
        cur = mask
        # Accumulate smooth linear ramp across feather_steps_lr rings (total 1 + feather_steps)
        denom = float(feather_steps_lr + 1)
        for _ in range(int(feather_steps_lr)):
            cur = F.max_pool2d(cur, 3, stride=1, padding=1)
            weight_lr = weight_lr + cur
        weight_lr = (weight_lr / denom).clamp(0.0, 1.0)
        # Mild 3x3 smoothing in LR removes blocky max-pool octagon corners
        weight_lr = F.avg_pool2d(F.pad(weight_lr, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        # Keep saturated core strictly at 1.0
        weight_lr = torch.maximum(weight_lr, mask)
    else:
        weight_lr = mask

    weight_hr = F.interpolate(weight_lr, size=pred.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    base_hr = site_upsample(guide_lr)
    blended = weight_hr * base_hr + (1.0 - weight_hr) * pred

    if cap_white:
        orig_hr = site_upsample(original_lr)
        ceiling = torch.where((orig_hr < 1.0) & (base_hr < 1.0), 1.0 - 2**-24, torch.maximum(orig_hr, base_hr).clamp_max(1.0))
        blended = torch.where(weight_hr > 0.0, torch.minimum(blended, ceiling), blended)
    return blended


def cfa_chroma_guard(
    pred: torch.Tensor,
    guide_lr: torch.Tensor,
    *,
    edge_threshold: float = 0.02,
    chroma_reduction: float = 0.65,
    green_reduction: float = 0.75,
) -> torch.Tensor:
    """Damp cross-channel chroma & G1/G2 residual ringing on strong luminance edges.

    Operates on the common red-site lattice via `align_planes`, preserves the shared
    luminance sharpening residual, and resamples the correction back to native CFA sites.
    """
    base = site_upsample(guide_lr)
    residual = align_planes(pred - base)
    common = align_planes(base)
    lum_grad = edge_energy(luminance(common))
    # Smooth quadratic ramp: almost 0 on weak midtone textures, saturates at 1 on strong steps
    ratio = (lum_grad / max(edge_threshold, 1e-6)).clamp(0.0, 1.0)
    gate = ratio * ratio * (3.0 - 2.0 * ratio)  # smoothstep

    lum_res = luminance(residual)
    chroma_res = residual - lum_res
    correction = chroma_reduction * chroma_res

    g_diff = 0.5 * (residual[:, 1:2] - residual[:, 3:4])
    extra_g = (green_reduction - chroma_reduction) * g_diff
    correction = correction.clone()
    correction[:, 1:2] = correction[:, 1:2] + extra_g
    correction[:, 3:4] = correction[:, 3:4] - extra_g

    return pred - align_planes(gate * correction, inverse=True)


def compressive_range_guard(
    pred: torch.Tensor,
    guide_lr: torch.Tensor,
    *,
    radius_lr: int = 1,
    domain_scale: float = 0.02,
    slack_abs: float = 0.0015,
    slack_rel: float = 0.025,
    positive_floor_frac: float = 0.35,
    soft_knee: float = 0.45,
    cap_unclipped_ceiling: bool = True,
    preserve_point_sources: bool = True,
) -> torch.Tensor:
    """Bound same-channel SR overshoots/undershoots smoothly in compressive asinh space.

    Unlike symmetric linear slack (which permits severe negative undershoots on
    0.003 -> 1.0 night-scene edges), compressive-domain slack automatically shrinks
    the allowed linear undershoot near dark shadows while keeping midtone edge
    sharpening intact.

    When `preserve_point_sources=True`, the *lower* bound and `<= 1.0` white ceiling
    remain unconditional (eliminating all negative black/cyan/red halos and >1.0
    blowups), while the *upper* local-max bound relaxes smoothly for dim isolated
    point sources (e.g. sub-pixel stars) whose true HR peak exceeds the blurred LR max.
    """
    r = int(radius_lr)
    k = 2 * r + 1
    padded = F.pad(guide_lr, (r, r, r, r), mode="replicate")
    hi_lr = F.max_pool2d(padded, k, stride=1)
    lo_lr = -F.max_pool2d(-padded, k, stride=1)
    hi_lin = site_upsample(hi_lr)
    lo_lin = site_upsample(lo_lr)

    z_pred = signed_asinh(pred, scale=domain_scale)
    z_hi = signed_asinh(hi_lin, scale=domain_scale)
    z_lo = signed_asinh(lo_lin, scale=domain_scale)

    z_abs = signed_asinh(pred.new_tensor(slack_abs), scale=domain_scale)
    z_slack = z_abs + slack_rel * (z_hi - z_lo).clamp_min(0.0)

    z_hi_bound = z_hi + z_slack
    z_lo_bound = z_lo - z_slack

    if preserve_point_sources:
        # Extended contrast support in 7x7 LR neighborhood: high on extended edges,
        # low on isolated 1-2px stars. Bright highlights (>=0.70) always keep tight bound.
        contrast_lr = hi_lr - lo_lr
        support_lr = F.avg_pool2d(
            F.pad((contrast_lr > 0.01).to(pred.dtype), (3, 3, 3, 3), mode="replicate"),
            7,
            stride=1,
        )
        edge_gate = ((support_lr - 0.22) / 0.18).clamp(0.0, 1.0)
        bright_gate = ((hi_lr - 0.55) / 0.15).clamp(0.0, 1.0)
        tight_gate = site_upsample(torch.maximum(edge_gate, bright_gate)).clamp(0.0, 1.0)
        z_point_cap = signed_asinh((2.3 * hi_lin + 0.015).clamp_max(1.0), scale=domain_scale)
        z_hi_bound = tight_gate * z_hi_bound + (1.0 - tight_gate) * torch.maximum(z_hi_bound, z_point_cap)

    # Guarantee positive local shadows cannot undershoot into negative black-clip halos,
    # while preserving signed below-black read noise where lo_lin <= 0.
    if positive_floor_frac > 0.0:
        pos_floor = signed_asinh((positive_floor_frac * lo_lin).clamp_min(0.0), scale=domain_scale)
        z_lo_bound = torch.where(lo_lin > 0.0, torch.maximum(z_lo_bound, pos_floor), z_lo_bound)

    if cap_unclipped_ceiling:
        z_one = signed_asinh(pred.new_tensor(1.0), scale=domain_scale)
        z_hi_bound = torch.where(hi_lin <= 1.0, torch.minimum(z_hi_bound, z_one), z_hi_bound)

    # Soft C1 knee inside the slack band so well-behaved pixels are untouched
    knee = (soft_knee * z_slack).clamp_min(1e-6)
    z_out = _soft_upper_bound(z_pred, z_hi_bound, knee)
    z_out = _soft_lower_bound(z_out, z_lo_bound, knee)
    return inverse_asinh(z_out, scale=domain_scale)


def refine_sr_prediction(
    pred: torch.Tensor,
    guide_lr: torch.Tensor,
    original_lr: torch.Tensor,
    config: RefineConfig | None = None,
) -> torch.Tensor:
    """Apply Track A inference-side refinement pipeline to a 2x SR prediction."""
    cfg = config or RefineConfig()
    out = pred
    if cfg.use_highlight_blend:
        out = highlight_blend_sr(
            out,
            guide_lr,
            original_lr,
            threshold=cfg.highlight_threshold,
            core_dilate_lr=cfg.highlight_core_dilate_lr,
            feather_steps_lr=cfg.highlight_feather_steps_lr,
            cap_white=cfg.highlight_cap_white,
        )
    if cfg.use_chroma_guard:
        out = cfa_chroma_guard(
            out,
            guide_lr,
            edge_threshold=cfg.chroma_edge_threshold,
            chroma_reduction=cfg.chroma_reduction,
            green_reduction=cfg.green_reduction,
        )
    if cfg.use_range_guard:
        out = compressive_range_guard(
            out,
            guide_lr,
            radius_lr=cfg.range_radius_lr,
            domain_scale=cfg.range_domain_scale,
            slack_abs=cfg.range_slack_abs,
            slack_rel=cfg.range_slack_rel,
            positive_floor_frac=cfg.range_positive_floor_frac,
            soft_knee=cfg.range_soft_knee,
            cap_unclipped_ceiling=cfg.range_cap_unclipped_ceiling,
            preserve_point_sources=cfg.range_preserve_point_sources,
        )
    return out
