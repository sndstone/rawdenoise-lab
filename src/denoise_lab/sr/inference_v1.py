"""Isolated v1 SR checkpoint loader, TLC pooling adapter, and feathered tiler.

Compatible with frozen research checkpoints:
- unet_fusion_noise_24k_s42/best.pt
- nvidia_input_fit_20261104/weights/control_s48_step024000.pt
- nvidia_input_fit_20261104/weights/control_s49_step024000.pt
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..model import NAFBlock, NAFNet
from ..tiling import reflected_bayer_pad
from .geometry import site_upsample


class PhotoSRv1(nn.Module):
    """Exact architectural reproduction of the frozen v1 UNet-NAF PhotoSR."""

    def __init__(
        self,
        mode: str = "fusion",
        width: int = 32,
        depth: int = 4,
        arch: str = "unet",
        base: str = "guide",
        context_apron: int = 16,
        tile_center: int = 80,
        tile_overlap: int = 40,
        **_,
    ):
        super().__init__()
        if mode not in ("raw", "guide", "fusion") or arch != "unet":
            raise ValueError("PhotoSRv1 supports arch='unet' with mode in {raw, guide, fusion}")
        self.mode = mode
        self.base = base
        self.arch = arch
        self.context_apron = int(context_apron)
        self.tile_center = int(tile_center)
        self.tile_overlap = int(tile_overlap)
        self.unet = NAFNet(
            in_channels=8 if mode == "fusion" else 4,
            out_channels=16,
            width=width,
            enc_blk_nums=(1, 1, 2),
            middle_blk_num=depth,
            dec_blk_nums=(1, 1, 1),
        )
        nn.init.zeros_(self.unet.ending.weight)
        nn.init.zeros_(self.unet.ending.bias)

    def forward(self, noisy: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if self.mode == "fusion":
            x = torch.cat((noisy, guide), dim=1)
        elif self.mode == "raw":
            x = noisy
        else:
            x = guide
        base = noisy if (self.mode == "raw" or self.base == "raw") else guide
        return site_upsample(base) + F.pixel_shuffle(self.unet(x), 2)


class LocalAvgPool2d(nn.Module):
    """Test-time Local Converter (TLC) drop-in replacement for AdaptiveAvgPool2d(1).

    Replaces global tile-wide pooling with a bounded local sliding-window average
    (reflect-padded) so bright specular lights in one region cannot shift channel
    attention gains far away.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        k = int(kernel_size)
        if k < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = k if k % 2 == 1 else k + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        kh = min(self.kernel_size, h if h % 2 == 1 else max(1, h - 1))
        kw = min(self.kernel_size, w if w % 2 == 1 else max(1, w - 1))
        if kh <= 1 and kw <= 1:
            return x
        py, px = kh // 2, kw // 2
        padded = F.pad(x, (px, px, py, py), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=(kh, kw), stride=1)


def apply_tlc(model: PhotoSRv1, base_kernel: int = 33, inplace: bool = False) -> PhotoSRv1:
    """Replace global AdaptiveAvgPool2d(1) in every NAFBlock.sca with LocalAvgPool2d.

    Scales the local window proportionally to each U-Net resolution stage:
    stage s uses odd window ~ max(3, base_kernel // 2**s).
    """
    target = model if inplace else copy.deepcopy(model)

    def _patch_seq(seq: nn.Sequential, k: int) -> None:
        for blk in seq:
            if isinstance(blk, NAFBlock) and isinstance(blk.sca, nn.Sequential):
                if isinstance(blk.sca[0], (nn.AdaptiveAvgPool2d, LocalAvgPool2d)):
                    blk.sca[0] = LocalAvgPool2d(k)

    for s, enc in enumerate(target.unet.encoders):
        _patch_seq(enc, max(3, base_kernel // (2**s)))
    bot_scale = 2 ** len(target.unet.encoders)
    _patch_seq(target.unet.middle_blks, max(3, base_kernel // bot_scale))
    for i, dec in enumerate(target.unet.decoders):
        s = len(target.unet.decoders) - 1 - i
        _patch_seq(dec, max(3, base_kernel // (2**s)))
    return target


def load_v1_checkpoint(path: str | Path, device: str = "cpu") -> tuple[PhotoSRv1, dict]:
    """Load a frozen v1 PhotoSR checkpoint (either campaign best.pt or control_s48/49)."""
    path = Path(path)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        cp = torch.load(path, map_location="cpu", weights_only=True)
    if "model_metadata" in cp:
        cfg = dict(cp["model_metadata"]["config"])
        meta = dict(cp["model_metadata"])
        meta["checkpoint_kind"] = "nvidia_input_fit_control"
    elif "config" in cp:
        cfg = dict(cp["config"])
        meta = {
            "config": cfg,
            "step": cp.get("step"),
            "checkpoint_kind": "campaign_v1",
            "guide_contract": cp.get("guide_contract"),
        }
    else:
        raise ValueError(f"Unrecognized v1 SR checkpoint layout: {path}")
    model = PhotoSRv1(**cfg)
    model.load_state_dict(cp["model"], strict=True)
    model.to(device).eval()
    return model, meta


def _feather_1d(length: int, overlap: int, mode: str, device: torch.device) -> torch.Tensor:
    axis = torch.ones(length, device=device, dtype=torch.float32)
    if overlap <= 0:
        return axis
    ramp = (torch.arange(overlap, device=device, dtype=torch.float32) + 1.0) / (overlap + 1.0)
    if mode == "cosine":
        ramp = 0.5 - 0.5 * torch.cos(math.pi * ramp)
    elif mode != "linear":
        raise ValueError(f"Unsupported feather_mode: {mode}")
    axis[:overlap] = ramp
    axis[-overlap:] = torch.minimum(axis[-overlap:], ramp.flip(0))
    return axis


@torch.inference_mode()
def context_tiled_sr(
    model: nn.Module,
    noisy: torch.Tensor,
    guide: torch.Tensor,
    center: int = 80,
    apron: int = 16,
    overlap: int = 40,
    feather_mode: str = "cosine",
    tile_batch_size: int = 1,
) -> torch.Tensor:
    """Fixed-context paired CFA tiling with site-aware Bayer reflection and feathering."""
    if noisy.shape != guide.shape or noisy.ndim != 4 or noisy.shape[1] != 4:
        raise ValueError("Expected matched Bx4xHxW noisy and guide tensors")
    if center <= 0 or apron < 0 or not 0 <= overlap < center:
        raise ValueError("Invalid fixed-context SR center, apron, or overlap")
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be >= 1")

    b, _, h, w = noisy.shape
    step = center - overlap
    hh = center + math.ceil(max(0, h - center) / step) * step
    ww = center + math.ceil(max(0, w - center) / step) * step

    y = reflected_bayer_pad(noisy, apron, apron + ww - w, apron, apron + hh - h)
    d = reflected_bayer_pad(guide, apron, apron + ww - w, apron, apron + hh - h)

    out_shape = (b, 4, hh * 2, ww * 2)
    result = (torch.zeros if overlap else torch.empty)(out_shape, device=noisy.device, dtype=torch.float32)
    if overlap:
        axis = _feather_1d(center * 2, overlap * 2, feather_mode, noisy.device)
        weight = (axis[:, None] * axis[None, :])[None, None]
        weights = torch.zeros((1, 1, hh * 2, ww * 2), device=noisy.device, dtype=torch.float32)

    tile_span = center + 2 * apron
    coords = [(top, left) for top in range(0, hh - center + 1, step) for left in range(0, ww - center + 1, step)]

    for i in range(0, len(coords), tile_batch_size):
        batch_coords = coords[i : i + tile_batch_size]
        y_tiles = [y[..., t : t + tile_span, l : l + tile_span] for t, l in batch_coords]
        d_tiles = [d[..., t : t + tile_span, l : l + tile_span] for t, l in batch_coords]
        yt = y_tiles[0] if len(y_tiles) == 1 else torch.cat(y_tiles, dim=0)
        dt = d_tiles[0] if len(d_tiles) == 1 else torch.cat(d_tiles, dim=0)
        preds = model(yt, dt)
        for (top, left), pred in zip(batch_coords, preds.split(b, dim=0)):
            patch = pred[..., 2 * apron : 2 * (apron + center), 2 * apron : 2 * (apron + center)]
            if overlap:
                result[..., 2 * top : 2 * (top + center), 2 * left : 2 * (left + center)] += patch * weight
                weights[..., 2 * top : 2 * (top + center), 2 * left : 2 * (left + center)] += weight
            else:
                result[..., 2 * top : 2 * (top + center), 2 * left : 2 * (left + center)] = patch

    if overlap:
        result.div_(weights)
    return result[..., : 2 * h, : 2 * w]
