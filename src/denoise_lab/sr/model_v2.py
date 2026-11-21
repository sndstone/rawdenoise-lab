"""Track B: Principled 2x RAW-to-RAW SR architecture (PhotoSRv2).

Addresses all four diagnosed v1 failure modes structurally:
1. Compressive residual domain (`asinh` default, smooth at 0, preserves signed below-black
   values, compresses 200:1 highlight steps) with smooth residual ceiling.
2. Explicit per-plane 4-channel saturation indicators so per-CFA clipping (e.g. R
   saturated while B unclipped on warm streetlights) is visible to the network.
3. Shift-equivariant Local-Window SCA (`LocalSCA`) replacing tile-wide AdaptiveAvgPool2d(1),
   eliminating tile breathing and tile-size mismatch.
4. Optional post-PixelShuffle HR refinement tail (`hr_channels > 0`) allowing the 4
   high-resolution subpixels to coordinate spatially and eliminate 2x2 checkerboard /
   G1-G2 zippering.
5. Zero-initialized output conv on top of exact CFA-site-aware upsampled guide.

Clean-room: trained strictly from random initialization; never loads external weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..model import GAMMA, LayerNorm2d, SimpleGate
from .geometry import from_compressive, site_upsample, to_compressive


class LocalSCA(nn.Module):
    """Shift-equivariant local-window Simple Channel Attention.

    Uses a bounded reflect-padded local average pool instead of global
    AdaptiveAvgPool2d(1), guaranteeing identical interior predictions across
    arbitrary tile sizes (zero tile-size breathing).
    """

    def __init__(self, channels: int, kernel_size: int = 15):
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("LocalSCA kernel_size must be a positive odd integer")
        self.kernel_size = k
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            pooled = x
        else:
            h, w = x.shape[-2:]
            kh = min(self.kernel_size, h if h % 2 == 1 else max(1, h - 1))
            kw = min(self.kernel_size, w if w % 2 == 1 else max(1, w - 1))
            py, px = kh // 2, kw // 2
            pooled = F.avg_pool2d(F.pad(x, (px, px, py, py), mode="reflect"), (kh, kw), stride=1)
        return self.proj(pooled)


class LocalNAFBlock(nn.Module):
    """NAFBlock with shift-equivariant LocalSCA instead of global AdaptiveAvgPool2d(1)."""

    def __init__(
        self,
        channels: int,
        sca_kernel: int = 15,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        dw, ff = channels * dw_expand, channels * ffn_expand
        if dw % 2 or ff % 2:
            raise ValueError("SimpleGate needs even expanded widths")
        d_out, f_out = dw // 2, ff // 2
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(d_out, channels, 1)
        self.sca = LocalSCA(d_out, kernel_size=sca_kernel)
        self.gate = SimpleGate()
        self.conv4 = nn.Conv2d(channels, ff, 1)
        self.conv5 = nn.Conv2d(f_out, channels, 1)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.gate(self.conv2(self.conv1(self.norm1(x))))
        y = x + self.beta * self.drop1(self.conv3(z * self.sca(z)))
        z = self.conv5(self.gate(self.conv4(self.norm2(y))))
        return y + self.gamma * self.drop2(z)


class LocalNAFNet(nn.Module):
    """Multi-scale U-Net built from shift-equivariant LocalNAFBlocks."""

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 64,
        width: int = 32,
        enc_blk_nums: tuple[int, ...] = (1, 1, 2),
        middle_blk_num: int = 4,
        dec_blk_nums: tuple[int, ...] = (1, 1, 1),
        sca_kernels: tuple[int, ...] = (15, 11, 7, 5),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("Encoder and decoder stage counts must match")
        if len(sca_kernels) != len(enc_blk_nums) + 1:
            raise ValueError("sca_kernels must provide one odd kernel size per encoder stage plus bottleneck")
        self.multiple = 2 ** len(enc_blk_nums)
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = [nn.ModuleList() for _ in range(4)]

        def make_blocks(c: int, n: int, k: int) -> nn.Sequential:
            return nn.Sequential(
                *(LocalNAFBlock(c, sca_kernel=k, dw_expand=dw_expand, ffn_expand=ffn_expand, dropout=dropout) for _ in range(n))
            )

        c = width
        for s, n in enumerate(enc_blk_nums):
            self.encoders.append(make_blocks(c, n, sca_kernels[s]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2))
            c *= 2
        self.middle_blks = make_blocks(c, middle_blk_num, sca_kernels[-1])
        for i, n in enumerate(dec_blk_nums):
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            s = len(dec_blk_nums) - 1 - i
            self.decoders.append(make_blocks(c, n, sca_kernels[s]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self.multiple or w % self.multiple:
            x = F.pad(x, (0, (-w) % self.multiple, 0, (-h) % self.multiple), mode="reflect")
        x, skips = self.intro(x), []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(up(x) + skip)
        return self.ending(x)[..., :h, :w]


@dataclass(frozen=True)
class PhotoSRv2Config:
    architecture: str = "PhotoSRv2_LocalNAF_Compressive_2x"
    mode: str = "fusion"  # "fusion" | "raw" | "guide"
    width: int = 32
    enc_blk_nums: tuple[int, ...] = (1, 1, 2)
    middle_blk_num: int = 4
    dec_blk_nums: tuple[int, ...] = (1, 1, 1)
    sca_kernels: tuple[int, ...] = (15, 11, 7, 5)
    hr_channels: int = 16  # Post-PixelShuffle HR spatial refinement width (0 = direct 16->4 shuffle)
    domain: str = "asinh"  # "asinh" | "gamma"
    domain_scale: float = 0.05
    gamma_exp: float = GAMMA
    use_sat_channel: bool = True
    sat_per_plane: bool = True  # 4 per-CFA saturation indicators (R/G1/B/G2) vs 1 shared
    sat_threshold: float = 0.98
    residual_limit: float = 1.5  # Smooth tanh ceiling in normalized compressive domain (0 disables)
    soft_white_cap: bool = True  # Differentiable C1 ceiling preventing sinh blowup >1.0
    soft_white_cap_start: float = 1.0  # Exact identity for all linear <= 1.0 when start=1.0
    soft_white_cap_knee: float = 0.003  # Strictly <= 1.003 in asinh (~1.011 linear)
    dw_expand: int = 2
    ffn_expand: int = 2
    dropout: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("enc_blk_nums", "dec_blk_nums", "sca_kernels"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PhotoSRv2Config":
        merged = dict(d)
        # Backward compatibility with earlier v2 checkpoints
        if "soft_white_cap" not in merged:
            merged["soft_white_cap"] = False
        elif "soft_white_cap_start" not in merged:
            merged["soft_white_cap_start"] = 0.975
            merged["soft_white_cap_knee"] = 0.025
        for k in ("enc_blk_nums", "dec_blk_nums", "sca_kernels"):
            if k in merged:
                merged[k] = tuple(merged[k])
        valid = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in merged.items() if k in valid})


class PhotoSRv2(nn.Module):
    """Compressive-domain, shift-equivariant 2x Bayer RAW super-resolution network."""

    def __init__(self, config: PhotoSRv2Config | None = None, **overrides):
        super().__init__()
        if config is None:
            config = PhotoSRv2Config(**overrides)
        elif overrides:
            merged = {**asdict(config), **overrides}
            config = PhotoSRv2Config.from_dict(merged)
        if config.mode not in ("fusion", "raw", "guide"):
            raise ValueError(f"Unsupported mode: {config.mode}")
        if config.domain not in ("asinh", "gamma"):
            raise ValueError(f"Unsupported compressive domain: {config.domain}")

        self.config = config
        sat_ch = (4 if config.sat_per_plane else 1) if config.use_sat_channel else 0
        in_ch = (8 if config.mode == "fusion" else 4) + sat_ch
        out_ch = 4 * config.hr_channels if config.hr_channels > 0 else 16

        self.body = LocalNAFNet(
            in_channels=in_ch,
            out_channels=out_ch,
            width=config.width,
            enc_blk_nums=config.enc_blk_nums,
            middle_blk_num=config.middle_blk_num,
            dec_blk_nums=config.dec_blk_nums,
            sca_kernels=config.sca_kernels,
            dw_expand=config.dw_expand,
            ffn_expand=config.ffn_expand,
            dropout=config.dropout,
        )
        if config.hr_channels > 0:
            self.hr_tail = nn.Sequential(
                nn.Conv2d(config.hr_channels, config.hr_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(config.hr_channels, 4, 3, padding=1),
            )
            # Zero-init final HR conv => exact site_upsample(base) at initialization
            nn.init.zeros_(self.hr_tail[-1].weight)
            nn.init.zeros_(self.hr_tail[-1].bias)
        else:
            self.hr_tail = None
            nn.init.zeros_(self.body.ending.weight)
            nn.init.zeros_(self.body.ending.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return to_compressive(x, domain=self.config.domain, scale=self.config.domain_scale, gamma_exp=self.config.gamma_exp)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return from_compressive(z, domain=self.config.domain, scale=self.config.domain_scale, gamma_exp=self.config.gamma_exp)

    def forward(self, noisy: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if noisy.shape != guide.shape or noisy.ndim != 4 or noisy.shape[1] != 4:
            raise ValueError("Expected matched Bx4xHxW noisy and guide tensors")
        cfg = self.config
        z_y = self._encode(noisy)
        z_d = self._encode(guide)

        if cfg.mode == "fusion":
            feats = [z_y, z_d]
        elif cfg.mode == "raw":
            feats = [z_y]
        else:
            feats = [z_d]

        if cfg.use_sat_channel:
            lo_thr = cfg.sat_threshold - 0.04
            peak = torch.maximum(noisy, guide)
            if not cfg.sat_per_plane:
                peak = peak.amax(dim=1, keepdim=True)
            sat = ((peak - lo_thr) / 0.04).clamp(0.0, 1.0)
            feats.append(sat)

        up = F.pixel_shuffle(self.body(torch.cat(feats, dim=1)), 2)
        delta_z = self.hr_tail(up) if self.hr_tail is not None else up
        if cfg.residual_limit > 0:
            delta_z = cfg.residual_limit * torch.tanh(delta_z / cfg.residual_limit)

        base_lin = site_upsample(noisy if cfg.mode == "raw" else guide)
        z_base = self._encode(base_lin)
        z_out = z_base + delta_z
        if cfg.soft_white_cap:
            start = float(cfg.soft_white_cap_start)
            knee = max(float(cfg.soft_white_cap_knee), 1e-5)
            z_capped = torch.where(
                z_out > start,
                start + knee * torch.tanh((z_out - start).clamp_min(0.0) / knee),
                z_out,
            )
            z_out = torch.where(z_base <= 1.0 + 1e-6, z_capped, z_out)
        return self._decode(z_out)


@torch.inference_mode()
def infer_photosr_v2(
    model: PhotoSRv2,
    noisy_lr: torch.Tensor,
    guide_lr: torch.Tensor,
    *,
    tta: bool = True,
    physical_floor_guard: bool = True,
) -> torch.Tensor:
    """Production-ready PhotoSRv2 inference with exact 2-fold Bayer transpose TTA.

    - Shift-equivariant (`LocalSCA`): runs directly at arbitrary resolution with
      zero tile-size breathing (or wrap with `context_tiled_sr` for >12MP images).
    - `tta=True`: averages identity + exact Bayer transpose (swapping G1<->G2),
      which preserves exact R/G1/B/G2 CFA sensor coordinates at 2x scale.
    - `physical_floor_guard=True`: applies only a zero-blur local shadow floor
      and <=1.0 white ceiling (no highlight masking or upper edge clamping).
    """
    p0 = model(noisy_lr, guide_lr)
    if tta:
        yt = noisy_lr.transpose(-2, -1)[:, [0, 3, 2, 1]]
        dt = guide_lr.transpose(-2, -1)[:, [0, 3, 2, 1]]
        pt = model(yt, dt)[:, [0, 3, 2, 1]].transpose(-2, -1)
        out = 0.5 * (p0 + pt)
    else:
        out = p0

    if physical_floor_guard:
        padded_g = F.pad(guide_lr, (1, 1, 1, 1), mode="replicate")
        lo_lin = -site_upsample(F.max_pool2d(-padded_g, 3, stride=1))
        hi_lin = site_upsample(F.max_pool2d(padded_g, 3, stride=1))
        floor = torch.where(lo_lin > 0.0, 0.25 * lo_lin, lo_lin - 0.002)
        out = torch.maximum(out, floor)
        out = torch.where(hi_lin <= 1.0, out.clamp_max(1.0), out)
    return out
