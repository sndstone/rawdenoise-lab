"""Track B physical forward degradation simulator for 2x RAW SR.

Replaces the coarse 3-sigma isotropic proxy with:
1. Continuous anisotropic 2D optical PSF (sigma_x, sigma_y, orientation theta).
2. Explicit LR photosite pixel-aperture box integration before CFA-site subsampling.
3. Realistic sensor highlight saturation clipping (capped at 1.0, signed below-black
   values retained) plus synthetic specular structures (lamps, rails, backlit signs,
   sub-pixel stars) composited before optical blur & clipping.
4. Realistic camera physics noise via `denoise_lab.clean_targets.iso_coupled_parameters`
   and `denoise_lab.physics_noise` (heavy-tailed read noise, row/col banding, DN
   quantization) plus a fast GPU-batched path for online MPS/CUDA training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..bayer import pack, unpack
from ..clean_targets import iso_coupled_parameters
from ..physics_noise import banding, correlate, moment_read, quantize_dn
from .geometry import SITES


@dataclass(frozen=True)
class DegradationParams:
    sigma_x: float = 0.55
    sigma_y: float = 0.55
    theta: float = 0.0
    aperture_fill: float = 0.65  # LR pixel-aperture integration width in HR packed units
    exposure: float = 1.0
    iso: float = 1600.0
    shot_k: float = 1.7e-4
    read_std: float = 1.5e-4
    read_excess_kurtosis: float = 0.5
    row_std: float = 0.0
    col_std: float = 0.0
    lag1_h: float = 0.0
    clip_highlights: bool = True
    quantize_bits: int = 0  # 0 disables; 14 or 16 enables DN quantization
    white_dn: float = 65535.0
    black_dn: float = 2048.0

    def to_dict(self) -> dict:
        return asdict(self)


def anisotropic_psf_kernel(
    sigma_x: float,
    sigma_y: float,
    theta: float,
    radius: int = 4,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Normalized 2D rotated anisotropic Gaussian PSF kernel of shape (2r+1, 2r+1)."""
    sx = max(float(sigma_x), 0.05)
    sy = max(float(sigma_y), 0.05)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    c, s = math.cos(theta), math.sin(theta)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    kernel = torch.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return kernel / kernel.sum().clamp_min(1e-12)


def _box_aperture_kernel(
    fill: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Separable 3x3 pixel-aperture integration kernel in HR packed coordinates."""
    f = float(np.clip(fill, 0.0, 1.0))
    w_side = 0.25 * f
    w_center = 1.0 - 2.0 * w_side
    k1d = torch.tensor([w_side, w_center, w_side], device=device, dtype=dtype)
    return k1d[:, None] * k1d[None, :]


def _cfa_subsample_2x(blurred: torch.Tensor) -> torch.Tensor:
    """Exact LR CFA sensor-center subsampling from Bx4x2Hx2W to Bx4xHxW."""
    h, w = blurred.shape[-2:]
    yy = torch.arange(h // 2, device=blurred.device, dtype=blurred.dtype)
    xx = torch.arange(w // 2, device=blurred.device, dtype=blurred.dtype)
    planes = []
    for c, (sy, sx) in enumerate(SITES):
        gy = 2.0 * yy + sy / 2.0 + 0.25
        gx = 2.0 * xx + sx / 2.0 + 0.25
        grid_y, grid_x = torch.meshgrid(
            2.0 * gy / (h - 1) - 1.0,
            2.0 * gx / (w - 1) - 1.0,
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(blurred.shape[0], -1, -1, -1)
        planes.append(
            F.grid_sample(
                blurred[:, c : c + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
    return torch.cat(planes, dim=1)


def forward_degrade_v2(
    target_hr: torch.Tensor,
    params: DegradationParams,
    *,
    psf_radius: int = 4,
) -> torch.Tensor:
    """Apply optical PSF + pixel-aperture box integration + exact CFA-site 2x reduction."""
    if target_hr.ndim != 4 or target_hr.shape[1] != 4:
        raise ValueError("Expected Bx4xHxW packed HR tensor")
    h, w = target_hr.shape[-2:]
    if h < 4 or w < 4 or h % 2 or w % 2:
        raise ValueError("HR packed dimensions must be even and >= 4")

    psf = anisotropic_psf_kernel(
        params.sigma_x,
        params.sigma_y,
        params.theta,
        radius=psf_radius,
        device=target_hr.device,
        dtype=target_hr.dtype,
    )
    k_psf = psf[None, None].expand(4, 1, -1, -1)
    blurred = F.conv2d(F.pad(target_hr, (psf_radius,) * 4, mode="reflect"), k_psf, groups=4)

    if params.aperture_fill > 1e-4:
        box = _box_aperture_kernel(params.aperture_fill, device=target_hr.device, dtype=target_hr.dtype)
        k_box = box[None, None].expand(4, 1, -1, -1)
        blurred = F.conv2d(F.pad(blurred, (1, 1, 1, 1), mode="reflect"), k_box, groups=4)

    return _cfa_subsample_2x(blurred)


def inject_specular_highlights(
    hr_packed: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Composite physically realistic bright specular features onto an HR packed crop.

    Adds sharp, CFA-consistent bright structures (streetlamps, backlit signs,
    metallic specular rails, and sub-pixel stars) with linear radiance > 1.0
    BEFORE optical PSF blur, noise, and 1.0 sensor clipping, directly teaching
    PhotoSRv2 to avoid ringing on 200:1 night-scene transitions.
    """
    _, h, w = hr_packed.shape
    sh, sw = 2 * h, 2 * w
    device, dtype = hr_packed.device, hr_packed.dtype
    y, x = torch.meshgrid(
        torch.arange(sh, device=device, dtype=dtype),
        torch.arange(sw, device=device, dtype=dtype),
        indexing="ij",
    )
    out = hr_packed.clone()
    kind = str(rng.choice(["lamp", "sign", "rail", "stars", "mixed"]))

    def _paint(mask_2d: torch.Tensor, rgb_color: np.ndarray) -> None:
        # Apply exact CFA-site sampling from the 2Hx2W HR sensor mask
        colors = (0, 1, 2, 1)
        for c, ((sy, sx), col) in enumerate(zip(SITES, colors)):
            m = mask_2d[sy::2, sx::2]
            out[c] = out[c] + m * float(rgb_color[col])

    # Warm-to-cool realistic light chromaticity (sodium lamp, LED white, cyan neon)
    palettes = [
        np.array([1.0, 0.85, 0.45], dtype=np.float32),  # warm streetlight (clips R/G before B!)
        np.array([1.0, 1.0, 1.0], dtype=np.float32),    # neutral white lamp/sign
        np.array([0.35, 0.90, 1.0], dtype=np.float32),  # cool/cyan backlit sign
        np.array([1.0, 0.40, 0.25], dtype=np.float32),  # red/amber tail-light
    ]

    if kind in ("lamp", "mixed"):
        for _ in range(int(rng.integers(1, 4))):
            cy = float(rng.uniform(0.15 * sh, 0.85 * sh))
            cx = float(rng.uniform(0.15 * sw, 0.85 * sw))
            rad = float(rng.uniform(3.0, 18.0))
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-6)
            # Soft glowing halo + crisp saturated core
            core = torch.sigmoid((rad - dist) * 1.2)
            glow = 0.22 * torch.exp(-0.5 * (dist / (rad * 2.2)) ** 2)
            intensity = float(rng.uniform(1.2, 3.8))
            col = palettes[int(rng.integers(len(palettes)))] * intensity
            _paint(core + glow, col)

    if kind in ("sign", "mixed"):
        cy = float(rng.uniform(0.2 * sh, 0.8 * sh))
        cx = float(rng.uniform(0.2 * sw, 0.8 * sw))
        hh = float(rng.uniform(10.0, 40.0))
        ww = float(rng.uniform(16.0, 64.0))
        ang = float(rng.uniform(-0.35, 0.35))
        ca, sa = math.cos(ang), math.sin(ang)
        xr = ca * (x - cx) + sa * (y - cy)
        yr = -sa * (x - cx) + ca * (y - cy)
        edge = torch.sigmoid((ww * 0.5 - xr.abs()) * 2.0) * torch.sigmoid((hh * 0.5 - yr.abs()) * 2.0)
        intensity = float(rng.uniform(1.1, 2.8))
        col = palettes[int(rng.integers(len(palettes)))] * intensity
        _paint(edge, col)

    if kind == "rail":
        # Periodic thin bright specular vertical/slanted rails (mirrors night railing!)
        ang = float(rng.uniform(-0.25, 0.25))
        period = float(rng.uniform(8.0, 20.0))
        width = float(rng.uniform(1.5, 4.0))
        coord = x * math.cos(ang) + y * math.sin(ang) + float(rng.uniform(0, period))
        phase = torch.remainder(coord, period) - 0.5 * period
        bars = torch.sigmoid((0.5 * width - phase.abs()) * 2.5)
        y0 = float(rng.uniform(0.35 * sh, 0.65 * sh))
        band = torch.sigmoid((y - y0) * 1.5)
        intensity = float(rng.uniform(0.9, 2.4))
        col = palettes[int(rng.integers(2))] * intensity
        _paint(bars * band, col)

    if kind in ("stars", "mixed"):
        for _ in range(int(rng.integers(3, 10))):
            cy = float(rng.uniform(12.0, sh - 12.0))
            cx = float(rng.uniform(12.0, sw - 12.0))
            sigma = float(rng.uniform(0.7, 1.5))
            star = torch.exp(-0.5 * (((x - cx) ** 2 + (y - cy) ** 2) / (sigma**2)))
            intensity = float(rng.uniform(0.25, 1.4))
            col = palettes[int(rng.integers(2))] * intensity
            _paint(star, col)

    return out


def sample_degradation_params(
    rng: np.random.Generator,
    *,
    iso_range: tuple[float, float] = (100.0, 25600.0),
    sigma_range: tuple[float, float] = (0.15, 1.05),
    anisotropy_max: float = 1.6,
    aperture_range: tuple[float, float] = (0.25, 0.85),
    highlight_boost_prob: float = 0.20,
    highlight_boost_range: tuple[float, float] = (1.5, 5.5),
    clean_prob: float = 0.10,
) -> DegradationParams:
    """Sample realistic continuous optical + exposure + camera-physics noise parameters."""
    sigma_base = float(rng.uniform(*sigma_range))
    ratio = float(np.exp(rng.uniform(-math.log(anisotropy_max), math.log(anisotropy_max))))
    sigma_x = float(np.clip(sigma_base * math.sqrt(ratio), sigma_range[0], sigma_range[1]))
    sigma_y = float(np.clip(sigma_base / math.sqrt(ratio), sigma_range[0], sigma_range[1]))
    theta = float(rng.uniform(0.0, math.pi))
    aperture_fill = float(rng.uniform(*aperture_range))

    if rng.random() < clean_prob:
        return DegradationParams(
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            theta=theta,
            aperture_fill=aperture_fill,
            exposure=float(rng.uniform(0.4, 1.5)),
            iso=100.0,
            shot_k=0.0,
            read_std=0.0,
            read_excess_kurtosis=0.0,
            row_std=0.0,
            col_std=0.0,
            clip_highlights=True,
        )

    coupled, exposure, iso = iso_coupled_parameters(rng, iso_range=iso_range)
    if rng.random() < highlight_boost_prob:
        exposure = float(np.exp(rng.uniform(math.log(highlight_boost_range[0]), math.log(highlight_boost_range[1]))))

    kurt = float(rng.uniform(0.0, 2.5)) if rng.random() < 0.6 else 0.0
    col_std = float(rng.uniform(0.0, 0.08)) * coupled["read_std"] if rng.random() < 0.4 else 0.0
    lag1_h = float(rng.uniform(-0.08, 0.15)) if rng.random() < 0.35 else 0.0
    quantize_bits = 14 if rng.random() < 0.25 else 0

    return DegradationParams(
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta=theta,
        aperture_fill=aperture_fill,
        exposure=float(exposure),
        iso=float(iso),
        shot_k=float(coupled["K"]),
        read_std=float(coupled["read_std"]),
        read_excess_kurtosis=kurt,
        row_std=float(coupled["row_std"]),
        col_std=col_std,
        lag1_h=lag1_h,
        clip_highlights=True,
        quantize_bits=quantize_bits,
    )


def simulate_lr_observation(
    target_hr_packed: np.ndarray | torch.Tensor,
    params: DegradationParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (noisy_lr, target_hr_clipped, sat_mask_hr) from a single HR packed target."""
    if isinstance(target_hr_packed, np.ndarray):
        hr_t = torch.from_numpy(np.asarray(target_hr_packed, dtype=np.float32))
    else:
        hr_t = target_hr_packed.detach().to(dtype=torch.float32, device="cpu")
    if hr_t.ndim == 3:
        hr_t = hr_t[None]

    hr_scaled = hr_t * float(params.exposure)
    clean_lr = forward_degrade_v2(hr_scaled, params)[0].numpy()

    signal = np.maximum(clean_lr, 0.0)
    shot = rng.normal(size=clean_lr.shape).astype(np.float32) * np.sqrt(params.shot_k * signal)
    if params.read_std > 0.0:
        if params.read_excess_kurtosis > 0.0:
            read = moment_read(clean_lr.shape, params.read_std**2, params.read_excess_kurtosis, rng).astype(np.float32)
        else:
            read = (rng.normal(size=clean_lr.shape) * params.read_std).astype(np.float32)
        if abs(params.lag1_h) > 1e-5:
            read = correlate(read, params.lag1_h, axis=-1).astype(np.float32)
    else:
        read = np.zeros_like(clean_lr)

    noisy_lr = clean_lr + shot + read
    if params.row_std > 0.0 or params.col_std > 0.0:
        h, w = clean_lr.shape[-2:]
        band = banding(
            (2 * h, 2 * w),
            {"row_std": params.row_std, "col_std": params.col_std},
            rng,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        noisy_lr = noisy_lr + pack(band).astype(np.float32)

    if params.quantize_bits > 0:
        noisy_lr = pack(
            quantize_dn(
                unpack(noisy_lr),
                {
                    "white_dn": params.white_dn,
                    "black_dn": params.black_dn,
                    "effective_bits": params.quantize_bits,
                    "sensor_bits": 16,
                },
            )
        ).astype(np.float32)

    hr_np = hr_scaled[0].numpy()
    sat_mask_hr = hr_np >= 1.0
    if params.clip_highlights:
        noisy_lr = np.minimum(noisy_lr, 1.0).astype(np.float32)
        hr_np = np.minimum(hr_np, 1.0).astype(np.float32)

    return noisy_lr.astype(np.float32), hr_np.astype(np.float32), sat_mask_hr


@torch.no_grad()
def synthesize_batch_on_device(
    hr_batch: torch.Tensor,
    rng: np.random.Generator,
    *,
    specular_prob: float = 0.25,
    sigma_range: tuple[float, float] = (0.15, 1.05),
    aperture_range: tuple[float, float] = (0.25, 0.85),
    psf_radius: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast GPU/MPS-batched physical degradation + optional specular highlight synthesis.

    Input:  hr_batch (Bx4x2Hx2W) clean float32 packed HR targets on target device.
    Output: (noisy_lr [Bx4xHxW], target_hr_clipped [Bx4x2Hx2W]) on same device.
    """
    b, _, h, w = hr_batch.shape
    device, dtype = hr_batch.device, hr_batch.dtype
    noisy_list = []
    target_list = []

    for i in range(b):
        p = sample_degradation_params(rng, sigma_range=sigma_range, aperture_range=aperture_range)
        hr = hr_batch[i] * float(p.exposure)
        if specular_prob > 0.0 and rng.random() < specular_prob:
            # Darken background slightly half the time to mimic high-contrast night scenes
            if rng.random() < 0.6:
                hr = hr * float(rng.uniform(0.08, 0.45))
            hr = inject_specular_highlights(hr, rng)

        clean_lr = forward_degrade_v2(hr[None], p, psf_radius=psf_radius)[0]
        shot_std = (p.shot_k * clean_lr.clamp_min(0.0)).sqrt()
        noisy = clean_lr + torch.randn_like(clean_lr) * shot_std

        if p.read_std > 0.0:
            if p.read_excess_kurtosis > 0.0:
                excess = max(0.0, p.read_excess_kurtosis)
                prob = min(0.01, 1.0 / (2.0 + 2.0 * excess / 3.0))
                delta = math.sqrt(excess / (3.0 * prob * (1.0 - prob)))
                low = max(0.0, 1.0 - prob * delta)
                high = low + delta
                scales = torch.where(torch.rand_like(clean_lr) < prob, high, low)
                read = torch.randn_like(clean_lr) * (p.read_std * scales.sqrt())
            else:
                read = torch.randn_like(clean_lr) * p.read_std
            noisy = noisy + read

        if p.row_std > 0.0:
            noisy = noisy + torch.randn((4, clean_lr.shape[-2], 1), device=device, dtype=dtype) * p.row_std
        if p.col_std > 0.0:
            noisy = noisy + torch.randn((4, 1, clean_lr.shape[-1]), device=device, dtype=dtype) * p.col_std

        if p.clip_highlights:
            noisy = noisy.clamp_max(1.0)
            hr = hr.clamp_max(1.0)

        noisy_list.append(noisy)
        target_list.append(hr)

    return torch.stack(noisy_list, dim=0), torch.stack(target_list, dim=0)
