#!/usr/bin/env python3
"""Track B: Clean-room online training harness for PhotoSRv2 (MPS / CUDA / CPU).

- Random initialization only (zero external pretrained weights).
- Parallel RAM preloading of clean binned X1D II Bayer targets (`bootstrap_binned`),
  with `resite_binned()` restoring native CFA offsets and strict group/source split
  isolation from validation and demo DNGs.
- Fast GPU-batched physical degradation: continuous anisotropic PSF, pixel-aperture
  integration, synthetic specular highlight compositing (lamps, rails, signs, stars),
  ISO-coupled physics noise, and saturation clipping at 1.0.
- Online frozen `r4_plain` guide (`correct_defects` + `r4._call` + `protect_highlights`)
  computed strictly from the noisy LR observation.
- Evaluates periodically on a fixed disjoint held-out validation suite and saves
  `best_ema.pt` + `last.pt` under the run directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from denoise_lab.bayer import augment_mosaic, pack, random_crop, read_mosaic
from denoise_lab.clean_targets import apply_channel_gains, resite_binned
from denoise_lab.defects import correct_defects
from denoise_lab.inference import Denoiser, protect_highlights
from denoise_lab.sr.degradation_v2 import (
    DegradationParams,
    forward_degrade_v2,
    inject_specular_highlights,
    synthesize_batch_on_device,
)
from denoise_lab.sr.geometry import align_planes, edge_energy, luminance, signed_asinh, site_upsample
from denoise_lab.sr.losses_v2 import SRv2Loss
from denoise_lab.sr.model_v2 import PhotoSRv2, PhotoSRv2Config

DEFAULT_DATA_ROOT = Path(os.environ.get("RAWDENOISE_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")))
DEFAULT_R4 = Path(
    os.environ.get("RAWDENOISE_R4_WEIGHTS", str(Path(__file__).resolve().parents[1] / "models/r4_plain_weights.pt"))
)


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _procedural_scene_mosaic(seed: int, sensor_size: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:sensor_size, :sensor_size].astype(np.float32) / max(sensor_size - 1, 1)
    rgb = np.full((3, sensor_size, sensor_size), rng.uniform(0.01, 0.08), dtype=np.float32)
    for _ in range(5):
        ang = rng.uniform(0.0, math.pi)
        freq = rng.uniform(2.0, 20.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        wave = np.sin(2.0 * math.pi * freq * (x * math.cos(ang) + y * math.sin(ang)) + phase)
        rgb += rng.uniform(0.01, 0.05) * wave[None]
    for _ in range(4):
        x0, y0 = rng.uniform(0.05, 0.75, 2)
        ww, hh = rng.uniform(0.08, 0.30, 2)
        m = (x > x0) & (x < x0 + ww) & (y > y0) & (y < y0 + hh)
        rgb[:, m] += rng.uniform(-0.04, 0.25, (3, 1))
    cx, cy = rng.uniform(0.25, 0.75, 2)
    rad = rng.uniform(0.03, 0.08)
    lamp = ((x - cx) ** 2 + (y - cy) ** 2) < rad**2
    rgb[:, lamp] = rng.uniform(0.95, 2.2, (3, 1))
    rgb = np.clip(rgb, 0.001, 2.5)
    mosaic = np.empty((sensor_size, sensor_size), dtype=np.float32)
    mosaic[0::2, 0::2] = rgb[0, 0::2, 0::2]
    mosaic[0::2, 1::2] = rgb[1, 0::2, 1::2]
    mosaic[1::2, 0::2] = rgb[1, 1::2, 0::2]
    mosaic[1::2, 1::2] = rgb[2, 1::2, 1::2]
    return mosaic


def preload_binned_mosaics(
    manifest_paths: list[Path],
    data_root: Path,
    *,
    max_per_manifest: int | None = None,
    resite: bool = True,
    workers: int = 16,
    seed: int = 42,
) -> list[np.ndarray]:
    """Load owner_archive binned Bayer mosaics in parallel into RAM as float16 (544, 544)."""
    selected: list[tuple[Path, int]] = []
    for mp in manifest_paths:
        rows = [r for r in json.loads(Path(mp).read_text(encoding="utf-8")) if r.get("license") == "owner_archive"]
        # Stratify deterministically across sources so every scene is represented
        by_src: dict[str, list[dict]] = {}
        for r in rows:
            by_src.setdefault(r["source_sha256"], []).append(r)
        ordered: list[dict] = []
        max_len = max(len(v) for v in by_src.values())
        keys = sorted(by_src.keys())
        for k in range(max_len):
            for s in keys:
                if k < len(by_src[s]):
                    ordered.append(by_src[s][k])
        if max_per_manifest is not None and len(ordered) > max_per_manifest:
            ordered = ordered[:max_per_manifest]
        for r in ordered:
            selected.append((data_root / r["clean"], int(r.get("bin_factor", 1))))

    def _load_one(item: tuple[Path, int]) -> np.ndarray:
        path, bf = item
        arr = read_mosaic(path)
        if resite and bf > 1:
            arr = resite_binned(arr, bf)
        return arr.astype(np.float16)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        mosaics = list(ex.map(_load_one, selected))
    print(f"Preloaded {len(mosaics)} resited binned mosaics in {time.perf_counter() - t0:.1f}s", flush=True)
    return mosaics


@torch.no_grad()
def compute_r4_guide_batch(noisy_lr: torch.Tensor, r4_denoiser: Denoiser | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run defect prepass + frozen r4_plain + protect_highlights directly on GPU/MPS.

    Returns (prepassed_noisy_lr, guide_lr). Both tensors stay on `noisy_lr.device`.
    """
    if r4_denoiser is None:
        smooth = F.avg_pool2d(F.pad(noisy_lr, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        return noisy_lr, (0.4 * noisy_lr + 0.6 * smooth).clamp_max(1.0)

    # Pre-pass runs on CPU before device upload to avoid MPS->CPU->MPS roundtrips
    pre = (
        correct_defects(noisy_lr, k=8, abs_floor=0.01, adjacency_ratio=0.1)[0]
        if noisy_lr.device.type == "cpu"
        else correct_defects(noisy_lr.detach().cpu(), k=8, abs_floor=0.01, adjacency_ratio=0.1)[0]
    )
    pre_dev = pre.to(r4_denoiser.device)
    # r4._call expects spatial dims divisible by 16
    h, w = pre_dev.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    inp = F.pad(pre_dev, (0, pw, 0, ph), mode="reflect") if (ph or pw) else pre_dev
    guide = r4_denoiser._call(inp)[..., :h, :w]
    guide = protect_highlights(guide, pre_dev)
    return pre_dev, guide


@torch.no_grad()
def build_fixed_val_bank(
    val_mosaics: list[np.ndarray],
    r4_denoiser: Denoiser | None,
    device: str,
    *,
    hr_packed_size: int = 192,
    seed: int = 20261121,
) -> list[dict]:
    """Build deterministic multi-condition held-out validation bank (6 realistic regimes)."""
    conditions = [
        ("low_sharp", DegradationParams(sigma_x=0.30, sigma_y=0.35, theta=0.4, aperture_fill=0.45, exposure=1.0, shot_k=1.2e-5, read_std=1.5e-4, read_excess_kurtosis=0.3), False),
        ("mid_nominal", DegradationParams(sigma_x=0.55, sigma_y=0.60, theta=1.1, aperture_fill=0.65, exposure=0.8, shot_k=1.7e-4, read_std=3.0e-4, read_excess_kurtosis=0.5), False),
        ("high_iso25600", DegradationParams(sigma_x=0.55, sigma_y=0.55, theta=0.0, aperture_fill=0.65, exposure=0.25, shot_k=2.7e-3, read_std=2.4e-3, read_excess_kurtosis=1.2, row_std=1.5e-4), False),
        ("aniso_blur", DegradationParams(sigma_x=0.95, sigma_y=0.45, theta=0.8, aperture_fill=0.80, exposure=0.9, shot_k=1.7e-4, read_std=3.0e-4, read_excess_kurtosis=0.5), False),
        ("specular_clipped", DegradationParams(sigma_x=0.50, sigma_y=0.50, theta=0.2, aperture_fill=0.60, exposure=0.25, shot_k=4.0e-5, read_std=2.0e-4, read_excess_kurtosis=0.3), True),
        ("shadow_dark", DegradationParams(sigma_x=0.55, sigma_y=0.55, theta=0.5, aperture_fill=0.65, exposure=0.12, shot_k=5.0e-4, read_std=8.0e-4, read_excess_kurtosis=0.8), False),
    ]
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device="cpu").manual_seed(seed)
    bank = []
    # 5 distinct val mosaics per condition = 30 diverse held-out pairs
    n_per_cond = min(5, max(1, len(val_mosaics) // len(conditions)))
    idx_cursor = 0
    for cname, params, add_spec in conditions:
        for _ in range(n_per_cond):
            m = val_mosaics[idx_cursor % len(val_mosaics)].astype(np.float32)
            idx_cursor += 1
            crop = random_crop(m, packed_size=hr_packed_size, rng=rng)
            hr = torch.from_numpy(pack(crop).astype(np.float32)) * float(params.exposure)
            if add_spec:
                hr = inject_specular_highlights(hr, rng)
            clean_lr = forward_degrade_v2(hr[None], params)[0]
            std = (params.shot_k * clean_lr.clamp_min(0.0) + params.read_std**2).sqrt()
            noisy_lr = clean_lr + torch.randn(clean_lr.shape, generator=torch_gen) * std
            if params.row_std > 0:
                noisy_lr = noisy_lr + torch.randn((4, clean_lr.shape[-2], 1), generator=torch_gen) * params.row_std
            noisy_lr = noisy_lr.clamp_max(1.0)
            target_hr = hr.clamp_max(1.0)
            pre, guide = compute_r4_guide_batch(noisy_lr[None].to(device), r4_denoiser)
            bank.append(
                {
                    "condition": cname,
                    "noisy": pre[0].cpu(),
                    "guide": guide[0].cpu(),
                    "target": target_hr.cpu(),
                }
            )
    return bank


@torch.inference_mode()
def evaluate_val_bank(model: nn.Module, bank: list[dict], device: str, border: int = 12) -> dict:
    model.eval()
    by_cond: dict[str, list[dict]] = {}
    for item in bank:
        y = item["noisy"][None].to(device)
        d = item["guide"][None].to(device)
        t = item["target"][None].to(device)
        p = model(y, d)

        pr = p[..., border:-border, border:-border]
        tr = t[..., border:-border, border:-border]
        lin_mse = float((pr - tr).square().mean())
        zp, zt = signed_asinh(pr, 0.05), signed_asinh(tr, 0.05)
        comp_mse = float((zp - zt).square().mean())

        pa, ta = align_planes(zp), align_planes(zt)
        chroma_rms = float(((pa - luminance(pa)) - (ta - luminance(ta))).square().mean().sqrt())
        g12_rms = float(((pa[:, 1:2] - pa[:, 3:4]) - (ta[:, 1:2] - ta[:, 3:4])).square().mean().sqrt())
        grad_ratio = float(
            edge_energy(luminance(pa)).square().mean().sqrt()
            / edge_energy(luminance(ta)).square().mean().sqrt().clamp_min(1e-12)
        )
        neg_halo = float(F.relu(-pr)[tr > 0.002].mean()) if (tr > 0.002).any() else 0.0
        white_over = float(F.relu(pr - 1.0).max())

        by_cond.setdefault(item["condition"], []).append(
            {
                "lin_mse": lin_mse,
                "comp_mse": comp_mse,
                "chroma_rms": chroma_rms,
                "g12_rms": g12_rms,
                "grad_ratio": grad_ratio,
                "neg_halo": neg_halo,
                "white_over": white_over,
            }
        )

    summary = {"conditions": {}}
    all_comp = []
    all_lin = []
    all_chroma = []
    all_g12 = []
    all_grad = []
    for cname, rows in by_cond.items():
        cm = float(np.mean([r["comp_mse"] for r in rows]))
        lm = float(np.mean([r["lin_mse"] for r in rows]))
        summary["conditions"][cname] = {
            "comp_psnr_db": float(-10.0 * math.log10(max(cm, 1e-20))),
            "lin_psnr_db": float(-10.0 * math.log10(max(lm, 1e-20))),
            "chroma_rms": float(np.mean([r["chroma_rms"] for r in rows])),
            "g12_rms": float(np.mean([r["g12_rms"] for r in rows])),
            "grad_ratio": float(np.mean([r["grad_ratio"] for r in rows])),
            "neg_halo": float(np.mean([r["neg_halo"] for r in rows])),
            "white_over_max": float(np.max([r["white_over"] for r in rows])),
        }
        all_comp.append(cm)
        all_lin.append(lm)
        all_chroma.extend(r["chroma_rms"] for r in rows)
        all_g12.extend(r["g12_rms"] for r in rows)
        all_grad.extend(r["grad_ratio"] for r in rows)

    summary["mean_comp_psnr_db"] = float(-10.0 * math.log10(max(float(np.mean(all_comp)), 1e-20)))
    summary["mean_lin_psnr_db"] = float(-10.0 * math.log10(max(float(np.mean(all_lin)), 1e-20)))
    summary["mean_chroma_rms"] = float(np.mean(all_chroma))
    summary["mean_g12_rms"] = float(np.mean(all_g12))
    summary["mean_grad_ratio"] = float(np.mean(all_grad))
    return summary


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.mul_(decay).add_(p, alpha=1.0 - decay)


def sample_hr_batch(
    mosaics: list[np.ndarray],
    batch_size: int,
    hr_packed_size: int,
    rng: np.random.Generator,
    channel_gain_prob: float = 0.5,
) -> torch.Tensor:
    out = []
    n = len(mosaics)
    for _ in range(batch_size):
        m = mosaics[int(rng.integers(n))].astype(np.float32)
        m = augment_mosaic(m, *rng.integers(0, 2, 3).astype(bool))
        m = random_crop(m, packed_size=hr_packed_size, rng=rng)
        if channel_gain_prob > 0 and rng.random() < channel_gain_prob:
            m = apply_channel_gains(m, rng)
        out.append(torch.from_numpy(pack(m).astype(np.float32)))
    return torch.stack(out, dim=0)


def train_v2(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    cfg = PhotoSRv2Config(
        mode=args.mode,
        width=args.width,
        enc_blk_nums=tuple(int(x) for x in args.enc_blocks.split(",")),
        middle_blk_num=args.mid_blocks,
        dec_blk_nums=tuple(int(x) for x in args.dec_blocks.split(",")),
        sca_kernels=tuple(int(x) for x in args.sca_kernels.split(",")),
        hr_channels=args.hr_channels,
        domain=args.domain,
        domain_scale=args.domain_scale,
    )
    model = PhotoSRv2(cfg).to(device)
    if args.resume_weights is not None:
        cp = torch.load(args.resume_weights, map_location="cpu", weights_only=True)
        state = cp.get("ema", cp["model"]) if args.resume_from_ema else cp["model"]
        model.load_state_dict(state, strict=True)
        print(f"Resumed own PhotoSRv2 weights from {args.resume_weights}", flush=True)

    ema = copy.deepcopy(model).eval()
    criterion = SRv2Loss(
        domain=cfg.domain,
        domain_scale=cfg.domain_scale,
        compressive_weight=args.w_comp,
        linear_weight=args.w_lin,
        g12_weight=args.w_g12,
        chroma_weight=args.w_chroma,
        grad_weight=args.w_grad,
        fft_weight=args.w_fft,
        overshoot_weight=args.w_ov,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay)

    r4_denoiser = None
    if not args.smoke:
        r4_path = Path(args.r4_weights) if args.r4_weights else DEFAULT_R4
        r4_denoiser = Denoiser.load(r4_path, device=device, weights="model")

    if args.smoke:
        train_mosaics = [_procedural_scene_mosaic(args.seed + i, args.hr_packed_size * 2 + 16).astype(np.float16) for i in range(24)]
        val_mosaics = [_procedural_scene_mosaic(args.seed + 1000 + i, args.hr_packed_size * 2 + 16).astype(np.float16) for i in range(6)]
    else:
        bb = Path(args.data_root) / "bootstrap_binned"
        train_manifests = [Path(p) for p in args.manifests] if args.manifests else [
            bb / "x1d2_bin4_train_r5.json",
            bb / "x1d2_bin2_train_r5.json",
        ]
        val_manifests = [Path(p) for p in args.val_manifests] if args.val_manifests else [
            bb / "x1d2_bin4_val_r5.json",
            bb / "x1d2_bin2_val_r5.json",
        ]
        train_mosaics = preload_binned_mosaics(
            train_manifests,
            Path(args.data_root),
            max_per_manifest=args.max_per_manifest,
            resite=True,
            seed=args.seed,
        )
        val_mosaics = preload_binned_mosaics(
            val_manifests,
            Path(args.data_root),
            max_per_manifest=40,
            resite=True,
            seed=args.seed + 1,
        )

    val_bank = build_fixed_val_bank(val_mosaics, r4_denoiser, device, hr_packed_size=192)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_val = evaluate_val_bank(ema, val_bank, device)
    print(
        f"Init val (base guide): comp_psnr={init_val['mean_comp_psnr_db']:.2f} dB, "
        f"lin_psnr={init_val['mean_lin_psnr_db']:.2f} dB, grad_ratio={init_val['mean_grad_ratio']:.3f}, "
        f"params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    best_val_psnr = -1e9
    history = []
    running_losses: list[float] = []
    sigma_range = (args.sigma_min, args.sigma_max)
    aperture_range = (args.aperture_min, args.aperture_max)

    def _make_cpu_batch(batch_seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        local_rng = np.random.default_rng(batch_seed)
        hr_batch_cpu = sample_hr_batch(train_mosaics, args.batch_size, args.hr_packed_size, local_rng)
        noisy_lr_cpu, target_hr_cpu = synthesize_batch_on_device(
            hr_batch_cpu,
            local_rng,
            specular_prob=args.specular_prob,
            sigma_range=sigma_range,
            aperture_range=aperture_range,
        )
        if r4_denoiser is not None:
            pre_lr_cpu = correct_defects(noisy_lr_cpu, k=8, abs_floor=0.01, adjacency_ratio=0.1)[0]
        else:
            pre_lr_cpu = noisy_lr_cpu
        return pre_lr_cpu, target_hr_cpu

    # Async 2-worker CPU prefetch overlaps degradation+defect-prepass with MPS/CUDA compute
    prefetch_pool = ThreadPoolExecutor(max_workers=2)
    step_seeds = rng.integers(0, 2**62, size=args.steps + 2)
    pending = [prefetch_pool.submit(_make_cpu_batch, int(step_seeds[0])), prefetch_pool.submit(_make_cpu_batch, int(step_seeds[1]))]
    next_submit = 2

    try:
      for step in range(1, args.steps + 1):
        pre_lr_cpu, target_hr_cpu = pending.pop(0).result()
        if next_submit < len(step_seeds):
            pending.append(prefetch_pool.submit(_make_cpu_batch, int(step_seeds[next_submit])))
            next_submit += 1

        pre_lr = pre_lr_cpu.to(device)
        if r4_denoiser is None:
            smooth = F.avg_pool2d(F.pad(pre_lr, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            guide_lr = (0.4 * pre_lr + 0.6 * smooth).clamp_max(1.0)
        else:
            with torch.no_grad():
                h_lr, w_lr = pre_lr.shape[-2:]
                ph, pw = (-h_lr) % 16, (-w_lr) % 16
                inp = F.pad(pre_lr, (0, pw, 0, ph), mode="reflect") if (ph or pw) else pre_lr
                guide_lr = protect_highlights(r4_denoiser._call(inp)[..., :h_lr, :w_lr], pre_lr)
        target_hr = target_hr_cpu.to(device)

        if step <= args.warmup_steps:
            lr = args.lr * (step / max(1, args.warmup_steps))
        else:
            prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr = args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        pred = model(pre_lr, guide_lr)
        loss, comps = criterion(pred, target_hr, return_components=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {comps}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ema_decay = min(args.ema_decay, (step + 1.0) / (step + 10.0))
        update_ema(ema, model, decay=ema_decay)
        running_losses.append(comps["total"])

        if device == "mps":
            if step % 20 == 0:
                torch.mps.synchronize()
            if step % 100 == 0:
                torch.mps.empty_cache()

        if step == 1 or step % max(1, args.log_every) == 0 or step == args.steps:
            mean_loss = float(np.mean(running_losses))
            running_losses = []
            pd = pred.detach()
            print(
                f"[step {step:5d}/{args.steps}] lr={lr:.2e} loss={mean_loss:.5f} "
                f"comp={comps['compressive']:.5f} grad={comps.get('grad', 0):.5f} ov={comps.get('overshoot', 0):.5f} "
                f"range=[{float(pd.min()):.4f}, {float(pd.max()):.4f}] t={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

        if (args.eval_every > 0 and step % args.eval_every == 0) or step == args.steps:
            val_ema = evaluate_val_bank(ema, val_bank, device)
            val_raw = evaluate_val_bank(model, val_bank, device)
            score = val_ema["mean_comp_psnr_db"]
            rec = {
                "step": step,
                "elapsed_s": time.perf_counter() - t0,
                "lr": lr,
                "val_ema": val_ema,
                "val_model": val_raw,
            }
            history.append(rec)
            spec_c = val_ema["conditions"].get("specular_clipped", {})
            print(
                f"  VAL@{step:5d} EMA: comp={val_ema['mean_comp_psnr_db']:.2f}dB lin={val_ema['mean_lin_psnr_db']:.2f}dB "
                f"chroma={val_ema['mean_chroma_rms']:.5f} g12={val_ema['mean_g12_rms']:.5f} grad={val_ema['mean_grad_ratio']:.3f} | "
                f"spec_clip={spec_c.get('comp_psnr_db', 0):.2f}dB neg={spec_c.get('neg_halo', 0):.5f} over={spec_c.get('white_over_max', 0):.4f} "
                f"(model comp={val_raw['mean_comp_psnr_db']:.2f}dB)",
                flush=True,
            )
            if args.save_checkpoint and score > best_val_psnr:
                best_val_psnr = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "config": cfg.to_dict(),
                        "step": step,
                        "val_ema": val_ema,
                        "random_init": args.resume_weights is None,
                    },
                    out_dir / "best_ema.pt",
                )
    finally:
        prefetch_pool.shutdown(wait=False, cancel_futures=True)

    summary = {
        "architecture": cfg.architecture,
        "config": cfg.to_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "smoke": bool(args.smoke),
        "steps": args.steps,
        "seed": args.seed,
        "random_init": args.resume_weights is None,
        "resume_weights": str(args.resume_weights) if args.resume_weights else None,
        "elapsed_s": time.perf_counter() - t0,
        "init_val": init_val,
        "best_val_comp_psnr_db": best_val_psnr,
        "history": history,
    }
    if args.save_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "config": cfg.to_dict(),
                "step": args.steps,
                "random_init": args.resume_weights is None,
            },
            out_dir / "last.pt",
        )
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--manifests", nargs="*", default=[])
    parser.add_argument("--val-manifests", nargs="*", default=[])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--r4-weights", type=Path, default=None)
    parser.add_argument("--max-per-manifest", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=Path("runs/raw_sr_v2/train"))
    parser.add_argument("--mode", default="fusion", choices=("fusion", "raw", "guide"))
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--enc-blocks", default="1,1,2")
    parser.add_argument("--mid-blocks", type=int, default=4)
    parser.add_argument("--dec-blocks", default="1,1,1")
    parser.add_argument("--sca-kernels", default="15,11,7,5")
    parser.add_argument("--hr-channels", type=int, default=16)
    parser.add_argument("--domain", default="asinh", choices=("asinh", "gamma"))
    parser.add_argument("--domain-scale", type=float, default=0.05)
    parser.add_argument("--hr-packed-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--specular-prob", type=float, default=0.25)
    parser.add_argument("--sigma-min", type=float, default=0.15)
    parser.add_argument("--sigma-max", type=float, default=1.05)
    parser.add_argument("--aperture-min", type=float, default=0.25)
    parser.add_argument("--aperture-max", type=float, default=0.85)
    parser.add_argument("--w-comp", type=float, default=1.0)
    parser.add_argument("--w-lin", type=float, default=0.25)
    parser.add_argument("--w-g12", type=float, default=0.15)
    parser.add_argument("--w-chroma", type=float, default=0.10)
    parser.add_argument("--w-grad", type=float, default=0.12)
    parser.add_argument("--w-fft", type=float, default=0.03)
    parser.add_argument("--w-ov", type=float, default=0.40)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--resume-weights", type=Path, default=None)
    parser.add_argument("--resume-from-ema", action="store_true")
    args = parser.parse_args()
    train_v2(args)


if __name__ == "__main__":
    main()
