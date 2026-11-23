#!/usr/bin/env python3
"""Unified Track A vs Track B quantitative & visual benchmark harness.

Compares:
- `raw` (CFA-site bilinear 2x of original RAW)
- `denoise` (CFA-site bilinear 2x of production frozen r4_plain)
- `track_a_s42` (frozen v1 s42 + Track A refined inference)
- `track_a_s48` (frozen v1 control_s48 + Track A refined inference)
- Any number of trained Track B `PhotoSRv2` checkpoints (`--v2 name=path`)

Evaluates identically on:
1. Disjoint held-out photographic validation bank (`bootstrap_binned` val_r5, 6 conditions)
2. Analytic fixtures (`clipped_step`, `neutral_step`, `weak_texture`, `star`)
3. Approved demo DNGs (`night`, `room`, `mushroom`) with exact uint16 DNG roundtrip
   verification and side-by-side visual crops.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import rawpy
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from denoise_lab.bayer import PHASES, normalize_raw, pack, unpack
from denoise_lab.defects import correct_defects
from denoise_lab.dng import RawMetadata, sensor_codes, write_dng
from denoise_lab.inference import Denoiser
from denoise_lab.sr.degradation_v2 import DegradationParams, forward_degrade_v2
from denoise_lab.sr.geometry import align_planes, edge_energy, luminance, site_upsample
from denoise_lab.sr.inference_v1 import context_tiled_sr, load_v1_checkpoint
from denoise_lab.sr.model_v2 import PhotoSRv2, PhotoSRv2Config
from denoise_lab.sr.postprocess import refine_sr_prediction
from sr_track_a_eval import CASES, DEFAULT_SNAP, analytic_fixture, make_grid, raw_stats, render_dng_srgb8
from sr_v2_train import DEFAULT_DATA_ROOT, build_fixed_val_bank, evaluate_val_bank, preload_binned_mosaics


def load_v2_checkpoint(path: Path, device: str, use_ema: bool = True) -> PhotoSRv2:
    cp = torch.load(path, map_location="cpu", weights_only=True)
    cfg = PhotoSRv2Config.from_dict(cp["config"])
    model = PhotoSRv2(cfg)
    state = cp["ema"] if (use_ema and "ema" in cp) else cp["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class _CallableAdapter(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, y, d):
        return self.fn(y, d)


@torch.inference_mode()
def run_analytic_for_callable(fn, r4: Denoiser, device: str) -> dict:
    results = {}
    params = DegradationParams(sigma_x=0.55, sigma_y=0.55, theta=0.0, aperture_fill=0.65)
    for kind in ("clipped_step", "neutral_step", "weak_texture", "star"):
        target = analytic_fixture(kind, n=96)
        clean_lr = forward_degrade_v2(target, params)
        noisy_lr = clean_lr.clamp_max(1.0) if kind == "clipped_step" else clean_lr
        target_eval = target.clamp_max(1.0) if kind == "clipped_step" else target
        guide_np = r4.denoise_packed(
            noisy_lr[0].numpy(), prepass=True, tile_size=512, overlap=60, strength=1.0, highlights="protect"
        )
        y_t = noisy_lr.to(device)
        d_t = torch.from_numpy(guide_np)[None].to(device)
        p = fn(y_t, d_t).cpu()
        b = 12
        pr, tr = p[..., b:-b, b:-b], target_eval[..., b:-b, b:-b]
        err = pr - tr
        pa, ta = align_planes(p), align_planes(target_eval)
        chroma_p = (pa - luminance(pa))[..., b:-b, b:-b]
        m = {
            "psnr_db": float(-10.0 * math.log10(max(float(err.square().mean()), 1e-20))),
            "min": float(pr.min()),
            "max": float(pr.max()),
            "neutral_chroma_rms": float(chroma_p.square().mean().sqrt()),
            "grad_ratio": float(
                edge_energy(luminance(pa))[..., b:-b, b:-b].square().mean().sqrt()
                / edge_energy(luminance(ta))[..., b:-b, b:-b].square().mean().sqrt().clamp_min(1e-12)
            ),
        }
        if kind == "star":
            cy, cx = p.shape[-2] // 2, p.shape[-1] // 2
            loc_p = p[..., cy - 6 : cy + 6, cx - 6 : cx + 6] - 0.02
            loc_t = target_eval[..., cy - 6 : cy + 6, cx - 6 : cx + 6] - 0.02
            m["point_flux_ratio"] = float(loc_p.sum() / loc_t.sum().clamp_min(1e-12))
            m["point_peak_ratio"] = float(loc_p.max() / loc_t.max().clamp_min(1e-12))
        results[kind] = m
    return results


def zoom_crop(img: np.ndarray, y0: int, x0: int, sz: int, out_sz: int = 384) -> np.ndarray:
    crop = img[y0 : y0 + sz, x0 : x0 + sz]
    return np.asarray(Image.fromarray(crop).resize((out_sz, out_sz), Image.Resampling.NEAREST))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", action="append", default=[], help="name=checkpoint.pt pairs to benchmark")
    parser.add_argument("--v2-tta", action="append", default=[], help="name=checkpoint.pt pairs with 4x Bayer TTA (+ _clean variant)")
    parser.add_argument("--output", type=Path, default=REPO / "runs/raw_sr_v2/benchmark")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--use-model-weights", action="store_true", help="Use raw model weights instead of EMA")
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = (
        ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        if args.device == "auto"
        else args.device
    )
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dng = out_dir / "_tmp_check.dng"

    r4_path = DEFAULT_SNAP / "models/r4_plain_weights.pt"
    r4 = Denoiser.load(r4_path, device=device, weights="model")
    s42, _ = load_v1_checkpoint(
        DEFAULT_SNAP / "runs/raw_sr_research/six_hour_20261030/experiments/unet_fusion_noise_24k_s42/best.pt",
        device=device,
    )
    s48, _ = load_v1_checkpoint(
        DEFAULT_SNAP / "runs/raw_sr_research/nvidia_input_fit_20261104/weights/control_s48_step024000.pt",
        device=device,
    )

    candidates: dict[str, callable] = {
        "denoise": lambda y, d: site_upsample(d),
        "track_a_s42": lambda y, d: refine_sr_prediction(
            context_tiled_sr(s42, y, d, center=80, apron=16, overlap=40, feather_mode="cosine", tile_batch_size=4),
            d,
            y,
        ),
        "track_a_s48": lambda y, d: refine_sr_prediction(
            context_tiled_sr(s48, y, d, center=80, apron=16, overlap=40, feather_mode="cosine", tile_batch_size=4),
            d,
            y,
        ),
    }

    for spec in args.v2:
        name, path_str = spec.split("=", 1)
        m = load_v2_checkpoint(Path(path_str), device=device, use_ema=not args.use_model_weights)
        candidates[name] = (lambda model: lambda y, d: model(y, d))(m)

    if args.v2_tta:
        for spec in args.v2_tta:
            name, path_str = spec.split("=", 1)
            m = load_v2_checkpoint(Path(path_str), device=device, use_ema=not args.use_model_weights)

            def _make_tta(model, light_guard: bool = False):
                def _run(y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
                    # Exact 2-fold Bayer-preserving transpose self-ensemble:
                    # Transpose swaps G1<->G2 (channels 1 & 3) while keeping R(0,0) and B(1,1) exact at 2x!
                    p0 = model(y, d)
                    yt = y.transpose(-2, -1)[:, [0, 3, 2, 1]]
                    dt = d.transpose(-2, -1)[:, [0, 3, 2, 1]]
                    pt = model(yt, dt)[:, [0, 3, 2, 1]].transpose(-2, -1)
                    out = 0.5 * (p0 + pt)
                    if light_guard:
                        # Zero-blur physical shadow floor + <=1.0 ceiling only (keeps 100% of sharp edges & stars!)
                        padded_g = F.pad(d, (1, 1, 1, 1), mode="replicate")
                        lo_lin = -site_upsample(F.max_pool2d(-padded_g, 3, stride=1))
                        hi_lin = site_upsample(F.max_pool2d(padded_g, 3, stride=1))
                        floor = torch.where(lo_lin > 0.0, 0.25 * lo_lin, lo_lin - 0.002)
                        out = torch.maximum(out, floor)
                        out = torch.where(hi_lin <= 1.0, out.clamp_max(1.0), out)
                    return out
                return _run

            candidates[name] = _make_tta(m, light_guard=False)
            candidates[name + "_clean"] = _make_tta(m, light_guard=True)

    report = {"device": device, "val": {}, "analytic": {}, "demos": {}}

    if not args.skip_val:
        bb = DEFAULT_DATA_ROOT / "bootstrap_binned"
        val_mosaics = preload_binned_mosaics(
            [bb / "x1d2_bin4_val_r5.json", bb / "x1d2_bin2_val_r5.json"],
            DEFAULT_DATA_ROOT,
            max_per_manifest=40,
            resite=True,
        )
        val_bank = build_fixed_val_bank(val_mosaics, r4, device, hr_packed_size=192)
        for name, fn in candidates.items():
            res = evaluate_val_bank(_CallableAdapter(fn), val_bank, device)
            report["val"][name] = res
            spec = res["conditions"]["specular_clipped"]
            hi = res["conditions"]["high_iso25600"]
            lo = res["conditions"]["low_sharp"]
            print(
                f"[VAL] {name:16s} | comp={res['mean_comp_psnr_db']:.2f}dB lin={res['mean_lin_psnr_db']:.2f}dB "
                f"chroma={res['mean_chroma_rms']:.5f} g12={res['mean_g12_rms']:.5f} grad={res['mean_grad_ratio']:.3f} | "
                f"low={lo['comp_psnr_db']:.2f} hi25k={hi['comp_psnr_db']:.2f} spec={spec['comp_psnr_db']:.2f} (neg={spec['neg_halo']:.5f})",
                flush=True,
            )

    for name, fn in candidates.items():
        report["analytic"][name] = run_analytic_for_callable(fn, r4, device)

    # Evaluate on the 3 approved demo DNGs (including raw baseline for visual comparison!)
    center_rows = []
    night_renders = {}
    room_renders = {}
    mushroom_renders = {}
    for case_id, label, filename, display_lift in CASES:
        source = REPO / "examples/raw" / filename
        with rawpy.imread(str(source)) as raw:
            meta = RawMetadata.from_raw(raw, source)
            oy, ox = PHASES[meta.phase]
            rows_idx, cols_idx = (np.arange(2) + oy) % 2, (np.arange(2) + ox) % 2
            meta = replace(
                meta,
                pattern=np.array([[0, 1], [1, 2]], dtype=np.uint8),
                black=np.asarray(meta.black)[np.ix_(rows_idx, cols_idx)],
                white=np.asarray(meta.white)[np.ix_(rows_idx, cols_idx)],
            )
            y_np = pack(normalize_raw(raw))

        d_np = r4.denoise_packed(y_np, prepass=True, tile_size=512, overlap=60, strength=1.0, highlights="protect")
        y_raw_t = torch.from_numpy(y_np)[None]
        y_pre_t = correct_defects(y_raw_t, k=8, abs_floor=0.01, adjacency_ratio=0.1)[0].to(device)
        d_t = torch.from_numpy(d_np)[None].to(device)
        denoise_hr = site_upsample(d_t).cpu()

        case_methods = {}
        renders = {}
        # Include raw baseline first
        raw_hr = site_upsample(y_raw_t)
        rgb8_raw, exact_raw = render_dng_srgb8(unpack(raw_hr[0].numpy()), meta, tmp_dng, display_lift)
        renders["raw"] = rgb8_raw
        st_raw = raw_stats(raw_hr, denoise_hr, y_pre_t.cpu())
        st_raw["dng_codes_exact"] = exact_raw
        case_methods["raw"] = st_raw

        for name, fn in candidates.items():
            with torch.inference_mode():
                pred = fn(y_pre_t, d_t).cpu()
            mosaic = unpack(pred[0].numpy())
            rgb8, exact = render_dng_srgb8(mosaic, meta, tmp_dng, display_lift)
            renders[name] = rgb8
            st = raw_stats(pred, denoise_hr, y_pre_t.cpu())
            st["dng_codes_exact"] = exact
            case_methods[name] = st

        report["demos"][case_id] = case_methods
        h, w = renders["denoise"].shape[:2]
        cy, cx = h // 2, w // 2
        c_slice = (slice(cy - 256, cy + 256), slice(cx - 256, cx + 256))
        order = ["raw", "denoise"] + [k for k in candidates.keys() if k != "denoise"]
        center_rows.append([(f"{case_id} / {k}", renders[k][c_slice]) for k in order])

        if case_id == "night":
            night_renders = renders
        elif case_id == "room":
            room_renders = renders
        elif case_id == "mushroom":
            mushroom_renders = renders

        print(
            f"[DEMO {case_id:8s}] "
            + " | ".join(
                f"{k}=[{case_methods[k]['min_signed_raw']:.4f},{case_methods[k]['max_signed_raw']:.4f}] edge={case_methods[k]['unclipped_edge_gain_vs_denoise']:.2f}x"
                for k in order
            ),
            flush=True,
        )

    tmp_dng.unlink(missing_ok=True)
    make_grid(center_rows, out_dir / "compare_center.png")

    order = ["raw", "denoise"] + [k for k in candidates.keys() if k != "denoise"]
    if night_renders:
        make_grid(
            [
                [(f"lamp / {k}", zoom_crop(night_renders[k], 840, 770, 256, 384)) for k in order],
                [(f"rail / {k}", zoom_crop(night_renders[k], 1030, 820, 256, 384)) for k in order],
                [(f"brick / {k}", zoom_crop(night_renders[k], 768, 1024, 256, 384)) for k in order],
            ],
            out_dir / "compare_night_zooms.png",
        )
    if room_renders and mushroom_renders:
        make_grid(
            [
                [(f"room / {k}", zoom_crop(room_renders[k], 896, 896, 256, 384)) for k in order],
                [(f"mush / {k}", zoom_crop(mushroom_renders[k], 896, 896, 256, 384)) for k in order],
            ],
            out_dir / "compare_room_mushroom_zooms.png",
        )

    report["total_seconds"] = time.perf_counter() - t0
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark saved to {out_dir / 'benchmark.json'} ({report['total_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
