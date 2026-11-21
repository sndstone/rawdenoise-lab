#!/usr/bin/env python3
"""Track A diagnostic & refined inference benchmark for frozen 2x RAW SR checkpoints.

Read-only with respect to frozen checkpoints and approved demos in examples/raw/.
Writes all outputs under a fresh local run directory (default: runs/raw_sr_v2/track_a).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageCms, ImageDraw
import rawpy
import torch
from torch.nn import functional as F

from denoise_lab.bayer import PHASES, normalize_raw, pack, unpack
from denoise_lab.defects import correct_defects
from denoise_lab.dng import RawMetadata, sensor_codes, write_dng
from denoise_lab.inference import Denoiser
from denoise_lab.sr.geometry import align_planes, edge_energy, luminance, site_upsample
from denoise_lab.sr.inference_v1 import apply_tlc, context_tiled_sr, load_v1_checkpoint
from denoise_lab.sr.postprocess import (
    RefineConfig,
    cfa_chroma_guard,
    compressive_range_guard,
    highlight_blend_sr,
    refine_sr_prediction,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SNAP = Path(os.environ.get("RAWDENOISE_SNAP_DIR", str(REPO)))
CASES = [
    ("night", "Night waterfront", "sample_iso100_night_river.dng", 4.0),
    ("room", "Living room", "sample_iso25600_living_room.dng", 2.0),
    ("mushroom", "Mushroom scene", "sample_iso3200_mushroom.dng", 1.0),
]


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def render_dng_srgb8(
    mosaic: np.ndarray,
    meta: RawMetadata,
    dng_path: Path,
    display_lift: float,
) -> tuple[np.ndarray, bool]:
    """Write synthesized 2x Bayer DNG, verify exact uint16 roundtrip, render sRGB8."""
    write_dng(dng_path, mosaic, meta, metadata_mode="minimal", preview=False)
    expected = sensor_codes(mosaic, meta)
    with rawpy.imread(str(dng_path)) as raw:
        exact = bool(np.array_equal(raw.raw_image_visible, expected))
        rgb16_lin = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.sRGB,
            adjust_maximum_thr=0.0,
        )
    linear = np.clip(rgb16_lin.astype(np.float32) * (display_lift / 65535.0), 0.0, 1.0)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * (linear ** (1.0 / 2.4)) - 0.055)
    rgb8 = np.rint(encoded * 255.0).astype(np.uint8)
    return rgb8, exact


def raw_stats(
    pred: torch.Tensor,
    guide_hr: torch.Tensor,
    original_lr: torch.Tensor,
    nov7_ref: torch.Tensor | None = None,
) -> dict:
    """Compute signed linear RAW diagnostics + unclipped sharpness & halo metrics."""
    mosaic = unpack(pred[0].cpu().numpy())
    # Mask unclipped interior regions (far from any >=0.95 photosite) to measure sharpness retention
    sat_lr = (original_lr >= 0.95).any(dim=1, keepdim=True).float()
    if sat_lr.any():
        for _ in range(8):
            sat_lr = F.max_pool2d(sat_lr, 3, stride=1, padding=1)
    unclipped_hr = (F.interpolate(sat_lr, size=pred.shape[-2:], mode="nearest") < 0.5)
    clipped_halo_hr = ~unclipped_hr

    pa = align_planes(pred)
    ga = align_planes(guide_hr)
    lp = luminance(pa)
    lg = luminance(ga)
    grad_p = edge_energy(lp)
    grad_g = edge_energy(lg)
    chroma_res = (pa - lp) - (ga - lg)
    g12_res = (pa[:, 1:2] - pa[:, 3:4]) - (ga[:, 1:2] - ga[:, 3:4])

    # Unclipped edge sharpening ratio vs denoised guide
    u_mask = unclipped_hr[..., 16:-16, 16:-16]
    gp_u = grad_p[..., 16:-16, 16:-16][u_mask]
    gg_u = grad_g[..., 16:-16, 16:-16][u_mask]
    edge_gain_unclipped = float(gp_u.square().mean().sqrt() / gg_u.square().mean().sqrt().clamp_min(1e-12))

    out = {
        "min_signed_raw": float(mosaic.min()),
        "max_signed_raw": float(mosaic.max()),
        "p001_signed_raw": float(np.percentile(mosaic, 0.01)),
        "p999_signed_raw": float(np.percentile(mosaic, 99.99)),
        "negative_fraction": float((mosaic < 0.0).mean()),
        "below_neg005_count": int((mosaic < -0.005).sum()),
        "above_one_fraction": float((mosaic > 1.0).mean()),
        "above_105_count": int((mosaic > 1.05).sum()),
        "unclipped_edge_gain_vs_denoise": edge_gain_unclipped,
        "unclipped_chroma_res_rms": float(chroma_res[..., 16:-16, 16:-16][u_mask.expand_as(chroma_res[..., 16:-16, 16:-16])].square().mean().sqrt()),
        "unclipped_g12_res_rms": float(g12_res[..., 16:-16, 16:-16][u_mask].square().mean().sqrt()),
    }
    if clipped_halo_hr.any():
        c_mask = clipped_halo_hr[..., 16:-16, 16:-16]
        if c_mask.any():
            out["halo_zone_min"] = float(pred[..., 16:-16, 16:-16][c_mask.expand_as(pred[..., 16:-16, 16:-16])].min())
            out["halo_zone_max"] = float(pred[..., 16:-16, 16:-16][c_mask.expand_as(pred[..., 16:-16, 16:-16])].max())
            out["halo_zone_chroma_res_rms"] = float(
                chroma_res[..., 16:-16, 16:-16][c_mask.expand_as(chroma_res[..., 16:-16, 16:-16])].square().mean().sqrt()
            )
    if nov7_ref is not None:
        diff_u = (pred - nov7_ref)[..., 16:-16, 16:-16][u_mask.expand_as(pred[..., 16:-16, 16:-16])]
        out["unclipped_rms_diff_from_nov7"] = float(diff_u.square().mean().sqrt())
    return out


def make_grid(tiles: list[list[tuple[str, np.ndarray]]], out_path: Path, label_h: int = 26) -> None:
    """Save a 2D grid of labeled RGB uint8 crops."""
    rows = len(tiles)
    cols = max(len(r) for r in tiles)
    h, w = tiles[0][0][1].shape[:2]
    canvas = Image.new("RGB", (cols * w, rows * (h + label_h)), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(tiles):
        for c, (label, img) in enumerate(row):
            x0 = c * w
            y0 = r * (h + label_h)
            draw.rectangle([x0, y0, x0 + w, y0 + label_h], fill=(24, 27, 32))
            draw.text((x0 + 8, y0 + 6), label, fill=(240, 240, 240))
            canvas.paste(Image.fromarray(img), (x0, y0 + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def analytic_fixture(kind: str, n: int = 96) -> torch.Tensor:
    """Exact float32 analytic HR sensor fields (1x4xnxn) for quantitative guard verification."""
    y, x = torch.meshgrid(torch.arange(n * 2, dtype=torch.float32), torch.arange(n * 2, dtype=torch.float32), indexing="ij")
    if kind == "clipped_step":
        # 200:1 specular night-lamp step (0.004 -> 1.3 clipped to 1.0 in LR)
        field = 0.004 + 1.30 * (x + 0.31 * y > 1.31 * n).float()
    elif kind == "neutral_step":
        field = 0.06 + 0.54 * (x + 0.37 * y > 1.37 * n).float()
    elif kind == "weak_texture":
        field = 0.15 + 0.008 * torch.sin(2.0 * math.pi * x / 9.0) * torch.cos(2.0 * math.pi * y / 13.0)
    elif kind == "star":
        field = 0.02 + 0.72 * torch.exp(-((x - n) ** 2 + (y - n) ** 2) / (2.0 * 0.95**2))
    else:
        raise ValueError(kind)
    sites = ((0, 0), (0, 1), (1, 1), (1, 0))
    return torch.stack([field[sy::2, sx::2] for sy, sx in sites], dim=0)[None]


@torch.inference_mode()
def run_analytic_suite(r4: Denoiser, s42: torch.nn.Module, device: str) -> dict:
    """Verify Track A refinement on analytic clipped_step, neutral_step, weak_texture, star."""
    from denoise_lab.sr.degradation_v2 import DegradationParams, forward_degrade_v2

    results = {}
    params = DegradationParams(sigma_x=0.65, sigma_y=0.65, theta=0.0, aperture_fill=0.75)
    for kind in ("clipped_step", "neutral_step", "weak_texture", "star"):
        target = analytic_fixture(kind, n=96)
        clean_lr = forward_degrade_v2(target, params)
        if kind == "clipped_step":
            noisy_lr = clean_lr.clamp_max(1.0)
            target_eval = target.clamp_max(1.0)
        else:
            noisy_lr = clean_lr
            target_eval = target
        guide_np = r4.denoise_packed(
            noisy_lr[0].numpy(),
            prepass=True,
            tile_size=512,
            overlap=60,
            strength=1.0,
            highlights="protect",
        )
        y_t = noisy_lr.to(device)
        d_t = torch.from_numpy(guide_np)[None].to(device)
        pred_raw = context_tiled_sr(s42, y_t, d_t, center=80, apron=16, overlap=40).cpu()
        pred_ref = refine_sr_prediction(pred_raw, d_t.cpu(), y_t.cpu())
        base_hr = site_upsample(d_t.cpu())

        row = {}
        for name, p in (("denoise", base_hr), ("v1_unrefined", pred_raw), ("track_a_refined", pred_ref)):
            b = 12
            pr = p[..., b:-b, b:-b]
            tr = target_eval[..., b:-b, b:-b]
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
            row[name] = m
        results[kind] = row
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAP)
    parser.add_argument("--output", type=Path, default=REPO / "runs/raw_sr_v2/track_a")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--tile-batch", type=int, default=4)
    parser.add_argument("--keep-full-outputs", action="store_true", help="Retain full-res 2048px DNGs and PNGs (~185 MB)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"Track A evaluation on device={device}", flush=True)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    dng_dir = out_dir / "dngs"
    dng_dir.mkdir(exist_ok=True)

    r4_path = args.snapshot / "models/r4_plain_weights.pt"
    ckpt_paths = {
        "s42": args.snapshot / "runs/raw_sr_research/six_hour_20261030/experiments/unet_fusion_noise_24k_s42/best.pt",
        "s48": args.snapshot / "runs/raw_sr_research/nvidia_input_fit_20261104/weights/control_s48_step024000.pt",
        "s49": args.snapshot / "runs/raw_sr_research/nvidia_input_fit_20261104/weights/control_s49_step024000.pt",
    }

    r4 = Denoiser.load(r4_path, device=device, weights="model")
    models = {}
    metas = {}
    for name, p in ckpt_paths.items():
        m, meta = load_v1_checkpoint(p, device=device)
        models[name] = m
        metas[name] = {"path": str(p), "sha256": sha256(p), **meta}
    s42_tlc = apply_tlc(models["s42"], base_kernel=33, inplace=False).to(device).eval()

    print("Running analytic fixture checks...", flush=True)
    analytic_results = run_analytic_suite(r4, models["s42"], device=device)

    case_records = []
    center_rows = []
    night_ablation_renders = {}
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    for case_id, label, filename, display_lift in CASES:
        ct0 = time.perf_counter()
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

        # Two guide contracts: Nov 7 legacy (tile=160, ov=32, highlights=off) vs production (tile=512, ov=60, highlights=protect)
        d_nov7 = r4.denoise_packed(
            y_np, prepass=True, tile_size=160, overlap=32, strength=1.0, highlights="off"
        )
        d_prod = r4.denoise_packed(
            y_np, prepass=True, tile_size=512, overlap=60, strength=1.0, highlights="protect"
        )
        guide_diff_rms = float(np.sqrt(np.mean((d_prod - d_nov7) ** 2)))
        guide_diff_max = float(np.max(np.abs(d_prod - d_nov7)))

        y_t = torch.from_numpy(y_np)[None].to(device)
        # Apply r4's exact defect prepass to y for refined SR so isolated hot pixels
        # removed from d_prod are not re-injected into PhotoSR's raw branch.
        y_pre_t = correct_defects(y_t.cpu(), k=8, abs_floor=0.01, adjacency_ratio=0.1)[0].to(device)
        d_nov7_t = torch.from_numpy(d_nov7)[None].to(device)
        d_prod_t = torch.from_numpy(d_prod)[None].to(device)

        with torch.inference_mode():
            pred_raw_hr = site_upsample(y_t).cpu()
            pred_denoise_hr = site_upsample(d_prod_t).cpu()

            # Exact Nov 7 baseline reproduction (uncorrected y, r4 tile=160 highlights=off, ov=0)
            pred_nov7_s42 = context_tiled_sr(
                models["s42"], y_t, d_nov7_t, center=80, apron=16, overlap=0, tile_batch_size=args.tile_batch
            ).cpu()

            # Production guide + defect-prepassed y + feathered overlap=40
            pred_s42_tile = context_tiled_sr(
                models["s42"], y_pre_t, d_prod_t, center=80, apron=16, overlap=40, feather_mode="cosine", tile_batch_size=args.tile_batch
            ).cpu()
            pred_s42_tlc = context_tiled_sr(
                s42_tlc, y_pre_t, d_prod_t, center=80, apron=16, overlap=40, feather_mode="cosine", tile_batch_size=args.tile_batch
            ).cpu()

            # Control s48 & s49 (raw + refined)
            pred_s48_tile = context_tiled_sr(
                models["s48"], y_pre_t, d_prod_t, center=80, apron=16, overlap=40, feather_mode="cosine", tile_batch_size=args.tile_batch
            ).cpu()
            pred_s49_tile = context_tiled_sr(
                models["s49"], y_pre_t, d_prod_t, center=80, apron=16, overlap=40, feather_mode="cosine", tile_batch_size=args.tile_batch
            ).cpu()

        y_cpu = y_pre_t.cpu()
        d_cpu = d_prod_t.cpu()

        # Ablations on s42
        pred_s42_hl_only = highlight_blend_sr(pred_s42_tile, d_cpu, y_cpu)
        pred_s42_refined = refine_sr_prediction(pred_s42_tile, d_cpu, y_cpu)
        pred_s42_tlc_refined = refine_sr_prediction(pred_s42_tlc, d_cpu, y_cpu)
        pred_s48_refined = refine_sr_prediction(pred_s48_tile, d_cpu, y_cpu)
        pred_s49_refined = refine_sr_prediction(pred_s49_tile, d_cpu, y_cpu)

        methods = {
            "raw": pred_raw_hr,
            "denoise": pred_denoise_hr,
            "nov7_s42": pred_nov7_s42,
            "s42_tile_guide": pred_s42_tile,
            "s42_tlc": pred_s42_tlc,
            "s42_hl_only": pred_s42_hl_only,
            "s42_refined": pred_s42_refined,
            "s42_tlc_refined": pred_s42_tlc_refined,
            "s48_unrefined": pred_s48_tile,
            "s48_refined": pred_s48_refined,
            "s49_unrefined": pred_s49_tile,
            "s49_refined": pred_s49_refined,
        }

        method_metrics = {}
        renders = {}
        for mname, tensor in methods.items():
            mosaic = unpack(tensor[0].numpy())
            dng_path = dng_dir / f"{case_id}_{mname}.dng"
            rgb8, exact = render_dng_srgb8(mosaic, meta, dng_path, display_lift)
            renders[mname] = rgb8
            stats = raw_stats(tensor, pred_denoise_hr, y_cpu, nov7_ref=pred_nov7_s42)
            stats["dng_codes_exact"] = exact
            method_metrics[mname] = stats
            # Remove intermediate DNGs after roundtrip verification unless --keep-full-outputs
            if not (args.keep_full_outputs and mname in ("denoise", "nov7_s42", "s42_refined", "s48_refined", "s49_refined")):
                dng_path.unlink(missing_ok=True)

        if args.keep_full_outputs:
            for mname in ("denoise", "nov7_s42", "s42_refined", "s48_refined", "s49_refined"):
                Image.fromarray(renders[mname]).save(out_dir / f"{case_id}_{mname}.png", icc_profile=icc)

        # Center 512x512 crop (matches center-comparisons.png geometry!)
        h, w = renders["denoise"].shape[:2]
        cy, cx = h // 2, w // 2
        c_slice = (slice(cy - 256, cy + 256), slice(cx - 256, cx + 256))
        center_rows.append(
            [
                (f"{case_id} / denoise", renders["denoise"][c_slice]),
                (f"{case_id} / nov7 s42", renders["nov7_s42"][c_slice]),
                (f"{case_id} / s42 refined", renders["s42_refined"][c_slice]),
                (f"{case_id} / s48 refined", renders["s48_refined"][c_slice]),
                (f"{case_id} / s49 refined", renders["s49_refined"][c_slice]),
            ]
        )

        if case_id == "night":
            night_ablation_renders = renders

        case_records.append(
            {
                "id": case_id,
                "label": label,
                "source": filename,
                "source_sha256": sha256(source),
                "display_lift": display_lift,
                "guide_160_vs_512_rms": guide_diff_rms,
                "guide_160_vs_512_max": guide_diff_max,
                "guide_nov7_min_max": [float(d_nov7.min()), float(d_nov7.max())],
                "guide_prod_min_max": [float(d_prod.min()), float(d_prod.max())],
                "methods": method_metrics,
                "seconds": time.perf_counter() - ct0,
            }
        )
        print(
            f"[{case_id}] done in {time.perf_counter() - ct0:.2f}s | "
            f"nov7_s42=[{method_metrics['nov7_s42']['min_signed_raw']:.4f}, {method_metrics['nov7_s42']['max_signed_raw']:.4f}] -> "
            f"s42_refined=[{method_metrics['s42_refined']['min_signed_raw']:.4f}, {method_metrics['s42_refined']['max_signed_raw']:.4f}] | "
            f"s48_refined=[{method_metrics['s48_refined']['min_signed_raw']:.4f}, {method_metrics['s48_refined']['max_signed_raw']:.4f}]",
            flush=True,
        )

    # Save center before/after grid across all 3 demos
    make_grid(center_rows, out_dir / "center_before_after.png")

    # Save zoomed 2x crops of night (row 1: lamp/sign/railing clipped halo zone; row 2: brickwork/pedestrian sign detail zone)
    if night_ablation_renders:
        # Center crop is [768:1280, 768:1280].
        # Lamp + glowing sign + top of railing sits around y=820..1200, x=780..1160.
        # Brick wall + triangular flag + pedestrian sign sits around y=768..1120, x=1040..1360.
        def zoom_crop(img: np.ndarray, y0: int, x0: int, sz: int, out_sz: int = 384) -> np.ndarray:
            crop = img[y0 : y0 + sz, x0 : x0 + sz]
            return np.asarray(Image.fromarray(crop).resize((out_sz, out_sz), Image.Resampling.NEAREST))

        ablation_keys = [
            ("denoise", "denoise (r4 512)"),
            ("nov7_s42", "nov7 s42 (ov=0)"),
            ("s42_tile_guide", "s42 + r4_512 + ov40"),
            ("s42_tlc", "s42 + TLC(33)"),
            ("s42_hl_only", "s42 + hl_blend only"),
            ("s42_refined", "s42 Track-A refined"),
            ("s48_refined", "s48 Track-A refined"),
            ("s49_refined", "s49 Track-A refined"),
        ]
        # 2x4 grid for lamps/railings + 2x4 grid for brickwork/signs
        lamp_row1 = [(lbl, zoom_crop(night_ablation_renders[k], 840, 770, 256, 384)) for k, lbl in ablation_keys[:4]]
        lamp_row2 = [(lbl, zoom_crop(night_ablation_renders[k], 840, 770, 256, 384)) for k, lbl in ablation_keys[4:]]
        make_grid([lamp_row1, lamp_row2], out_dir / "night_halo_ablation.png")

        railing_row1 = [(lbl, zoom_crop(night_ablation_renders[k], 1030, 820, 256, 384)) for k, lbl in ablation_keys[:4]]
        railing_row2 = [(lbl, zoom_crop(night_ablation_renders[k], 1030, 820, 256, 384)) for k, lbl in ablation_keys[4:]]
        make_grid([railing_row1, railing_row2], out_dir / "night_railing_ablation.png")

        detail_row1 = [(lbl, zoom_crop(night_ablation_renders[k], 768, 1024, 256, 384)) for k, lbl in ablation_keys[:4]]
        detail_row2 = [(lbl, zoom_crop(night_ablation_renders[k], 768, 1024, 256, 384)) for k, lbl in ablation_keys[4:]]
        make_grid([detail_row1, detail_row2], out_dir / "night_detail_ablation.png")

    summary = {
        "device": device,
        "total_seconds": time.perf_counter() - t0,
        "r4_path": str(r4_path),
        "r4_sha256": sha256(r4_path),
        "checkpoints": metas,
        "refine_config": RefineConfig().__dict__,
        "analytic_fixtures": analytic_results,
        "cases": case_records,
    }
    if not args.keep_full_outputs and dng_dir.exists() and not any(dng_dir.iterdir()):
        dng_dir.rmdir()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Track A report to {out_dir / 'summary.json'} ({summary['total_seconds']:.2f}s total)", flush=True)


if __name__ == "__main__":
    main()
