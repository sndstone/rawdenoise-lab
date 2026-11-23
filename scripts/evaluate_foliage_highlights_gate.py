#!/usr/bin/env python3
"""Council Stage-1 regression gate for green foliage grids and clipped highlights.

Measures:
  1. Green moss/foliage G1-G2 relative offset (%) and packed-green power-of-2
     periodic FFT energy (grid_p248 = sum of col/row periods 2, 4, 8), both
     default (tta_transpose=False) and optionally with tta_transpose=True.
  2. Clipped highlights (core & 1-4 packed-px ring) signed Lab mean_da / mean_db
     and raw (R+B)/2 - G shift with highlights='off' and highlights='protect',
     including Sony/X2D >1.0 headroom exactness under protect.

Read-only on all input RAWs; writes JSON + Markdown report under --output.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np
import rawpy
import torch
from torch.nn import functional as F

from denoise_lab.bayer import normalize_raw, pack, unpack
from denoise_lab.defects import correct_defects
from denoise_lab.dng import RawMetadata, write_dng
from denoise_lab.inference import Denoiser


def to_srgb(linear):
    a = np.clip(linear, 0.0, 1.0)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(np.maximum(a, 0.0031308), 1.0 / 2.4) - 0.055)


def srgb_to_lab(srgb):
    lin = np.where(srgb <= 0.04045, srgb / 12.92, np.power(np.maximum((srgb + 0.055) / 1.055, 0.0), 2.4))
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
    xyz = (lin.astype(np.float64) @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(np.maximum(xyz, 0.0)), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def demosaic_ahd(packed, meta, exposure=1.0, headroom_scale=4.0):
    """Demosaic packed RGGB while preserving >1.0 sensor headroom and WB headroom.

    Scaling the mosaic by 1/headroom_scale before uint16 DNG encoding and using
    HighlightMode.Ignore ensures >1.0 headroom codes (e.g. Sony 1.069) and
    post-WB linear RGB values never clip inside LibRaw before applying exposure.
    """
    scale = float(max(headroom_scale, 1.0))
    scaled = np.clip(packed / scale, 0.0, 1.0).astype(np.float32)
    c_meta = replace(meta, pattern=np.array([[0, 1], [1, 2]], dtype=np.uint8),
                     black=np.zeros((2, 2)), white=np.full((2, 2), 65535.0), orientation=1)
    with tempfile.NamedTemporaryFile(suffix=".dng", delete=True) as tf:
        write_dng(Path(tf.name), unpack(scaled), c_meta, metadata_mode="minimal", preview=False)
        with rawpy.imread(tf.name) as raw:
            lin = raw.postprocess(
                use_camera_wb=True, no_auto_bright=True, output_bps=16,
                output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1),
                user_flip=0, adjust_maximum_thr=0.0,
                highlight_mode=rawpy.HighlightMode.Ignore,
            ).astype(np.float32) / 65535.0
    return to_srgb(lin * (float(exposure) * scale))


def foliage_metrics(packed):
    g1, g2 = packed[1], packed[3]
    g = 0.5 * (g1 + g2)
    mean_g = float(np.mean(g))
    g12_rel_pct = 100.0 * float(np.mean(g1 - g2)) / max(mean_g, 1e-6)
    h, w = g.shape
    ch, cw = (h // 32) * 32, (w // 32) * 32
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    gc = g[y0:y0 + ch, x0:x0 + cw].astype(np.float64)
    gc -= np.mean(gc)
    col_fft = np.abs(np.fft.rfft(gc.mean(axis=0))) / cw
    row_fft = np.abs(np.fft.rfft(gc.mean(axis=1))) / ch

    def amp(spec, n, p):
        idx = n // p
        return float(spec[idx]) if 0 < idx < len(spec) else 0.0

    grid_p248 = sum(amp(col_fft, cw, p) + amp(row_fft, ch, p) for p in (2, 4, 8))
    return {"g1_minus_g2_rel_pct": g12_rel_pct, "grid_p248": grid_p248}


def highlight_metrics(original, denoised, meta, exposure, *, headroom_scale=4.0, prepass_ref=None):
    core = (original >= 0.98).any(axis=0)
    if not core.any():
        return {"has_clip": False}
    c_t = torch.from_numpy(core.astype(np.float32))[None, None]
    ring = (F.max_pool2d(c_t, 9, stride=1, padding=4)[0, 0].numpy() > 0) & (~core)
    img_in = demosaic_ahd(original, meta, exposure, headroom_scale=headroom_scale)
    img_out = demosaic_ahd(denoised, meta, exposure, headroom_scale=headroom_scale)
    dl = srgb_to_lab(img_out) - srgb_to_lab(img_in)
    scale = img_in.shape[0] // original.shape[1]
    m_core = core.repeat(scale, axis=0).repeat(scale, axis=1)
    m_ring = ring.repeat(scale, axis=0).repeat(scale, axis=1)

    pre_core = (prepass_ref >= 0.98).any(axis=0) if prepass_ref is not None else None

    def region_stats(pmask, rmask, *, pre_mask=None):
        if not pmask.any():
            return None
        g_in = 0.5 * (original[1][pmask] + original[3][pmask])
        g_out = 0.5 * (denoised[1][pmask] + denoised[3][pmask])
        rb_in = 0.5 * (original[0][pmask] + original[2][pmask])
        rb_out = 0.5 * (denoised[0][pmask] + denoised[2][pmask])
        stats = {
            "mean_da": float(np.mean(dl[..., 1][rmask])),
            "mean_db": float(np.mean(dl[..., 2][rmask])),
            "raw_rb_minus_g_shift": float(np.mean((rb_out - g_out) - (rb_in - g_in))),
            "max_abs_diff": float(np.max(np.abs(denoised[:, pmask] - original[:, pmask]))),
        }
        if prepass_ref is not None:
            cmask = pre_mask if pre_mask is not None else pmask
            if cmask.any():
                g_pre = 0.5 * (prepass_ref[1][cmask] + prepass_ref[3][cmask])
                rb_pre = 0.5 * (prepass_ref[0][cmask] + prepass_ref[2][cmask])
                g_out_p = 0.5 * (denoised[1][cmask] + denoised[3][cmask])
                rb_out_p = 0.5 * (denoised[0][cmask] + denoised[2][cmask])
                stats["prepass_raw_rb_minus_g_shift"] = float(np.mean((rb_out_p - g_out_p) - (rb_pre - g_pre)))
                stats["prepass_max_abs_diff"] = float(np.max(np.abs(denoised[:, cmask] - prepass_ref[:, cmask])))
        return stats

    return {
        "has_clip": True,
        "headroom_scale": float(headroom_scale),
        "core": region_stats(core, m_core, pre_mask=pre_core),
        "ring_1_4": region_stats(ring, m_ring),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--moss-raw", type=Path, required=True, help="Dense green foliage/moss RAW")
    ap.add_argument("--moss-crop-yx", type=int, nargs=2, default=[1760, 1376])
    ap.add_argument("--highlight-raw", action="append", default=[], metavar="NAME=PATH[:EXPOSURE]")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--check-transpose-tta", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    denoisers = {spec.split("=", 1)[0]: Denoiser.load(spec.split("=", 1)[1], device=args.device) for spec in args.checkpoint}

    # 1. Green moss gate
    with rawpy.imread(str(args.moss_raw)) as raw:
        moss_packed = pack(normalize_raw(raw))
    cy, cx = args.moss_crop_yx
    sz, ctx = 384, 64
    _, h, w = moss_packed.shape
    y0, y1 = max(0, cy - ctx), min(h, cy + sz + ctx)
    x0, x1 = max(0, cx - ctx), min(w, cx + sz + ctx)
    patch = moss_packed[:, y0:y1, x0:x1]
    dy, dx = cy - y0, cx - x0
    raw_moss = foliage_metrics(moss_packed[:, cy:cy + sz, cx:cx + sz])

    moss_out = {"raw_input": raw_moss}
    for name, den in denoisers.items():
        out = den.denoise_packed(patch, tile_size=args.tile_size, highlights="protect")[:, dy:dy + sz, dx:dx + sz]
        moss_out[name] = foliage_metrics(out)
        if args.check_transpose_tta:
            out_t = den.denoise_packed(patch, tile_size=args.tile_size, highlights="protect", tta_transpose=True)[:, dy:dy + sz, dx:dx + sz]
            moss_out[f"{name}_tta_transpose"] = foliage_metrics(out_t)

    # 2. Clipped highlights gate
    hl_out = {}
    for entry in args.highlight_raw:
        hname, rest = entry.split("=", 1)
        if ":" in rest and not rest.endswith(":"):
            pstr, estr = rest.rsplit(":", 1)
            try:
                exp = float(estr)
                hpath = Path(pstr)
            except ValueError:
                hpath, exp = Path(rest), 1.0
        else:
            hpath, exp = Path(rest), 1.0
        with rawpy.imread(str(hpath)) as raw:
            meta = RawMetadata.from_raw(raw, hpath)
            packed = pack(normalize_raw(raw))
        _, hh, ww = packed.shape
        hsz = min(384, hh, ww)
        m98 = (packed >= 0.98).any(axis=0)
        ys = sorted(set(list(range(0, max(1, hh - hsz + 1), 192)) + [hh - hsz]))
        xs = sorted(set(list(range(0, max(1, ww - hsz + 1), 192)) + [ww - hsz]))
        _, hy, hx = max((int(m98[y:y + hsz, x:x + hsz].sum()), y, x) for y in ys for x in xs)
        crop = packed[:, hy:hy + hsz, hx:hx + hsz].copy()
        prepass_ref = correct_defects(torch.from_numpy(crop)[None], k=8, abs_floor=0.01, adjacency_ratio=0.1)[0][0].numpy()

        outputs = {}
        peak = float(np.max(crop))
        for name, den in denoisers.items():
            off = den.denoise_packed(crop, tile_size=args.tile_size, highlights="off")
            pro = den.denoise_packed(crop, tile_size=args.tile_size, highlights="protect")
            outputs[name] = (off, pro)
            peak = max(peak, float(np.max(off)), float(np.max(pro)))
        # Shared headroom-preserving scale across input and all variants for this crop.
        headroom_scale = max(4.0, float(np.ceil(peak * 4.0)))

        hl_out[hname] = {
            "path": str(hpath),
            "exposure": exp,
            "headroom_scale": headroom_scale,
            "orig_ch_max": [float(crop[c].max()) for c in range(4)],
            "models": {},
        }
        for name, (off, pro) in outputs.items():
            hl_out[hname]["models"][name] = {
                "off": highlight_metrics(crop, off, meta, exp, headroom_scale=headroom_scale, prepass_ref=prepass_ref),
                "protect": highlight_metrics(crop, pro, meta, exp, headroom_scale=headroom_scale, prepass_ref=prepass_ref),
            }

    result = {"moss": moss_out, "highlights": hl_out}
    (args.output / "gate_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
