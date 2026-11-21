#!/usr/bin/env python3
"""Real paired evaluation from a tripod burst: noisy high-ISO frames vs a mean of base-ISO frames.

The spec JSON lists groups: {"name", "noisy": [files], "clean": [files], "shift": [dy, dx]}.
`shift` is in packed pixels; aligned_noisy[:, y, x] = noisy[:, y+dy, x+dx]. Integer packed shifts
keep the CFA phase. By default these measured shifts are used unchanged.
--align refines each frame with subpixel packed-plane translation and NLF motion masks.
--alignment-only saves mask/noise diagnostics without running a model.

Per group:
- Brightness: per-channel affine fit noisy ~ (a(x, y))*clean + b. a(x, y) is quadratic in position
  to absorb small aperture/vignetting differences. Fitted on static blocks only.
- Motion mask: blocks whose base-ISO burst varies beyond its own noise, or whose noisy-group mean
  departs from the fitted clean target beyond noise, are excluded (dilated).
- Metrics on static pixels: MSE corrected for the clean mean's own noise (from the burst's
  temporal variance), noise reduction in dB, PSNR against the target's 99.5th percentile and
  SSIM on peak-normalised planes.
- Control: synthetic Poisson-Gaussian noise at each frame's blind NLF on the same target,
  denoised by the same model. A real-vs-synthetic gap means real noise departs from the prior.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_cameras import gauss, load_packed, render
from denoise_lab.burst_eval import block_view, dilate, fit_gain, shifted, upsample  # noqa: F401 (re-exported)
from denoise_lab.burst_align import warp_packed




def temporal_nlf(frames, block=16, bins=30):
    """Noise curve var = K*mu + c from a static burst: per-block temporal variance vs mean.

    The per-brightness median rejects moving blocks. Returns K and read_std per channel.
    """
    mu = block_view(frames.mean(0), block).mean((-3, -1))
    tv = block_view(frames.var(0, ddof=1), block).mean((-3, -1))
    out = []
    for c in range(frames.shape[1]):
        m, v = mu[c].ravel(), tv[c].ravel()
        ok = np.isfinite(m) & np.isfinite(v) & (m < .9)
        m, v = m[ok], v[ok]
        edges = np.unique(np.quantile(m, np.linspace(0, 1, bins+1)))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (m >= lo) & (m < hi)
            if sel.sum() >= 20:
                xs.append(np.median(m[sel]))
                ys.append(np.median(v[sel]))
        A = np.stack([np.maximum(xs, 0), np.ones(len(xs))], 1)
        K, c0 = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
        out.append({"K": float(K), "read_std": float(np.sqrt(max(c0, 0.))), "points": len(xs)})
    return out


def ssim_np(x, y, peak, mask=None, legacy_metrics=False):
    from denoise_lab.metrics import ssim
    if mask is not None and legacy_metrics:
        x = np.where(mask, x, y)
    t = lambda a: torch.from_numpy(np.nan_to_num(np.clip(a/peak, 0, 1)).astype(np.float32))[None]
    valid = torch.from_numpy(np.asarray(mask, dtype=bool)) if mask is not None and not legacy_metrics else None
    return ssim(t(x), t(y), mask=valid)


def prepare_pair(clean_frames, noisy, shift, block=16, align=False, legacy_metrics=False):
    """Prepare integer-shift pairs, or opt in to registered NLF references."""
    b = block
    if align:
        from denoise_lab.burst_align import (align_burst, estimate_burst_noise,
                                            interpolation_variance, robust_merge, match_cross_iso)
        registered, clean_shifts = align_burst(clean_frames)
        cq = [interpolation_variance(s) for s in clean_shifts]
        cp = estimate_burst_noise(registered, cq, b)
        merged = robust_merge(registered, cp, cq, block=b)
        aligned, shifts = align_burst(noisy, reference=clean_frames[0], initial=shift)
        nq = [interpolation_variance(s) for s in shifts]
        np_ = estimate_burst_noise(aligned, nq, b)
        target, target_var, static, coeffs = match_cross_iso(aligned, merged, np_, fit_gain, nq, b)
        info = {"clean_shifts": clean_shifts.tolist(), "noisy_shifts": shifts.tolist(),
                "clean_nlf": cp, "noisy_nlf": np_, "clean_static_fraction": float(merged.static.mean()),
                "clean_noise_std": float(np.sqrt(np.nanmean(merged.variance[:, merged.static]))),
                "effective_frames": float(np.median(merged.effective_count[:, merged.static]))
                                    if merged.static.any() else 0.}
        return aligned, target, target_var, static, coeffs, shifts, info
    clean = clean_frames.mean(0)
    n_clean = len(clean_frames)
    tvar = clean_frames.var(0, ddof=1)
    dy, dx = shift
    aligned = np.stack([shifted(a, dy, dx) for a in noisy])
    valid = np.isfinite(aligned).all((0, 1))
    valid &= (clean < .95).all(0) & (np.nan_to_num(aligned) < .95).all((0, 1))
    # Burst motion: block temporal variance vs the burst's own robust noise-level fit.
    tv_b = block_view(tvar, b).mean((-3, -1))
    mu_b = block_view(clean, b).mean((-3, -1))
    # Expected variance = median block variance at the same brightness (40 quantile bins), which
    # includes read noise without a parametric fit; burst motion stands out above it.
    moving = np.zeros(mu_b.shape[1:], bool)
    for c in range(4):
        edges = np.quantile(mu_b[c], np.linspace(0, 1, 41))
        idx = np.clip(np.searchsorted(edges, mu_b[c], side="right")-1, 0, 39)
        expected = np.array([np.median(tv_b[c][idx == k]) if (idx == k).any() else np.inf for k in range(40)])
        moving |= tv_b[c] > 3*expected[idx]
    static = upsample(~dilate(moving, 2), b, valid.shape) & valid
    noisy_mean = np.nanmean(aligned, 0)
    target, coeffs, gain_map = fit_gain(noisy_mean, clean, static, b, return_gain=True)
    # Group motion: noisy-group mean vs target beyond the expected noise of a block mean.
    diff_b = block_view(noisy_mean-target, b).mean((-3, -1))
    nvar_b = block_view(np.nanvar(aligned, 0, ddof=1), b).mean((-3, -1))
    z = np.abs(diff_b)/np.sqrt(np.maximum(nvar_b/(len(noisy)*b*b), 1e-16))
    moved = (np.nan_to_num(z, nan=99) > 5).any(0)
    static &= upsample(~dilate(moved, 2), b, valid.shape)
    if legacy_metrics:
        gain_map = np.array([sol[0] for sol in coeffs])[:, None, None]
    target_var = (gain_map**2)*tvar/n_clean  # residual noise of the reference
    info = {"clean_noise_std": float(np.sqrt(np.mean(tvar[:, static]/n_clean)))
                                if static.any() else None}
    return aligned, target, target_var, static, coeffs, np.tile(shift, (len(noisy), 1)), info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--model")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-static", type=float, default=.05, help="Minimum static frame fraction to score a group")
    p.add_argument("--align", action="store_true", help="Subpixel packed registration and NLF robust targets")
    p.add_argument("--legacy-metrics", action="store_true",
                   help="Restore v1 target-filled SSIM and global-only non-aligned reference variance")
    p.add_argument("--alignment-only", "--mask-only", dest="alignment_only", action="store_true",
                   help="Save reference/mask diagnostics without model inference")
    args = p.parse_args()
    protocol = "paired-burst-v1" if args.legacy_metrics else "paired-burst-v2"
    if not args.alignment_only and not args.model:
        p.error("--model is required unless --alignment-only")
    from denoise_lab.inference import Denoiser
    from denoise_lab.bootstrap import add_bootstrap_noise
    from denoise_lab.bayer import pack, unpack
    from denoise_lab.rawstats import estimate_noise_parameters
    from PIL import Image
    spec = json.loads(Path(args.spec).read_text())
    root = Path(spec["root"])
    out = Path(args.output)
    (out/"crops").mkdir(parents=True, exist_ok=True)
    den = None if args.alignment_only else Denoiser.load(args.model, device=args.device)
    rng = np.random.default_rng(args.seed)
    b = args.block
    results, skipped, nlf_rows, alignment_rows = [], [], [], []
    for group in spec["groups"]:
        clean_frames = np.stack([load_packed(root/f)[0] for f in group["clean"]])
        dy, dx = group["shift"]
        noisy, wb = [], None
        for f in group["noisy"]:
            a, wb = load_packed(root/f)
            noisy.append(a)
        noisy = np.stack(noisy)
        try:
            aligned, target, target_var, static, coeffs, shifts, info = prepare_pair(
                clean_frames, noisy, (dy, dx), b, args.align, args.legacy_metrics)
        except ValueError as exc:  # e.g. fit_gain without enough static blocks
            print(f"{group['name']}: not scored ({exc})", flush=True)
            skipped.append({"group": group["name"], "static_fraction": 0., "reason": str(exc)})
            continue
        info.update({"group": group["name"], "static_fraction": float(static.mean()), "gain_fit": coeffs,
                     "matched_target_noise_std": float(np.sqrt(np.nanmean(target_var[:, static])))
                                                 if static.any() else None})
        alignment_rows.append(info)
        np.save(out/f"{group['name']}_static_mask.npy", static)
        print(f"{group['name']}: usable {static.mean():.2%}; target noise std={info['matched_target_noise_std']}", flush=True)
        if args.alignment_only:
            continue
        m = static[None].repeat(4, 0)
        peak = float(np.percentile(target[m], 99.5)) if m.any() else float("nan")
        tv_mean = float(target_var[m].mean())
        nlf_rows.append({"group": group["name"], "noisy": temporal_nlf(noisy), "clean": temporal_nlf(clean_frames)})
        if static.mean() < args.min_static:
            print(f"{group['name']}: static {static.mean():.1%} of frame, below --min-static; crops only", flush=True)
            pred = warp_packed(den.denoise_packed(noisy[0]), shifts[0]) if args.align else shifted(den.denoise_packed(noisy[0]), dy, dx)
            for name, (cy, cx) in spec.get("crops", {}).items():
                sl = (slice(None), slice(cy, cy+192), slice(cx, cx+192))
                scale = .8/max(np.percentile((target[sl][1]+target[sl][3])/2, 99), 1e-4)
                tiles = [render(np.nan_to_num(x[sl]), wb, scale) for x in (aligned[0], pred, target)]
                sep = np.full((192, 4, 3), 255, np.uint8)
                row_img = np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)
                Image.fromarray(row_img).resize((row_img.shape[1]*2, row_img.shape[0]*2), Image.NEAREST).save(
                    out/"crops"/f"{group['name']}_{name}.png")
            skipped.append({"group": group["name"], "static_fraction": float(static.mean())})
            continue
        print(f"{group['name']}: static {static.mean():.1%} of frame, peak {peak:.4f}", flush=True)
        for i, (f, a) in enumerate(zip(group["noisy"], aligned)):
            raw = noisy[i]
            est = estimate_noise_parameters(raw)
            pred = warp_packed(den.denoise_packed(raw), shifts[i]) if args.align else shifted(den.denoise_packed(raw), dy, dx)
            mse_in = float(((a-target)[m]**2).mean())-tv_mean
            mse_out = float(((pred-target)[m]**2).mean())-tv_mean
            # Synthetic control: same target, white Poisson-Gaussian noise at this frame's blind NLF.
            t0 = np.nan_to_num(target)
            synth, _ = add_bootstrap_noise(unpack(t0), rng, fixed_parameters={
                "K": est["K"], "read_std": est["read_std"], "row_std": 0.})
            synth = pack(synth)
            spred = den.denoise_packed(synth)
            s_in = float(((synth-t0)[m]**2).mean())
            s_out = float(((spred-t0)[m]**2).mean())
            row = {"group": group["name"], "file": f, "K": est["K"], "read_std": est["read_std"],
                   "static_fraction": float(static.mean()), "peak": peak, "target_noise_var": tv_mean,
                   "mse_in": mse_in, "mse_out": mse_out,
                   "noise_reduction_db": 10*np.log10(mse_in/mse_out) if mse_out > 0 else None,
                   "psnr_in": 10*np.log10(peak**2/mse_in), "psnr_out": 10*np.log10(peak**2/mse_out) if mse_out > 0 else None,
                   "ssim_in": ssim_np(a, target, peak, m, args.legacy_metrics),
                   "ssim_out": ssim_np(pred, target, peak, m, args.legacy_metrics),
                   "synthetic_psnr_in": 10*np.log10(peak**2/s_in), "synthetic_psnr_out": 10*np.log10(peak**2/s_out),
                   "synthetic_noise_reduction_db": 10*np.log10(s_in/s_out)}
            results.append(row)
            print(f"  {f}: PSNR {row['psnr_in']:.2f} -> {row['psnr_out']:.2f} dB | synthetic "
                  f"{row['synthetic_psnr_in']:.2f} -> {row['synthetic_psnr_out']:.2f} dB", flush=True)
            if i == 0:
                for name, (cy, cx) in spec.get("crops", {}).items():
                    sl = (slice(None), slice(cy, cy+192), slice(cx, cx+192))
                    scale = .8/max(np.percentile((target[sl][1]+target[sl][3])/2, 99), 1e-4)
                    tiles = [render(np.nan_to_num(x[sl]), wb, scale) for x in (a, pred, target, spred)]
                    sep = np.full((192, 4, 3), 255, np.uint8)
                    row_img = np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2], sep, tiles[3]], 1)
                    Image.fromarray(row_img).resize((row_img.shape[1]*2, row_img.shape[0]*2), Image.NEAREST).save(
                        out/"crops"/f"{group['name']}_{name}.png")
        np.save(out/f"{group['name']}_static_mask.npy", static)
    (out/"alignment.json").write_text(json.dumps({"metric_protocol": protocol, "align": args.align,
                                                "spec": spec, "groups": alignment_rows}, indent=2))
    if args.alignment_only:
        return
    (out/"results.json").write_text(json.dumps({"metric_protocol": protocol,
                                                "model": args.model, "spec": spec, "frames": results,
                                                "skipped": skipped, "temporal_nlf": nlf_rows}, indent=1))
    groups = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)
    lines = [f"Metric protocol: `{protocol}`", "",
             "| Group | Frames | Static area | Real PSNR in→out | Noise reduction dB | SSIM in→out | "
             "Synthetic PSNR in→out | Synthetic noise reduction dB |", "|"+"---|"*8]
    for g, rs in groups.items():
        mean = lambda k: float(np.mean([r[k] for r in rs]))
        def mean_ssim(k):
            values = [r[k] for r in rs if r[k] is not None]
            return f"{np.mean(values):.3f}" if values else "n/a"
        lines.append(f"| {g} | {len(rs)} | {mean('static_fraction'):.0%} | {mean('psnr_in'):.2f}→{mean('psnr_out'):.2f} | "
                     f"{mean('noise_reduction_db'):.2f} | {mean_ssim('ssim_in')}→{mean_ssim('ssim_out')} | "
                     f"{mean('synthetic_psnr_in'):.2f}→{mean('synthetic_psnr_out'):.2f} | {mean('synthetic_noise_reduction_db'):.2f} |")
    for r in skipped:
        lines.append(f"| {r['group']} | – | {r['static_fraction']:.1%} | not scored: scene changed between groups | | | | |")
    lines += ["", "| Group | Burst | K (R, G1, B, G2) | read σ (R, G1, B, G2) |", "|---|---|---|---|"]
    for r in nlf_rows:
        for kind in ("noisy", "clean"):
            v = r[kind]
            gains = ', '.join(f"{c['K']:.2e}" for c in v)
            reads = ', '.join(f"{c['read_std']:.2e}" for c in v)
            lines.append(f"| {r['group']} | {kind} | {gains} | {reads} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
