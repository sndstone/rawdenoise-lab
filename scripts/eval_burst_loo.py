#!/usr/bin/env python3
"""Real-noise evaluation from same-ISO tripod bursts: leave-one-out burst-mean references.

For each group (same ISO, exposure and framing) and each chosen test frame i, the reference is the
mean of the other N-1 frames. Every frame is first brightness-matched to the group (per-channel
gain on static blocks; light changes during a burst). Static pixels are blocks whose burst
temporal variance stays within 3x the median at the same brightness (wind/motion excluded,
dilated). MSE is corrected for the reference's own noise: the per-pixel variance of the other
frames divided by N-1. So scores reflect real sensor noise of that ISO without a base-ISO
reference or exposure matching between groups.

Spec JSON: {"root": dir, "groups": [{"name", "files": [...], "test": [indices]}]}
Models: --model name=path (repeatable). The same frames and masks are used for every model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_cameras import gauss, load_packed, render
from eval_real_burst import block_view, dilate, upsample


def pooled_psnr(rows):
    """Sample-weighted corrected MSE, converted to PSNR once; sensitivity only."""
    if not rows or any(r['mse_out_all'] is None or not np.isfinite(r['mse_out_all'])
                       or r['mse_out_all'] <= 0 for r in rows):
        return None
    weights = np.array([r.get('static_samples', 1) for r in rows])
    mse = np.average([r['mse_out_all'] for r in rows], weights=weights)
    peak2 = np.average([r['peak']**2 for r in rows], weights=weights)
    return float(10*np.log10(peak2/mse))


def pooled_delta(candidate, baseline):
    c, b = pooled_psnr(candidate), pooled_psnr(baseline)
    return c-b if c is not None and b is not None else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--model", action="append", required=True, help="name=path")
    p.add_argument("--device", default="auto")
    p.add_argument("--weights", choices=["model", "ema"], default="model")
    p.add_argument("--output", required=True)
    p.add_argument("--block", type=int, default=16)
    p.add_argument("--crops", default="{}", help='JSON {"name": [y, x]} packed crop origins (192 px)')
    evaluate(p.parse_args())


def evaluate(args):
    """Run LOO scoring; an in-memory spec allows bounded evaluation without RAW copies."""
    from denoise_lab.inference import Denoiser
    from PIL import Image
    spec = args.spec if isinstance(args.spec, dict) else json.loads(Path(args.spec).read_text())
    root = Path(spec["root"])
    out = Path(args.output)
    (out/"crops").mkdir(parents=True, exist_ok=True)
    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        models[name] = Denoiser.load(path, device=args.device,
            **({"weights": args.weights} if getattr(args, "weights", "model") != "model" else {}))
    crops = json.loads(args.crops)
    b = args.block
    def load_frame(filename):
        packed, wb = load_packed(root/filename)
        size = getattr(args, 'packed_crop_size', None)
        if size:
            y = max(0, (packed.shape[1]-size)//2)
            x = max(0, (packed.shape[2]-size)//2)
            packed = packed[:, y:y+size, x:x+size].copy()
        return packed, wb
    def fmt(value):
        return "unavailable" if value is None else f"{value:.2f}"
    rows = []
    for group in spec["groups"]:
        files = group["files"]
        n = len(files)
        # Pass 1: block means per frame for brightness matching.
        bm, wb = [], None
        for f in files:
            a, wb = load_frame(f)
            bm.append(block_view(a, b).mean((-3, -1)))
        bm = np.stack(bm)
        ref_b = bm.mean(0)
        gains = np.ones((n, 4))
        ok = np.isfinite(ref_b).all(0) & (ref_b.min(0) > 1e-3) & (ref_b.max(0) < .9)
        for i in range(n):
            for c in range(4):
                r = bm[i, c][ok]/ref_b[c][ok]
                gains[i, c] = float(np.median(r))
        # Pass 2: sums of brightness-matched frames; keep test frames.
        S = SS = None
        test = {}
        for i, f in enumerate(files):
            a = load_frame(f)[0].astype(np.float64)/gains[i][:, None, None]
            S = a if S is None else S+a
            SS = a*a if SS is None else SS+a*a
            if i in group["test"]:
                test[i] = (a.astype(np.float32), load_frame(f)[0], gains[i])
        mean_all = S/n
        var_all = (SS-n*mean_all**2)/(n-1)
        tv_b = block_view(var_all, b).mean((-3, -1))
        mu_b = block_view(mean_all, b).mean((-3, -1))
        moving = np.zeros(mu_b.shape[1:], bool)
        for c in range(4):
            edges = np.quantile(mu_b[c], np.linspace(0, 1, 41))
            idx = np.clip(np.searchsorted(edges, mu_b[c], side="right")-1, 0, 39)
            expected = np.array([np.median(tv_b[c][idx == k]) if (idx == k).any() else np.inf for k in range(40)])
            moving |= tv_b[c] > 3*expected[idx]
        clipped = (mean_all > .9).any(0)
        static = upsample(~dilate(moving, 2), b, mean_all.shape[1:]) & ~clipped
        green = (mean_all[1]+mean_all[3])/2
        grad = np.hypot(*np.gradient(gauss(green[None], 2.)[0]))
        tex_thr = np.quantile(grad[static], .5) if static.any() else 0
        textured = static & (grad > tex_thr)
        flat = static & ~textured
        m = static[None].repeat(4, 0)
        peak = float(np.percentile(mean_all[m], 99.5))
        print(f"{group['name']}: n={n}, static {static.mean():.1%}, gains {gains[:, 1].min():.3f}-{gains[:, 1].max():.3f}", flush=True)
        for i, (x, raw, g) in test.items():
            target = (S-x.astype(np.float64))/(n-1)
            var_t = (SS-x.astype(np.float64)**2-(n-1)*target**2)/(n-2)/(n-1)
            vt = {k: float(var_t[:, msk].mean()) for k, msk in (("all", static), ("flat", flat), ("tex", textured))}
            err_in = (x-target)**2
            for name, den in models.items():
                pred = den.denoise_packed(raw)/g[:, None, None]
                err = (pred-target)**2
                row = {"group": group["name"], "frame": files[i], "model": name, "n": n,
                       "static_fraction": float(static.mean()), "peak": peak,
                       "static_samples": int(m.sum()), "weights": getattr(args, "weights", "model"),
                       "center_crop_packed": getattr(args, 'packed_crop_size', None)}
                for k, msk in (("all", static), ("flat", flat), ("tex", textured)):
                    mi = float(err_in[:, msk].mean())-vt[k]
                    mo = float(err[:, msk].mean())-vt[k]
                    row[f"mse_in_{k}"], row[f"mse_out_{k}"] = mi, mo
                    row[f"gain_db_{k}"] = 10*np.log10(mi/mo) if mi > 0 and mo > 0 else None
                row["psnr_out"] = 10*np.log10(peak**2/row["mse_out_all"]) if row["mse_out_all"] > 0 else None
                row["psnr_in"] = 10*np.log10(peak**2/row["mse_in_all"]) if row["mse_in_all"] > 0 else None
                rows.append(row)
                print(f"  {files[i]} {name:10s} PSNR {fmt(row['psnr_in'])} -> {fmt(row['psnr_out'])}  "
                      f"gain {fmt(row['gain_db_all'])} dB (flat {fmt(row['gain_db_flat'])}, tex {fmt(row['gain_db_tex'])})", flush=True)
                if i == group["test"][0]:
                    for cname, (cy, cx) in crops.items():
                        sl = (slice(None), slice(cy, cy+192), slice(cx, cx+192))
                        scale = .8/max(np.percentile((target[sl][1]+target[sl][3])/2, 99), 1e-4)
                        tiles = [render(v[sl], wb, scale) for v in (x, pred, target)]
                        sep = np.full((192, 4, 3), 255, np.uint8)
                        img = np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)
                        Image.fromarray(img).resize((img.shape[1]*2, img.shape[0]*2), Image.NEAREST).save(
                            out/"crops"/f"{group['name']}_{cname}_{name}.png")
    baseline = next(iter(models))
    reference = {(r['group'], r['frame']): r for r in rows if r['model'] == baseline}
    for row in rows:
        ref = reference[(row['group'], row['frame'])]['psnr_out']
        row['baseline_model'] = baseline
        row['psnr_delta_db'] = row['psnr_out']-ref if row['psnr_out'] is not None and ref is not None else None
    (out/"results.json").write_text(json.dumps(rows, indent=1))
    lines = ["| Group | Model | Frames | Static | PSNR in→out | Gain dB | Gain flat | Gain texture | Pooled corrected-MSE PSNR (sensitivity) | Pooled delta vs first model | Frame PSNR deltas vs first model |", "|"+"---|"*11]
    for gname in dict.fromkeys(r["group"] for r in rows):
        for mname in models:
            rs = [r for r in rows if r["group"] == gname and r["model"] == mname]
            mean = lambda k: float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else float("nan")
            lines.append(f"| {gname} | {mname} | {len(rs)} | {rs[0]['static_fraction']:.0%} | "
                         f"{mean('psnr_in'):.2f}→{mean('psnr_out'):.2f} | {mean('gain_db_all'):.2f} | "
                         f"{mean('gain_db_flat'):.2f} | {mean('gain_db_tex'):.2f} | "
                         f"{fmt(pooled_psnr(rs))} | {fmt(pooled_delta(rs, [r for r in rows if r['group'] == gname and r['model'] == baseline]))} | "
                         +", ".join(f"{r['frame']}: {fmt(r['psnr_delta_db'])}" for r in rs)+" |")
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    main()
