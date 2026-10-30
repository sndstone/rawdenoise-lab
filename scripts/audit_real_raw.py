#!/usr/bin/env python3
"""Real-RAW statistics for source images and our normalized predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rawpy

from denoise_lab.bayer import pack, unpack
from denoise_lab import rawstats as rs

BINS = [(-1., .002), (.002, .01), (.01, .05), (.05, 2.)]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_codes(path):
    with rawpy.imread(str(path)) as raw:
        desc = raw.color_desc.decode("ascii")
        colors = raw.raw_colors_visible
        phase = "".join(desc[int(c)] for c in colors[:2, :2].ravel())
        meta = raw.camera_white_level_per_channel
        info = {"black": [float(b) for b in raw.black_level_per_channel],
                "white": [float(w) for w in (meta if meta is not None else [raw.white_level]*4)],
                "phase": phase, "visible": list(raw.raw_image_visible.shape),
                "raw": list(raw.raw_image.shape), "margins": [raw.sizes.top_margin, raw.sizes.left_margin]}
        codes = raw.raw_image_visible.astype(np.float32)
        full = raw.raw_image.astype(np.float32)
        return codes, colors.copy(), info, full


def normalize(codes, colors, black, white):
    b, w = np.asarray(black, np.float32), np.asarray(white, np.float32)
    return (codes-b[colors])/(w[colors]-b[colors])


def detect_ob(codes, black, tol=12., min_run=8):
    """Leading/trailing rows/cols whose mean is within tol DN of black."""
    b = float(np.mean(black))
    rows, cols = codes.mean(1), codes.mean(0)
    def run(m, rev=False):
        idx = range(len(m)-1, -1, -1) if rev else range(len(m))
        n = 0
        for i in idx:
            if abs(m[i]-b) <= tol:
                n += 1
            else:
                break
        return n if n >= min_run else 0
    return {"top": run(rows), "bottom": run(rows, True), "left": run(cols), "right": run(cols, True)}


def per_bin(mu, values, fn=np.median):
    out = {}
    for lo, hi in BINS:
        sel = (mu >= lo) & (mu < hi)
        out[f"{lo:g}..{hi:g}"] = {"blocks": int(sel.sum()), "value": float(fn(values[sel])) if sel.any() else None}
    return out


def analyze_output(name, inp, out, nlf_map, region, gradient_energy, mu_blocks, nlf_fn):
    y0, y1, x0, x1 = region
    i, o = inp[:, y0:y1, x0:x1], out[:, y0:y1, x0:x1]
    r = i-o
    res = {"name": name, "finite": bool(np.isfinite(o).all())}
    mu, diff, z, valid = rs.block_mean_shift(i, o, block=32, noise_var=nlf_fn)
    m4 = mu.mean(0)[valid]
    res["block_mean_shift"] = {f"ch{c}": {"median_diff_by_signal": per_bin(m4, diff[c][valid]),
                                          "mean_diff_by_signal": per_bin(m4, diff[c][valid], np.mean),
                                          "median_z_by_signal": per_bin(m4, z[c][valid]),
                                          "frac_abs_z_gt3_by_signal": per_bin(m4, (np.abs(z[c][valid]) > 3).astype(float), np.mean)}
                               for c in range(4)}
    # Residual variance relative to the NLF bracket, by 16x16 block signal level.
    rb = rs._blocks(r, 16)
    rvar = rb.var(-1).mean(0).ravel()
    res["residual_var_over_nlf"] = {q: per_bin(mu_blocks, rvar/np.maximum(nlf_map[q], 1e-14)) for q in nlf_map}
    # Texture leakage: within signal bins, rank correlation of residual energy with
    # local structure energy of a smoothed input (independent of output).
    leak = {}
    for lo, hi in BINS:
        sel = (mu_blocks >= lo) & (mu_blocks < hi)
        if sel.sum() > 50:
            a = np.argsort(np.argsort(rvar[sel]))
            b = np.argsort(np.argsort(gradient_energy[sel]))
            leak[f"{lo:g}..{hi:g}"] = float(np.corrcoef(a, b)[0, 1])
    res["residual_texture_rank_corr"] = leak
    res["residual_lag1"] = rs.lag1_autocorr(r)
    res["output_rows"] = rs.line_structure(o, axis=-1)
    res["output_cols"] = rs.line_structure(o, axis=-2)
    res["residual_rows"] = rs.line_structure(r, axis=-1)["ratio"]
    res["residual_cols"] = rs.line_structure(r, axis=-2)["ratio"]
    return res, r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--ours", action="append", default=[], metavar="NAME=NORMALIZED_NPY")
    p.add_argument("--crop", action="append", default=[], metavar="NAME:X:Y:SIZE",
                   help="Sensor-pixel crop for residual PNGs")
    p.add_argument("--label", required=True)
    p.add_argument("--image-box", help="Sensor TOP:BOTTOM:LEFT:RIGHT of exposed image inside the visible array")
    p.add_argument("--ob-region", action="append", default=[], metavar="NAME:Y0:Y1:X0:X1",
                   help="Optical-black box in visible sensor coordinates (evidence documented in report)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    codes, colors, info, full = read_codes(args.source)
    if info["phase"] != "RGGB":
        raise ValueError("Audit expects an RGGB visible origin")
    black, white = info["black"], info["white"]
    scale = float(np.mean(white)-np.mean(black))
    report = {"label": args.label, "source": str(args.source), "source_sha256": sha256(args.source),
              "evaluation_only": True, "geometry": info,
              "caveats": ["Single frame: temporal noise, texture, PRNU, shading and fixed pattern are confounded.",
                          "NLF brackets are spatial diagnostics, not photon-transfer calibration."]}
    # Code distribution.
    report["codes"] = {f"ch{c}": {"below_black_frac": float((codes[colors == c] < black[c]).mean()),
                                  "zero_frac": float((codes[colors == c] == 0).mean()),
                                  "ge_white_frac": float((codes[colors == c] >= white[c]).mean()),
                                  "p0_01": float(np.percentile(codes[colors == c], .01))} for c in range(4)}
    ob = detect_ob(codes, black)
    report["ob_autodetect"] = ob
    if args.image_box:
        t, b_, l, r_ = [int(v) for v in args.image_box.split(":")]
        ob = {"top": t, "bottom": codes.shape[0]-b_, "left": l, "right": codes.shape[1]-r_}
    report["ob_in_visible"] = ob
    report["image_box_sensor"] = args.image_box
    # Optical black: within visible area (X2D-style) or raw margins (LibRaw-cropped).
    obstats = {}
    def colorplanes(arr, oy, ox):
        out = {}
        for c in range(4):
            sy, sx = [(0, 0), (0, 1), (1, 1), (1, 0)][c]
            yy, xx = (sy-oy) % 2, (sx-ox) % 2
            out[c] = arr[yy::2, xx::2]
        return out
    for spec in args.ob_region:
        name, ya, yb, xa, xb = spec.split(":")
        ya, yb, xa, xb = int(ya), int(yb), int(xa), int(xb)
        band = codes[ya:yb, xa:xb]
        obstats[name] = {f"ch{c}": rs.dark_region_stats(a, black[c], white[c]-black[c])
                         for c, a in colorplanes(band, ya, xa).items()}
        obstats[name]["sensor_box"] = [ya, yb, xa, xb]
    tm, lm = info["margins"]
    if lm >= 24:
        band = full[tm+64:tm+info["visible"][0]-64, 4:lm-4]
        obstats["raw_left_margin"] = {f"ch{c}": rs.dark_region_stats(a, black[c], white[c]-black[c])
                                      for c, a in colorplanes(band, tm+64, 4).items()}
    report["optical_black"] = obstats
    # Image region (exclude OB plus a small apron), in packed coordinates.
    y0, x0 = (ob["top"]+8)//2, (ob["left"]+8)//2
    hp, wp = codes.shape[0]//2, codes.shape[1]//2
    y1, x1 = hp-(ob["bottom"]+8)//2, wp-(ob["right"]+8)//2
    region = (y0, y1, x0, x1)
    report["analysis_region_packed"] = region
    src = pack(normalize(codes, colors, black, white)).astype(np.float32)
    del full
    sub = src[:, y0:y1, x0:x1]
    mu, var = rs.green_pair_blocks(sub, 16)
    nlf = {}
    rng = np.random.default_rng(0)
    for q in (.1, .25, .5):
        fit = rs.fit_nlf(mu, var, quantile=q)
        boots = []
        for _ in range(30):
            idx = rng.integers(0, len(mu), len(mu))
            b = rs.fit_nlf(mu[idx], var[idx], quantile=q)
            boots.append((b["K"], b["c"]))
        boots = np.asarray(boots)
        nlf[f"q{q:g}"] = {"K": fit["K"], "read_equiv_std": float(np.sqrt(fit["c"])),
                          "K_ci90": np.percentile(boots[:, 0], [5, 95]).tolist(),
                          "read_equiv_std_ci90": np.sqrt(np.percentile(boots[:, 1], [5, 95])).tolist(),
                          "points": fit["points"]}
    report["green_pair_nlf"] = nlf
    report["nlf_note"] = ("var((G1-G2)/sqrt2) per 16x16 packed block vs green signal; q=quantile across blocks per "
                          "signal bin. Low q under-reads (sampling scatter), high q over-reads (texture). "
                          "Intercept includes read, row/column differences and dark fixed pattern.")
    # Per-block quantities shared by outputs.
    rb = rs._blocks(sub, 16)
    mu_blocks = rb.mean(-1).mean(0).ravel()
    nlf_map = {k: np.maximum(v["K"]*np.maximum(mu_blocks, 0)+v["read_equiv_std"]**2, 1e-14) for k, v in nlf.items()}
    med = nlf["q0.5"]
    def nlf_fn(m):
        return med["K"]*np.maximum(m, 0)+med["read_equiv_std"]**2
    smooth = rs.box_mean(sub.mean(0, keepdims=True), 5)[0]
    gy, gx = np.diff(smooth, axis=0, append=smooth[-1:]), np.diff(smooth, axis=1, append=smooth[:, -1:])
    gradient_energy = rs._blocks((gx*gx+gy*gy)[None], 16)[0].mean(-1).ravel()
    sigma_px = np.sqrt(np.maximum(nlf["q0.5"]["K"]*np.maximum(rs.box_mean(sub, 5), 0)+nlf["q0.5"]["read_equiv_std"]**2, 1e-14)).astype(np.float32)
    pos_in, neg_in = rs.impulses(sub, sigma_px, 8.)
    report["input_impulses_8sigma"] = {"positive_frac": float(pos_in.mean()), "negative_frac": float(neg_in.mean()),
                                       "per_channel_positive": pos_in.mean((1, 2)).tolist()}
    report["input_lines"] = {"rows": rs.line_structure(sub, -1), "cols": rs.line_structure(sub, -2),
                             "green_pair_rows": rs.line_structure(((sub[1]-sub[3])/np.sqrt(2))[None], -1)["ratio"],
                             "green_pair_cols": rs.line_structure(((sub[1]-sub[3])/np.sqrt(2))[None], -2)["ratio"]}
    outputs = []
    for spec in args.ours:
        name, path = spec.split("=", 1)
        mosaic = np.load(path, mmap_mode="r", allow_pickle=False)
        if mosaic.shape != tuple(info["visible"]):
            raise ValueError(f"{name}: geometry {mosaic.shape} != {info['visible']}")
        outputs.append((name, "ours", path, pack(np.asarray(mosaic, np.float32))))
    results, crops = [], {}
    for name, kind, path, out in outputs:
        res, r = analyze_output(name, src, out, nlf_map, region, gradient_energy, mu_blocks, nlf_fn)
        res["kind"], res["path"] = kind, path
        o = out[:, y0:y1, x0:x1]
        pos_out, _ = rs.impulses(o, sigma_px, 8.)
        res["impulses"] = {"input_positive_surviving_frac": float(pos_out[pos_in].mean()) if pos_in.any() else None,
                           "output_positive_frac": float(pos_out.mean())}
        results.append(res)
        for spec in args.crop:
            cname, cx, cy, size = spec.split(":")
            cx, cy, size = int(cx)//2, int(cy)//2, int(size)//2
            crops.setdefault(cname, {})[name] = (out[:, cy:cy+size, cx:cx+size].copy(),
                                                 src[:, cy:cy+size, cx:cx+size]-out[:, cy:cy+size, cx:cx+size])
        del out, r
        print("ANALYZED", name, round(time.monotonic()-started, 1), flush=True)
    report["outputs"] = results
    args.output.mkdir(parents=True, exist_ok=True)
    # Residual PNGs: green residual, +-4 sigma of the median-q NLF read term, same scale for all outputs.
    if crops:
        from PIL import Image
        s = 4*nlf["q0.5"]["read_equiv_std"]+1e-9
        for cname, items in crops.items():
            tiles = []
            for name, (_, resid) in items.items():
                g = (resid[1]+resid[3])/2
                tiles.append(np.clip((g/s+1)*127.5, 0, 255).astype(np.uint8))
            Image.fromarray(np.concatenate(tiles, axis=1)).save(args.output/f"{args.label}_{cname}_green_residual.png")
            report.setdefault("residual_pngs", {})[cname] = {"file": f"{args.label}_{cname}_green_residual.png",
                                                             "order": list(items), "scale_pm": s,
                                                             "note": "input minus output, mean of G1/G2, grey=0"}
    report["seconds"] = time.monotonic()-started
    (args.output/f"{args.label}.json").write_text(json.dumps(report, indent=1, default=float)+"\n")
    print("SAVED", args.output/f"{args.label}.json")


if __name__ == "__main__":
    main()
