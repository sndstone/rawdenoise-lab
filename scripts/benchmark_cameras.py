#!/usr/bin/env python3
"""Cross-camera benchmark of one checkpoint on owner RAWs. Source access is READ ONLY.

Three steps:
  select  Pick a stratified set of real noisy photos and clean base-ISO photos per
          camera from a per-file EXIF scan. Real photos skip every capture day in
          data/PROVENANCE.csv; clean sources skip its train-split days. Copy read-only.
  run     Denoise every real photo and score fixed ROIs with no-reference proxies.
          Build clean-reference synthetic pairs from 2x2-binned base-ISO crops with
          each camera's own blind-estimated noise per ISO band. Save visual crops.
  report  Write summary JSON/Markdown tables and an HTML contact sheet.

No-reference proxies are scene-confounded diagnostics, not quality measurements. The
synthetic test uses white Poisson-Gaussian noise. It cannot reproduce compression or
in-camera RAW filtering, which the real-photo proxies are meant to expose.
"""
import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

BANDS = [("iso400-640", 400, 640), ("iso800-1250", 800, 1250), ("iso1600-2500", 1600, 2500),
         ("iso3200-5000", 3200, 5000), ("iso6400-25600", 6400, 25600)]
BASE_ISO = {"ILCE-5100": 100, "DSC-RX100M5": 125, "ILCE-7M3": 100, "X1D II 50C": 100}
SHORT = {"ILCE-5100": "a5100", "DSC-RX100M5": "rx100m5", "ILCE-7M3": "a7m3", "X1D II 50C": "x1d2"}
LADDER = (800, 3200, 12800)
SNR_BINS = [(0, 3), (3, 10), (10, 30), (30, 10**9)]
# Training prior of r4_plain (train_config.yaml): broad sources plus the ISO-coupled X1D II source.
PRIOR = {"K": (1e-5, 2e-3), "read_std": (3e-5, 6e-3)}


def band_of(iso):
    for name, lo, hi in BANDS:
        if lo <= iso <= hi:
            return name
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(rows, n, rng):
    """Round-robin over capture days so one scene cannot dominate a stratum."""
    days = defaultdict(list)
    for r in rows:
        days[r["day"]].append(r)
    order = sorted(days)
    rng.shuffle(order)
    for d in order:
        rng.shuffle(days[d])
    out = []
    while len(out) < n and any(days.values()):
        for d in order:
            if days[d] and len(out) < n:
                out.append(days[d].pop())
    return out


def cmd_select(args):
    rng, clean_rng = random.Random(args.seed), random.Random(args.seed+1)
    scan = json.loads(Path(args.scan).read_text())
    excluded, trained = set(), set()
    import csv
    with open(args.provenance) as f:
        for row in csv.DictReader(f):
            excluded.add((row["camera"], row["capture_date"][:10]))
            if row["split"] == "train":
                trained.add((row["camera"], row["capture_date"][:10]))
    rows = []
    for r in scan:
        if r.get("model") not in BASE_ISO or r["ext"] == ".dng" or "error" in r or not r.get("date"):
            continue
        try:
            iso = int(str(r["iso"]).split()[0])
        except (TypeError, ValueError):
            continue
        day = r["date"][:10]
        if (r["model"], day) in trained:
            continue
        rows.append({**r, "iso_int": iso, "day": day, "dev_day": (r["model"], day) in excluded})
    selection = []
    for camera in BASE_ISO:
        cam = [r for r in rows if r["model"] == camera and not r["dev_day"]]
        used_days = set()
        for name, _, _ in BANDS:
            got = pick([r for r in cam if band_of(r["iso_int"]) == name], args.per_band, rng)
            used_days |= {r["day"] for r in got}
            selection += [{**r, "set": "real", "band": name} for r in got]
            print(f"{camera:12s} {name:14s} {len(got)}")
        # Clean base-ISO sources: days not used by the real set; validation days are allowed.
        base = [r for r in rows if r["model"] == camera and r["iso_int"] == BASE_ISO[camera]
                and r["day"] not in used_days]
        got = pick(base, args.clean_per_camera, clean_rng)
        selection += [{**r, "set": "clean", "band": f"iso{BASE_ISO[camera]}"} for r in got]
        print(f"{camera:12s} {'clean base':14s} {len(got)}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, r in enumerate(selection, 1):
        src = Path(args.source_root)/r["path"]
        tmp = out/".partial"
        with open(src, "rb") as fin, open(tmp, "wb") as fout:  # source opened read-only
            shutil.copyfileobj(fin, fout, 1 << 22)
        digest = sha256(tmp)
        if tmp.stat().st_size != r["bytes"]:
            raise ValueError(f"Incomplete copy: {src}")
        dest = out/SHORT[r["model"]]/f"{r['set']}_{r['band']}_{digest[:12]}{r['ext']}"
        dest.parent.mkdir(exist_ok=True)
        tmp.replace(dest)
        manifest.append({"file": str(dest.relative_to(out)), "source": r["path"], "sha256": digest,
                         "camera": r["model"], "iso": r["iso_int"], "set": r["set"], "band": r["band"],
                         "capture_date": r["date"], "bytes": r["bytes"], "provenance_val_day": r["dev_day"]})
        print(f"[{i}/{len(selection)}] {r['path']}", flush=True)
    (out/"manifest.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": args.seed, "source_access": "READ_ONLY",
        "excluded_capture_days": ("real set: every (camera, day) in data/PROVENANCE.csv; "
                                  "clean set: train-split days only (validation days allowed)"),
        "note": "Evaluation only. Exclude these capture days from any future training.",
        "files": manifest}, indent=1))


# ---------------------------------------------------------------- metrics


from denoise_lab.noref import gauss


def hf(a):
    from denoise_lab.rawstats import box_mean
    return np.asarray(a, np.float64)-box_mean(a, 3)


def blocks_mask(score, block, frac):
    """Boolean pixel mask of the lowest-score `frac` of block x block blocks."""
    h, w = (score.shape[0]//block)*block, (score.shape[1]//block)*block
    b = score[:h, :w].reshape(h//block, block, w//block, block).mean((1, 3))
    keep = b <= np.quantile(b, frac)
    mask = np.zeros(score.shape, bool)
    mask[:h, :w] = np.repeat(np.repeat(keep, block, 0), block, 1)
    return mask


def roi_metrics(inp, out, nlf):
    """No-reference proxies on one packed ROI [4,H,W]; nlf = (K, read_std) of the input."""
    from denoise_lab.rawstats import robust_std, impulses
    inp, out = inp.astype(np.float64), out.astype(np.float64)
    res = inp-out
    green = (inp[1]+inp[3])/2
    smooth = gauss(inp, 2)
    grad = np.hypot(*np.gradient(gauss(green, 2.)))
    flat = blocks_mask(grad, 16, .25)
    tex = ~blocks_mask(grad, 16, .5)
    hin, hout = hf(inp), hf(out)
    noise_left = float(np.median([robust_std(hout[c][flat])/max(robust_std(hin[c][flat]), 1e-12)
                                  for c in range(4)]))
    expected = nlf[0]*np.maximum(smooth, 0)+nlf[1]**2
    res_over_nlf = float(np.median(res[:, flat]**2)/np.median(expected[:, flat])/0.4549)  # chi2(1) median
    # Removed detail shows up as residual energy beyond the noise level where the scene is textured.
    z = res**2/np.maximum(expected, 1e-14)
    flat_z = np.median(z[:, flat])
    detail_excess = float(np.median(z[:, tex])/flat_z) if flat_z > 0 else None
    def chroma(x):
        g = (x[1]+x[3])/2
        return np.stack([x[0]-g, x[2]-g])
    def band(x):
        return gauss(x, 2)-gauss(x, 8)
    # Chroma blotch in noise units: band-passed R-G/B-G spread in flat areas divided by the spread
    # the same filter gives on white noise at the input's NLF.
    white = np.random.default_rng(0).normal(size=inp.shape)*np.sqrt(expected)
    cin, cout, cw = band(chroma(inp)), band(chroma(out)), band(chroma(white))
    unit = [max(robust_std(cw[c][flat]), 1e-12) for c in range(2)]
    chroma_in = float(np.mean([robust_std(cin[c][flat])/unit[c] for c in range(2)]))
    chroma_out = float(np.mean([robust_std(cout[c][flat])/unit[c] for c in range(2)]))
    d = (inp[1]-inp[3])/np.sqrt(2)
    d = d-gauss(d[None], 4)[0]
    dh = np.where(flat[:, 1:] & flat[:, :-1])
    noise_corr = float(np.corrcoef(d[:, 1:][dh], d[:, :-1][dh])[0, 1]) if len(dh[0]) > 100 else None
    sigma = np.sqrt(expected)
    flags_in = np.logical_or(*impulses(inp, sigma)) & flat  # isolated outliers in flat areas only
    survival = None
    if flags_in.any():
        survival = float(np.logical_or(*impulses(out, sigma))[flags_in].mean())
    return {"noise_left": noise_left, "residual_over_nlf": res_over_nlf,
            "detail_excess": detail_excess,
            "chroma_in_noise_units": chroma_in, "chroma_out_noise_units": chroma_out,
            "input_noise_lag1": noise_corr,
            "snr": float(np.median(green)/np.sqrt(nlf[0]*max(np.median(green), 0)+nlf[1]**2)),
            "defect_survival": survival, "defects_flagged": int(flags_in.sum()),
            "median_green": float(np.median(green))}


def choose_rois(packed, size, quantiles=(.15, .4, .65, .9)):
    """Fixed ROIs from the INPUT only: valid tiles at chosen intensity quantiles."""
    _, h, w = packed.shape
    cand = []
    for y in range(0, h-size+1, size//2):
        for x in range(0, w-size+1, size//2):
            t = packed[:, y:y+size, x:x+size]
            g = (t[1]+t[3])/2
            if (t >= .98).mean() > .001 or np.median(g) < .0005:
                continue
            cand.append((float(np.median(g)), y, x))
    if not cand:
        return []
    cand.sort()
    picks, seen = [], set()
    for q in quantiles:
        m, y, x = cand[min(int(q*len(cand)), len(cand)-1)]
        if (y, x) not in seen:
            seen.add((y, x))
            picks.append((y, x))
    return picks


def denoise_rois(den, packed, size, context, *, limit=None):
    """Yield input-selected coordinates and paired crops, with shared inference context."""
    rois = choose_rois(packed, size)
    for y, x in rois[:limit]:
        y0, x0 = max(0, y-context), max(0, x-context)
        y1, x1 = min(packed.shape[1], y+size+context), min(packed.shape[2], x+size+context)
        pred = den.denoise_packed(np.ascontiguousarray(packed[:, y0:y1, x0:x1]))
        yield y, x, packed[:, y:y+size, x:x+size], pred[:, y-y0:y-y0+size, x-x0:x-x0+size]


def render(packed, wb, scale):
    """Half-resolution RGB (R, mean G, B) with camera WB, fixed exposure and sRGB gamma."""
    rgb = np.stack([packed[0]*wb[0], (packed[1]+packed[3])/2*wb[1], packed[2]*wb[2]], -1)*scale
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*np.power(np.maximum(rgb, .0031308), 1/2.4)-.055)
    return (rgb*255+.5).astype(np.uint8)


def load_packed(path):
    import rawpy
    from denoise_lab.bayer import normalize_raw, pack
    with rawpy.imread(str(path)) as raw:
        packed = pack(normalize_raw(raw)).astype(np.float32)
        wb = np.asarray(raw.camera_whitebalance[:3], np.float64)
    wb = wb/wb[1] if wb[1] > 0 else np.ones(3)
    return packed, wb


def cmd_run(args):
    from PIL import Image
    from denoise_lab.bootstrap import add_bootstrap_noise
    from denoise_lab.clean_targets import bin_mosaic
    from denoise_lab.bayer import pack, unpack
    from denoise_lab.inference import Denoiser
    from denoise_lab.metrics import psnr, ssim
    from denoise_lab.rawstats import estimate_noise_parameters
    import torch
    root = Path(args.bench)
    manifest = json.loads((root/"manifest.json").read_text())["files"]
    out = Path(args.output)
    (out/"crops").mkdir(parents=True, exist_ok=True)
    den = Denoiser.load(args.model, device=args.device)
    results_path = out/"real.json"
    real = json.loads(results_path.read_text()) if results_path.exists() else {}
    for i, f in enumerate([m for m in manifest if m["set"] == "real"], 1):
        if f["sha256"] in real:
            continue
        t0 = time.monotonic()
        packed, wb = load_packed(root/f["file"])
        est = estimate_noise_parameters(packed)
        nlf = (est["K"], est["read_std"])
        rois = choose_rois(packed, args.roi)
        if not rois:
            real[f["sha256"]] = {**f, "error": "no valid ROI"}
            continue
        from denoise_lab.noref import crop_metrics, bootstrap_crops
        metrics, crops, noref = [], [], []
        for j, (y, x, n, o) in enumerate(denoise_rois(den, packed, args.roi, args.context)):
            noref.append(crop_metrics(n, o))
            m = roi_metrics(n, o, nlf)
            metrics.append({"roi_yx": [y, x], **m})
            # Visual crop: centre 192x192 packed, both sides with the same exposure.
            c = (args.roi-192)//2
            ni, oi = n[:, c:c+192, c:c+192], o[:, c:c+192, c:c+192]
            g = np.percentile((oi[1]+oi[3])/2, 99)
            scale = .8/max(g, 1e-4)
            pair = np.concatenate([render(ni, wb, scale), np.full((192, 4, 3), 255, np.uint8),
                                   render(oi, wb, scale)], 1)
            name = f"{SHORT[f['camera']]}_{f['band']}_{f['sha256'][:12]}_r{j}.png"
            Image.fromarray(pair).resize((pair.shape[1]*2, pair.shape[0]*2), Image.NEAREST).save(out/"crops"/name)
            crops.append(name)
        real[f["sha256"]] = {**f, "nlf": {"K": nlf[0], "read_std": nlf[1]}, "rois": metrics, "crops": crops,
                             "noref_crops": noref, "noref": bootstrap_crops(noref),
                             "shape": list(packed.shape), "seconds": round(time.monotonic()-t0, 1)}
        results_path.write_text(json.dumps(real, indent=1))
        print(f"[real {i}] {f['camera']} ISO {f['iso']} {time.monotonic()-t0:.1f}s", flush=True)

    # Synthetic clean-reference pairs: each camera's own median blind NLF per band.
    nlf_band = defaultdict(list)
    for r in real.values():
        if "nlf" in r:
            nlf_band[(r["camera"], r["band"])].append((r["nlf"]["K"], r["nlf"]["read_std"], r["iso"]))
    synth = []
    rng = np.random.default_rng(args.seed)
    for f in [m for m in manifest if m["set"] == "clean"]:
        packed, _ = load_packed(root/f["file"])
        clean_full = pack(bin_mosaic(unpack(packed), 2))
        rois = choose_rois(clean_full, args.synth_crop, quantiles=(.3, .5, .7, .9))
        for (y, x) in rois:
            clean = clean_full[:, y:y+args.synth_crop, x:x+args.synth_crop]
            own = [(band, *(float(np.median([v[k] for v in vals])) for k in range(3)))
                   for (cam, band), vals in sorted(nlf_band.items()) if cam == f["camera"]]
            # Common ladder: identical noise for every camera (X1D II ISO-coupled anchor), so
            # differences reflect image content and file format, not noise level.
            ladder = [(f"ladder_iso{iso}", 6.7e-4*iso/6400, 6e-4*iso/6400, iso) for iso in LADDER]
            for band, K, read, iso in own+ladder:
                cam = f["camera"]
                e = min(1., 100/iso if band.startswith("ladder") else BASE_ISO[cam]/iso)
                noisy, _ = add_bootstrap_noise(unpack(clean*e), rng, fixed_parameters={
                    "K": K, "read_std": read, "row_std": 0.})
                noisy = pack(noisy)
                pred = den.denoise_packed(np.ascontiguousarray(noisy))
                t = lambda a: torch.from_numpy(np.ascontiguousarray(a/e, dtype=np.float32))[None]
                synth.append({"camera": cam, "band": band, "iso": iso, "K": K, "read_std": read,
                              "source": f["sha256"], "roi_yx": [y, x],
                              "psnr_in": psnr(t(noisy), t(clean*e)), "psnr_out": psnr(t(pred), t(clean*e)),
                              "ssim_in": ssim(t(noisy), t(clean*e)), "ssim_out": ssim(t(pred), t(clean*e))})
        print(f"[synth] {f['camera']} {len(synth)} pairs", flush=True)
    (out/"synthetic.json").write_text(json.dumps(synth, indent=1))
    (out/"run_info.json").write_text(json.dumps({"model": str(args.model), "backend": den.description,
                                                 "roi": args.roi, "context": args.context,
                                                 "synth_crop": args.synth_crop, "seed": args.seed}, indent=1))


# ---------------------------------------------------------------- report


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def fmt(x, p=3):
    return "–" if x is None else f"{x:.{p}g}"


def cmd_report(args):
    out = Path(args.output)
    real = json.loads((out/"real.json").read_text())
    synth = json.loads((out/"synthetic.json").read_text())
    cams = list(BASE_ISO)
    keys = ["noise_left", "chroma_in_noise_units", "chroma_out_noise_units", "detail_excess",
            "residual_over_nlf", "input_noise_lag1", "defect_survival", "snr"]
    synth_row = lambda ss: {"synth_pairs": len(ss), "psnr_in": med([s["psnr_in"] for s in ss]),
                            "psnr_out": med([s["psnr_out"] for s in ss]),
                            "psnr_gain": med([s["psnr_out"]-s["psnr_in"] for s in ss]),
                            "ssim_out": med([s["ssim_out"] for s in ss])}
    by_band, by_snr, ladder = [], [], []
    for cam in cams:
        for name, _, _ in BANDS:
            rs = [r for r in real.values() if r["camera"] == cam and r["band"] == name and "rois" in r]
            if not rs:
                continue
            rois = [m for r in rs for m in r["rois"]]
            Ks = [r["nlf"]["K"] for r in rs]
            reads = [r["nlf"]["read_std"] for r in rs]
            by_band.append({"camera": cam, "band": name, "photos": len(rs), "rois": len(rois),
                            **{k: med([m[k] for m in rois]) for k in keys},
                            "K": med(Ks), "read_std": med(reads),
                            "outside_prior": sum(not (PRIOR["K"][0] <= k <= PRIOR["K"][1]) or
                                                 not (PRIOR["read_std"][0] <= r <= PRIOR["read_std"][1])
                                                 for k, r in zip(Ks, reads)),
                            **synth_row([s for s in synth if s["camera"] == cam and s["band"] == name])})
        for lo, hi in SNR_BINS:
            rois = [m for r in real.values() if r["camera"] == cam and "rois" in r
                    for m in r["rois"] if lo <= m["snr"] < hi]
            if rois:
                by_snr.append({"camera": cam, "snr_bin": f"{lo}-{hi}", "rois": len(rois),
                               **{k: med([m[k] for m in rois]) for k in keys}})
        for iso in LADDER:
            ss = [s for s in synth if s["camera"] == cam and s["band"] == f"ladder_iso{iso}"]
            if ss:
                ladder.append({"camera": cam, "noise": f"ISO {iso} (X1D II level)", **synth_row(ss)})
    (out/"summary.json").write_text(json.dumps({"by_band": by_band, "by_snr": by_snr, "ladder": ladder}, indent=1))
    lines = ["## Real photos by ISO band, plus each camera's own noise on clean crops", "",
             "| Camera | ISO band | Photos | SNR | Noise left | Chroma in (noise units) | Chroma out (noise units) | "
             "Detail removed | Residual/NLF | Hot px surviving | K | read σ | Outside prior | Synth PSNR in→out | SSIM out |",
             "|"+"---|"*15]
    for r in by_band:
        lines.append(f"| {r['camera']} | {r['band']} | {r['photos']} | {fmt(r['snr'])} | {fmt(r['noise_left'])} | "
                     f"{fmt(r['chroma_in_noise_units'])} | {fmt(r['chroma_out_noise_units'])} | {fmt(r['detail_excess'])} | "
                     f"{fmt(r['residual_over_nlf'])} | {fmt(r['defect_survival'])} | {fmt(r['K'])} | {fmt(r['read_std'])} | "
                     f"{r['outside_prior']}/{r['photos']} | {fmt(r['psnr_in'], 4)}→{fmt(r['psnr_out'], 4)} | {fmt(r['ssim_out'])} |")
    lines += ["", "## Real-photo ROIs grouped by signal-to-noise ratio", "",
              "| Camera | ROI SNR | ROIs | Noise left | Chroma in | Chroma out | Detail removed | Residual/NLF | Hot px surviving |",
              "|"+"---|"*9]
    for r in by_snr:
        lines.append(f"| {r['camera']} | {r['snr_bin']} | {r['rois']} | {fmt(r['noise_left'])} | {fmt(r['chroma_in_noise_units'])} | "
                     f"{fmt(r['chroma_out_noise_units'])} | {fmt(r['detail_excess'])} | {fmt(r['residual_over_nlf'])} | "
                     f"{fmt(r['defect_survival'])} |")
    lines += ["", "## Identical synthetic noise on every camera's clean crops", "",
              "| Camera | Noise | Pairs | PSNR in→out | Gain dB | SSIM out |", "|"+"---|"*6]
    for r in ladder:
        lines.append(f"| {r['camera']} | {r['noise']} | {r['synth_pairs']} | {fmt(r['psnr_in'], 4)}→{fmt(r['psnr_out'], 4)} | "
                     f"{fmt(r['psnr_gain'])} | {fmt(r['ssim_out'])} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    parts = ["<!doctype html><meta charset=utf-8><title>Camera benchmark</title><style>"
             "body{font:13px system-ui;margin:20px;background:#111;color:#ddd}img{margin:2px;image-rendering:pixelated}"
             "h2{margin-top:28px}table{border-collapse:collapse}td,th{border:1px solid #444;padding:3px 6px}</style>",
             "<h1>Cross-camera benchmark</h1><p>Each crop: noisy input (left), denoised (right), "
             "same WB and exposure, half-resolution R/G/B, 2× nearest.</p>"]
    for cam in cams:
        parts.append(f"<h2>{html.escape(cam)}</h2>")
        for name, _, _ in BANDS:
            rs = [r for r in real.values() if r["camera"] == cam and r["band"] == name and "crops" in r]
            if not rs:
                continue
            parts.append(f"<h3>{name}</h3><div>")
            for r in rs:
                for c in r["crops"][:2]:
                    parts.append(f"<img src='crops/{c}' width=388 title='ISO {r['iso']} {html.escape(r['source'])}'>")
            parts.append("</div>")
    (out/"report.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--scan", required=True, help="Per-file EXIF scan JSON (path, model, iso, date, ext, bytes)")
    s.add_argument("--source-root", required=True, help="Read-only root the scan paths are relative to")
    s.add_argument("--provenance", default="data/PROVENANCE.csv")
    s.add_argument("--output", default="data/camera_bench")
    s.add_argument("--per-band", type=int, default=6)
    s.add_argument("--clean-per-camera", type=int, default=4)
    s.add_argument("--seed", type=int, default=20261031)
    r = sub.add_parser("run")
    r.add_argument("--bench", default="data/camera_bench")
    r.add_argument("--model", required=True)
    r.add_argument("--device", default="auto")
    r.add_argument("--output", default="runs/camera_bench")
    r.add_argument("--roi", type=int, default=512)
    r.add_argument("--context", type=int, default=64)
    r.add_argument("--synth-crop", type=int, default=256)
    r.add_argument("--seed", type=int, default=20261031)
    q = sub.add_parser("report")
    q.add_argument("--output", default="runs/camera_bench")
    args = p.parse_args()
    {"select": cmd_select, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
