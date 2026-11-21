#!/usr/bin/env python3
"""Per-camera noise profiles from dark-frame sets, plus interim shot-noise gains.

Dark frames give, per (camera, shutter, ISO): read-noise std and Tukey-lambda shape (rows/columns
removed), temporal row/column offset std, fixed row/column pattern std, black offset, code step,
lag-1 horizontal/vertical correlation and hot-pixel density. All values are in normalised units
((DN - black)/(white - black)).

The shot-noise gain K needs flat fields, which do not exist yet. K here comes from a per-camera
model passed in the spec (a slope per ISO, an optional cap ISO and explicit overrides), estimated
from tripod bursts and blind real-photo fits. Every profile records `K_source` so flat-field
results can replace it later.

Spec JSON: {"cameras": {name: {"dir": ..., "shutter": optional filter, "default_shutter": optional,
             "K_per_iso": slope,
             "K_cap_iso": optional, "K_override": {iso: K}, "quantization": optional}}}
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rawpy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoise_lab.bayer import normalize_raw, pack  # noqa: E402
from denoise_lab.rawstats import robust_std  # noqa: E402
from denoise_lab.physics_noise import fit_physics  # noqa: E402
from calibrate_noise import ppcc_shape  # noqa: E402


def load(path, crop_size=None):
    with rawpy.imread(str(path)) as raw:
        codes = raw.raw_image_visible
        if crop_size:
            # Central, CFA-aligned crop bounds memory without decimating pixels:
            # decimation would destroy lag-1 and quantisation measurements.
            h, w = codes.shape
            y, x = max(0, (h-crop_size)//4*2), max(0, (w-crop_size)//4*2)
            codes = codes[y:y+crop_size, x:x+crop_size]
        u = np.unique(codes[::3, ::3])
        step = int(np.min(np.diff(u))) if len(u) > 1 else 1
        black = float(raw.black_level_per_channel[0])
        white = float(raw.white_level)
        if crop_size:
            from denoise_lab.bayer import unify
            colors = raw.raw_colors_visible[y:y+crop_size, x:x+crop_size]
            desc = raw.color_desc.decode("ascii")
            phase = "".join(desc[int(c)] for c in colors[:2, :2].ravel())
            levels = np.asarray(raw.black_level_per_channel)[colors]
            whites = raw.camera_white_level_per_channel or [white]*4
            whites = np.asarray(whites)[colors]
            whites = np.where(whites > levels, whites, white)
            mosaic = unify((codes.astype(np.float64)-levels)/(whites-levels), phase)
            return pack(mosaic), step, black, white
        return pack(normalize_raw(raw)).astype(np.float64), step, black, white


def lag1(d):
    g = d - d.mean()
    v = (g*g).mean()
    if v == 0:
        return 0., 0.
    return float((g[:, 1:]*g[:, :-1]).mean()/v), float((g[1:]*g[:-1]).mean()/v)


def profile(files, crop_size=None):
    if len(files) < 2:
        raise ValueError("Calibration requires at least two dark frames per shutter/ISO")
    frames, step, black, white = [], 1, 0., 1.
    for f in files:
        a, step, black, white = load(f, crop_size)
        frames.append(a)
    st = np.stack(frames)
    n = len(st)
    reads, rows, cols, h, v = [], [], [], [], []
    for a, b in zip(st[0::2], st[1::2]):
        d = (a-b)/np.sqrt(2)
        r = d.mean(-1, keepdims=True)
        c = d.mean(-2, keepdims=True)
        pix = d-r-c
        reads.append(np.mean([robust_std(pix[k][::2, ::2]) for k in range(4)]))
        rows.append(float(np.std(r)))
        cols.append(float(np.std(c)))
        center = pix[1][200:-200, 200:-200] if min(pix.shape[-2:]) > 404 else pix[1]
        hh, vv = lag1(center)
        h.append(hh)
        v.append(vv)
    mean = st.mean(0)
    centered = st-mean[None]
    centered -= centered.mean(-1, keepdims=True)
    sample = centered[:, :, ::7, ::7].ravel()*np.sqrt(n/(n-1))
    shape, ppcc = ppcc_shape(sample)
    read = float(np.mean(reads))
    med = np.median(st, 0) if n >= 3 else mean
    hot = float(((med-np.median(med)) > 8*read).mean())
    return {"read_std": read, "tukey_lambda": float(shape), "ppcc": float(ppcc),
            "row_std": float(np.mean(rows)), "col_std": float(np.mean(cols)),
            "row_fixed_std": float(np.std(mean.mean(-1))), "col_fixed_std": float(np.std(mean.mean(-2))),
            "black_offset": float(np.median(mean)), "lag1_h": float(np.mean(h)), "lag1_v": float(np.mean(v)),
            "hot_density": hot, "code_step": step/(white-black), "black_dn": black, "white_dn": white,
            "frames": n, "physics": fit_physics(st, step, black, white)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--crop-size", type=int, help="Even central sensor crop; default uses full frames")
    args = p.parse_args()
    if args.crop_size is not None and (args.crop_size < 16 or args.crop_size % 2):
        p.error("--crop-size must be even and >=16")
    spec = json.loads(Path(args.spec).read_text())
    out = {"units": "normalized", "source": "dark frames 2026-11-01; K interim (no flats)", "cameras": {}}
    for name, cam in spec["cameras"].items():
        profiles = {}
        shutters = {}
        if cam.get("dir"):
            root = Path(cam["dir"])
            man = json.loads((root/"manifest.json").read_text())
            groups = {}
            for r in man:
                if cam.get("shutter") and r.get("shutter") != cam["shutter"]:
                    continue
                groups.setdefault((r.get("shutter", "unspecified"), int(r["iso"])), []).append(root/r["file"])
            for (shutter, iso), files in sorted(groups.items()):
                prof = profile(sorted(files), args.crop_size)
                shutters.setdefault(shutter, {"iso": {}})["iso"][iso] = prof
                prof["shutter"] = shutter
                prof["crop_size"] = args.crop_size
                print(f"{name} ISO {iso}: read {prof['read_std']:.3e} lambda {prof['tukey_lambda']:+.3f} "
                      f"row {prof['row_std']:.2e} col {prof['col_std']:.2e} lag1 h {prof['lag1_h']:+.3f} "
                      f"hot {prof['hot_density']:.1e} n={prof['frames']}", flush=True)
            # Never form a default ISO series by mixing shutter modes.
            default = cam.get("shutter", cam.get("default_shutter", next(iter(shutters), None)))
            if default is not None:
                profiles = dict(shutters[default]["iso"])
        for iso, values in cam.get("manual_profiles", {}).items():
            profiles[int(iso)] = dict(values)
        cap = cam.get("K_cap_iso")
        all_profiles = list(profiles.items())+[(iso, prof) for mode in shutters.values() for iso, prof in mode["iso"].items()]
        for iso, prof in all_profiles:
            k_iso = min(iso, cap) if cap else iso
            k_iso = max(k_iso, cam.get("K_floor_iso", 0))
            prof["K"] = float(cam.get("K_override", {}).get(str(iso), cam["K_per_iso"]*k_iso))
            prof["K_source"] = cam["K_source"]
        out["cameras"][name] = {"iso": profiles, **{k: cam[k] for k in ("quantization", "processed_isos", "note")
                                                   if k in cam}}
        if shutters:
            out["cameras"][name]["shutters"] = shutters
        if shutters:
            out["cameras"][name]["shutter"] = default
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
