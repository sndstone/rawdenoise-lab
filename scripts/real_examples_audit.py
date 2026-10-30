#!/usr/bin/env python3
"""Unpaired real high-ISO crops: noise envelope + residual diagnostics + previews.

Uses only our own cached validation crops (data/bootstrap/real_examples). No
clean truth exists, so there is no PSNR. Noise-map models receive the blind
green-pair estimate (the realistic path). Using these crops' statistics to
shape a prior would make them development cases; this round does not.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from denoise_lab.bayer import pack, read_mosaic
from denoise_lab.metrics import quick_srgb

from denoise_lab.arch_registry import build_model
from denoise_lab.noise import noise_std
from denoise_lab.tiling import tiled_inference
from denoise_lab import rawstats as rs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    manifest = json.loads((args.root/"bootstrap/real_examples/manifest.json").read_text())
    models = {}
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        m = build_model(saved["model_config"]).to(args.device).eval()
        m.load_state_dict(saved["model"])
        models[name] = (m, hashlib.sha256(Path(path).read_bytes()).hexdigest(), path)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "own held-out real high-ISO crops, unpaired, descriptive only", "crops": []}
    for e in manifest:
        packed = pack(read_mosaic(args.root/e["crop"])).astype(np.float32)
        mu, var = rs.green_pair_blocks(packed, 16)
        nlf = {f"q{q:g}": rs.fit_nlf(mu, var, quantile=q) for q in (.1, .25, .5)}
        est = rs.estimate_noise_parameters(packed)
        nvar = lambda m: nlf["q0.5"]["K"]*np.maximum(m, 0)+nlf["q0.5"]["c"]
        rec = {"camera": e["camera"], "iso": e["iso"], "exposure": e["exposure"], "sha256": e["sha256"],
               "shape": list(packed.shape),
               "nlf": {k: {"K": v["K"], "read_equiv_std": float(np.sqrt(v["c"]))} for k, v in nlf.items()},
               "median_green": float(np.median((packed[1]+packed[3])/2)), "outputs": {}}
        x = torch.from_numpy(packed)[None].to(args.device)
        sigma = np.sqrt(np.maximum(nvar(rs.box_mean(packed, 5)), 1e-14)).astype(np.float32)
        pos_in, _ = rs.impulses(packed, sigma, 8.)
        rec["input_positive_impulse_frac_8sigma"] = float(pos_in.mean())
        crop = (slice(None), slice(packed.shape[1]//2-128, packed.shape[1]//2+128),
                slice(packed.shape[2]//2-128, packed.shape[2]//2+128))
        green = np.median(packed[1][crop[1:]])
        gain = .18/max(float(green), 1e-4)
        tiles = [packed[crop]]
        for name, (m, digest, path) in models.items():
            cond = None
            if m.cond == "noise_map":
                cond = torch.from_numpy(noise_std(est, np.maximum(packed, 0).mean(0, keepdims=True)))[None].to(args.device)
            with torch.inference_mode():
                if max(packed.shape[1:]) > 512:
                    y = tiled_inference(m, x, tile_size=512, overlap=64, feather_power=2.2, noise_map=cond)
                else:
                    y = m(x, cond)
            out = y[0].float().cpu().numpy()
            r = packed-out
            _, diff, z, valid = rs.block_mean_shift(packed, out, 32, noise_var=nvar)
            rb = rs._blocks(r, 16).var(-1).mean(0).ravel()
            mb = rs._blocks(packed, 16).mean(-1).mean(0).ravel()
            pos_out, _ = rs.impulses(out, sigma, 8.)
            rec["outputs"][name] = {
                "checkpoint_sha256": digest,
                "mean_block_shift_per_channel": [float(diff[c][valid].mean()) for c in range(4)],
                "median_block_z_per_channel": [float(np.median(z[c][valid])) for c in range(4)],
                "residual_var_over_nlf_median": float(np.median(rb/np.maximum(nvar(mb), 1e-14))),
                "residual_lag1": rs.lag1_autocorr(r),
                "impulse_survival_8sigma": float(pos_out[pos_in].mean()) if pos_in.any() else None,
                "estimated_map_params": est if m.cond == "noise_map" else None}
            tiles.append(out[crop])
        imgs = [(quick_srgb(torch.from_numpy(t*gain)[None])[0].permute(1, 2, 0).numpy()*255).astype(np.uint8) for t in tiles]
        name = f"{e['camera'].replace(' ', '_')}_iso{e['iso']}_{e['sha256'][:8]}.png"
        Image.fromarray(np.concatenate(imgs, 1)).save(args.output/name)
        rec["preview"] = {"file": name, "order": ["input", *models], "gain": gain,
                          "note": "bilinear demosaic, neutral WB, no colour matrix: diagnostic only"}
        report["crops"].append(rec)
        print("CROP", e["camera"], e["iso"], json.dumps(rec["nlf"]["q0.5"]), flush=True)
    (args.output/"real_examples_audit.json").write_text(json.dumps(report, indent=1)+"\n")
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
