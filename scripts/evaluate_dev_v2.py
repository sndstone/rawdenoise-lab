#!/usr/bin/env python3
"""DEV evaluation v2: all validation sources, several 256-packed crops each.

Development diagnostics against imperfect low-ISO references plus added
synthetic noise. The validation split has been inspected for earlier model
selection, so nothing here is an untouched acceptance test.

Slices: the legacy exposure x level grid; out-of-prior stress slices
(structured tails, larger magnitude); and a 2x2 same-channel binned-reference
slice (half resolution, ~half native reference noise; NOT ground truth).
Noise-map models are scored with oracle, x0.5, x2 and blind-estimated maps.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from denoise_lab.bayer import unpack, pack
from denoise_lab.bootstrap import add_bootstrap_noise
from denoise_lab.data import BootstrapDataset
from denoise_lab.metrics import psnr
from denoise_lab.model import gamma
from denoise_lab.arch_registry import build_model
from denoise_lab.noise import noise_std
from denoise_lab.rawstats import estimate_noise_parameters

LEVELS = {"low": {"K": .00002, "read_std": .0002, "row_std": .00005},
          "medium": {"K": .0002, "read_std": .001, "row_std": .0002},
          "high": {"K": .001, "read_std": .004, "row_std": .0005}}
# Stress slices. "tails" mirrors the structure seen in one development
# file's optical black (heavy tails, column offsets) at in-prior magnitude;
# "extreme" is outside the training magnitude prior. Both are uncalibrated.
STRESS = {"tails": {"K": .001, "read_std": .001, "row_std": .00005, "col_std": .00015, "impulse_p": .005},
          "extreme": {"K": .004, "read_std": .008, "row_std": .0005}}
SLICES = ([(e, lv, "native") for e in (1., 1/16, 1/64) for lv in LEVELS] +
          [(1/16, "tails", "native"), (1/64, "tails", "native"), (1/16, "extreme", "native"),
           (1., "low", "binned"), (1., "high", "binned")])
MAP_MODES = {"oracle": 1., "half": .5, "double": 2., "estimated": None}


def color_error(prediction, target):
    d = prediction-target
    green = (d[:, 1]+d[:, 3])/2
    return float(torch.stack((d[:, 0]-green, d[:, 2]-green), 1).square().mean())


def masks(target):
    g = gamma(target.clamp_min(0)).mean(1)[0]
    gy = torch.zeros_like(g)
    gx = torch.zeros_like(g)
    gy[1:-1] = (g[2:]-g[:-2])/2
    gx[:, 1:-1] = (g[:, 2:]-g[:, :-2])/2
    m = (gx.square()+gy.square()).sqrt()
    lo, hi = torch.quantile(m.flatten(), torch.tensor([.5, .75], device=m.device))
    return m <= lo, m >= hi


def masked_psnr(p, t, mask):
    mse = (p.double()-t.double()).square().mean(1)[0][mask].mean().item()
    return 10*np.log10(1/mse) if mse > 0 else float("inf")


def binned_item(clean_packed, parameters, seed):
    """2x2 same-channel average of the reference, then fresh noise."""
    c = clean_packed.numpy()
    h, w = c.shape[-2]//2*2, c.shape[-1]//2*2
    b = c[:, :h, :w].reshape(4, h//2, 2, w//2, 2).mean((2, 4))
    mosaic = unpack(b)
    noisy, p = add_bootstrap_noise(mosaic, np.random.default_rng(seed), fixed_parameters=parameters)
    inp = pack(noisy)
    std = noise_std(p, np.maximum(inp, 0).mean(0, keepdims=True))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))
    return {"noisy": t(inp), "clean": t(b), "noise_map": t(std)}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def paired(rows, a, b, key="model_psnr"):
    by = defaultdict(dict)
    for r in rows:
        if r["model"] in (a, b):
            by[r["source_sha256"]].setdefault(r["model"], []).append(r[key])
    d = np.array([np.mean(v[a])-np.mean(v[b]) for v in by.values() if a in v and b in v])
    if len(d) == 0:
        return None
    sem = float(d.std(ddof=1)/np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"a": a, "b": b, "sources": int(len(d)), "mean_delta_db": float(d.mean()), "sem": sem,
            "ci95": [float(d.mean()-1.96*sem), float(d.mean()+1.96*sem)],
            "median_delta_db": float(np.median(d)), "sources_a_worse": int((d < 0).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--camera", choices=["x1d2", "a7m3"], required=True)
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--crops-per-source", type=int, default=3)
    ap.add_argument("--references", default="baseline,control,balanced")
    ap.add_argument("--previous-evaluation", type=Path, help="Legacy evaluation JSON, to flag previously scored sources")
    args = ap.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    train = json.loads((args.root/"bootstrap"/f"{args.camera}_train.json").read_text())
    validation = json.loads((args.root/"bootstrap"/f"{args.camera}_val.json").read_text())
    for key in ["source_sha256", "sensor_sha256", "group"]:
        if {e[key] for e in train} & {e[key] for e in validation}:
            raise ValueError(f"Train/validation overlap: {key}")
    grouped = defaultdict(list)
    for e in validation:
        grouped[e["source_sha256"]].append(e)
    rng = random.Random(20261028)
    entries = []
    for _, values in sorted(grouped.items()):
        values = sorted(values, key=lambda e: e["clean"])
        entries += rng.sample(values, min(args.crops_per_source, len(values)))
    previous = set()
    if args.previous_evaluation and args.previous_evaluation.exists():
        previous = {r["source_sha256"] for r in json.loads(args.previous_evaluation.read_text())["rows"]}
    models, provenance = {}, {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(saved["model_config"]).to(args.device).eval()
        model.load_state_dict(saved["model"])
        models[label] = model
        provenance[label] = {"checkpoint": path, "sha256": sha(path), "stage_step": saved.get("step"),
                             "model_config": saved["model_config"], "init": saved.get("init"),
                             "training_data": saved.get("config", {}).get("data")}
    rows, estimator = [], defaultdict(list)
    for si, (exposure, level, ref) in enumerate(SLICES):
        parameters = {**LEVELS, **STRESS}[level]
        dataset = BootstrapDataset(entries, root=args.root, crop_size=256, augment=False, deterministic=True,
                                   seed=20261028+1000*si, fixed_parameters=parameters, exposure_scale=exposure)
        slice_name = f"{ref}|{exposure:g}|{level}"
        for i in range(len(dataset)):
            item = dataset[i]
            if ref == "binned":
                item = binned_item(item["clean"], parameters, 20261028+1000*si+i)
            target = item["clean"][None].to(args.device)
            noisy = item["noisy"][None].to(args.device)
            oracle = item["noise_map"][None].to(args.device)
            flat, texture = masks(target)
            common = {"slice": slice_name, "exposure_scale": exposure, "level": level, "reference": ref,
                      "source_sha256": entries[i]["source_sha256"], "crop": entries[i]["clean"],
                      "previously_scored": entries[i]["source_sha256"] in previous,
                      "input_psnr": psnr(noisy, target), "zero_psnr": psnr(torch.zeros_like(target), target)}
            est_map = None
            for label, model in models.items():
                modes = MAP_MODES if model.cond == "noise_map" else {None: None}
                for mode, factor in modes.items():
                    if mode is None:
                        cond, name = None, label
                    elif factor is not None:
                        cond, name = oracle*factor, f"{label}@{mode}"
                    else:
                        if est_map is None:
                            est = estimate_noise_parameters(item["noisy"].numpy())
                            est_map = torch.from_numpy(noise_std(est, np.maximum(item["noisy"].numpy(), 0).mean(0, keepdims=True)))[None].to(args.device)
                            ratio = float(est_map.median()/oracle.median().clamp_min(1e-12))
                            estimator[slice_name].append(ratio)
                        cond, name = est_map, f"{label}@{mode}"
                    with torch.inference_mode():
                        pred = model(noisy, cond)
                    if not torch.isfinite(pred).all():
                        raise ValueError(f"Nonfinite output: {name}")
                    mp = psnr(pred, target)
                    rows.append({**common, "model": name, "model_psnr": mp, "gain_db": mp-common["input_psnr"],
                                 "over_zero_db": mp-common["zero_psnr"],
                                 "flat_psnr": masked_psnr(pred, target, flat),
                                 "texture_psnr": masked_psnr(pred, target, texture),
                                 "color_mse": color_error(pred, target),
                                 "channel_bias": (pred-target).mean(dim=(0, 2, 3)).cpu().tolist()})
        print("EVALUATED", slice_name, len(dataset), round(time.monotonic()-started), flush=True)
    labels = sorted({r["model"] for r in rows})
    refs = [r for r in args.references.split(",") if r in labels]
    summary, comparisons = [], []
    for s in dict.fromkeys(r["slice"] for r in rows):
        srows = [r for r in rows if r["slice"] == s]
        for m in labels:
            mr = [r for r in srows if r["model"] == m]
            per_source = defaultdict(list)
            for r in mr:
                per_source[r["source_sha256"]].append(r)
            def mean_of(k):
                return float(np.mean([np.mean([r[k] for r in v]) for v in per_source.values()]))
            summary.append({"slice": s, "model": m, "sources": len(per_source), "crops": len(mr),
                            **{k: mean_of(k) for k in ["input_psnr", "model_psnr", "gain_db", "over_zero_db",
                                                       "flat_psnr", "texture_psnr", "color_mse"]},
                            "mean_channel_bias": np.mean([r["channel_bias"] for r in mr], 0).tolist()})
        for ref in refs:
            for m in labels:
                if m != ref:
                    for key in ["model_psnr", "flat_psnr", "texture_psnr"]:
                        c = paired(srows, m, ref, key)
                        if c:
                            comparisons.append({"slice": s, "metric": key, **c})
                    sub = [r for r in srows if not r["previously_scored"]]
                    if previous and sub:
                        c = paired(sub, m, ref)
                        if c:
                            comparisons.append({"slice": s, "metric": "model_psnr", "subset": "not_previously_scored", **c})
    result = {"protocol": "dev_v2", "status": "DEVELOPMENT diagnostics; validation split previously inspected; not acceptance",
              "camera": args.camera, "sources": len(grouped), "crops": len(entries), "crop_packed": 256,
              "previously_scored_sources": len(previous & set(grouped)),
              "slices": SLICES, "levels": LEVELS, "stress": STRESS, "map_modes": list(MAP_MODES),
              "estimator_median_std_ratio": {k: {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                                                 "p90": float(np.percentile(v, 90))} for k, v in estimator.items()},
              "checkpoints": provenance, "script_sha256": sha(__file__),
              "summary": summary, "comparisons": comparisons, "rows": rows,
              "seconds": time.monotonic()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1)+"\n")
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
