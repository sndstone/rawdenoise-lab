#!/usr/bin/env python3
"""Matched development checks of inference changes against a frozen snapshot.

References are held-out low-ISO photos plus synthetic noise, not independent real
clean ground truth. Reports per-source and per-capture-day paired differences.
No third-party outputs are read by this evaluator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from denoise_lab.arch_registry import build_model
from denoise_lab.checkpoint import checkpoint_weights
from denoise_lab.data import BootstrapDataset
from denoise_lab.ensemble import bayer_self_ensemble
from denoise_lab.metrics import psnr
from denoise_lab.model import gamma

SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("quality_dev_reference", SCRIPT.with_name("evaluate_dev_v2.py"))
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)

# Orientation-isolation stress tests, deliberately outside measured calibration.
# The offset variance is identical in all three arms; only its direction changes.
ORIENTATION_LEVELS = {
    "row_only": {"K": .0002, "read_std": .001, "row_std": .002, "col_std": 0.},
    "column_only": {"K": .0002, "read_std": .001, "row_std": 0., "col_std": .002},
    "balanced_axes": {"K": .0002, "read_std": .001, "row_std": .002/np.sqrt(2), "col_std": .002/np.sqrt(2)},
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_summary(rows, candidate, grouping):
    groups = defaultdict(list)
    for row in rows:
        if row["candidate"] == candidate:
            groups[row[grouping]].append(row["delta_psnr_db"])
    differences = np.array([np.mean(values) for values in groups.values()], dtype=np.float64)
    if not len(differences):
        raise ValueError("No paired observations")
    if len(differences) > 1:
        rng = np.random.default_rng(20261030)
        boot = np.mean(rng.choice(differences, (4000, len(differences)), replace=True), axis=1)
        ci = np.quantile(boot, [.025, .975]).tolist()
    else:
        ci = None
    return {"grouping": grouping, "groups": len(groups), "mean_delta_db": float(differences.mean()),
            "ci95": ci, "groups_worse": int((differences < 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", action="append", choices=["ema", "transpose", "d4", "fp16mixed", "fp16io"], default=[])
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH", help="Additional independently trained unconditioned Bayer checkpoint")
    parser.add_argument("--slices", nargs="+", help="Exact native|exposure|level names")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sources", type=int, help="Deterministic source subset, for screening only")
    parser.add_argument("--crops-per-source", type=int, default=2)
    parser.add_argument("--realizations", type=int, default=2)
    parser.add_argument("--orientation-stress", action="store_true", help="Uncalibrated matched-variance row/column stress slices")
    args = parser.parse_args()
    if args.crops_per_source < 1 or args.realizations < 1 or (args.sources is not None and args.sources < 2):
        parser.error("Require positive crop/draw counts and at least two screened sources")
    if len(set(args.candidate)) != len(args.candidate):
        parser.error("Duplicate candidate")
    if not args.candidate and not args.checkpoint:
        parser.error("Provide --candidate or --checkpoint")
    torch.set_num_threads(2)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    entries = json.loads(args.manifest.read_text())
    train = json.loads(args.train_manifest.read_text())
    for key in ("source_sha256", "sensor_sha256", "group"):
        if any(not e.get(key) for e in entries+train):
            raise ValueError(f"Missing split provenance: {key}")
        if {e[key] for e in entries} & {e[key] for e in train}:
            raise ValueError(f"Training/evaluation overlap: {key}")
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["source_sha256"]].append(entry)
    rng = random.Random(20261030)
    source_ids = sorted(grouped)
    if args.sources:
        source_ids = sorted(rng.sample(source_ids, min(args.sources, len(source_ids))))
    selected = []
    for source_id in source_ids:
        values = sorted(grouped[source_id], key=lambda row: row["clean"])
        selected.extend(rng.sample(values, min(args.crops_per_source, len(values))))
    saved = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    baseline = build_model(saved["model_config"]).to(args.device).eval()
    baseline.load_state_dict(checkpoint_weights(saved))
    models = {}
    for candidate in args.candidate:
        if candidate == "ema":
            model = build_model(saved["model_config"]).to(args.device).eval()
            model.load_state_dict(checkpoint_weights(saved, "ema"))
        elif candidate == "transpose":
            model = lambda image: bayer_self_ensemble(baseline, image)
        elif candidate == "d4":
            model = lambda image: bayer_self_ensemble(baseline, image, tuple(itertools.product([False, True], repeat=3)))
        else:
            export_spec = importlib.util.spec_from_file_location("quality_onnx_precision", SCRIPT.with_name("export_onnx.py"))
            export = importlib.util.module_from_spec(export_spec)
            export_spec.loader.exec_module(export)
            model = build_model(saved["model_config"]).to(args.device).eval()
            model.load_state_dict(checkpoint_weights(saved))
            model = (export.MixedPrecisionIO(model) if candidate == "fp16mixed" else export.FloatIO(model)).eval()
        models[candidate] = model
    provenance = {}
    for item in args.checkpoint:
        label, separator, path = item.partition("=")
        if not separator or not label or label in models or label in {"snapshot", "baseline"}:
            parser.error("Additional checkpoints require unique NAME=PATH labels")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["model_config"]
        if config.get("cond", "none") != "none" or config.get("arch", "nafnet") != "nafnet":
            parser.error("Additional checkpoints must be unconditioned Bayer NAFNet")
        model = build_model(config).to(args.device).eval()
        model.load_state_dict(checkpoint_weights(checkpoint))
        models[label] = model
        args.candidate.append(label)
        provenance[label] = {"path": path, "sha256": sha(path), "weights": "model", "step": checkpoint.get("step"),
                             "model_config": config, "init": checkpoint.get("init")}
    known = {f"{ref}|{exposure:g}|{level}": (exposure, level, ref) for exposure, level, ref in dev.SLICES}
    orientation = {f"native|{exposure:g}|{level}": (exposure, level, "native")
                   for exposure in (1., 1/16) for level in ORIENTATION_LEVELS}
    names = args.slices or (list(orientation) if args.orientation_stress else list(known))
    if args.orientation_stress:
        known.update(orientation)
    if set(names)-known.keys():
        parser.error("Unknown slice")
    started, rows = time.monotonic(), []
    with torch.inference_mode():
        for name in names:
            exposure, level, ref = known[name]
            parameters = {**dev.LEVELS, **dev.STRESS, **ORIENTATION_LEVELS}[level]
            # Seed depends on the frozen slice order, regardless of partition.
            slice_index = list(known).index(name)
            for draw in range(args.realizations):
                seed = 20261030 + 10000*slice_index + 1000*draw
                dataset = BootstrapDataset(selected, root=args.root, crop_size=256, augment=False,
                    deterministic=True, seed=seed, fixed_parameters=parameters, exposure_scale=exposure)
                for index, entry in enumerate(selected):
                    item = dataset[index]
                    if ref == "binned":
                        item = dev.binned_item(item["clean"], parameters, seed+index)
                    target, image = [item[key][None].to(args.device) for key in ("clean", "noisy")]
                    expected = baseline(image)
                    if not torch.isfinite(expected).all():
                        raise ValueError("Nonfinite baseline")
                    baseline_psnr = psnr(expected, target)
                    flat, texture = dev.masks(target)
                    for candidate, model in models.items():
                        begin = time.monotonic()
                        prediction = model(image)
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                        seconds = time.monotonic()-begin
                        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                            raise ValueError("Invalid candidate output: "+candidate)
                        score = psnr(prediction, target)
                        difference = prediction-expected
                        rows.append({"slice": name, "candidate": candidate, "source_sha256": entry["source_sha256"],
                            "group": entry["group"], "crop": entry["clean"], "draw": draw, "seed": seed+index,
                            "baseline_psnr": baseline_psnr, "model_psnr": score, "delta_psnr_db": score-baseline_psnr,
                            "gamma_mse": float((gamma(prediction)-gamma(target)).square().mean()),
                            "baseline_gamma_mse": float((gamma(expected)-gamma(target)).square().mean()),
                            "flat_psnr_delta": dev.masked_psnr(prediction, target, flat)-dev.masked_psnr(expected, target, flat),
                            "texture_psnr_delta": dev.masked_psnr(prediction, target, texture)-dev.masked_psnr(expected, target, texture),
                            "color_mse": dev.color_error(prediction, target), "baseline_color_mse": dev.color_error(expected, target),
                            "channel_bias": (prediction-target).mean((0, 2, 3)).cpu().tolist(),
                            "max_delta_dn16": float(difference.abs().max()*65535),
                            "rms_delta_dn16": float(difference.square().mean().sqrt()*65535),
                            "seconds": seconds})
                print(f"EVALUATED {name} draw={draw} crops={len(dataset)} seconds={time.monotonic()-started:.1f}", flush=True)
    summary = []
    for name in names:
        for candidate in args.candidate:
            subset = [row for row in rows if row["slice"] == name and row["candidate"] == candidate]
            summary.append({"slice": name, "candidate": candidate, "observations": len(subset),
                "by_source": paired_summary(subset, candidate, "source_sha256"),
                "by_capture_day": paired_summary(subset, candidate, "group"),
                **{key: float(np.mean([r[key] for r in subset])) for key in ("baseline_psnr", "model_psnr", "flat_psnr_delta",
                    "texture_psnr_delta", "color_mse", "baseline_color_mse", "gamma_mse", "baseline_gamma_mse", "seconds")},
                "max_delta_dn16": max(r["max_delta_dn16"] for r in subset),
                "mean_rms_delta_dn16": float(np.mean([r["rms_delta_dn16"] for r in subset]))})
    report = {"status": "DEVELOPMENT; synthetic noise on imperfect held-out references, not acceptance",
        "protocol": "frozen_snapshot_inference_candidates_v1", "snapshot_sha256": sha(args.snapshot),
        "manifest_sha256": sha(args.manifest), "train_manifest_sha256": sha(args.train_manifest),
        "script_sha256": sha(SCRIPT), "ensemble_sha256": sha(SCRIPT.parents[1]/"src/denoise_lab/ensemble.py"),
        "snapshot": str(args.snapshot), "candidates": args.candidate, "sources": len(source_ids),
        "additional_checkpoints": provenance,
        "crops": len(selected), "realizations": args.realizations, "torch": torch.__version__, "device": args.device,
        "dn16_definition": "normalized full-scale delta multiplied by 65535; not a camera-specific DN conversion",
        "orientation_stress": "uncalibrated matched-variance direction diagnostic" if args.orientation_stress else None,
        "summary": summary, "rows": rows, "seconds": time.monotonic()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
