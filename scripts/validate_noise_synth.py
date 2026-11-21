#!/usr/bin/env python3
"""Compare flat-dark synthesis with measured calibration moments, without training.

References are the actual dark-pair statistics stored by build_dark_calibration.
This is an in-sample model-fit check, not independent held-out validation. Flat
signal defaults to zero: positive-signal shot noise needs flat-field captures.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from denoise_lab.bayer import pack
from denoise_lab.camera_noise import CameraNoise
from denoise_lab.physics_noise import dark_statistics

OPTIONS = {"banding": True, "quantization": True, "horizontal_correlation": True,
           "vertical_correlation": True, "moment_read": True}
METRICS = ("variance", "excess_kurtosis", "lag1_h", "lag1_v", "row_power", "col_power",
           "row_fixed_power", "col_fixed_power", "code_step_dn")


def tolerance(metric, real, reference, shape, frames):
    """Predeclared targets: 15% variance, 35% tails/power plus sampling floors."""
    if metric == "code_step_dn":
        return 1e-3
    if metric == "variance":
        return .15*real
    if metric == "excess_kurtosis":
        return max(2., .35*abs(real))
    if metric.startswith("lag1"):
        return .035
    dim = shape[1]//2 if metric.startswith("row") else shape[0]//2
    other = shape[0]//2 if metric.startswith("row") else shape[1]//2
    # Mean-band estimates contain temporal band noise as well as white-pixel
    # leakage. A fixed-pattern estimate has only one temporal mean, not N pairs.
    temporal = reference["row_power" if metric.startswith("row") else "col_power"]
    floor = 4*(temporal+reference["variance"]/dim)*np.sqrt(2/(4*other*max(1, frames//2)))
    if "fixed" in metric:
        floor = 4*(temporal+reference["variance"]/dim)/frames*np.sqrt(2/(4*other))
    return .35*abs(real)+floor


def validate(calibration, output, frames=12, size=1024, seed=20261104, legacy=False, model_calibration=None):
    cal = yaml.safe_load(Path(calibration).read_text())
    results, skipped = [], []
    cases = [(name, {**cam, **mode}, shutter)
             for name, cam in cal["cameras"].items()
             for shutter, mode in cam.get("shutters", {None: {}}).items()]
    for name, cam, shutter in cases:
        model = CameraNoise(model_calibration or calibration, {name: 1}, K_jitter_log=0, read_jitter_log=0,
                            physics_options={} if legacy else OPTIONS,
                            shutter_modes={name: shutter} if shutter and not model_calibration else None)
        for iso, profile in cam["iso"].items():
            phys = profile.get("physics", {})
            reference = phys.get("statistics")
            if reference is None:
                skipped.append(f"{name}/{iso}: no measured dark moments")
                continue
            rng = np.random.default_rng(seed+int(iso))
            p = model.at(name, iso, processed=int(iso) in cam.get("processed_isos", []))
            # Use the calibration's crop and frame count when possible so band
            # estimators and their finite-burst uncertainty are directly comparable.
            count = max(2, profile.get("frames", frames))
            shape = (size, size)
            clean = np.zeros(shape, np.float32)
            stack = np.stack([pack(model.apply(clean, p, rng)) for _ in range(count)])
            measured = dark_statistics(stack)
            span = p["white_dn"]-p["black_dn"]
            dn = np.rint(stack.astype(np.float64)*span+p["black_dn"]).astype(np.int64)
            codes = np.unique(dn)
            measured["code_step_dn"] = int(np.gcd.reduce(np.diff(codes))) if len(codes) > 1 else 0
            real = {**reference, "code_step_dn": phys["code_step_dn"]}
            for metric in METRICS:
                tol = tolerance(metric, real[metric], reference, shape, count)
                results.append({"camera": name, "shutter": profile.get("shutter", "unspecified"),
                                "iso": int(iso), "metric": metric, "real": real[metric],
                                "synthetic": measured[metric], "tolerance": tol,
                                "pass": bool(abs(measured[metric]-real[metric]) <= tol)})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps({"calibration": str(calibration), "model_calibration": str(model_calibration or calibration), "legacy": legacy,
        "seed": seed, "size": size, "results": results, "skipped": skipped}, indent=2))
    lines = ["# Dark-noise synthesis validation", "", "In-sample fit check on zero-signal flat fields; no shot-gain validation.",
             "Variance: 15%; excess kurtosis: max(2, 35%); lag-1: 0.035 absolute;", 
             "band powers: 35% plus four estimated sampling errors; code step: 0.001 DN.",
             "Lag-1 distances are in same-colour packed pixels (two sensor pixels).",
             "Fixed patterns repeat per camera/shutter and crop size. Spatial defects/shading are outside this fit.", "",
             "| Camera / shutter | ISO | Statistic | Real | Synthetic | Tolerance | Pass |",
             "|---|---:|---|---:|---:|---:|:---:|"]
    lines += [f"| {r['camera']} / {r['shutter']} | {r['iso']} | {r['metric']} | {r['real']:.5g} | {r['synthetic']:.5g} | {r['tolerance']:.3g} | {'yes' if r['pass'] else 'NO'} |" for r in results]
    lines += ["", f"Passed {sum(r['pass'] for r in results)}/{len(results)} comparisons.", "", *skipped]
    output.with_suffix(".md").write_text("\n".join(lines)+"\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default="calib/darks_physnoise_20261101.yaml")
    parser.add_argument("--output", default="runs/noise_synth/validation")
    parser.add_argument("--model-calibration", help="Optional separate synthesis profiles, e.g. original round-6 calibration")
    parser.add_argument("--frames", type=int, default=12, help="Fallback when calibration has no frame count")
    parser.add_argument("--size", type=int, default=1024, help="Even sensor crop size; match calibration for band powers")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--legacy", action="store_true", help="Evaluate unchanged round-6 synthesis against the same references")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any declared target is missed")
    args = parser.parse_args()
    if args.size < 16 or args.size % 2 or args.frames < 2:
        parser.error("size must be even and >=16; frames must be >=2")
    results = validate(args.calibration, args.output, args.frames, args.size, args.seed, args.legacy, args.model_calibration)
    failures = sum(not r["pass"] for r in results)
    print(f"{len(results)-failures}/{len(results)} within tolerance; {failures} failures. Wrote {args.output}.md/.json")
    if args.strict and (failures or not results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
