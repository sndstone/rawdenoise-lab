#!/usr/bin/env python3
"""Clean-reference test with camera-calibrated synthetic noise on held-out (validation) crops.

Uses the same dark-frame profiles as camera-noise training, so it favours models trained on them;
read it as "does the model handle each camera's measured noise character", not as real-photo quality.
Exposure is fixed at 1/4 and 1/32 of the crop (mid-tones and shadows). PSNR is on packed linear
RAW after dividing by the exposure (peak 1).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from denoise_lab.bayer import pack, read_mosaic
from denoise_lab.camera_noise import CameraNoise
from denoise_lab.inference import Denoiser
from denoise_lab.metrics import psnr

CASES = {"x1d2": ([400, 1600, 6400, 25600], "data/bootstrap_binned/x1d2_bin4_val_r4.json", [], 6400),
         "a7m3": ([400, 1600, 6400, 25600], "data/bootstrap_binned_a7/a7m3_bin4_val.json", [65535], None),
         "a5100": ([400, 1600, 6400], "data/bootstrap_binned_a7/a7m3_bin4_val.json", [25600], None),
         "rx100m5": ([400, 1600, 6400, 12800], "data/bootstrap_binned_a7/a7m3_bin4_val.json", [], None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", required=True, help="name=path")
    p.add_argument("--calibration", default="calib/darks_20261101.yaml")
    p.add_argument("--crops", type=int, default=24)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    models = {m.split("=", 1)[0]: Denoiser.load(m.split("=", 1)[1], device=args.device) for m in args.model}
    rows = []
    for cam, (isos, manifest, processed, cap) in CASES.items():
        cn = CameraNoise(args.calibration, {cam: 1})
        entries = json.loads(Path(manifest).read_text())
        idx = np.random.default_rng(0).choice(len(entries), min(args.crops, len(entries)), replace=False)
        cleans = []
        for i in idx:
            m = read_mosaic(Path("data")/entries[i]["clean"])
            h, w = (min(m.shape[0], 512)//2)*2, (min(m.shape[1], 512)//2)*2
            cleans.append(m[:h, :w])
        for iso, proc in [(i, False) for i in isos]+[(i, True) for i in processed]:
            for exposure in (1/4, 1/32):
                e = exposure*(cap/iso if cap and iso > cap else 1)
                prm = cn.at(cam, iso, processed=proc)
                acc = {name: [] for name in models}
                acc_in = []
                for k, clean in enumerate(cleans):
                    rng = np.random.default_rng(1000+k)
                    c = (clean*e).astype(np.float32)
                    noisy = pack(cn.apply(c, prm, rng))
                    t = lambda a: torch.from_numpy(np.ascontiguousarray(a/e, dtype=np.float32))[None]
                    target = pack(c)
                    acc_in.append(psnr(t(noisy), t(target)))
                    for name, den in models.items():
                        acc[name].append(psnr(t(den.denoise_packed(noisy)), t(target)))
                row = {"camera": cam, "iso": iso, "processed": proc, "exposure": exposure,
                       "psnr_in": float(np.mean(acc_in)), **{n: float(np.mean(v)) for n, v in acc.items()}}
                rows.append(row)
                print(json.dumps(row), flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/"results.json").write_text(json.dumps(rows, indent=1))
    names = list(models)
    lines = ["| Camera | ISO | Exposure | Input | "+" | ".join(names)+" |", "|"+"---|"*(4+len(names))]
    for r in rows:
        lines.append(f"| {r['camera']} | {r['iso']}{' (processed)' if r['processed'] else ''} | 1/{round(1/r['exposure'])} | "
                     f"{r['psnr_in']:.2f} | "+" | ".join(f"{r[n]:.2f}" for n in names)+" |")
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
