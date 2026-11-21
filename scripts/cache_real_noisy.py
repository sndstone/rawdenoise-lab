#!/usr/bin/env python3
"""Cache packed crops of real high-ISO archive RAWs for self-supervised training. Source READ ONLY.

Selects files from a per-file EXIF scan by camera and minimum ISO and drops excluded capture days
(test sets). Each file gets a whole-frame blind noise estimate, then random 256x256 packed crops,
avoiding clipped and near-black areas. Crops are stored as float16 packed R/G1/B/G2 (normalised
units), with per-crop sigma = sqrt(K*mean + read^2) for loss normalisation.
These crops have no clean reference. They are for self-supervised (neighbour-pair) losses only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random

import numpy as np

CROP = 256


def work(job):
    path, out_dir, camera, iso, n_crops, seed = job
    import rawpy
    from denoise_lab.bayer import normalize_raw, pack
    from denoise_lab.rawstats import estimate_noise_parameters
    try:
        with open(path, "rb") as f:  # read-only
            data = f.read()
        digest = hashlib.sha256(data).hexdigest()
        with rawpy.imread(path) as raw:
            packed = pack(normalize_raw(raw)).astype(np.float32)
    except Exception as exc:
        return {"source": path, "error": repr(exc)[:200]}
    est = estimate_noise_parameters(packed)
    rng = np.random.default_rng(seed)
    _, h, w = packed.shape
    rows, tries = [], 0
    while len(rows) < n_crops and tries < n_crops*20:
        tries += 1
        y, x = int(rng.integers(0, h-CROP)), int(rng.integers(0, w-CROP))
        c = packed[:, y:y+CROP, x:x+CROP]
        g = (c[1]+c[3])/2
        if (c >= .98).mean() > .001 or np.median(g) < .0005:
            continue
        name = f"{camera}/{digest[:16]}_{len(rows):02d}.npy"
        np.save(Path(out_dir)/name, c.astype(np.float16))
        mean = float(np.maximum(c, 0).mean())
        rows.append({"crop": name, "camera": camera, "iso": iso, "source_sha256": digest, "yx": [y, x],
                     "K": est["K"], "read_std": est["read_std"],
                     "sigma": float(np.sqrt(max(est["K"]*mean+est["read_std"]**2, 1e-12)))})
    return {"source": path, "crops": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output", default="data/selfsup_real")
    p.add_argument("--exclude-days", nargs="*", default=[], help="manifests whose capture DAYS are excluded")
    p.add_argument("--exclude-files", nargs="*", default=[], help="manifests whose source FILES are excluded")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=20261101)
    args = p.parse_args()
    # camera: (min ISO, max files, crops per file)
    plan = {"ILCE-5100": (800, 10**6, 12, "a5100"), "DSC-RX100M5": (800, 10**6, 6, "rx100m5"),
            "X1D II 50C": (800, 10**6, 6, "x1d2"), "ILCE-7M3": (1600, 900, 3, "a7m3")}
    excluded = set()
    for m in args.exclude_days:
        data = json.loads(Path(m).read_text())
        for r in (data["files"] if isinstance(data, dict) else data):
            cam = r.get("camera") or r.get("model")
            date = r.get("capture_date") or r.get("capture_time") or r.get("date")
            if cam and date:
                excluded.add((cam, str(date)[:10]))
    excluded_files = set()
    for m in args.exclude_files:
        data = json.loads(Path(m).read_text())
        for r in (data["files"] if isinstance(data, dict) else data):
            excluded_files.add(str(r.get("source") or r.get("path")).replace("\\", "/"))
    rng = random.Random(args.seed)
    scan = json.loads(Path(args.scan).read_text())
    jobs = []
    out = Path(args.output)
    for model, (min_iso, cap, n, short) in plan.items():
        (out/"cache"/short).mkdir(parents=True, exist_ok=True)
        cands = []
        for r in scan:
            if r.get("model") != model or r["ext"] == ".dng" or not r.get("date"):
                continue
            try:
                iso = int(str(r["iso"]).split()[0])
            except (TypeError, ValueError):
                continue
            if iso >= min_iso and (model, r["date"][:10]) not in excluded and r["path"] not in excluded_files:
                cands.append((r, iso))
        rng.shuffle(cands)
        cands = cands[:cap]
        print(f"{model}: {len(cands)} files x {n} crops", flush=True)
        jobs += [(str(Path(args.source_root)/r["path"]), str(out/"cache"), short, iso, n, rng.randrange(2**31))
                 for r, iso in cands]
    manifest, errors = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(work, jobs, chunksize=2), 1):
            if "error" in res:
                errors.append(res)
            else:
                for row in res["crops"]:
                    row["crop"] = "cache/"+row["crop"]
                    row["source"] = res["source"]
                manifest += res["crops"]
            if i % 100 == 0:
                print(f"[{i}/{len(jobs)}] crops {len(manifest)} errors {len(errors)}", flush=True)
    (out/"manifest.json").write_text(json.dumps(manifest))
    (out/"errors.json").write_text(json.dumps(errors, indent=1))
    print(f"done: {len(manifest)} crops from {len(jobs)-len(errors)} files; {len(errors)} errors")


if __name__ == "__main__":
    main()
