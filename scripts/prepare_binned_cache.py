#!/usr/bin/env python3
"""Cache Bayer-binned crops from full low-ISO X1D II frames. NAS access is READ ONLY.

Averaging k x k same-colour photosites divides the reference's own ISO 100 noise
variance by k^2 (k=2: 1/4, k=4: 1/16) at 1/k resolution. Full frames are needed for
k=4 because the existing 544-pixel crops shrink to 136 pixels. Targets are still
single exposures, now with much less residual noise; they are not measured clean pairs.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import tempfile
import threading

import numpy as np
import rawpy
from denoise_lab.bayer import normalize_raw
from denoise_lab.bootstrap import capture_group, screen_crop, split_for_group
from denoise_lab.cache_precision import (
    CachePrecisionError, add_cache_dtype_argument, manifest_rows, preflight_cache, save_cache_array,
)
from denoise_lab.clean_targets import bin_mosaic

SIZE = 544


def select(catalog, exclude_groups, per_split, per_group, seed=20261029, camera="X1D II 50C", min_bytes=0):
    seen, groups = set(), defaultdict(list)
    for item in catalog:
        if item.get("camera") != camera or item.get("iso") != "100" or "error" in item:
            continue
        if item.get("bytes", 0) < min_bytes:
            continue
        try:
            if not 0 < float(Fraction(item["exposure"])) <= .25:
                continue
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        header = (item["bytes"], item.get("prefix_64k_sha256"))
        if not header[1] or header in seen:
            continue
        seen.add(header)
        item = dict(item, group=capture_group(item["camera"], item.get("capture_time"), item["path"]))
        if item["group"] in exclude_groups:
            continue
        item["split"] = split_for_group(item["group"])
        groups[item["group"]].append(item)
    rng, chosen = random.Random(seed), []
    for split, limit in per_split.items():
        buckets = [sorted(v, key=lambda r: r["path"]) for g, v in sorted(groups.items()) if v[0]["split"] == split]
        for b in buckets:
            rng.shuffle(b)
            del b[per_group:]
        picked = []
        while buckets and len(picked) < limit:
            for b in buckets:
                if b and len(picked) < limit:
                    picked.append(b.pop())
            buckets = [b for b in buckets if b]
        chosen += picked
    return chosen


def crops_from(mosaic, count, sha, stride=272):
    options = []
    for y in range(0, mosaic.shape[0]-SIZE+1, stride):
        for x in range(0, mosaic.shape[1]-SIZE+1, stride):
            score = screen_crop(mosaic[y:y+SIZE, x:x+SIZE])
            if score is not None:
                options.append({"x": x, "y": y, **score})
    options.sort(key=lambda r: r["gradient_proxy"])
    rng = np.random.default_rng(int(sha[:16], 16))
    order = [list(rng.permutation(b)) for b in np.array_split(np.arange(len(options)), 4) if len(b)]
    chosen = []
    while any(order) and len(chosen) < count:
        for b in order:
            if b and len(chosen) < count:
                chosen.append(options[b.pop()])
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=Path("runs/bootstrap/catalog.jsonl"))
    p.add_argument("--server", required=True)
    p.add_argument("--share", required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--password-file", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/bootstrap_binned"))
    add_cache_dtype_argument(p)
    p.add_argument("--train-images", type=int, default=400)
    p.add_argument("--val-images", type=int, default=60)
    p.add_argument("--per-group", type=int, default=24)
    p.add_argument("--exclude-group", action="append", default=[],
                   help="capture groups held out (test/dev photo days)")
    p.add_argument("--crops", default="2:8,4:10", help="factor:count per image")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--camera", default="X1D II 50C")
    p.add_argument("--slug", default="x1d2", help="manifest prefix")
    p.add_argument("--min-bytes", type=int, default=0, help="e.g. 40000000 keeps uncompressed A7 III files")
    a = p.parse_args()
    try:
        documents = preflight_cache(a.output, a.cache_dtype)
    except CachePrecisionError as error:
        p.error(str(error))
    catalog = [json.loads(l) for l in a.catalog.read_text().splitlines()]
    items = select(catalog, set(a.exclude_group), {"train": a.train_images, "val": a.val_images}, a.per_group,
                   camera=a.camera, min_bytes=a.min_bytes)
    plan = [tuple(map(int, s.split(":"))) for s in a.crops.split(",")]
    print("SELECTED", dict(Counter(i["split"] for i in items)), "groups", len({i["group"] for i in items}), flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    journal = a.output/"sources.jsonl"
    records = documents.get(journal.resolve(), [])
    # Preserve historical resume behavior: every journalled source is done,
    # including errors. Cached arrays were checked before any network access.
    done = {r["path"] for r in records}
    password = a.password_file.read_text().rstrip("\r\n")
    lock, state, seen = threading.Lock(), threading.local(), set()

    def prepare(item):
        try:
            if not hasattr(state, "conn"):
                from impacket.smbconnection import SMBConnection
                state.conn = SMBConnection(a.server, a.server, timeout=60)
                state.conn.login(a.username, password)
                state.tree = state.conn.connectTree(a.share)
            from sample_smb_metadata import ReadOnlySMBFile
            digest = hashlib.sha256()
            with tempfile.TemporaryDirectory(prefix="raw-", dir=a.output) as folder:
                local = Path(folder)/("source"+item["suffix"])
                with ReadOnlySMBFile(state.conn, state.tree, item["path"], item["bytes"],
                                     budget=item["bytes"]+1048576, block_size=1048576) as src, local.open("wb") as dst:
                    while chunk := src.read(4*1048576):
                        dst.write(chunk); digest.update(chunk); src.cache.clear()
                if local.stat().st_size != item["bytes"]:
                    raise ValueError("Incomplete read or NAS file changed during read")
                sha = digest.hexdigest()
                with rawpy.imread(str(local)) as raw:
                    sensor = hashlib.sha256(raw.raw_image_visible.tobytes()).hexdigest()
                    mosaic = normalize_raw(raw)
            with lock:
                if sensor in seen:
                    return {**item, "status": "duplicate_pixels"}
                seen.add(sensor)
            entries = []
            for factor, count in plan:
                binned = bin_mosaic(np.asarray(mosaic, np.float32), factor)
                directory = a.output/"cache"/(f"bin{factor}" if a.slug == "x1d2" else f"{a.slug}_bin{factor}")/sha[:2]
                directory.mkdir(parents=True, exist_ok=True)
                for index, crop in enumerate(crops_from(binned, count, sha)):
                    path = directory/f"{sha}_{index:02d}.npy"
                    actual_dtype = save_cache_array(
                        path, binned[crop["y"]:crop["y"]+SIZE, crop["x"]:crop["x"]+SIZE], a.cache_dtype)
                    entries.append({"clean": str(path.relative_to(a.output.parent)), "source_sha256": sha,
                                    "cache_dtype": actual_dtype,
                                    "sensor_sha256": sensor, "group": item["group"], "split": item["split"],
                                    "camera": item["camera"], "source_iso": 100, "bin_factor": factor,
                                    "reference_kind": f"bayer_binned_{factor}x{factor}_low_iso_single_frame",
                                    "license": "owner_archive", "sensor_crop": crop})
            return {**item, "status": "cached", "sha256": sha, "sensor_sha256": sensor,
                    "cache_dtype": a.cache_dtype, "crops": entries}
        except Exception as error:
            return {**item, "status": "error", "error": f"{type(error).__name__}: {error}"}

    pending = [i for i in items if i["path"] not in done]
    with journal.open("a", buffering=1) as out, ThreadPoolExecutor(a.workers) as pool:
        for n, future in enumerate(as_completed([pool.submit(prepare, i) for i in pending]), 1):
            record = future.result()
            out.write(json.dumps(record)+"\n")
            records.append(record)
            if n % 20 == 0 or record["status"] == "error":
                print(n, len(pending), record["status"], record.get("error", ""), flush=True)
    for split in ("train", "val"):
        for factor, _ in plan:
            entries = [c for r in records if r.get("status") == "cached" for c in r["crops"]
                       if c["split"] == split and c["bin_factor"] == factor and c["camera"] == a.camera]
            manifest_rows(entries, a.cache_dtype)
            (a.output/f"{a.slug}_bin{factor}_{split}.json").write_text(json.dumps(entries))
            print("MANIFEST", split, factor, len(entries), flush=True)
    print("STATUS", dict(Counter(r["status"] for r in records)), flush=True)


if __name__ == "__main__":
    main()
