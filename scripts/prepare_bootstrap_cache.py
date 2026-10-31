#!/usr/bin/env python3
"""Curate low-ISO NAS files into a bounded local crop cache. NAS access is READ ONLY.

Targets remain noisy single exposures. These crops support provisional synthetic
pretraining and must never be described as measured clean-reference pairs.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import threading
import time

import numpy as np
import rawpy
from denoise_lab.bayer import normalize_raw
from denoise_lab.bootstrap import capture_group, screen_crop, split_for_group
from denoise_lab.cache_precision import (
    CachePrecisionError, add_cache_dtype_argument, manifest_rows, preflight_cache, save_cache_array,
)

CAMERAS = {"X1D II 50C": "x1d2", "ILCE-7M3": "a7m3"}


def candidates(catalog, max_images):
    groups, seen_headers, duplicates = defaultdict(list), set(), 0
    for item in catalog:
        if item.get("camera") not in CAMERAS or item.get("iso") != "100":
            continue
        try:
            if not 0 < float(Fraction(item["exposure"])) <= .25:
                continue
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        # A7 III compressed files are about 25 MB; conservatively use the larger
        # camera-original files for this first pass. This size filter is a proxy.
        if item["camera"] == "ILCE-7M3" and item["bytes"] < 40_000_000:
            continue
        if not item.get("prefix_64k_sha256"):
            continue
        header = (item["camera"], item["bytes"], item["prefix_64k_sha256"])
        if header in seen_headers:
            duplicates += 1
            continue
        seen_headers.add(header)
        item = dict(item)
        item["group"] = capture_group(item["camera"], item.get("capture_time"), item["path"])
        item["split"] = split_for_group(item["group"])
        groups[item["group"]].append(item)
    rng = random.Random(20261026)
    buckets = list(groups.values())
    rng.shuffle(buckets)
    for bucket in buckets:
        rng.shuffle(bucket)
        del bucket[8:]  # cap individual shooting days before round-robin selection
    selected, counts = [], Counter()
    while buckets:
        for bucket in buckets:
            item = bucket.pop()
            if counts[item["camera"]] < max_images*4:
                selected.append(item)
                counts[item["camera"]] += 1
        buckets = [bucket for bucket in buckets if bucket]
    return selected, duplicates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("runs/bootstrap/catalog.jsonl"))
    parser.add_argument("--server", required=True)
    parser.add_argument("--share", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/bootstrap"))
    add_cache_dtype_argument(parser)
    parser.add_argument("--images-per-camera", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=12)
    args = parser.parse_args()
    try:
        documents = preflight_cache(args.output, args.cache_dtype)
    except CachePrecisionError as error:
        parser.error(str(error))
    catalog = {}
    for line in args.catalog.read_text().splitlines():
        item = json.loads(line)
        if "error" not in item:
            catalog[item["path"]] = item
    selected, header_duplicates = candidates(catalog.values(), args.images_per_camera)
    print("CANDIDATES", json.dumps(dict(Counter(item["camera"] for item in selected))),
          "PREFIX_DUPLICATE_CANDIDATES", header_duplicates, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output / "sources.jsonl"
    existing = {}
    for item in documents.get(journal.resolve(), []):
        if item.get("status") != "error":
            existing[item["path"]] = item
    counts = Counter(item["camera"] for item in existing.values() if item["status"] == "cached")
    seen_sensors = {item["sensor_sha256"] for item in existing.values() if item.get("sensor_sha256")}
    seen_sources = {item["sha256"] for item in existing.values() if item.get("sha256")}
    lock, stop, state = threading.Lock(), threading.Event(), threading.local()
    sessions, password = [], args.password_file.read_text().rstrip("\r\n")
    started = time.monotonic()

    def prepare(item):
        reserved = False
        with lock:
            if stop.is_set() or counts[item["camera"]] >= args.images_per_camera:
                return None
        if shutil.disk_usage(args.output).free < args.reserve_gib*1024**3:
            stop.set()
            return {**item, "status": "disk_reserve"}
        try:
            if not hasattr(state, "connection"):
                from impacket.smbconnection import SMBConnection
                state.connection = SMBConnection(args.server, args.server, timeout=60)
                state.connection.login(args.username, password)
                state.tree = state.connection.connectTree(args.share)
                with lock:
                    sessions.append((state.connection, state.tree))
            from sample_smb_metadata import ReadOnlySMBFile
            digest = hashlib.sha256()
            with tempfile.TemporaryDirectory(prefix="raw-", dir=args.output) as folder:
                local = Path(folder)/("source"+item["suffix"])
                with ReadOnlySMBFile(state.connection, state.tree, item["path"], item["bytes"],
                                     budget=item["bytes"]+1048576, block_size=1048576) as source, local.open("wb") as target:
                    while chunk := source.read(4*1048576):
                        target.write(chunk)
                        digest.update(chunk)
                        source.cache.clear()
                if local.stat().st_size != item["bytes"]:
                    raise ValueError("Incomplete read or NAS file changed during read")
                sha = digest.hexdigest()
                with lock:
                    if sha in seen_sources:
                        return {**item, "sha256": sha, "status": "duplicate_file"}
                with rawpy.imread(str(local)) as raw:
                    sensor_sha = hashlib.sha256(raw.raw_image_visible.tobytes()).hexdigest()
                    with lock:
                        if sensor_sha in seen_sensors:
                            return {**item, "sha256": sha, "sensor_sha256": sensor_sha, "status": "duplicate_pixels"}
                        seen_sensors.add(sensor_sha)
                        seen_sources.add(sha)
                    normalization = {"black": list(raw.black_level_per_channel), "white": int(raw.white_level),
                                     "camera_white": raw.camera_white_level_per_channel}
                    mosaic = normalize_raw(raw)
                options, size = [], 544  # extra border permits Bayer-aware flips and later packed 256 crops
                for y in range(32, mosaic.shape[0]-size-32+1, size):
                    for x in range(32, mosaic.shape[1]-size-32+1, size):
                        score = screen_crop(mosaic[y:y+size, x:x+size])
                        if score is not None:
                            options.append({"x": x, "y": y, **score})
                if len(options) < 8:
                    return {**item, "sha256": sha, "sensor_sha256": sensor_sha, "status": "exposure_rejected", "acceptable_crops": len(options)}
                # Mix smooth and textured regions rather than selecting only fine edges.
                options.sort(key=lambda record: record["gradient_proxy"])
                rng = np.random.default_rng(int(sha[:16], 16))
                buckets = np.array_split(np.arange(len(options)), 4)
                order = [list(rng.permutation(bucket)) for bucket in buckets]
                chosen = []
                while any(order) and len(chosen) < args.patches_per_image:
                    for bucket in order:
                        if bucket and len(chosen) < args.patches_per_image:
                            chosen.append(options[bucket.pop()])
                slug = CAMERAS[item["camera"]]
                with lock:
                    if counts[item["camera"]] >= args.images_per_camera:
                        return None
                    counts[item["camera"]] += 1
                    reserved = True
                directory = args.output / "cache" / slug / sha[:2]
                directory.mkdir(parents=True, exist_ok=True)
                crops = []
                for index, crop in enumerate(chosen):
                    path = directory / f"{sha}_{index:02d}.npy"
                    actual_dtype = save_cache_array(
                        path, mosaic[crop["y"]:crop["y"]+size, crop["x"]:crop["x"]+size], args.cache_dtype)
                    crops.append({"clean": str(path.relative_to(args.output.parent)),
                                  "cache_dtype": actual_dtype,
                                  "source_sha256": sha, "sensor_sha256": sensor_sha,
                                  "group": item["group"], "split": item["split"], "camera": item["camera"],
                                  "source_iso": 100, "reference_kind": "low_iso_single_frame_not_verified_clean",
                                  "license": "owner_archive", "sensor_crop": crop})
                return {**item, "status": "cached", "sha256": sha, "sensor_sha256": sensor_sha,
                        "cache_dtype": args.cache_dtype,
                        "normalization": normalization, "crops": crops,
                        "source_url": f"smb://{args.server}/{args.share}/{item['path'].replace(chr(92), '/')}"}
        except Exception as error:
            if reserved:
                with lock:
                    counts[item["camera"]] -= 1
            return {**item, "status": "error", "error": f"{type(error).__name__}: {error}"}

    pending = [item for item in selected if item["path"] not in existing]
    try:
        with journal.open("a", buffering=1) as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare, item) for item in pending]
            completed = 0
            for future in as_completed(futures):
                record = future.result()
                if record is None:
                    continue
                existing[record["path"]] = record
                output.write(json.dumps(record)+"\n")
                completed += 1
                if completed % 10 == 0:
                    progress = {"cached_sources": dict(counts), "processed_this_run": completed,
                                "elapsed_seconds": round(time.monotonic()-started, 1),
                                "free_gib": round(shutil.disk_usage(args.output).free/1024**3, 2)}
                    (args.output/"progress.json").write_text(json.dumps(progress, indent=2))
                    print("CACHE", json.dumps(progress), flush=True)
    finally:
        for connection, tree in sessions:
            connection.disconnectTree(tree)
            connection.logoff()
            connection.close()
    stats = {"source_status": dict(Counter(item["status"] for item in existing.values())),
             "prefix_duplicate_candidates_collapsed": header_duplicates, "cameras": {},
             "nas_access": "READ_ONLY", "reference_kind": "low_iso_single_frame_not_verified_clean",
             "cache_dtype": args.cache_dtype,
             "noise_model": "uncalibrated_bootstrap_prior", "elapsed_seconds": time.monotonic()-started}
    for camera, slug in CAMERAS.items():
        crops = [crop for item in existing.values() if item.get("status") == "cached" and item["camera"] == camera for crop in item["crops"]]
        for split in ["train", "val"]:
            rows = manifest_rows([crop for crop in crops if crop["split"] == split], args.cache_dtype)
            (args.output/f"{slug}_{split}.json").write_text(json.dumps(rows, indent=2)+"\n")
        train = {crop["group"] for crop in crops if crop["split"] == "train"}
        val = {crop["group"] for crop in crops if crop["split"] == "val"}
        if train & val:
            raise RuntimeError("Capture-day leakage between splits")
        stats["cameras"][slug] = {"source_photos": len({crop["source_sha256"] for crop in crops}),
                       "train_crops": sum(crop["split"] == "train" for crop in crops),
                       "val_crops": sum(crop["split"] == "val" for crop in crops),
                       "train_capture_days": len(train), "val_capture_days": len(val)}
    (args.output/"summary.json").write_text(json.dumps(stats, indent=2)+"\n")
    print("CACHE_COMPLETE", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
