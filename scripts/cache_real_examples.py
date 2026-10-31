#!/usr/bin/env python3
"""Read a few actual high-ISO RAWs via read-only SMB for qualitative review."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import rawpy
from denoise_lab.bayer import normalize_raw
from denoise_lab.bootstrap import capture_group, split_for_group
from denoise_lab.cache_precision import (
    CachePrecisionError, add_cache_dtype_argument, manifest_rows, preflight_cache, save_cache_array,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("runs/bootstrap/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/bootstrap/real_examples"))
    add_cache_dtype_argument(parser)
    parser.add_argument("--share", required=True)
    args = parser.parse_args()
    try:
        documents = preflight_cache(args.output, args.cache_dtype)
    except CachePrecisionError as error:
        parser.error(str(error))
    items = [json.loads(line) for line in args.catalog.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    connection, tree = None, None
    records = documents.get((args.output/"manifest.json").resolve(), [])
    manifest_rows(records, args.cache_dtype)
    try:
        for camera in ["ILCE-7M3", "X1D II 50C"]:
            candidates = []
            for item in items:
                if item.get("camera") != camera or item.get("iso") not in {"1600", "3200", "6400"}:
                    continue
                if camera == "ILCE-7M3" and item["bytes"] < 40_000_000:
                    continue
                try:
                    if not 0 < float(Fraction(item["exposure"])) <= 1:
                        continue
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
                group = capture_group(camera, item.get("capture_time"), item["path"])
                candidates.append({**item, "group": group, "split": split_for_group(group)})
            candidates.sort(key=lambda item: (item["split"] != "val", -int(item["iso"]),
                                              hashlib.sha256(item["path"].encode()).hexdigest()))
            previous = [r for r in records if r["camera"] == camera]
            chosen_groups, count = {r["group"] for r in previous}, len(previous)
            done = {r["path"] for r in records}
            for item in candidates[:30]:
                if count >= 3:
                    break
                if item["group"] in chosen_groups or item["path"] in done:
                    continue
                if connection is None:
                    from impacket.smbconnection import SMBConnection
                    connection = SMBConnection(args.server, args.server, timeout=60)
                    connection.login(args.username, args.password_file.read_text().rstrip("\r\n"))
                    tree = connection.connectTree(args.share)
                from sample_smb_metadata import ReadOnlySMBFile
                with tempfile.TemporaryDirectory(dir=args.output) as folder:
                    path = Path(folder)/("input"+item["suffix"])
                    digest = hashlib.sha256()
                    with ReadOnlySMBFile(connection, tree, item["path"], item["bytes"],
                                         budget=item["bytes"]+1048576, block_size=1048576) as source, path.open("wb") as target:
                        while chunk := source.read(4*1048576):
                            target.write(chunk)
                            digest.update(chunk)
                            source.cache.clear()
                    if path.stat().st_size != item["bytes"]:
                        raise ValueError("Incomplete read or NAS source changed")
                    with rawpy.imread(str(path)) as raw:
                        mosaic = normalize_raw(raw)
                options, size = [], 1024
                for y in range(32, mosaic.shape[0]-size-32, size):
                    for x in range(32, mosaic.shape[1]-size-32, size):
                        tile = mosaic[y:y+size, x:x+size]
                        median = float(np.median(tile[0::2, 1::2]))
                        if .005 < median < .4 and np.mean(tile > .99) < .005:
                            options.append((abs(np.log(median/.08)), x, y))
                if not options:
                    continue
                _, x, y = min(options)
                sha = digest.hexdigest()
                destination = args.output/f"{sha}.npy"
                actual_dtype = save_cache_array(destination, mosaic[y:y+size, x:x+size], args.cache_dtype)
                records.append({**item, "sha256": sha, "crop_x": x, "crop_y": y,
                                "cache_dtype": actual_dtype,
                                "crop": str(destination.relative_to(args.output.parents[1])),
                                "purpose": "unpaired_real_inference_visual_only", "nas_access": "READ_ONLY"})
                chosen_groups.add(item["group"])
                count += 1
                print("REAL_EXAMPLE", camera, "ISO", item["iso"], "exposure", item["exposure"], flush=True)
                if count == 3:
                    break
        manifest_rows(records, args.cache_dtype)
        (args.output/"manifest.json").write_text(json.dumps(records, indent=2)+"\n")
    finally:
        if connection is not None:
            if tree is not None:
                connection.disconnectTree(tree)
            connection.logoff()
            connection.close()


if __name__ == "__main__":
    main()
