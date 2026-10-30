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
from impacket.smbconnection import SMBConnection
from sample_smb_metadata import ReadOnlySMBFile
from denoise_lab.bayer import normalize_raw
from denoise_lab.bootstrap import capture_group, split_for_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("runs/bootstrap/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/bootstrap/real_examples"))
    parser.add_argument("--share", required=True)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.catalog.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    connection = SMBConnection(args.server, args.server, timeout=60)
    connection.login(args.username, args.password_file.read_text().rstrip("\r\n"))
    tree = connection.connectTree(args.share)
    records = []
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
            chosen_groups, count = set(), 0
            for item in candidates[:30]:
                if item["group"] in chosen_groups:
                    continue
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
                np.save(destination, mosaic[y:y+size, x:x+size].astype(np.float16))
                records.append({**item, "sha256": sha, "crop_x": x, "crop_y": y,
                                "crop": str(destination.relative_to(args.output.parents[1])),
                                "purpose": "unpaired_real_inference_visual_only", "nas_access": "READ_ONLY"})
                chosen_groups.add(item["group"])
                count += 1
                print("REAL_EXAMPLE", camera, "ISO", item["iso"], "exposure", item["exposure"], flush=True)
                if count == 3:
                    break
        (args.output/"manifest.json").write_text(json.dumps(records, indent=2)+"\n")
    finally:
        connection.disconnectTree(tree)
        connection.logoff()
        connection.close()


if __name__ == "__main__":
    main()
