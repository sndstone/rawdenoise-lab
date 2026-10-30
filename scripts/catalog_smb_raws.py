#!/usr/bin/env python3
"""Resume a complete RAW metadata catalog using read-only SMB handles."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import exifread
from impacket.smbconnection import SMBConnection
from sample_smb_metadata import ReadOnlySMBFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/bootstrap/catalog.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--share", required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    files = [item for item in inventory["raw_files"] if item["suffix"] in {".arw", ".3fr"}]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            try:
                item = json.loads(line)
                if "error" not in item:
                    previous[item["path"]] = item
            except json.JSONDecodeError:
                pass
    pending = [item for item in files if item["path"] not in previous]
    password = args.password_file.read_text().rstrip("\r\n")
    state, sessions, lock = threading.local(), [], threading.Lock()
    keys = {"camera": "Image Model", "iso": "EXIF ISOSpeedRatings", "exposure": "EXIF ExposureTime",
            "aperture": "EXIF FNumber", "capture_time": "EXIF DateTimeOriginal", "software": "Image Software"}

    def inspect(item):
        try:
            if not hasattr(state, "connection"):
                state.connection = SMBConnection(args.server, args.server, timeout=30)
                state.connection.login(args.username, password)
                state.tree = state.connection.connectTree(args.share)
                with lock:
                    sessions.append((state.connection, state.tree))
            with ReadOnlySMBFile(state.connection, state.tree, item["path"], item["bytes"]) as source:
                tags = exifread.process_file(source, details=False, extract_thumbnail=False)
                source.seek(0)
                fingerprint = hashlib.sha256(source.read(min(65536, item["bytes"]))).hexdigest()
                return {**item, **{name: str(tags[key]) for name, key in keys.items() if key in tags},
                        "prefix_64k_sha256": fingerprint, "metadata_bytes_read": source.bytes_read,
                        "source_access": "READ_ONLY; FILE_OPEN existing only"}
        except Exception as error:
            return {**item, "error": f"{type(error).__name__}: {error}"}

    results = dict(previous)
    errors, started = 0, time.monotonic()
    try:
        with args.output.open("a", buffering=1) as output, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(inspect, item) for item in pending]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                output.write(json.dumps(record)+"\n")
                results[record["path"]] = record
                errors += "error" in record
                if index % 500 == 0 or index == len(pending):
                    status = {"done": len(results), "total": len(files), "errors_this_run": errors,
                              "elapsed_seconds": round(time.monotonic()-started, 1)}
                    args.output.with_suffix(".progress.json").write_text(json.dumps(status, indent=2))
                    print("CATALOG", json.dumps(status), flush=True)
    finally:
        for connection, tree in sessions:
            connection.disconnectTree(tree)
            connection.logoff()
            connection.close()
    cameras = Counter(item.get("camera", "unknown") for item in results.values())
    isos = defaultdict(Counter)
    for item in results.values():
        isos[item.get("camera", "unknown")][item.get("iso", "unknown")] += 1
    summary = {"records": len(results), "expected": len(files), "camera_counts": dict(cameras),
               "iso_counts": {key: dict(value) for key, value in isos.items()},
               "errors": sum("error" in item for item in results.values()),
               "metadata_bytes_read": sum(item.get("metadata_bytes_read", 0) for item in results.values()),
               "nas_access": "READ_ONLY", "complete": len(results) == len(files)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
