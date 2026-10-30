#!/usr/bin/env python3
"""Sample NAS RAW metadata through strictly read-only SMB file handles.

FILE_OPEN only opens existing files; desiredAccess excludes all write/delete
rights. No NAS mount is created. Local reports/copies never go to SMB paths.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random

import exifread
from impacket.smbconnection import SMBConnection
from impacket.smb3structs import FILE_READ_DATA, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_SHARE_DELETE, FILE_OPEN


class ReadOnlySMBFile(io.RawIOBase):
    """Seekable, block-cached reader with a per-file network byte budget."""
    def __init__(self, connection, tree, path, size, budget=4*1024*1024, block_size=65536):
        super().__init__()
        self.connection, self.tree, self.size, self.budget = connection, tree, size, budget
        self.position, self.bytes_read, self.cache = 0, 0, {}
        self.block_size = block_size
        self.handle = connection.openFile(tree, path,
                                         desiredAccess=FILE_READ_DATA | FILE_READ_ATTRIBUTES,
                                         shareMode=FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                         creationDisposition=FILE_OPEN)

    def readable(self):
        return True

    def writable(self):
        return False

    def write(self, data):
        raise io.UnsupportedOperation("NAS files are read-only")

    def truncate(self, size=None):
        raise io.UnsupportedOperation("NAS files are read-only")

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0:
            return b""
        chunks = []
        while size:
            block_start = self.position // self.block_size * self.block_size
            if block_start not in self.cache:
                count = min(self.block_size, self.size-block_start)
                if self.bytes_read+count > self.budget:
                    raise ValueError("Metadata read budget exceeded")
                data = self.connection.readFile(self.tree, self.handle, offset=block_start,
                                                bytesToRead=count, singleCall=False)
                self.cache[block_start] = data
                self.bytes_read += len(data)
            data = self.cache[block_start]
            offset = self.position-block_start
            chunk = data[offset:offset+size]
            if not chunk:
                break
            chunks.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self):
        if not self.closed:
            self.connection.closeFile(self.tree, self.handle)
            super().close()


def choose(files, count, rng):
    """Round-robin shuffled folders to avoid sampling one shoot exclusively."""
    groups = defaultdict(list)
    for item in files:
        groups[item["path"].rsplit("\\", 1)[0]].append(item)
    buckets = list(groups.values())
    rng.shuffle(buckets)
    for bucket in buckets:
        rng.shuffle(bucket)
    selected = []
    while buckets and len(selected) < count:
        for bucket in buckets:
            selected.append(bucket.pop())
            if len(selected) >= count:
                break
        buckets = [bucket for bucket in buckets if bucket]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-file", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--samples-per-format", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("runs/nas_metadata_sample.json"))
    parser.add_argument("--copy-selection", type=Path, help="Instead, copy this list of inventoried paths to local storage")
    parser.add_argument("--copy-output", type=Path, default=Path("data/nas_samples"))
    parser.add_argument("--share", required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    connection = SMBConnection(args.server, args.server, timeout=30)
    password = args.password_file.read_text().rstrip("\r\n")
    connection.login(args.username, password)
    del password
    tree = connection.connectTree(args.share)
    try:
        if args.copy_selection:
            paths = json.loads(args.copy_selection.read_text())
            indexed = {item["path"]: item for item in inventory["raw_files"]}
            if len(paths) > 12 or sum(indexed[p]["bytes"] for p in paths) > 1024**3:
                raise ValueError("Copy selection exceeds 12 files or 1 GiB")
            args.copy_output.mkdir(parents=True, exist_ok=True)
            provenance = []
            for path in paths:
                item = indexed[path]
                name = hashlib.sha256(path.encode()).hexdigest()[:12] + "_" + path.rsplit("\\", 1)[-1]
                destination = args.copy_output / name
                temporary = destination.with_suffix(destination.suffix + ".part")
                digest = hashlib.sha256()
                try:
                    with ReadOnlySMBFile(connection, tree, path, item["bytes"], budget=item["bytes"]+65536) as source, temporary.open("wb") as target:
                        while chunk := source.read(1024*1024):
                            target.write(chunk)
                            digest.update(chunk)
                            # Full-copy reads are sequential; do not retain the entire RAW in RAM.
                            source.cache.clear()
                    if temporary.stat().st_size != item["bytes"]:
                        raise ValueError("NAS file changed or copy was incomplete")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                provenance.append({"source": f"smb://{args.server}/{inventory['share']}/{path.replace(chr(92), '/')}",
                                   "path": str(destination.resolve()), "sha256": digest.hexdigest(),
                                   "bytes": item["bytes"], "access": "FILE_READ_DATA|FILE_READ_ATTRIBUTES; FILE_OPEN",
                                   "training_eligible": False, "license": "owner_archive_pending_review"})
                print("COPIED_READ_ONLY", name, item["bytes"], flush=True)
            (args.copy_output / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
            return
        rng = random.Random(42)
        selected = []
        for suffix in [".3fr", ".arw"]:
            selected += choose([item for item in inventory["raw_files"] if item["suffix"] == suffix], args.samples_per_format, rng)
        records, errors = [], []
        keys = {"camera": "Image Model", "make": "Image Make", "iso": "EXIF ISOSpeedRatings",
                "exposure": "EXIF ExposureTime", "aperture": "EXIF FNumber", "capture_time": "EXIF DateTimeOriginal",
                "software": "Image Software"}
        for index, item in enumerate(selected):
            try:
                with ReadOnlySMBFile(connection, tree, item["path"], item["bytes"]) as source:
                    tags = exifread.process_file(source, details=False, extract_thumbnail=False)
                    record = {**item, **{name: str(tags[key]) for name, key in keys.items() if key in tags},
                              "metadata_bytes_read": source.bytes_read}
                records.append(record)
            except Exception as error:
                errors.append({"path": item["path"], "error": f"{type(error).__name__}: {error}"})
            if (index+1) % 20 == 0:
                print("METADATA_PROGRESS", index+1, "/", len(selected), flush=True)
        models = Counter(record.get("camera", "unknown") for record in records)
        isos = defaultdict(Counter)
        for record in records:
            isos[record.get("camera", "unknown")][record.get("iso", "unknown")] += 1
        result = {"server": args.server, "share": inventory["share"], "access": "READ_ONLY",
                  "open_access": "FILE_READ_DATA|FILE_READ_ATTRIBUTES", "open_disposition": "FILE_OPEN (existing only)",
                  "sampling": f"seed 42; round-robin folders; up to {args.samples_per_format} per RAW format; not an archive-wide ISO census",
                  "records": records, "errors": errors, "camera_counts": dict(models),
                  "iso_counts": {camera: dict(counts) for camera, counts in isos.items()},
                  "metadata_bytes_read": sum(record["metadata_bytes_read"] for record in records)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print("CAMERAS", json.dumps(dict(models)), flush=True)
        print("ISOS", json.dumps(result["iso_counts"]), flush=True)
        print("READ_BYTES", result["metadata_bytes_read"], "ERRORS",len(errors), flush=True)
    finally:
        connection.disconnectTree(tree)
        connection.logoff()
        connection.close()


if __name__ == "__main__":
    main()
