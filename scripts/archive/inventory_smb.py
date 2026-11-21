#!/usr/bin/env python3
"""READ ONLY SMB share listing and RAW inventory. Reports are written locally.

No SMB file-open, upload, delete, rename, directory creation, or attribute-write
operation is used. The remote operations are authentication and directory queries.
Install the optional `nas` dependencies to run this script.
"""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
from pathlib import Path

from impacket.smbconnection import SMBConnection

SUFFIXES = {".arw", ".3fr", ".fff", ".dng", ".nef", ".cr2", ".cr3", ".raf", ".rw2", ".orf"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--share", required=True)
    parser.add_argument("--path", default="")
    parser.add_argument("--max-directories", type=int, default=300)
    parser.add_argument("--max-entries", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("runs/nas_inventory.json"))
    args = parser.parse_args()
    password = args.password_file.read_text().rstrip("\r\n")
    connection = SMBConnection(args.server, args.server, timeout=20)
    try:
        connection.login(args.username, password)
        del password
        shares = [{"name": item["shi1_netname"].rstrip("\x00"),
                   "comment": item["shi1_remark"].rstrip("\x00")}
                  for item in connection.listShares()]
        result = {"server": args.server, "mode": "READ_ONLY_DIRECTORY_QUERIES",
                  "recorded_at": datetime.now(timezone.utc).isoformat(), "shares": shares}
        print("SHARES", json.dumps(shares), flush=True)
        if args.share:
            queue = deque([(args.path.replace("/", "\\").strip("\\"), 0)])
            directories, raws, errors = [], [], []
            entries = visited = depth_limited = 0
            while queue and visited < args.max_directories and entries < args.max_entries:
                current, depth = queue.popleft()
                visited += 1
                try:
                    children = connection.listPath(args.share, (current + "\\*" if current else "*"))
                except Exception as error:
                    errors.append({"path": current, "error": str(error)})
                    continue
                for child in children:
                    name = child.get_longname()
                    if name in {".", ".."}:
                        continue
                    entries += 1
                    path = current + "\\" + name if current else name
                    if child.is_directory():
                        if depth < 2:
                            directories.append(path)
                        if depth < args.max_depth and not name.startswith((".", "@")) and name not in {"#recycle", "$RECYCLE.BIN"}:
                            queue.append((path, depth+1))
                        elif depth >= args.max_depth:
                            depth_limited += 1
                    elif Path(name).suffix.lower() in SUFFIXES:
                        raws.append({"path": path, "bytes": child.get_filesize(), "suffix": Path(name).suffix.lower()})
                    if entries >= args.max_entries:
                        break
            counts = Counter(item["suffix"] for item in raws)
            result.update(share=args.share, root=args.path, raw_files=raws, raw_count=len(raws),
                          raw_bytes=sum(item["bytes"] for item in raws), extension_counts=dict(counts),
                          top_directories=directories, directories_queried=visited, entries_seen=entries,
                          truncated=bool(queue) or entries >= args.max_entries or depth_limited > 0,
                          depth_limited_directories=depth_limited, errors=errors)
            print("RAW_COUNTS", json.dumps(dict(counts)), "BYTES", result["raw_bytes"], flush=True)
            print("TOP_DIRECTORIES", json.dumps(directories[:60]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("LOCAL_REPORT", args.output, flush=True)
    finally:
        try:
            connection.logoff()
        finally:
            connection.close()


if __name__ == "__main__":
    main()
