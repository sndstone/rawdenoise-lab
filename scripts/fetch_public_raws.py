#!/usr/bin/env python3
"""Fetch exact-camera CC0 samples for analysis, with verified SHA-256 provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

CATALOG = "https://raw.pixls.us/json/getrepository.php?set=all"
SAMPLES = {2414, 2415, 2418, 3874, 3941}


def fetch(url):
    return urlopen(Request(quote(url, safe=":/?=&%"),
                           headers={"User-Agent": "rawdenoise-lab/0.1 sample-analysis"}), timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/public_raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with fetch(CATALOG) as response:
        payload = response.read()
    (args.output / "catalog.json").write_bytes(payload)
    rows = []
    for row in json.loads(payload)["data"]:
        match = re.search(r"getfile.php/(\d+)/", row[7])
        if not match or int(match[1]) not in SAMPLES:
            continue
        if "creativecommons.org/publicdomain/zero/1.0/" not in row[5]:
            raise ValueError(f"Sample {match[1]} does not have the required CC0 declaration")
        rows.append((int(match[1]), row))
    if {item[0] for item in rows} != SAMPLES:
        raise ValueError("The catalog no longer contains all expected samples")

    def download(item):
        sample_id, row = item
        url = html.unescape(re.search(r"href='([^']+)'", row[7])[1])
        exif_url = html.unescape(re.search(r"href='([^']+)'", row[8])[1])
        sha = re.search(r"[a-f0-9]{64}", row[7])[0]
        suffix = Path(url).suffix.lower()
        path = args.output / f"{sample_id}{suffix}"
        existing_sha = None
        if path.exists():
            with path.open("rb") as source:
                existing_sha = hashlib.file_digest(source, "sha256").hexdigest()
        if existing_sha != sha:
            temporary = path.with_suffix(suffix + ".part")
            try:
                with fetch(url) as response, temporary.open("wb") as output:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 150 * 1024 * 1024:
                            raise ValueError("Sample exceeded the 150 MiB per-file budget")
                        output.write(chunk)
                with temporary.open("rb") as source:
                    actual = hashlib.file_digest(source, "sha256").hexdigest()
                if actual != sha:
                    raise ValueError(f"SHA-256 mismatch for sample {sample_id}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with fetch(exif_url) as response:
            (args.output / f"{sample_id}.exif.txt").write_bytes(response.read())
        result = {"sample_id": sample_id, "make": row[0], "camera": row[1],
                  "mode": row[2], "remark": row[4], "license": "CC0-1.0",
                  "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                  "license_evidence_html": row[5], "catalog_url": CATALOG,
                  "source_url": url, "source_exif_url": exif_url,
                  "sha256": sha, "bytes": path.stat().st_size, "path": str(path.resolve()),
                  "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "training_eligible": False, "purpose": "analysis_pending_quality_review"}
        print(f"Verified {sample_id}: {row[1]}, {row[2]}, {path.stat().st_size / 1e6:.1f} MB", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, rows))
    (args.output / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Saved provenance for {len(records)} samples; total {sum(r['bytes'] for r in records)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
