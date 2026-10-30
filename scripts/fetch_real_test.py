#!/usr/bin/env python3
"""Copy a stratified set of real noisy X1D II RAWs for no-reference evaluation. NAS READ ONLY.

These photos have no clean reference and are used only for evaluation (never training).
Strata: long-exposure low ISO (night), ISO 1600, 3200, 6400 and above. At most
--per-group photos per capture day so one scene cannot dominate.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

from impacket.smbconnection import SMBConnection
from sample_smb_metadata import ReadOnlySMBFile
from denoise_lab.bootstrap import capture_group

STRATA = {"night_low_iso": 6, "iso1600": 8, "iso3200": 8, "iso6400plus": 6}


def stratum(iso, exposure):
    if iso <= 400 and exposure >= 1:
        return "night_low_iso"
    if iso == 1600:
        return "iso1600"
    if iso == 3200:
        return "iso3200"
    if iso >= 6400:
        return "iso6400plus"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=Path("runs/bootstrap/catalog.jsonl"))
    p.add_argument("--server", required=True)
    p.add_argument("--share", required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--password-file", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/real_test"))
    p.add_argument("--per-group", type=int, default=3)
    a = p.parse_args()
    seen, pools = set(), defaultdict(lambda: defaultdict(list))
    for line in a.catalog.read_text().splitlines():
        r = json.loads(line)
        if r.get("camera") != "X1D II 50C" or "error" in r:
            continue
        header = (r["bytes"], r.get("prefix_64k_sha256"))
        if header in seen:
            continue
        seen.add(header)
        try:
            s = stratum(int(r["iso"]), float(Fraction(r["exposure"])))
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        if s:
            pools[s][capture_group(r["camera"], r.get("capture_time"), r["path"])].append(r)
    rng, chosen = random.Random(20261029), []
    for s, limit in STRATA.items():
        groups = [sorted(v, key=lambda r: r["path"]) for _, v in sorted(pools[s].items())]
        for g in groups:
            rng.shuffle(g)
            del g[a.per_group:]
        picked = []
        while groups and len(picked) < limit:
            for g in groups:
                if g and len(picked) < limit:
                    picked.append(g.pop())
            groups = [g for g in groups if g]
        chosen += [dict(r, stratum=s) for r in picked]
    a.output.mkdir(parents=True, exist_ok=True)
    conn = SMBConnection(a.server, a.server, timeout=60)
    conn.login(a.username, a.password_file.read_text().rstrip("\r\n"))
    tree = conn.connectTree(a.share)
    records = []
    for r in chosen:
        digest = hashlib.sha256()
        tmp = a.output/"partial.tmp"
        with ReadOnlySMBFile(conn, tree, r["path"], r["bytes"], budget=r["bytes"]+1048576,
                             block_size=1048576) as src, tmp.open("wb") as dst:
            while chunk := src.read(4*1048576):
                dst.write(chunk); digest.update(chunk); src.cache.clear()
        if tmp.stat().st_size != r["bytes"]:
            raise ValueError("Incomplete read: " + r["path"])
        sha = digest.hexdigest()
        local = a.output/f"{r['stratum']}_{sha[:12]}{r['suffix']}"
        tmp.rename(local)
        records.append(dict(r, sha256=sha, local=str(local), purpose="no_reference_evaluation_only",
                            group=capture_group(r["camera"], r.get("capture_time"), r["path"])))
        print(r["stratum"], r["iso"], r["exposure"], local.name, flush=True)
    conn.logoff()
    (a.output/"manifest.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
