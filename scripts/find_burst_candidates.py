#!/usr/bin/env python3
"""Find possible fixed-setting RAW bursts in an existing metadata catalog.

This is an offline metadata screen. It cannot establish registration, scene
stability, or fractional-pixel sampling diversity. No NAS or GPU is accessed.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

CAMERAS = {"ILCE-7M3", "X1D II 50C"}


def find_candidates(records, max_gap=2., max_duration=10., min_frames=4):
    unique, duplicates = {}, 0
    for record in records:
        if record.get("camera") not in CAMERAS or "error" in record:
            continue
        try:
            timestamp = datetime.strptime(record["capture_time"][:19], "%Y:%m:%d %H:%M:%S")
            iso = int(record["iso"])
            if not 1 <= iso <= 25600:
                continue
        except (KeyError, ValueError):
            continue
        fingerprint = (record["camera"], record.get("prefix_64k_sha256", record["path"]), record["bytes"])
        if fingerprint in unique:
            duplicates += 1
            continue
        unique[fingerprint] = {**record, "_timestamp": timestamp}
    by_camera = defaultdict(list)
    for record in unique.values():
        by_camera[record["camera"]].append(record)
    candidates = []

    def emit(group):
        if len(group) < min_frames:
            return
        first, last = group[0], group[-1]
        indices = sorted({round(i*(len(group)-1)/min(15, len(group)-1)) for i in range(min(16, len(group)))})
        candidates.append({"camera": first["camera"], "iso": int(first["iso"]),
                           "exposure": first.get("exposure"), "aperture": first.get("aperture"),
                           "start": first["capture_time"], "end": last["capture_time"],
                           "duration_seconds": (last["_timestamp"]-first["_timestamp"]).total_seconds(),
                           "frame_count": len(group), "sample_paths": [group[index]["path"] for index in indices],
                           "subpixel_shift_verified": False, "same_scene_verified": False,
                           "aperture_metadata_available": first.get("aperture") is not None})

    for camera, frames in by_camera.items():
        frames.sort(key=lambda record: (record["_timestamp"], record["path"]))
        group = []
        for frame in frames:
            if group:
                previous = group[-1]
                changed = any(frame.get(key) != previous.get(key) for key in ["iso", "exposure", "aperture"])
                gap = (frame["_timestamp"]-previous["_timestamp"]).total_seconds()
                duration = (frame["_timestamp"]-group[0]["_timestamp"]).total_seconds()
                if changed or gap > max_gap or duration > max_duration:
                    emit(group)
                    group = []
            group.append(frame)
        emit(group)
    candidates.sort(key=lambda item: (item["iso"] > 400, -min(item["frame_count"], 16), item["iso"], item["start"]))
    return {"metadata_candidates": candidates, "candidate_sequences": len(candidates),
            "camera_counts": dict(Counter(item["camera"] for item in candidates)),
            "low_iso_camera_counts": dict(Counter(item["camera"] for item in candidates if item["iso"] <= 400)),
            "prefix_duplicate_candidates_collapsed": duplicates,
            "deduplication": "camera + 64-KiB prefix hash + size; not full decoded-pixel verification",
            "limits": {"max_gap_seconds": max_gap, "max_duration_seconds": max_duration, "min_frames": min_frames},
            "status": "metadata candidates only; no burst verified for super-resolution",
            "priority": "secondary to calibrated RAW denoising"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("runs/bootstrap/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/super_resolution/burst_candidates.json"))
    args = parser.parse_args()
    records = {}
    for line in args.catalog.read_text().splitlines():
        record = json.loads(line)
        records[record["path"]] = record
    result = find_candidates(records.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "metadata_candidates"}, indent=2))


if __name__ == "__main__":
    main()
