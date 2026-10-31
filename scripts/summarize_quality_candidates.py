#!/usr/bin/env python3
"""Separate screened sources/days from additional development confirmations."""
import argparse
import json
from pathlib import Path

from evaluate_quality_candidates import paired_summary, sha


def summarize(confirmation, screening):
    if confirmation["snapshot_sha256"] != screening["snapshot_sha256"]:
        raise ValueError("Screening and confirmation snapshots differ")
    seen_sources = {r["source_sha256"] for r in screening["rows"]}
    seen_days = {r["group"] for r in screening["rows"]}
    partitions = {
        "additional_sources": [r for r in confirmation["rows"] if r["source_sha256"] not in seen_sources],
        "additional_capture_days": [r for r in confirmation["rows"] if r["group"] not in seen_days],
        "screened_sources": [r for r in confirmation["rows"] if r["source_sha256"] in seen_sources],
        "all_development_sources": confirmation["rows"],
    }
    result = {}
    for partition, rows in partitions.items():
        summary = []
        for name in dict.fromkeys(row["slice"] for row in rows):
            for candidate in confirmation["candidates"]:
                subset = [r for r in rows if r["slice"] == name and r["candidate"] == candidate]
                if not subset:
                    continue
                summary.append({"slice": name, "candidate": candidate,
                    "by_source": paired_summary(subset, candidate, "source_sha256"),
                    "by_capture_day": paired_summary(subset, candidate, "group"),
                    **{key: sum(r[key] for r in subset)/len(subset) for key in ("flat_psnr_delta", "texture_psnr_delta",
                        "color_mse", "baseline_color_mse", "gamma_mse", "baseline_gamma_mse")}})
        result[partition] = {"sources": len({r["source_sha256"] for r in rows}),
            "capture_days": len({r["group"] for r in rows}), "summary": summary}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", required=True, type=Path)
    parser.add_argument("--confirmation", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    screening = json.loads(args.screening.read_text(encoding="utf-8"))
    result = {"status": "DEVELOPMENT only; additional to this screen, not untouched acceptance data",
        "screening_sha256": sha(args.screening), "snapshot_sha256": screening["snapshot_sha256"],
        "primary_partition": "additional_sources", "capture_day_sensitivity_partition": "additional_capture_days",
        "confirmations": {str(p): {"sha256": sha(p), "partitions": summarize(json.loads(p.read_text(encoding="utf-8")), screening)}
                          for p in args.confirmation}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
