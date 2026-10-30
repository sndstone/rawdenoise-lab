#!/usr/bin/env python3
"""Inspect RAW sensor values and crop candidates; never certify clean targets.

The high-frequency residual is a texture-plus-noise proxy, not a calibrated
read-noise estimate. Single unrelated photographs cannot establish ground truth.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import exifread
import numpy as np
import rawpy
from PIL import Image
from denoise_lab.bayer import normalize_raw, pack

SUFFIXES = {".arw", ".3fr", ".fff", ".dng", ".nef", ".cr2", ".cr3", ".raf"}


def metadata(path):
    with path.open("rb") as source:
        tags = exifread.process_file(source, details=False)
    fields = {"make": "Image Make", "camera": "Image Model", "iso": "EXIF ISOSpeedRatings",
              "exposure": "EXIF ExposureTime", "aperture": "EXIF FNumber",
              "software": "Image Software", "capture_time": "EXIF DateTimeOriginal"}
    result = {name: str(tags[key]) for name, key in fields.items() if key in tags}
    # The archive's EXIF sidecar also covers tags not supported by ExifRead.
    sidecar = path.with_suffix(".exif.txt")
    aliases = {"make": "Exif.Image.Make", "camera": "Exif.Image.Model",
               "iso": "Exif.Photo.ISOSpeedRatings", "exposure": "Exif.Photo.ExposureTime",
               "aperture": "Exif.Photo.FNumber", "software": "Exif.Image.Software",
               "capture_time": "Exif.Photo.DateTimeOriginal"}
    if sidecar.exists():
        entries = {}
        for line in sidecar.read_text(errors="replace").splitlines():
            pieces = line.split(None, 1)
            if len(pieces) == 2:
                entries[pieces[0]] = pieces[1].strip()
        for name, key in aliases.items():
            if key in entries:
                result.setdefault(name, entries[key])
    return result


def inspect(path, output, crop_size):
    info = {"path": str(path.resolve()), "filename": path.name,
            "bytes": path.stat().st_size, **metadata(path)}
    with path.open("rb") as source:
        info["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    with rawpy.imread(str(path)) as raw:
        colors = raw.raw_colors_visible[:2, :2]
        desc = raw.color_desc.decode("ascii")
        info.update(raw_shape=list(raw.raw_image.shape), visible_shape=list(raw.raw_image_visible.shape),
                    cfa="".join(desc[int(c)] for c in colors.ravel()),
                    black_levels=list(raw.black_level_per_channel), white_level=int(raw.white_level),
                    camera_white_levels=raw.camera_white_level_per_channel,
                    rawpy_version=rawpy.__version__, libraw_version=list(rawpy.libraw_version))
        info["at_or_above_nominal_sensor_white_fraction"] = float(
            np.mean(raw.raw_image_visible >= raw.white_level))
        info["decoder_crop"] = {key: int(getattr(raw.sizes, key)) for key in
                                ("crop_left_margin", "crop_top_margin", "crop_width", "crop_height")}
        mosaic = normalize_raw(raw)
        preview = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
    thumb = Image.fromarray(preview)
    thumb.thumbnail((1200, 900))
    preview_path = output / f"{path.stem}_{info['sha256'][:10]}_preview.jpg"
    thumb.save(preview_path, quality=90)
    info["preview"] = str(preview_path.resolve())
    del preview
    planes = pack(mosaic)
    del mosaic
    info["packed_shape"] = list(planes.shape)
    info["channels"] = {}
    histograms = {}
    for label, plane in zip(["R", "G1", "B", "G2"], planes):
        sampled = plane[::4, ::4]
        info["channels"][label] = {
            "quantiles_001_01_50_99_999": np.quantile(sampled, [.001, .01, .5, .99, .999]).tolist(),
            "below_black_fraction": float(np.mean(plane < 0)),
            "signal_below_1pct_fraction": float(np.mean(plane < .01)),
            "at_or_above_normalization_white_fraction": float(np.mean(plane >= 1)),
            "signal_above_99pct_fraction": float(np.mean(plane >= .99))}
        histograms[label] = np.histogram(sampled, bins=np.linspace(-.01, 1.01, 257))[0].tolist()
    info["histograms"] = histograms
    candidates, measurements = [], []
    _, height, width = planes.shape
    for y in range(0, height - crop_size + 1, crop_size):
        for x in range(0, width - crop_size + 1, crop_size):
            crop = planes[:, y:y+crop_size, x:x+crop_size]
            green = crop[1]
            median = float(np.median(green))
            clipped = float(np.mean(crop >= .99))
            dark = float(np.mean(crop < .01))
            gradient = float(np.median(np.abs(np.diff(green, axis=0))) +
                             np.median(np.abs(np.diff(green, axis=1))))
            hh = (green[0::2, 0::2] - green[0::2, 1::2] -
                  green[1::2, 0::2] + green[1::2, 1::2]) / 2
            residual = float(np.median(np.abs(hh - np.median(hh))) / .67448975)
            record = {"sensor_x": x*2, "sensor_y": y*2, "sensor_size": crop_size*2,
                      "median_green": median, "bright_fraction": clipped, "dark_fraction": dark,
                      "gradient_proxy": gradient, "texture_plus_noise_proxy": residual}
            measurements.append(record)
            if .01 <= median <= .8 and clipped < .001 and dark < .5:
                candidates.append(record)
    info["patches_examined"] = len(measurements)
    info["candidate_patch_count"] = len(candidates)
    info["patch_measurements"] = measurements
    info["candidate_crops"] = candidates
    info["training_eligible"] = False
    info["limitations"] = ["Single frame: residual includes real texture; no noise calibration or clean-target certification.",
                           "Crop counts use exposure heuristics, not focus, motion, or target-noise verification.",
                           "No paired-reference PSNR/SSIM can be computed from unrelated samples."]
    info["limitations"].append("Decoder per-channel normalization white can differ from nominal sensor white; thresholds do not prove physical clipping.")
    if path.suffix.lower() == ".fff":
        info["limitations"].append("FFF processing history requires review before any training use.")
    return info


def plot(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(records), 3, figsize=(15, 3.6*len(records)), squeeze=False)
    for row, record in enumerate(records):
        axes[row, 0].imshow(Image.open(record["preview"]))
        axes[row, 0].axis("off")
        name = record["filename"]
        display_name = name if len(name) <= 42 else name[:22] + "…" + name[-16:]
        axes[row, 0].set_title(f"{display_name}\n{record.get('camera', '?')} | ISO {record.get('iso', '?')} | {record.get('exposure', '?')} s", fontsize=9)
        bins = np.linspace(-.01, 1.01, 257)
        for label, color in [("R", "#d1495b"), ("G1", "#338a50"), ("B", "#3066be"), ("G2", "#87b05a")]:
            axes[row, 1].plot((bins[1:]+bins[:-1])/2, record["histograms"][label], color=color, label=label, linewidth=1)
        axes[row, 1].set_yscale("symlog", linthresh=1)
        axes[row, 1].set(xlabel="Normalized RAW signal", ylabel="Sampled pixel count", title="Sensor-plane exposure distribution")
        axes[row, 1].legend(fontsize=8)
        patches = record["patch_measurements"]
        axes[row, 2].scatter([p["median_green"] for p in patches],
                             [p["texture_plus_noise_proxy"] for p in patches], s=9, alpha=.55)
        axes[row, 2].set(xlabel="Patch median green signal", ylabel="Texture + noise proxy (normalized)",
                         title=f"{record['candidate_patch_count']}/{record['patches_examined']} exposure-screened crops")
        axes[row, 2].set_yscale("symlog", linthresh=.0001)
    fig.suptitle("RAW sample audit — residuals are not calibrated sensor noise", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output / "raw_audit.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/public_raw_audit"))
    parser.add_argument("--crop-size", type=int, default=256, help="Packed crop size, even and >=2")
    args = parser.parse_args()
    if args.crop_size < 2 or args.crop_size % 2:
        parser.error("crop-size must be even and at least two")
    paths = sorted({f.resolve() for p in args.inputs for f in (p.rglob("*") if p.is_dir() else [p])
                    if f.is_file() and f.suffix.lower() in SUFFIXES})
    if not paths:
        parser.error("No RAW files found")
    args.output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for path in paths:
        try:
            record = inspect(path, args.output, args.crop_size)
            records.append(record)
            print(f"{path.name}: {record.get('camera')} ISO {record.get('iso')}, CFA {record['cfa']}, "
                  f"packed {record['packed_shape']}, crops {record['candidate_patch_count']}/{record['patches_examined']}", flush=True)
        except Exception as error:
            failures.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {path}: {error}", flush=True)
    payload = {"records": records, "failures": failures, "crop_size_packed": args.crop_size}
    (args.output / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["filename", "camera", "iso", "exposure", "bytes", "cfa", "candidate_patch_count", "patches_examined", "training_eligible"]
    with (args.output / "summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    if records:
        plot(records, args.output)
    print(f"Saved {len(records)} records and {len(failures)} failures to {args.output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
