#!/usr/bin/env python3
"""Denoise Bayer RAWs with PyTorch, ONNX Runtime, or macOS Core ML."""
import argparse
import json
import sys
import warnings
import zipfile
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import rawpy
import tifffile

from denoise_lab.bayer import PHASES, normalize_raw, pack, unpack
from denoise_lab.dng import RawMetadata, sensor_codes, write_dng
from denoise_lab.inference import Denoiser, COREML_COMPUTE_UNITS, DEFAULT_OVERLAP
from denoise_lab.models_registry import MODELS, fetch_model, model_hash, resolve_model
from denoise_lab.defects import (load_defect_map, find_defect_map, defect_map_mask,
                                 raw_defect_metadata)


RAW_EXTENSIONS = {'.3fr', '.arw', '.cr2', '.cr3', '.crw', '.dng', '.erf', '.fff',
                  '.iiq', '.kdc', '.mef', '.mos', '.mrw', '.nef', '.nrw', '.orf',
                  '.pef', '.raf', '.raw', '.rw2', '.rwl', '.sr2', '.srf', '.srw', '.x3f'}


def discover_inputs(inputs, recursive=False, pattern='*'):
    """Stable, deduplicated RAW discovery; directory outputs keep relative paths."""
    found, seen = [], set()
    for source in inputs:
        candidates = sorted(source.rglob(pattern) if recursive else source.glob(pattern)) if source.is_dir() else [source]
        for path in candidates:
            if source.is_dir() and (not path.is_file() or path.suffix.lower() not in RAW_EXTENSIONS
                                    or path.stem.endswith('_denoised')):
                continue
            if path.resolve() not in seen:
                seen.add(path.resolve())
                found.append((path, path.relative_to(source).parent if source.is_dir() else Path()))
    return found


def fetch_main(argv=None):
    parser = argparse.ArgumentParser(description='Download and verify a released model over HTTPS')
    parser.add_argument('--name', choices=MODELS, default='r4_plain')
    parser.add_argument('--format', choices=['onnx', 'pt', 'coreml'], default='onnx')
    parser.add_argument('--dir', type=Path, default=Path('models'))
    args = parser.parse_args(argv)
    try:
        print(fetch_model(args.name, args.format, args.dir))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Model download failed: {exc}\n')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['fetch-model']:
        return fetch_main(argv[1:])
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: rawdenoise photo.dng --model r4_plain --device cpu. Download: rawdenoise fetch-model --help")
    parser.add_argument("--version", action="version", version=f"rawdenoise {version('denoise_lab')}")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories of folder inputs")
    parser.add_argument("--glob", default="*", help="Folder glob filter, in addition to RAW extension detection")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing outputs instead of failing")
    parser.add_argument("--report", type=Path, help="Write a JSON processing report (never overwrite)")
    parser.add_argument("--chroma-strength", type=float, help="Chroma blend in [0,1]; defaults to --strength")
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more Bayer RAW files or folders")
    parser.add_argument("--model", required=True, type=Path, help="Model file (.pt, .onnx, .mlpackage) or registry name such as r4_plain")
    parser.add_argument("--output-dir", type=Path, help="Default: beside each input with _denoised suffix")
    parser.add_argument("--format", choices=["dng", "tiff", "npy"], default="dng", help="DNG mosaic, rendered 16-bit sRGB TIFF, or normalized float32 NPY")
    parser.add_argument("--metadata", choices=["full", "minimal"], default="full", help="Preserve standard EXIF/GPS and DNG colour metadata, or use the legacy minimal writer")
    parser.add_argument("--dng-preview", action=argparse.BooleanOptionalAction, default=True, help="Embed a small sRGB browsing preview in full-metadata DNGs")
    parser.add_argument("--backend", choices=["auto", "torch", "onnx", "coreml"], default="auto", help="Select runtime; auto uses the model suffix")
    parser.add_argument("--device", default="auto", help="auto or cpu; torch also accepts cuda, cuda:0, or mps (ONNX/Core ML: auto or cpu only)")
    parser.add_argument("--compute-units", choices=COREML_COMPUTE_UNITS,
                        help="Core ML only; default: all (cpu_only with --device cpu)")
    parser.add_argument('--prepass', choices=['auto', 'map', 'heuristic', 'map+heuristic', 'off'],
                        default='auto', help='auto: mapped correction when supplied, otherwise the existing heuristic')
    parser.add_argument('--defect-map', type=Path, help='Calibrated .npz map for this camera body/settings')
    parser.add_argument('--defect-map-dir', type=Path,
                        help='Auto-lookup by exact camera model and serial; unknown serial never auto-matches')
    parser.add_argument("--no-prepass", action="store_true", help="Disable hot/dead-pixel correction; recommended for isolated stars")
    parser.add_argument("--highlights", choices=["protect", "off"], default="protect",
                        help="Preserve input near clipping and limit highlight overshoot; off uses legacy processing")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Tile overlap in packed pixels (twice this extent in sensor pixels), 0–511")
    parser.add_argument("--align-tiles", action="store_true", help="Align tile origins to 16 packed pixels; round overlap up to a multiple of 16")
    parser.add_argument("--feather-mode", choices=["linear", "cosine"], default="linear", help="Tile blending window (experimental cosine option)")
    parser.add_argument("--tta-shift", type=int, default=1, help="Average N diagonal tile-grid offsets; roughly N times the model work")
    parser.add_argument("--strength", type=float, default=1., help="Blend processed result with input: 0 preserves input, 1 applies full processing")
    args = parser.parse_args(argv)
    args.chroma_strength = args.strength if args.chroma_strength is None else args.chroma_strength
    if not 0 <= args.chroma_strength <= 1:
        parser.error("chroma-strength must be in [0,1]")
    if not 0 <= args.strength <= 1 or not 0 <= args.overlap < 512:
        parser.error("strength must be in [0,1] and overlap in [0,511]")
    if args.no_prepass and args.prepass not in {'auto', 'off'}:
        parser.error('--no-prepass conflicts with an enabled --prepass')
    alignment = 16 if args.align_tiles else 1
    if not 1 <= args.tta_shift <= (512-args.overlap)//alignment:
        parser.error("tta-shift must be positive and fit the available aligned grid phases")
    inputs = discover_inputs(args.inputs, args.recursive, args.glob)
    if not inputs:
        parser.error('No RAW inputs matched')
    if args.report and args.report.exists():
        parser.error(f'Refusing to overwrite existing report: {args.report}')
    registry = MODELS.get(str(args.model))
    start = time.monotonic()
    try:
        args.model = resolve_model(args.model, args.backend)
        with warnings.catch_warnings(record=True) as load_warnings:
            warnings.simplefilter('always')
            denoiser = Denoiser.load(args.model, backend=args.backend, device=args.device,
                                     compute_units=args.compute_units)
        if denoiser.backend == "torch":
            import torch
            torch.set_num_threads(min(8, torch.get_num_threads()))
        digest = model_hash(args.model)
        if registry is None:
            registry = next((entry for entry in MODELS.values() if digest in entry['sha256'].values()), None)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        parser.exit(1, f"Model load failed: {exc}\n")
    print(f"Loaded {denoiser.description} in {time.monotonic()-start:.2f}s", flush=True)
    failures = skipped = succeeded = 0
    records = []
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key != 'inputs'}
    for index, (source, relative) in enumerate(inputs, 1):
        start = time.monotonic()
        destination = (args.output_dir/relative if args.output_dir else source.parent) / f"{source.stem}_denoised.{args.format}"
        print(f"[{index}/{len(inputs)}] {source} -> {destination}", flush=True)
        record = dict(input=str(source), output=str(destination), camera=None, iso=None,
                      backend=denoiser.backend,
                      device=denoiser.device if denoiser.backend == 'torch' else ', '.join(denoiser.providers),
                      providers=denoiser.providers,
                      compute_units=denoiser.compute_units if denoiser.backend == 'coreml' else None,
                      model_hash=digest,
                      settings=settings, warnings=[str(w.message) for w in load_warnings],
                      status='failed', timing_seconds=0.)
        records.append(record)
        reserved = False
        try:
            if destination.resolve() == source.resolve():
                raise ValueError(f'Refusing to overwrite input: {source}')
            if destination.exists() and args.skip_existing:
                skipped += 1
                record['status'] = 'skipped'
                print('  Skipped existing output', flush=True)
                continue
            if destination.exists():
                raise ValueError(f"Refusing to overwrite existing file: {destination}")
            with rawpy.imread(str(source)) as raw:
                meta = RawMetadata.from_raw(raw, source)
                identity, iso, exposure = raw_defect_metadata(raw, source)
                record.update(camera=f'{meta.make} {meta.model}', iso=iso)
                if registry is None:
                    record['warnings'].append('Model training scope unknown; camera and ISO are not validated')
                else:
                    if meta.model not in registry['cameras']:
                        record['warnings'].append('Camera not validated for this model')
                    if iso is None:
                        record['warnings'].append('ISO unavailable; cannot check training prior')
                    elif not registry['iso_range'][0] <= iso <= registry['iso_range'][1]:
                        record['warnings'].append('ISO outside training prior')
                for message in record['warnings']:
                    print(f'  WARNING: {message}', flush=True)
                packed = pack(normalize_raw(raw))
                mode = 'off' if args.no_prepass else args.prepass
                mask = None
                if mode in {'auto', 'map', 'map+heuristic'} and (args.defect_map or args.defect_map_dir):
                    map_path = args.defect_map or find_defect_map(args.defect_map_dir, identity)
                    if map_path is not None:
                        mask = defect_map_mask(load_defect_map(map_path), identity, iso=iso, exposure=exposure)
                        print(f'  Defect map: {map_path}; {int(mask.sum())} mapped sites', flush=True)
                    else:
                        message = 'No exact model/serial map match; auto uses the heuristic'
                        record['warnings'].append(message)
                        print(f'  WARNING: {message}', flush=True)
                with warnings.catch_warnings(record=True) as runtime_warnings:
                    warnings.simplefilter('always')
                    predicted = denoiser.denoise_packed(
                        packed, prepass=mode, defect_mask=mask, overlap=args.overlap,
                        strength=args.strength, chroma_strength=args.chroma_strength,
                        white_balance=1/np.asarray(meta.neutral), highlights=args.highlights,
                        grid_alignment=alignment, feather_mode=args.feather_mode, tta_shift=args.tta_shift)
                for warning in runtime_warnings:
                    record['warnings'].append(str(warning.message))
                    print(f'  WARNING: {warning.message}', flush=True)
                # Preserve borders cropped by unify for non-RGGB and odd dimensions.
                colors = raw.raw_colors_visible
                black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
                whites = raw.camera_white_level_per_channel
                white = np.asarray(whites if whites is not None else [raw.white_level]*len(black), dtype=np.float32)
                white = np.where(white > black, white, raw.white_level)
                visible = (raw.raw_image_visible.astype(np.float32)-black[colors])/(white[colors]-black[colors])
                y, x = PHASES[meta.phase]
                mosaic = unpack(predicted)
                visible[y:y+mosaic.shape[0], x:x+mosaic.shape[1]] = mosaic
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Reserve exclusively before writers open the path, preventing overwrite races.
                with destination.open('xb'):
                    pass
                reserved = True
                if args.format == "dng":
                    write_dng(destination, visible, meta, metadata_mode=args.metadata, preview=args.dng_preview)
                elif args.format == "npy":
                    # Full visible sensor geometry and source CFA, signed linear normalization.
                    np.save(destination, visible, allow_pickle=False)
                else:
                    raw.raw_image_visible[:] = sensor_codes(visible, meta)
                    rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                          output_color=rawpy.ColorSpace.sRGB, adjust_maximum_thr=0.)
                    tifffile.imwrite(destination, rgb, photometric="rgb", metadata=None)
            record['status'] = 'success'
            succeeded += 1
            print(f"  Done in {time.monotonic()-start:.2f}s ({denoiser.description})", flush=True)
        except Exception as exc:
            if reserved:
                destination.unlink(missing_ok=True)
            record['error'] = str(exc)
            failures += 1
            print(f"  FAILED: {exc}", flush=True)
        finally:
            record['timing_seconds'] = time.monotonic()-start
    print(f'{succeeded} succeeded, {skipped} skipped, {failures} failed', flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            json.dump(dict(files=records, summary=dict(succeeded=succeeded, skipped=skipped, failed=failures)),
                      stream, indent=2, allow_nan=False)
            stream.write('\n')
    if failures:
        parser.exit(1, f"{failures} file(s) failed\n")


if __name__ == "__main__":
    main()
