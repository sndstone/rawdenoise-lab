#!/usr/bin/env python3
"""Calibrate sparse sensor defect candidates from one body's dark RAWs.

Use a capture manifest (file, iso, exposure in seconds or rational, optional shutter,
model, serial). Without a manifest, standard TIFF/EXIF metadata supplies settings.
Each observed ISO/exposure/shutter group needs at least two independent captures.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import re

from denoise_lab.defects import (make_defect_map, read_dark_frame, save_defect_map,
                                 raw_defect_metadata, validate_defect_map)


def capture_groups(directory, manifest=None, shutter=None):
    """Deterministic groups; a manifest may supply metadata absent from RAW EXIF."""
    directory = Path(directory)
    manifest = Path(manifest) if manifest else directory/'manifest.json'
    if manifest.exists():
        rows = json.loads(manifest.read_text())
    else:
        import rawpy
        rows = []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {'.arw', '.3fr', '.dng', '.nef', '.cr2', '.cr3', '.raf'}:
                continue
            with rawpy.imread(str(path)) as raw:
                meta, iso, exposure = raw_defect_metadata(raw, path)
            rows.append(dict(file=path.name, iso=iso, exposure=exposure, **meta))
    groups = defaultdict(list)
    for row in rows:
        if shutter and row.get('shutter', '') != shutter:
            continue
        if row.get('iso') is None or row.get('exposure') is None:
            raise ValueError(f"Missing ISO/exposure for {row['file']}")
        iso, exposure = float(row['iso']), float(Fraction(str(row['exposure'])))
        path = (directory/row['file']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Manifest files must remain inside the dark directory')
        groups[(iso, exposure, row.get('shutter', ''))].append((path, row))
    return [(dict(iso_min=i, iso_max=i, exposure_min=e, exposure_max=e, shutter=s),
             sorted(files, key=lambda item: str(item[0]))) for (i, e, s), files in sorted(groups.items())]


def identity(first, model=None, serial=None):
    path, row = first
    image, meta, _, _ = read_dark_frame(path)
    del image
    for key, override in [('model', model), ('serial', serial)]:
        declared = override or row.get(key)
        if declared and meta.get(key) and meta[key] != declared:
            raise ValueError(f'Declared {key} disagrees with RAW metadata')
        meta[key] = declared or meta[key]
    return meta


def dark_frames(files, metadata):
    """Validate identity/geometry of every frame; never combine known body serials."""
    for path, row in files:
        image, meta, iso, exposure = read_dark_frame(path)
        for key in ('shape', 'phase', 'origin', 'raw_shape'):
            if meta[key] != metadata[key]:
                raise ValueError(f'{path.name}: mixed dark frame {key}')
        for key in ('model', 'serial'):
            observed, declared = meta.get(key), row.get(key)
            if observed and declared and observed != declared:
                raise ValueError(f'{path.name}: declared {key} disagrees with RAW')
            value = observed or declared
            if value and metadata.get(key) and value != metadata[key]:
                raise ValueError(f'{path.name}: mixed camera {key}')
            if value and not metadata.get(key):
                metadata[key] = value
        for key, value in [('iso', iso), ('exposure', exposure)]:
            if value is not None and row.get(key) is not None:
                declared = float(Fraction(str(row[key])))
                if abs(value-declared) > max(abs(declared)*1e-6, 1e-12):
                    raise ValueError(f'{path.name}: declared {key} disagrees with RAW')
        yield image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('calib/defect_maps'))
    parser.add_argument('--name', help='Output filename stem; default model and serial')
    parser.add_argument('--camera-model')
    parser.add_argument('--camera-serial')
    parser.add_argument('--shutter', help='Select one shutter mode from the manifest')
    args = parser.parse_args()
    groups = capture_groups(args.directory, args.manifest, args.shutter)
    if not groups:
        parser.error('No dark frames selected')
    metadata = identity(groups[0][1][0], args.camera_model, args.camera_serial)
    calibration = make_defect_map(((c, dark_frames(files, metadata)) for c, files in groups), metadata)
    name = args.name or re.sub(r'[^A-Za-z0-9_.-]+', '_',
                               metadata['model']+'_'+(metadata['serial'] or 'unknown_serial'))
    if Path(name).name != name or name in {'.', '..'}:
        parser.error('--name must be a filename stem')
    path = args.output_dir/(name+'.npz')
    if path.exists():
        parser.error(f'Refusing to overwrite {path}')
    save_defect_map(path, calibration)
    meta = validate_defect_map(calibration)
    print(f"Saved {path}: {len(calibration['kind'])} observations in {len(meta['conditions'])} groups; {path.stat().st_size} bytes")
    print('Dark-only candidates; dead photoresponse and exposure scaling need independent flat/long-exposure checks.')


if __name__ == '__main__':
    main()
