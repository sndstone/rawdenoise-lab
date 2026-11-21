#!/usr/bin/env python3
"""Screen fresh RAW captures: python scripts/check_burst.py FOLDER --output runs/check.

Reads recursively; never modifies captures. Reports group diagnostics and pairs each
higher-ISO group with the nearest minimum-ISO group (same aperture preferred).
Separate scenes into separate folders: metadata alone cannot identify scene identity.
"""
import argparse
import json
from pathlib import Path
from itertools import chain
import time

import numpy as np

from denoise_lab.burst_check import (analyze_group, compare_groups, group_frames,
                                     load_preview, read_metadata, write_report, analog_iso_table)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-side', type=int, default=0, help='0: native packed resolution (default); >0: approximate binned preview')
    p.add_argument('--block', type=int, default=16, help='Mask block size in input packed pixels (default 16)')
    p.add_argument('--alignment-spec', type=Path, help='Optional evaluator spec with measured integer packed shifts')
    p.add_argument('--max-gap', type=float, default=30, help='Seconds between burst frames, plus shutter time')
    p.add_argument('--min-static', type=float, default=.10, help='Static fraction for USABLE (default .10)')
    p.add_argument('--marginal-static', type=float, default=.03, help='Static fraction for MARGINAL (default .03)')
    p.add_argument('--min-blocks', type=int, default=20, help='Minimum complete static blocks (default 20)')
    p.add_argument('--calibration', type=Path, default=Path('calib/darks_20261101.yaml'),
                   help='Read cameras.CAMERA.highest_analog_iso if present')
    p.add_argument('--highest-analog-iso', action='append', default=[], metavar='CAMERA=ISO',
                   help='Override analog ceiling; repeat for cameras; 0 disables ceiling')
    args = p.parse_args()
    if (args.max_side != 0 and args.max_side < 32) or args.max_gap <= 0 or args.block < 1:
        p.error('Require max-side 0 or >=32, max-gap >0, block >=1')
    if not 0 <= args.marginal_static <= args.min_static <= 1 or args.min_blocks < 1:
        p.error('Require 0 <= marginal-static <= min-static <= 1 and min-blocks >= 1')
    try:
        analog_isos = analog_iso_table(args.calibration, args.highest_analog_iso)
    except (ValueError, TypeError) as e:
        p.error(str(e))
    if args.max_side:
        print('INFO: binned preview approximation; native evaluator masks can differ', flush=True)
    if not args.folder.is_dir():
        p.error('folder does not exist')
    if args.output.resolve().is_relative_to(args.folder.resolve()):
        p.error('output must be outside the read-only capture folder')
    alignments = []
    if args.alignment_spec:
        alignments = json.loads(args.alignment_spec.read_text())['groups']
    started = time.monotonic()
    records, errors, groups = [], [], []
    suffixes = {'.3fr', '.arw', '.dng', '.nef', '.cr2', '.cr3', '.orf', '.rw2', '.pef', '.raf'}
    for path in sorted(args.folder.rglob('*')):
        if path.suffix.lower() in suffixes and path.is_file():
            try:
                records.append(read_metadata(path))
            except (ValueError, OSError) as e:
                errors.append(str(e))
    for members in group_frames(records, args.max_gap):
        scales, clips, kept = [], [], []
        def frames():
            shape = None
            for r in members:
                start = time.monotonic()
                try:
                    a, scale, clip = load_preview(r['path'], args.max_side)
                    if shape is not None and (a.shape != shape or scale != scales[0]):
                        raise ValueError('packed dimensions changed within group')
                    shape = a.shape
                    scales.append(scale)
                    clips.append(clip)
                    kept.append(r)
                    print(f'{Path(r["path"]).name}: {time.monotonic()-start:.2f}s', flush=True)
                    yield a
                except (ValueError, OSError, RuntimeError) as e:
                    errors.append(f'{r["path"]}: {e}')
        stream = frames()
        first = next(stream, None)
        if first is not None:
            groups.append(analyze_group(chain([first], stream), kept, scales[0], clips,
                                        args.block, analog_isos))
        del first
    pairings = []
    for i, noisy in enumerate(groups):
        r = noisy['summary']['frames'][0]
        same_camera = [g for g in groups if g['summary']['frames'][0]['camera'] == r['camera']]
        base_iso = min(g['summary']['frames'][0]['iso'] for g in same_camera)
        if r['iso'] == base_iso:
            continue
        candidates = [g for g in same_camera if g['summary']['frames'][0]['iso'] == base_iso]
        clean = min(candidates, key=lambda g: (g['summary']['frames'][0]['aperture'] != r['aperture'],
                    abs(np.mean([f['timestamp'] for f in g['summary']['frames']])-
                        np.mean([f['timestamp'] for f in noisy['summary']['frames']]))))
        try:
            alignment = next((g['shift'] for g in alignments
                              if sorted(g['noisy']) == sorted(Path(f['path']).name for f in noisy['summary']['frames'])
                              and sorted(g['clean']) == sorted(Path(f['path']).name for f in clean['summary']['frames'])), None)
            pair_started = time.monotonic()
            result, mask = compare_groups(clean, noisy, args.min_static, args.marginal_static,
                                          args.min_blocks, analog_isos, alignment)
            result["pair_seconds"] = time.monotonic()-pair_started
            result.update(clean_group=next(j for j, g in enumerate(groups) if g is clean), noisy_group=i)
            pairings.append((result, mask))
            print(f'{result["verdict"]}: {Path(r["path"]).name}: {"; ".join(result["reasons"])}', flush=True)
        except ValueError as e:
            errors.append(f'Group {i} pairing failed: {e}')
    if not pairings:
        errors.append('No ISO pairings available; need minimum-ISO and higher-ISO groups from the same camera.')
    write_report(args.output, groups, pairings, errors)
    print(f'{len(records)} inputs, {len(groups)} groups, {len(pairings)} pairings, {len(errors)} errors; '
          f'{time.monotonic()-started:.1f}s; report: {args.output/"index.html"}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
