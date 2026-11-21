#!/usr/bin/env python3
"""Held-out dark repeatability and synthetic isolated-star map diagnostics.

No clean ground truth exists for real darks: precision/recall below measure agreement
with independent held-out temporal candidates, not physical defect ground truth.
Calibration uses full visible sensors; heuristic diagnostics use four fixed crops.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from denoise_lab.bayer import pack
from denoise_lab.defects import (DEFECT_KINDS, dark_statistics, detect_dark_defects,
    make_defect_map, validate_defect_map, save_defect_map, correct_defects,
    correct_mapped_defects, read_dark_frame)
from build_defect_map import capture_groups, identity, dark_frames


def metrics(predicted, truth):
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def star_report():
    """No lens PSF: deliberately tests the known isolated-photosite ambiguity."""
    rng = np.random.default_rng(20261104)
    darks = rng.normal(0, .0002, (8, 256, 256)).astype(np.float32)
    darks[:, 30, 32] += .5
    c = dict(iso_min=800, iso_max=800, exposure_min=1., exposure_max=1., shutter='')
    m = make_defect_map([(c, darks)], dict(shape=[256, 256], phase='RGGB'))
    image = rng.normal(.01, .0002, (256, 256)).astype(np.float32)
    image[30, 32] = .5
    stars = np.zeros(image.shape, bool)
    for y in range(20, 240, 20):
        for x in range(20, 240, 20):
            image[y, x] = .8
            stars[y, x] = True
    mapped = np.zeros(image.shape, bool)
    mapped[tuple(m['coordinates'].T)] = True
    tensor = torch.from_numpy(pack(image))[None]
    star_mask = torch.from_numpy(pack(stars))[None]
    fixed, _ = correct_mapped_defects(tensor, torch.from_numpy(pack(mapped))[None])
    heuristic, flags = correct_defects(tensor, k=8, abs_floor=.01, adjacency_ratio=.1)
    return dict(stars=int(stars.sum()), mapped_stars_changed=int((fixed[star_mask] != tensor[star_mask]).sum()),
                heuristic_stars_changed=int(flags[star_mask].sum()),
                mapped_defect_after=float(fixed[0, 0, 15, 16]),
                heuristic_defect_after=float(heuristic[0, 0, 15, 16]))


def merge_maps(maps):
    """Retain sparse observations separately, including settings with zero candidates."""
    meta = validate_defect_map(maps[0])
    conditions, group, offset = [], [], 0
    for m in maps:
        current = validate_defect_map(m)
        conditions.extend(current['conditions'])
        group.append(m['group']+offset)
        offset += len(current['conditions'])
    merged = {key: np.concatenate([m[key] for m in maps])
              for key in ('coordinates', 'kind', 'confidence', 'mean', 'std')}
    merged['group'] = np.concatenate(group)
    merged['metadata'] = np.array(json.dumps({**meta, 'conditions': conditions}))
    return merged


def evaluate_camera(name, directory, output_dir, crop_size):
    groups = capture_groups(directory)
    meta = identity(groups[0][1][0])
    maps, rows, splits = defaultdict(list), [], []
    for condition, files in groups:
        # Chronological build/test split. No frame contributes to both halves.
        split = max(2, (len(files)+1)//2)
        build, test = files[:split], files[split:]
        if not test:
            raise ValueError('Evaluation requires at least three frames per condition')
        m = make_defect_map([(condition, dark_frames(build, meta))], meta)
        maps[condition['shutter']].append(m)
        splits.append(dict(condition=condition, build=[p.name for p, _ in build], test=[p.name for p, _ in test]))
        counts = {label: int(np.count_nonzero(m['kind'] & bit)) for bit, label in DEFECT_KINDS.items()}
        row = dict(camera=name, **condition, build_frames=len(build), test_frames=len(test),
                   map_observations=len(m['kind']), kinds=counts)
        if len(test) < 2:
            row['unavailable'] = 'Only one held-out frame; no temporal reference'
        else:
            stats = dark_statistics(dark_frames(test, meta))
            truth, _ = detect_dark_defects(stats)
            del stats
            truth = truth != 0
            row['full_sensor_reference_candidates'] = int(truth.sum())
            image, _, _, _ = read_dark_frame(test[0][0])
            mapped = np.zeros(image.shape, bool)
            mapped[tuple(m['coordinates'].T)] = True
            size = min(crop_size, min(image.shape)//2)
            size -= size % 2
            ys = [2*int((image.shape[0]*f-size/2)//2) for f in (.25, .75)]
            xs = [2*int((image.shape[1]*f-size/2)//2) for f in (.25, .75)]
            predicted, expected, heuristic_masks = [], [], []
            for y in ys:
                for x in xs:
                    crop = image[y:y+size, x:x+size]
                    _, flags = correct_defects(torch.from_numpy(pack(crop))[None],
                                                k=8, abs_floor=.01, adjacency_ratio=.1)
                    predicted.append(pack(mapped[y:y+size, x:x+size]).ravel())
                    expected.append(pack(truth[y:y+size, x:x+size]).ravel())
                    heuristic_masks.append(flags.numpy().ravel())
            truth_crops = np.concatenate(expected)
            row['evaluated_sites'] = len(truth_crops)
            row['crop_origins'] = [[y, x] for y in ys for x in xs]
            row['map'] = metrics(np.concatenate(predicted), truth_crops)
            row['heuristic'] = metrics(np.concatenate(heuristic_masks), truth_crops)
            del image, truth, mapped
        rows.append(row)
        print(name, int(condition['iso_min']), condition['shutter'], row, flush=True)
    artifacts = []
    for shutter, values in maps.items():
        path = output_dir/f'{name}_{shutter or "unspecified"}_build.npz'
        merged = merge_maps(values)
        save_defect_map(path, merged)
        artifacts.append(dict(path=str(path), bytes=path.stat().st_size,
                              observations=len(merged['kind']), unique_sites=len(np.unique(merged['coordinates'], axis=0))))
    return rows, splits, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dark-dir', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/defect_maps'))
    parser.add_argument('--crop-size', type=int, default=1024, help='Sensor-pixel width of each of four fixed diagnostic crops')
    args = parser.parse_args()
    if args.crop_size < 16:
        parser.error('--crop-size must be at least 16')
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(protocol='dark_map_repeatability_v1', crop_size_sensor=args.crop_size, rows=[], splits={}, artifacts=[], stars=star_report(),
                  limitations=['Real precision/recall use repeated-dark candidates as a proxy, not physical ground truth.',
                    'Full sensors used for calibration/reference; heuristic measured on first held-out frame in four fixed crops.',
                    'Short exposure captures do not validate exposure scaling or long exposures.',
                    'Dead photoresponse needs illuminated flat fields; high variance does not confirm telegraph behaviour.',
                    'Chronological same-session holdout does not measure day/temperature stability.'])
    for spec in args.dark_dir:
        name, sep, directory = spec.partition('=')
        if not sep or not name or Path(name).name != name:
            parser.error('--dark-dir requires NAME=PATH')
        rows, splits, artifacts = evaluate_camera(name, Path(directory), args.output_dir, args.crop_size)
        report['rows'].extend(rows)
        report['splits'][name] = splits
        report['artifacts'].extend(artifacts)
        (args.output_dir/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_report(report, args.output_dir)


def write_report(report, output_dir):
    """Render reproducible metrics without rereading private frames."""
    lines = ['# Dark-frame defect map evaluation', '', 'Development diagnostics, 2026-11-04. No neural model or external reference outputs used.', '',
             'Build/test: chronological halves within each camera/ISO/exposure/shutter group; full visible sensors, no shared frames. '
             'Means use a 6-sigma / sqrt(N) threshold and 0.0005 normalized floor; persistent offsets need 75% support above 3 sigma. '
             'High variance exceeds 6 times typical temporal noise and 0.001. Hot means exceed 0.01; other positive candidates are warm. '
             'Confidence is frame-support evidence discounted by sample count, not a calibrated probability.', '',
             'Precision/recall are **held-out repeatability proxies**, against independent held-out temporal candidate masks. '
             'They are not physical defect ground truth. The heuristic acts on the first held-out frame, using its existing defaults '
             f"(k=8, floor=0.01, adjacency=0.1). Four fixed {report.get('crop_size_sensor', 1024)}-square sensor crops sample the quarter/three-quarter positions; "
             'exact locations, split filenames and confusion counts are in results.json. Undefined precision/recall is shown as —.', '',
             '| Camera | Shutter | ISO | Seconds | Build/test | Full map observations | Eval sites | Map P/R | Heuristic P/R |',
             '|---|---|---:|---:|---:|---:|---:|---|---|']
    def fmt(value):
        return '—' if value is None else f'{value:.4f}'
    for row in report['rows']:
        def pr(key):
            return '/'.join(fmt(row.get(key, {}).get(k)) for k in ('precision', 'recall'))
        lines.append(f"| {row['camera']} | {row['shutter']} | {row['iso_min']:g} | {row['exposure_min']:.6g} | "
                     f"{row['build_frames']}/{row['test_frames']} | {row['map_observations']} | {row.get('evaluated_sites', '—')} | {pr('map')} | {pr('heuristic')} |")
    lines += ['', '## Pooled repeatability', '',
              'Confusion counts pooled across evaluated settings (unevaluated one-frame holdouts excluded). '
              'High-variance fraction counts overlapping kind flags among build observations.', '',
              '| Camera | Map precision | Map recall | High-variance fraction |', '|---|---:|---:|---:|']
    for camera in dict.fromkeys(r['camera'] for r in report['rows']):
        rows = [r for r in report['rows'] if r['camera'] == camera and 'map' in r]
        totals = {key: sum(r['map'][key] for r in rows) for key in ('tp', 'fp', 'fn')}
        tp, fp, fn = (totals[k] for k in ('tp', 'fp', 'fn'))
        precision = tp/(tp+fp) if tp+fp else None
        recall = tp/(tp+fn) if tp+fn else None
        count = sum(r['map_observations'] for r in rows)
        variable = sum(r['kinds']['high_variance'] for r in rows)/count if count else None
        lines.append(f"| {camera} | {fmt(precision)} | {fmt(recall)} | {fmt(variable)} |")
    lines += ['', 'These are experimental candidate maps. Low held-out repeatability, especially among '
              'high-variance sites with few captures, does not support promoting these broad maps to default correction. '
              'Review repeated high-mean candidates separately and collect more independent frames before deployment.']
    lines += ['', '## Isolated synthetic stars', '', json.dumps(report['stars'], indent=2), '',
              'The synthetic frame contains independent Gaussian noise plus 121 isolated bright photosites, away from one calibrated hot site. '
              'Mapped correction must preserve every unlisted star exactly; the heuristic intentionally cannot distinguish such stars from defects.', '',
              '## Map sizes', '']
    for a in report['artifacts']:
        lines.append(f"- `{Path(a['path']).name}`: {a['bytes']} bytes; {a['observations']} condition observations; {a['unique_sites']} distinct sites.")
    lines += ['', '## Limits and follow-ups', ''] + ['- '+s for s in report['limitations']]
    lines += ['- X1D electronic-shutter captures have no saturated hot pixels; warm candidates may still exist. '
              'Sony top ISO applies in-camera processing; do not extrapolate lower-ISO maps there.',
              '- Serial numbers are read only from standard metadata when available. Unknown-serial maps require explicit selection.',
              '- Collect repeated long darks at several exposures and temperatures, plus flats; validate on a separate capture day before promotion.']
    (output_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
