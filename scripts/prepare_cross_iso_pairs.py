#!/usr/bin/env python3
"""Prepare development-evaluation pairs from one static scene's cross-ISO bursts.

Read-only capture discovery uses RAW metadata (or PATH.json sidecars). --dry-run
reads metadata only and writes nothing. Outputs are native registered normalized
RGGB mosaics, an independent low-ISO target, masks/counts and QA diagnostics.
Inspect QA and reserve independent scenes before using any pair for training.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from denoise_lab.bayer import pack, unpack, read_mosaic
from denoise_lab.burst_check import read_metadata, group_frames
from denoise_lab.burst_align import warp_packed, robust_merge, interpolation_variance
from eval_real_burst import prepare_pair


RAW_SUFFIXES = {'.3fr', '.arw', '.dng', '.nef', '.cr2', '.cr3', '.orf', '.rw2', '.pef', '.raf'}


def discover_groups(folder, max_gap=30.):
    paths = sorted(p for p in Path(folder).rglob('*')
                   if p.is_file() and p.suffix.lower() in RAW_SUFFIXES)
    if not paths:
        raise ValueError('No RAW capture files found')
    return group_frames([read_metadata(p) for p in paths], max_gap)


def group_report(groups):
    return [{'group': i, 'count': len(g),
             **{k: g[0][k] for k in ('camera', 'iso', 'shutter', 'aperture')},
             'frames': [Path(r['path']).name for r in g]} for i, g in enumerate(groups)]


def select_groups(groups, low=None, high=None):
    if low is None or high is None:
        if len(groups) != 2:
            raise ValueError('Need exactly two bursts, or select --low-group and --high-group from --dry-run')
        order = sorted(range(2), key=lambda i: groups[i][0]['iso'])
        low = order[0] if low is None else low
        high = order[1] if high is None else high
    if not 0 <= low < len(groups) or not 0 <= high < len(groups):
        raise ValueError('Selected group index is out of range')
    clean, noisy = groups[low], groups[high]
    if clean[0]['camera'] != noisy[0]['camera'] or clean[0]['iso'] >= noisy[0]['iso']:
        raise ValueError('Need the same camera and strictly lower target ISO')
    if min(len(clean), len(noisy)) < 2:
        raise ValueError('Need at least two independent frames in each burst')
    return clean, noisy


def prepare_arrays(clean_frames, noisy_frames, block=16, shift=(0, 0)):
    """Shared aligned reference preparation, plus independent half-target QA."""
    if min(len(clean_frames), len(noisy_frames)) < 2:
        raise ValueError('Need at least two frames per burst')
    if not np.isfinite(clean_frames).all() or not np.isfinite(noisy_frames).all():
        raise ValueError('Source mosaics must be finite before alignment')
    aligned, target, variance, valid, coeffs, shifts, info = prepare_pair(
        clean_frames, noisy_frames, shift, block=block, align=True)
    gains = np.asarray([c[0] for c in coeffs])
    if not np.isfinite(coeffs).all() or np.any(gains <= 0):
        raise ValueError('Cross-ISO gain fit must be finite and positive')
    registered = np.stack([warp_packed(a, s) for a, s in zip(clean_frames, info['clean_shifts'])])
    factors = [interpolation_variance(s) for s in info['clean_shifts']]
    merged = robust_merge(registered, info['clean_nlf'], factors, block=block)
    valid &= (merged.effective_count >= max(2, int(np.ceil(.75*len(clean_frames))))).all(0)
    valid &= np.isfinite(target).all(0) & np.isfinite(variance).all(0)
    valid &= np.isfinite(aligned).all((0, 1))
    valid &= np.isfinite(registered).all((0, 1))
    if not valid.any():
        raise ValueError('No static finite supported pixels after cross-ISO matching')
    # Compare disjoint interleaved halves on the same registered grid and apply
    # the full burst's frozen photometric fit; do not refit each half separately.
    pixels = registered[:, :, valid]
    half_delta = pixels[::2].mean(0)-pixels[1::2].mean(0)
    offset = np.asarray([c[1] for c in coeffs])[:, None, None]
    base = gains[:, None, None]*np.nan_to_num(merged.clean)+offset
    local = np.divide(target, base, out=np.ones_like(target), where=np.abs(base) > 1e-8)
    half_delta *= (gains[:, None, None]*local)[:, valid]
    qa = {**info, 'static_fraction': float(valid.mean()), 'gain_per_channel': gains.tolist(),
          'gain_fit': coeffs, 'noisy_shifts': np.asarray(shifts).tolist(),
          'frame_counts': {'clean': len(clean_frames), 'noisy': len(noisy_frames)},
          'half_burst_frame_counts': [len(pixels[::2]), len(pixels[1::2])],
          'half_burst_target_difference': {
              'rmse': float(np.sqrt(np.mean(half_delta**2))),
              'rmse_per_channel': np.sqrt(np.mean(half_delta**2, axis=1)).tolist(),
              'bias_per_channel': half_delta.mean(1).tolist()},
          'clipping_threshold': .98,
          'clipping_fraction': {'clean': float((clean_frames >= .98).mean()),
                                'noisy': float((noisy_frames >= .98).mean()),
                                'matched_target': float((target[:, valid] >= .98).mean())},
          'target_noise_std': float(np.sqrt(variance[:, valid].mean())),
          'support_min': float(merged.effective_count[:, valid].min()),
          'purpose': 'development_evaluation',
          'geometry': 'normalized RGGB mosaic; registration in packed CFA planes',
          'target_independence': 'target uses only the lower-ISO burst',
          'half_burst_note': 'disjoint half means; shared full-burst registration and gain fit'}
    return aligned, target, valid, merged.effective_count, qa


def write_pairs(output, aligned, target, valid, support, noisy_records, qa):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'clean.npy', unpack(np.where(np.isfinite(target), target, 0)))
    np.save(output/'valid.npy', valid.repeat(2, 0).repeat(2, 1))
    np.save(output/'support.npy', support)
    entries = []
    for i, (a, record) in enumerate(zip(aligned, noisy_records)):
        name = f'noisy_{i:03d}.npy'
        np.save(output/name, unpack(np.where(np.isfinite(a), a, 0)))
        entries.append({'noisy': str((output/name).resolve()),
                        'clean': str((output/'clean.npy').resolve()), 'iso': int(record['iso']),
                        'valid_mask': str((output/'valid.npy').resolve()),
                        'support': str((output/'support.npy').resolve()),
                        'source': record['path'], 'split': 'development_evaluation'})
    (output/'pairs.json').write_text(json.dumps(entries, indent=2, allow_nan=False))
    (output/'qa.json').write_text(json.dumps(qa, indent=2, allow_nan=False))
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true', help='Only group metadata; no decoding or writes')
    p.add_argument('--low-group', type=int)
    p.add_argument('--high-group', type=int)
    p.add_argument('--max-gap', type=float, default=30.)
    p.add_argument('--block', type=int, default=16, help='Packed-pixel mask block size')
    p.add_argument('--shift', type=float, nargs=2, default=(0., 0.), metavar=('DY', 'DX'))
    p.add_argument('--min-static', type=float, default=.1, help='Reject pairs below this usable fraction')
    args = p.parse_args()
    if not args.folder.is_dir() or args.max_gap <= 0 or args.block < 1 or not 0 <= args.min_static <= 1:
        p.error('Need a capture directory, positive max-gap/block and min-static in [0, 1]')
    groups = discover_groups(args.folder, args.max_gap)
    report = group_report(groups)
    print(json.dumps({'groups': report}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.output is None:
        p.error('--output is required unless --dry-run')
    if args.output.resolve().is_relative_to(args.folder.resolve()):
        p.error('output must be outside the read-only capture folder')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output directory must be empty to avoid overwriting prepared pairs')
    clean, noisy = select_groups(groups, args.low_group, args.high_group)
    load = lambda records: np.stack([pack(read_mosaic(r['path'])) for r in records])
    aligned, target, valid, support, qa = prepare_arrays(load(clean), load(noisy), args.block, args.shift)
    qa.update({'groups': report, 'clean_sources': [r['path'] for r in clean],
               'noisy_sources': [r['path'] for r in noisy], 'min_static': args.min_static})
    if qa['static_fraction'] < args.min_static:
        print(json.dumps({'rejected': qa}, indent=2, allow_nan=False))
        raise ValueError('Static fraction below --min-static; no pairs written')
    write_pairs(args.output, aligned, target, valid, support, noisy, qa)
    print(json.dumps({'pairs': len(noisy), 'static_fraction': qa['static_fraction'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
