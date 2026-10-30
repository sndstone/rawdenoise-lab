#!/usr/bin/env python3
"""Flat-field linear identity gate through the production reflected-apron tiler."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from denoise_lab.arch_registry import build_model
from denoise_lab.tiling import tiled_inference

LEVELS = [.001, .01, .05, .1, .18, .35, .5, .7, .9, 1.]
KINDS = {
    'equal': (1., 1., 1., 1.),
    'daylight': (.55, 1., .7, 1.),
    'tungsten': (.35, 1., 1.1, 1.),
    'ramp': (.55, 1., .7, 1.),
    'partial_clip': (.8, 1., .85, 1.),  # at level 1 the greens clip, R/B do not
}
THRESHOLD = 1e-3
# Pre-registered round-2 gate: all kinds gated (equal = clipped highlights at level 1).
RELATIVE_THRESHOLD = .05   # max abs error / level
BIAS_THRESHOLD = 2.5e-4    # |mean signed error|


def flat_frame(level, kind, height, width, device):
    ratios = torch.tensor(KINDS[kind], device=device).view(1, 4, 1, 1)
    image = torch.full((1, 4, height, width), level, device=device)*ratios
    if kind == 'ramp':
        image *= torch.linspace(.7, 1.3, width, device=device).view(1, 1, 1, width)
    return image


def error_metrics(error, level=None):
    # numpy avoids torch.quantile's element-count limit for full packed frames.
    values = error.detach().float().cpu().numpy().reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite identity output')
    absolute = np.abs(values)
    maximum = float(absolute.max())
    bias = float(values.mean(dtype=np.float64))
    passed = maximum <= THRESHOLD and abs(bias) <= BIAS_THRESHOLD
    if level is not None:
        passed = passed and maximum <= RELATIVE_THRESHOLD*level
    return dict(max_abs_error=maximum, mean_bias=bias, relative_max_error=None if level is None else maximum/level,
                p99_9_abs_error=float(np.quantile(absolute, .999)), passed=passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--weights', choices=['model', 'ema'], default='model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--frame-size', type=int, nargs=2, default=(1100, 1500), metavar=('H', 'W'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.frame_size) < 1:
        parser.error('--frame-size dimensions must be positive')
    torch.set_num_threads(2)
    rows, provenance = [], {}
    with torch.inference_mode():
        for spec in args.checkpoint:
            if '=' not in spec:
                parser.error('--checkpoint requires NAME=PATH')
            name, path = spec.split('=', 1)
            if name in provenance:
                raise ValueError('Duplicate checkpoint label: ' + name)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            key = 'ema_model' if args.weights == 'ema' else 'model'
            if key not in saved:
                raise ValueError(f'{path}: requested {args.weights} weights but checkpoint has no {key}')
            model = build_model(saved['model_config']).to(args.device).eval()
            model.load_state_dict(saved[key])
            provenance[name] = dict(checkpoint=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                    model_config=saved['model_config'], step=saved.get('step'), weights=args.weights)
            for kind in KINDS:
                for level in LEVELS:
                    image = flat_frame(level, kind, *args.frame_size, args.device)
                    noise_map = torch.zeros_like(image[:, :1]) if model.cond == 'noise_map' else None
                    prediction = tiled_inference(model, image, tile_size=512, overlap=58,
                                                 feather_power=2.2, noise_map=noise_map)
                    rows.append(dict(model=name, kind=kind, level=level, path='tiled',
                                     **error_metrics(prediction-image, level)))
                    tile = flat_frame(level, kind, 512, 512, args.device)
                    cond = torch.zeros_like(tile[:, :1]) if model.cond == 'noise_map' else None
                    prediction = model(tile, cond)
                    rows.append(dict(model=name, kind=kind, level=level, path='bare_interior',
                                     **error_metrics((prediction-tile)[..., 32:-32, 32:-32], level)))
                print(f'EVALUATED {name} {kind}', flush=True)
    summary = []
    for name in provenance:
        for kind in KINDS:
            for path in ('tiled', 'bare_interior'):
                group = [r for r in rows if (r['model'], r['kind'], r['path']) == (name, kind, path)]
                summary.append(dict(model=name, kind=kind, path=path,
                    max_abs_error=max(r['max_abs_error'] for r in group),
                    mean_bias=float(np.mean([r['mean_bias'] for r in group])),
                    p99_9_abs_error=max(r['p99_9_abs_error'] for r in group),
                    passed=all(r['passed'] for r in group)))
    output = dict(protocol='production_tiled_flat_identity', checkpoints=provenance,
                  frame_size=list(args.frame_size), tile_size=512, overlap=58, feather_power=2.2,
                  interior_border=32, levels=LEVELS, kinds=KINDS, threshold=THRESHOLD, relative_threshold=RELATIVE_THRESHOLD, bias_threshold=BIAS_THRESHOLD,
                  passed=all(r['passed'] for r in rows if r['path'] == 'tiled'),
                  summary_definition='Across levels: worst max and p99.9, arithmetic mean signed bias; pass requires every level.',
                  summary=summary, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    lines = ['Gate (tiled rows, every kind and level): max abs <= 1e-3, max abs <= 5% of level, |mean bias| <= 2.5e-4.',
             'Summary takes worst max/p99.9 across levels and arithmetic mean signed bias.', '',
             '| Model | Kind | Path | Max abs error | Mean bias | p99.9 abs error | Pass |',
             '|---|---|---|---:|---:|---:|---|']
    for row in summary:
        lines.append(f"| {row['model']} | {row['kind']} | {row['path']} | {row['max_abs_error']:.6g} | "
                     f"{row['mean_bias']:.6g} | {row['p99_9_abs_error']:.6g} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines += ['', '| Model | Kind | Level | Path | Max abs error | Mean bias | p99.9 abs error | Pass |',
              '|---|---|---:|---|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['kind']} | {row['level']:g} | {row['path']} | "
                     f"{row['max_abs_error']:.6g} | {row['mean_bias']:.6g} | {row['p99_9_abs_error']:.6g} | "
                     f"{'PASS' if row['passed'] else 'FAIL'} |")
    markdown = '\n'.join(lines)+'\n'
    args.out.with_suffix('.md').write_text(markdown)
    print(markdown)
    print('SAVED', args.out)


if __name__ == '__main__':
    main()
