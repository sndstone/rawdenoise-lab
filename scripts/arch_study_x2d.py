#!/usr/bin/env python3
"""Clean-room real-RAW architecture diagnostics; no reference outputs are loaded.

Coordinates: inference/seams are packed RGGB pixels; ROIs are RGB (x,y,size).
Region predictions contain only that region (origin is recorded in summary.json).
Rendering uses the full original RAW template once per variant, with only the
inferred rectangle replaced. No cropped RAW / altered CFA metadata is used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw

from denoise_lab.arch_registry import build_model
from denoise_lab.bayer import PHASES, SITES, normalize_raw, pack, unpack
from denoise_lab.diffusion import BridgeRefiner, UnrolledRefiner
from denoise_lab.model import gamma
from denoise_lab.tiling import reflected_bayer_pad, tiled_inference

ROOT = Path(__file__).resolve().parents[1]
TILING = dict(tile_size=512, overlap=58, feather_power=2.2, alpha=1.)
CHANNELS = ('R', 'G1', 'B', 'G2')


def block_mean_diagnostics(source, prediction, block=64):
    """Discard incomplete edge blocks; dark/rest classification is per channel."""
    source, prediction = np.asarray(source), np.asarray(prediction)
    if source.shape != prediction.shape or source.ndim != 3 or source.shape[0] != 4:
        raise ValueError('Expected matching 4xHxW arrays')
    h, w = source.shape[-2:]
    h, w = h//block*block, w//block*block
    if not h or not w:
        raise ValueError('Region must contain at least one complete block')
    def means(a):
        return a[:, :h, :w].reshape(4, h//block, block, w//block, block).mean((2, 4), dtype=np.float64)
    levels, delta = means(source), means(prediction)-means(source)
    result = {'block_size_packed': block, 'discarded_edge_packed': [source.shape[1]-h, source.shape[2]-w]}
    for label, mask in [('dark', levels < .01), ('rest', levels >= .01)]:
        result[label] = {}
        for c, channel in enumerate(CHANNELS):
            d = delta[c][mask[c]]
            result[label][channel] = dict(count=int(d.size), mean_abs=float(np.abs(d).mean()) if d.size else None,
                median_abs=float(np.median(np.abs(d))) if d.size else None,
                p99_abs=float(np.quantile(np.abs(d), .99)) if d.size else None,
                signed_mean=float(d.mean()) if d.size else None)
    return result


def robust_std(a):
    a = np.asarray(a)
    return float(np.median(np.abs(a-np.median(a)))/.6744897501960817)


def residual_diagnostics(source, prediction):
    result = {}
    for c, channel in enumerate(CHANNELS):
        residual = source[c]-prediction[c]
        scale = robust_std(residual)
        result[channel] = dict(std=float(residual.std(dtype=np.float64)), robust_std=scale,
            fraction_abs_gt_5_robust_std=float((np.abs(residual) > 5*scale).mean()))
    return result


def tile_count(h, w, tile_size=512, overlap=58):
    step = tile_size-overlap
    return math.prod(1+max(0, math.ceil((n+2*overlap-tile_size)/step)) for n in (h, w))


def boundary_mask(h, w, tile_size=512, overlap=58, shift=128, radius=8):
    """Union of tile start/end strips for both grids, including overlap edges."""
    mask = np.zeros((h, w), dtype=bool)
    step = tile_size-overlap
    for offset in (0, shift):
        for axis, n in enumerate((h, w)):
            for start in range(0, n+2*offset+2*overlap, step):
                for edge in (start-overlap-offset, start+tile_size-overlap-offset):
                    lo, hi = max(0, edge-radius), min(n, edge+radius)
                    if hi > lo:
                        if axis == 0:
                            mask[lo:hi, :] = True
                        else:
                            mask[:, lo:hi] = True
    return mask


@torch.inference_mode()
def shifted_grid_seam(model, image, shift=128, radius=8, **tiling):
    if not tiling:
        tiling = TILING
    h, w = image.shape[-2:]
    baseline = tiled_inference(model, image, **tiling)
    padded = reflected_bayer_pad(image, shift, shift, shift, shift)
    shifted = tiled_inference(model, padded, **tiling)[..., shift:shift+h, shift:shift+w]
    delta = (baseline-shifted).abs().mean((0, 1)).cpu().numpy()
    mask = boundary_mask(h, w, tiling.get('tile_size', 512), tiling.get('overlap', 58), shift, radius)
    near = float(delta[mask].mean()) if mask.any() else None
    away = float(delta[~mask].mean()) if (~mask).any() else None
    return dict(max_abs_diff=float((baseline-shifted).abs().max().item()),
                mean_abs_diff=float(delta.mean()), boundary_mean_abs_diff=near,
                elsewhere_mean_abs_diff=away,
                boundary_excess=near/away if away and near is not None else None,
                shift_packed=[shift, shift], boundary_radius_packed=radius,
                boundary_definition='union of starts and ends of tiles in both grids',
                zero_elsewhere_error=away == 0)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def roi_bounds(roi):
    x, y, size = roi
    return y//2, (y+size+1)//2, x//2, (x+size+1)//2


def contained(bounds, region):
    y0, y1, x0, x1 = bounds
    a, b, c, d = region
    return a <= y0 < y1 <= b and c <= x0 < x1 <= d


@torch.inference_mode()
def stochastic_rois(model, source, region, rois, name, args, device):
    result = {}
    for label, roi in rois.items():
        y0, y1, x0, x1 = roi_bounds(roi)
        a, b, c, d = region
        bounds = (max(a, y0-64), min(b, y1+64), max(c, x0-64), min(d, x1+64))
        v0, v1, u0, u1 = bounds
        crop = torch.from_numpy(source[:, v0-a:v1-a, u0-c:u1-c].copy())[None].to(device)
        # Online Welford moments avoid keeping seed-sized image stacks in memory.
        mean = variance = None
        start = time.monotonic()
        for seed in range(args.bridge_seeds):
            # MPS does not support a native Generator. Generate the explicit
            # epsilon fields on CPU with a fixed Generator and transfer them.
            generator = torch.Generator(device='cpu' if device.type == 'mps' else device).manual_seed(seed)
            if device.type == 'mps':
                fields = torch.randn((args.bridge_stochastic_steps, *crop.shape), generator=generator).to(device)
                sample = model(crop, steps=args.bridge_stochastic_steps, stochastic=True, eps_fields=fields)
                del fields
            else:
                sample = model(crop, steps=args.bridge_stochastic_steps, stochastic=True, generator=generator)
            value = gamma(sample, model.gamma_exponent)[0].cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite stochastic output')
            if mean is None:
                mean, variance = value.copy(), np.zeros_like(value)
            else:
                delta = value-mean
                mean += delta/(seed+1)
                variance += delta*(value-mean)
        std = np.sqrt(np.maximum(variance/(args.bridge_seeds-1), 0))
        prefix = f'{label}_{name}_stochastic_K{args.bridge_stochastic_steps}'
        np.save(args.out_dir/f'{prefix}_mean_gamma.npy', mean.astype(np.float16), allow_pickle=False)
        np.save(args.out_dir/f'{prefix}_std_gamma.npy', std.astype(np.float16), allow_pickle=False)
        green = std[[1, 3]].mean(0)
        p99 = float(np.quantile(green, .99))
        Image.fromarray((np.clip(green/max(p99, 1e-12), 0, 1)*255).astype(np.uint8)).save(args.out_dir/f'{prefix}_std.png')
        input_g = gamma(crop, model.gamma_exponent)[0].cpu().numpy()
        # Blind high-pass noise proxy, same gamma units as the seed std; scene
        # edges can inflate it. This is not a ground-truth noise estimate.
        noise = [robust_std(np.diff(plane, axis=1))/math.sqrt(2) for plane in input_g]
        average = [float(plane.mean()) for plane in std]
        result[label] = dict(bounds_packed=list(bounds), margin_packed_requested=64,
            seeds=list(range(args.bridge_seeds)), steps=args.bridge_stochastic_steps,
            mean_std_gamma=average, input_noise_std_gamma=noise,
            mean_std_over_input_noise_std=[s/n if n else None for s, n in zip(average, noise)],
            input_noise_estimator='horizontal first-difference MAD / (0.67448975 * sqrt(2)); scene structure may inflate',
            std_ddof=1, green_png_p99=p99, files_prefix=prefix, seconds=time.monotonic()-start)
        del crop, mean, variance, std, value, sample, input_g
        gc.collect()
    return result


def render(raw_path, predictions, region, rois, output):
    import rawpy
    settings = dict(use_camera_wb=True, no_auto_bright=True, no_auto_scale=False, output_bps=16,
        adjust_maximum_thr=0., demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        output_color=rawpy.ColorSpace.sRGB, median_filter_passes=0,
        fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off, exp_shift=4.,
        exp_preserve_highlights=0., bright=1., highlight_mode=rawpy.HighlightMode.Clip)
    timings = {}
    for name, path in [('before', None), *predictions.items()]:
        start = time.monotonic()
        with rawpy.imread(str(raw_path)) as raw:
            if path is not None:
                prediction = np.load(path, mmap_mode='r', allow_pickle=False)
                desc = raw.color_desc.decode('ascii')
                phase = ''.join(desc[int(c)] for c in raw.raw_colors_visible[:2, :2].ravel())
                dy, dx = PHASES[phase]
                y0, y1, x0, x1 = region
                view = raw.raw_image_visible[dy+2*y0:dy+2*y1, dx+2*x0:dx+2*x1]
                colors = raw.raw_colors_visible[dy:dy+2, dx:dx+2]
                black = raw.black_level_per_channel
                # Exactly the reference template settings and scalar white level.
                for sy, sx in SITES:
                    c = int(colors[sy, sx])
                    codes = prediction[sy::2, sx::2].astype(np.float32)*(raw.white_level-black[c])+black[c]
                    view[sy::2, sx::2] = np.clip(np.rint(codes), 0, 65535).astype(np.uint16)
                    del codes
                del prediction, view, colors
            rgb = raw.postprocess(**settings)
        for label, (x, y, size) in rois.items():
            crop = (rgb[y:y+size, x:x+size]//257).astype(np.uint8)
            if crop.shape[:2] != (size, size):
                raise ValueError(f'ROI {label} outside rendered RGB')
            Image.fromarray(crop).save(output/f'{label}_{name}.png')
        del rgb
        gc.collect()
        timings[name] = time.monotonic()-start
        print(f'RENDER {name}: {timings[name]:.2f}s', flush=True)
    for label, (_, _, size) in rois.items():
        names = ['before', *predictions]
        sheet = Image.new('RGB', (size*len(names), size+32), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate(names):
            draw.text((i*size+8, 8), name, fill='black')
            with Image.open(output/f'{label}_{name}.png') as image:
                sheet.paste(image, (i*size, 32))
        sheet.save(output/f'sheet_{label}.png')
        del sheet
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--rois', type=Path, default=ROOT/'runs/arch_study/protocol_addendum_x2d.json')
    parser.add_argument('--bridge-steps', default='1,4,16')
    parser.add_argument('--bridge-seeds', type=int, default=4)
    parser.add_argument('--bridge-stochastic-steps', type=int, default=4)
    parser.add_argument('--region', help='Packed Y0:Y1:X0:X1')
    parser.add_argument('--skip-render', action='store_true')
    args = parser.parse_args()
    steps = [int(k) for k in args.bridge_steps.split(',')]
    if min(steps) < 1 or len(set(steps)) != len(steps) or args.bridge_seeds < 2 or args.bridge_stochastic_steps < 1:
        parser.error('Steps must be positive/unique and bridge-seeds >= 2')
    checkpoints = {}
    for spec in args.checkpoint:
        name, sep, path = spec.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name == 'before' or name in checkpoints:
            parser.error('Checkpoints need unique safe names (excluding before): NAME=PATH')
        checkpoints[name] = Path(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    import rawpy
    with rawpy.imread(str(args.raw)) as raw:
        mosaic = normalize_raw(raw)
    source = pack(mosaic)
    del mosaic
    full_shape = source.shape
    region = tuple(map(int, args.region.split(':'))) if args.region else (0, source.shape[1], 0, source.shape[2])
    if len(region) != 4 or not contained(region, (0, source.shape[1], 0, source.shape[2])):
        parser.error('Invalid packed region')
    y0, y1, x0, x1 = region
    source = source[:, y0:y1, x0:x1].copy()
    if min(source.shape[1:]) < 64:
        parser.error('Region must be at least 64 packed pixels on each side')
    roi_data = json.loads(args.rois.read_text())
    all_rois = roi_data.get('rois_rgb_xy_size', roi_data)
    rois = {}
    for label, roi in all_rois.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
            parser.error('Unsafe ROI label')
        if contained(roi_bounds(roi), region):
            rois[label] = roi
        else:
            warnings.warn(f'Skipping ROI {label}: outside inference region')
    # Fixed top-left interior location; clamp for small smoke regions.
    sh, sw = min(1100, source.shape[1]), min(1300, source.shape[2])
    sy, sx = min(128, source.shape[1]-sh), min(128, source.shape[2]-sw)
    seam_bounds = [y0+sy, y0+sy+sh, x0+sx, x0+sx+sw]
    summary = dict(source=str(args.raw.resolve()), source_sha256=digest(args.raw),
        script_sha256=digest(__file__), full_packed_shape=list(full_shape), region_packed=list(region),
        rois=rois, requested_rois=all_rois, tiling=TILING, device=str(device),
        seam_region_packed=seam_bounds, checkpoints={}, variants={}, stochastic={},
        prediction_format='float16 unpacked normalized mosaic of region_packed',
        rendering='Original full RAW template, one postprocess per variant, exact render_comparison.py settings; +4 exp_shift; only inference region replaced',
        clean_room=True, quality_scope='Development cross-camera stress diagnostic; no ground truth quality claim')
    predictions = {}
    for name, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = build_model(saved['model_config']).to(device).eval()
        model.load_state_dict(saved['model'])
        summary['checkpoints'][name] = dict(path=str(path.resolve()), sha256=digest(path), model_config=saved['model_config'])
        del saved
        if getattr(model, 'cond', 'none') != 'none':
            raise ValueError('Architecture pilot expects unconditioned models')
        variants = [(f'{name}_K{k}', UnrolledRefiner(model, k)) for k in steps] if isinstance(model, BridgeRefiner) else [(name, model)]
        for variant, wrapper in variants:
            if variant in predictions or variant in checkpoints and variant != name:
                raise ValueError(f'Variant name collision: {variant}')
            image = torch.from_numpy(source)[None].to(device)
            synchronize(device)
            start = time.monotonic()
            output = tiled_inference(wrapper, image, **TILING)
            synchronize(device)
            seconds = time.monotonic()-start
            prediction = output[0].cpu().numpy()
            if not np.isfinite(prediction).all() or np.abs(prediction).max() > np.finfo(np.float16).max:
                raise ValueError(f'Nonfinite/float16-overflow prediction: {variant}')
            target = args.out_dir/f'{variant}_normalized.npy'
            np.save(target, unpack(prediction).astype(np.float16), allow_pickle=False)
            predictions[variant] = target
            diagnostics = dict(seconds=seconds, tiles=tile_count(*source.shape[-2:]),
                tiles_per_second=tile_count(*source.shape[-2:])/seconds,
                prediction=str(target), block_mean=block_mean_diagnostics(source, prediction),
                residual=residual_diagnostics(source, prediction))
            del output, prediction, image
            seam_image = torch.from_numpy(source[:, sy:sy+sh, sx:sx+sw].copy())[None].to(device)
            diagnostics['seam'] = shifted_grid_seam(wrapper, seam_image, **TILING)
            summary['variants'][variant] = diagnostics
            del seam_image
            print(f'INFERENCE {variant}: {seconds:.2f}s ({diagnostics["tiles_per_second"]:.3f} tiles/s)', flush=True)
            gc.collect()
        if isinstance(model, BridgeRefiner):
            summary['stochastic'][name] = stochastic_rois(model, source, region, rois, name, args, device)
        del variants, wrapper, model
        gc.collect()
    del source
    gc.collect()
    summary['render_seconds'] = {} if args.skip_render or not rois else render(args.raw, predictions, region, rois, args.out_dir)
    (args.out_dir/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['# X2D architecture diagnostics', '',
        'Development stress case; no ground-truth quality scores. Block deltas and residuals are linear sensor units.', '',
        '| Variant | Seconds | Tiles/s | Seam mean abs | Seam max abs | Boundary excess |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, d in summary['variants'].items():
        s = d['seam']
        ratio = f'{s["boundary_excess"]:.4g}' if s['boundary_excess'] is not None else 'undefined'
        lines.append(f'| {name} | {d["seconds"]:.2f} | {d["tiles_per_second"]:.3f} | {s["mean_abs_diff"]:.6g} | {s["max_abs_diff"]:.6g} | {ratio} |')
    lines += ['', 'Per-channel dark/rest block means, residual statistics, stochastic seed maps, hashes, coordinates and rendering timings: `summary.json`.',
        'Stochastic maps are packed signed-gamma-domain sample means/std (ddof=1); PNG green std is scaled to p99. Noise denominator is a blind first-difference MAD proxy.',
        'Boundary strips cover starts and ends of tiles in both grids. Undefined ratios have zero elsewhere error or no elsewhere pixels.', '']
    (args.out_dir/'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
