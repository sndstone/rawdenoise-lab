#!/usr/bin/env python3
"""Development crops with measured camera noise; diagnostic, not independent acceptance.

PSNR uses peak 1 in exposed packed RAW, as in DEV v2 (no exposure division).
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from denoise_lab.arch_registry import build_model
from denoise_lab.bayer import pack, read_mosaic
from denoise_lab.camera_noise import CameraNoise
from denoise_lab.checkpoints import load_checkpoint, model_config, resolve_weights
from denoise_lab.metrics import psnr
from denoise_lab.noise import noise_std

PHYSICS = dict(banding=True, quantization=True, horizontal_correlation=True,
               vertical_correlation=True, moment_read=True)
PROFILES = [('x1d2', 'leaf', [100, 3200, 25600], False),
            ('x1d2_electronic', 'electronic', [100, 800, 3200], False),
            ('a7m3', 'mechanical', [400, 800, 6400, 25600], False),
            ('a7m3', 'mechanical', [65535], True)]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def seed_for(seed, *parts):
    payload = json.dumps([seed, *parts], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


def profiles(calibration):
    """Require an exact measured ISO/shutter; CameraNoise.at otherwise clamps/interpolates."""
    result = []
    for camera, shutter, isos, processed in PROFILES:
        cam = calibration.get('cameras', {}).get(camera, {})
        selected = cam.get('shutters', {}).get(shutter)
        if selected is None and cam.get('shutter') == shutter:
            selected = cam
        available = {int(k): v for k, v in (selected or {}).get('iso', {}).items()}
        processed_isos = set((selected or {}).get('processed_isos', cam.get('processed_isos', [])))
        for iso in isos:
            reason = None
            if selected is None:
                reason = 'missing calibrated camera/shutter'
            elif iso not in available:
                reason = 'missing exact calibrated ISO'
            elif (iso in processed_isos) != processed:
                reason = 'calibrated processing mode does not match'
            elif not available[iso].get('physics'):
                reason = 'missing measured physics parameters'
            row = dict(camera=camera, shutter=shutter, iso=iso, processed=processed,
                       profile=f'{camera}/{shutter}/{iso}'+('/processed' if processed else ''),
                       status='SKIP' if reason else 'OK')
            if reason:
                row['reason'] = reason
            result.append(row)
    return result


def paired_deltas(candidate, baseline):
    """Pair whole source means; repeated crops never become independent scenes."""
    def index(rows):
        return {(r['slice'], r['source_sha256']): r for r in rows}
    c, b = index(candidate), index(baseline)
    rows = []
    for key in sorted(c.keys() | b.keys()):
        cv, bv = c.get(key, {}).get('model_psnr'), b.get(key, {}).get('model_psnr')
        rows.append(dict(slice=key[0], source_sha256=key[1], candidate=cv, baseline=bv,
                         delta_db=float(cv-bv) if cv is not None and bv is not None else None))
    return rows


def evaluate(args):
    started = time.monotonic()
    cfg = args.config
    size = int(cfg.get('crop_size', 256))
    count = 1 if args.quick else int(cfg.get('crops_per_source', 3))
    limit = cfg.get('quick_sources', 1) if args.quick else cfg.get('max_sources')
    exposures = cfg.get('exposures', [1., 1/16, 1/64])
    crop_seed, noise_seed = cfg.get('crop_seed', 20261109), cfg.get('noise_seed', 20261110)
    fixed_seed = cfg.get('fixed_pattern_seed', 20261111)
    if fixed_seed == 20261104:
        raise ValueError('measured evaluation fixed-pattern seed must differ from training')
    if size <= 0 or count <= 0 or (limit is not None and limit <= 0):
        raise ValueError('crop size, crop count and source limit must be positive')
    if not exposures or len(set(exposures)) != len(exposures) or any(not 0 < e <= 1 for e in exposures):
        raise ValueError('exposures must be distinct and in (0, 1]')
    calibration_path = Path(cfg['calibration'])
    calibration = yaml.safe_load(calibration_path.read_text())
    requested = profiles(calibration)
    physics = {**PHYSICS, 'fixed_seed': fixed_seed}
    noises = {}
    for p in requested:
        if p['status'] != 'OK' or p['camera'] in noises:
            continue
        shutters = calibration['cameras'][p['camera']].get('shutters', {})
        modes = {p['camera']: p['shutter']} if p['shutter'] in shutters else None
        noises[p['camera']] = CameraNoise(calibration_path, {p['camera']: 1}, K_jitter_log=0,
            read_jitter_log=0, physics_options=physics, shutter_modes=modes)
    for p in requested:
        if p['status'] == 'OK':
            p['parameters'] = noises[p['camera']].at(p['camera'], p['iso'], processed=p['processed'])
    # Validate the complete manifests before quick truncation or checkpoint loading.
    datasets = []
    train_keys, val_keys = defaultdict(set), defaultdict(set)
    for dataset in cfg.get('datasets', []):
        manifest, train_manifest = Path(dataset['manifest']), Path(dataset['train_manifest'])
        validation, train = json.loads(manifest.read_text()), json.loads(train_manifest.read_text())
        for key in ('source_sha256', 'sensor_sha256', 'group'):
            train_keys[key].update(e[key] for e in train)
            val_keys[key].update(e[key] for e in validation)
            if train_keys[key] & val_keys[key]:
                raise ValueError(f'Train/validation overlap: {key}')
        grouped = defaultdict(list)
        for e in validation:
            grouped[e['source_sha256']].append(e)
        if not grouped:
            raise ValueError('empty validation manifest')
        datasets.append((dataset, grouped, dict(name=dataset['name'], manifest_sha256=sha(manifest),
                         train_manifest_sha256=sha(train_manifest), available_sources=len(grouped))))
    if not datasets:
        raise ValueError('measured suite requires datasets')
    models, provenance = {}, {}
    for item in args.checkpoint:
        name, path = item.split('=', 1)
        if name in models:
            raise ValueError('duplicate checkpoint label: '+name)
        saved = load_checkpoint(path)
        model = build_model(model_config(saved)).to(args.device).eval()
        model.load_state_dict(resolve_weights(saved, args.weights))
        models[name] = model
        provenance[name] = dict(sha256=sha(path), weights=args.weights, model_config=saved['model_config'])
    rows, source_scores, summary = [], [], []
    root = Path(cfg.get('root', 'data'))
    for dataset, grouped, _ in datasets:
        selected_profiles = [p for p in requested if p['camera'] in dataset['cameras']]
        if not selected_profiles:
            raise ValueError('dataset has no requested measured profiles: '+dataset['name'])
        # Read each selected clean once; reuse its deterministic crops across every profile/exposure/model.
        for source, entries in sorted(grouped.items())[:limit]:
            entries = sorted(entries, key=lambda e: e['clean'])
            cache = {}
            for crop in range(count):
                cs = seed_for(crop_seed, dataset['name'], source, crop)
                rng = np.random.default_rng(cs)
                entry = entries[int(rng.integers(len(entries)))]
                filename = entry['clean']
                if filename not in cache:
                    cache[filename] = read_mosaic(root/filename)
                clean = cache[filename]
                if min(clean.shape) < size*2:
                    raise ValueError(f'clean reference too small for {size}-packed crop')
                y, x = [2*int(rng.integers((extent-size*2)//2+1)) for extent in clean.shape]
                clean = clean[y:y+2*size, x:x+2*size]
                if not np.isfinite(clean).all():
                    raise ValueError('nonfinite clean reference')
                for p in selected_profiles:
                    if p['status'] != 'OK':
                        continue
                    for exposure in exposures:
                        cap = cfg.get('gain_cap_iso', {'x1d2': 6400, 'x1d2_electronic': 6400}).get(p['camera'])
                        actual = exposure*min(1., cap/p['iso']) if cap else exposure
                        target = pack((clean*actual).astype(np.float32))
                        ns = seed_for(noise_seed, dataset['name'], source, crop, p['profile'], exposure)
                        noisy = pack(noises[p['camera']].apply(clean*actual, p['parameters'], np.random.default_rng(ns)))
                        t = torch.from_numpy(target.copy())[None].to(args.device)
                        inp = torch.from_numpy(noisy.copy())[None].to(args.device)
                        slice_name = f"{dataset['name']}/{p['profile']}/exposure={exposure:g}"
                        common = dict(slice=slice_name, source_sha256=source, clean=filename, crop=crop,
                            crop_yx=[y//2, x//2], crop_seed=cs, noise_seed=ns, profile=p['profile'],
                            exposure=exposure, actual_exposure=actual, input_psnr=psnr(inp, t))
                        for name, model in models.items():
                            cond = None
                            if model.cond == 'noise_map':
                                std = noise_std(p['parameters'], np.maximum(noisy, 0).mean(0, keepdims=True))
                                cond = torch.from_numpy(std)[None].to(args.device)
                            with torch.inference_mode():
                                pred = model(inp, cond)
                            if not torch.isfinite(pred).all():
                                raise ValueError('nonfinite measured output: '+name)
                            rows.append({**common, 'model': name, 'model_psnr': psnr(pred, t)})
        for p in selected_profiles:
            for exposure in exposures:
                s = f"{dataset['name']}/{p['profile']}/exposure={exposure:g}"
                for name in models:
                    rs = [r for r in rows if r['slice'] == s and r['model'] == name]
                    by_source = defaultdict(list)
                    for r in rs:
                        by_source[r['source_sha256']].append(r)
                    for source, values in sorted(by_source.items()):
                        source_scores.append(dict(slice=s, source_sha256=source, model=name,
                            model_psnr=float(np.mean([v['model_psnr'] for v in values])), crops=len(values)))
                    values = [r['model_psnr'] for r in source_scores if r['slice'] == s and r['model'] == name]
                    summary.append(dict(slice=s, model=name, model_psnr=float(np.mean(values)) if values else None,
                        sources=len(values), crops=len(rs), status=p['status'], reason=p.get('reason')))
    baseline = next(iter(models))
    comparisons = {name: paired_deltas([r for r in source_scores if r['model'] == name],
                                      [r for r in source_scores if r['model'] == baseline])
                   for name in models if name != baseline}
    result = dict(protocol='measured_dev_v1', weights=args.weights, quick=args.quick,
        metric='source-mean crop PSNR, peak 1, exposed packed RAW; no exposure normalization',
        calibration_sha256=sha(calibration_path), script_sha256=sha(__file__), profiles=requested,
        physics_options=physics, K_jitter_log=0, read_jitter_log=0, crop_seed=crop_seed, noise_seed=noise_seed,
        fixed_pattern_seed=fixed_seed, crop_packed=size, crops_per_source=count, exposures=exposures,
        datasets=[d[2] for d in datasets], checkpoints=provenance, summary=summary, rows=rows,
        source_scores=source_scores, source_paired_deltas=comparisons, seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True, metavar='NAME=PATH')
    p.add_argument('--config', type=Path, default=Path('configs/scoreboard/round9.yaml'))
    p.add_argument('--weights', choices=['model', 'ema'], default='model')
    p.add_argument('--device', default='cpu')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.config = yaml.safe_load(args.config.read_text())['suites']['measured']
    torch.set_num_threads(2)
    result = evaluate(args)
    for r in result['summary']:
        print(r['slice'], r['model'], r['model_psnr'] if r['status'] == 'OK' else r['reason'])


if __name__ == '__main__':
    main()
