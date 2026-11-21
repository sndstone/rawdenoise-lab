#!/usr/bin/env python3
"""Core ML latency and flat-field identity using the selected production runtime.

Repeat --compute-units to compare runtime configurations. --legacy-tiling
reproduces the previous gate's overlap 58 and torch blending implementation.
Identity isolates model/tiling errors with prepass and highlight protection off.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from denoise_lab.arch_registry import build_model
from denoise_lab.inference import (Denoiser, COREML_COMPUTE_UNITS, coreml_compute_units,
                                  DEFAULT_TILE_SIZE, DEFAULT_OVERLAP, DEFAULT_GRID_ALIGNMENT,
                                  DEFAULT_FEATHER_MODE, DEFAULT_TTA_SHIFT, DEFAULT_FEATHER_POWER)
from denoise_lab.tiling import tiled_inference

spec = importlib.util.spec_from_file_location('ident', Path(__file__).with_name('evaluate_identity.py'))
ident = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ident)


def tiling_settings(legacy=False):
    return dict(tile_size=DEFAULT_TILE_SIZE, overlap=58 if legacy else DEFAULT_OVERLAP,
                grid_alignment=DEFAULT_GRID_ALIGNMENT, feather_mode=DEFAULT_FEATHER_MODE,
                tta_shift=DEFAULT_TTA_SHIFT)


def argument_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', action='append', metavar='NAME=PATH')
    source.add_argument('--package', action='append', metavar='NAME=PATH',
                        help='Evaluate an existing package without reconversion')
    ap.add_argument('--compute-units', action='append', choices=COREML_COMPUTE_UNITS,
                    help='Repeat for side-by-side results; default: all')
    ap.add_argument('--legacy-tiling', action='store_true')
    ap.add_argument('--frame-size', type=int, nargs=2, default=(1100, 1500))
    ap.add_argument('--out', type=Path, required=True)
    return ap


def package_identity(path, frame_size=(1100, 1500), compute_units=None, legacy_tiling=False):
    """Apply existing thresholds to an artifact; time the same loaded runtime."""
    model = Denoiser.load(path, backend='coreml', compute_units=compute_units)
    settings = tiling_settings(legacy_tiling)
    tile = np.random.default_rng(20261109).normal(.05, .004, (1, 4, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)).astype(np.float32)
    for _ in range(5):
        model._call(tile)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        output = model._call(tile)
        times.append((time.perf_counter()-start)*1000)
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite benchmark output')
    rows = []
    with torch.inference_mode():
        for kind in ident.KINDS:
            for level in ident.LEVELS:
                image = ident.flat_frame(level, kind, *frame_size, 'cpu')
                if legacy_tiling:
                    def run(t):
                        return torch.from_numpy(model._call(t.numpy()))
                    pred = tiled_inference(run, image, tile_size=settings['tile_size'],
                                           overlap=settings['overlap'], feather_power=DEFAULT_FEATHER_POWER,
                                           grid_alignment=settings['grid_alignment'],
                                           feather_mode=settings['feather_mode'])
                else:
                    pred = torch.from_numpy(model.denoise_packed(
                        image[0].numpy(), prepass=False, highlights='off', **settings))[None]
                rows.append(dict(kind=kind, level=level, **ident.error_metrics(pred-image, level)))
    return dict(package=str(path), compute_units=model.compute_units, tiling=settings,
                legacy_tiling=legacy_tiling, prepass=False, highlights='off',
                median_ms=statistics.median(times), samples_ms=times,
                rows=rows, passed=all(row['passed'] for row in rows),
                max_abs_error=max(row['max_abs_error'] for row in rows))


def main():
    ap = argument_parser()
    a = ap.parse_args()
    if min(a.frame_size) < 1:
        ap.error('--frame-size dimensions must be positive')
    items = a.package or a.checkpoint
    if any('=' not in item for item in items):
        ap.error('Model arguments require NAME=PATH')
    names = [item.split('=', 1)[0] for item in items]
    if len(set(names)) != len(names):
        ap.error('Model labels must be unique')
    torch.set_num_threads(2)
    units = list(dict.fromkeys(a.compute_units or [coreml_compute_units()]))
    report = dict(precision='fp16', compute_units=units, frame_size=a.frame_size,
                  tiling=tiling_settings(a.legacy_tiling), legacy_tiling=a.legacy_tiling,
                  thresholds=dict(abs=ident.THRESHOLD, rel=ident.RELATIVE_THRESHOLD,
                                  bias=ident.BIAS_THRESHOLD), models={})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for item in items:
        name, path = item.split('=', 1)
        results = report['models'][name] = {}
        with tempfile.TemporaryDirectory(prefix='coreml_gate_') as scratch:
            package = path
            ref = x = None
            if a.checkpoint:
                import coremltools as ct
                saved = torch.load(path, map_location='cpu', weights_only=True)
                model = build_model(saved['model_config']).eval()
                model.load_state_dict(saved['model'])
                generator = torch.Generator().manual_seed(20261109)
                x = torch.rand(1, 4, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, generator=generator)*.05+.005
                x += torch.randn(x.shape, generator=generator)*.004
                with torch.no_grad():
                    traced = torch.jit.trace(model, x, check_trace=False)
                    ref = model(x).numpy()
                ml = ct.convert(traced, inputs=[ct.TensorType(name='inp', shape=x.shape)],
                                convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
                                minimum_deployment_target=ct.target.macOS13, skip_model_load=True)
                package = str(Path(scratch)/'model.mlpackage')
                ml.save(package)
            for unit in units:
                result = package_identity(package, a.frame_size, unit, a.legacy_tiling)
                if ref is not None:
                    runtime = Denoiser.load(package, backend='coreml', compute_units=unit)
                    result.update(checkpoint=path, step=saved.get('step'),
                                  fp16_vs_fp32_max_abs=float(np.abs(runtime._call(x.numpy())-ref).max()))
                    result.pop('package')
                results[unit] = result
                print(name, unit, result['passed'], flush=True)
                a.out.write_text(json.dumps(report, indent=1, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
