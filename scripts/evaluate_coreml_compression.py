#!/usr/bin/env python3
"""Export and measure local Core ML compression variants; never promote without dev/gate evidence.

Uses deterministic, previously inspected development references, not independent
acceptance data. Output-to-fp16 PSNR uses peak=1 and is not denoising quality.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from denoise_lab.bayer import pack, read_mosaic
from denoise_lab.data import BootstrapDataset
from denoise_lab.inference import Denoiser, COREML_COMPUTE_UNITS, coreml_compute_units, DEFAULT_TILE_SIZE
from export_coreml import COMPRESSION_MODES
from coreml_identity_gate import package_identity, tiling_settings
from evaluate_dev_v2 import LEVELS


def size_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def errors(prediction, reference):
    delta = prediction.astype(np.float64)-reference.astype(np.float64)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite prediction or reference')
    mse = float(np.mean(delta**2))
    return dict(psnr_db=None if mse == 0 else float(-10*np.log10(mse)),
                identical=mse == 0, max_abs=float(np.abs(delta).max()))


def dev_samples(root, count):
    """One fixed 256-packed crop/source, cycled over exposure and noise levels."""
    manifest = root/'bootstrap_binned/x1d2_bin2_val_r4.json'
    if not manifest.exists():
        return [], 'Development manifest absent'
    entries = json.loads(manifest.read_text())
    train = json.loads((root/'bootstrap_binned/x1d2_bin2_train_r4.json').read_text())
    for key in ('source_sha256', 'sensor_sha256', 'group'):
        if {e[key] for e in entries} & {e[key] for e in train}:
            raise ValueError(f'Train/development overlap: {key}')
    selected = {}
    for e in sorted(entries, key=lambda e: (e['source_sha256'], e['clean'])):
        selected.setdefault(e['source_sha256'], e)
    # Multiple deterministic crops/source if fewer than count independent sources exist.
    pool = list(selected.values())
    if not pool:
        return [], 'Empty development manifest'
    samples = []
    for i in range(count):
        entry = pool[i % len(pool)]
        level = list(LEVELS)[i % 3]
        exposure = (1., 1/16, 1/64)[(i//3) % 3]
        seed = 20261104+i
        dataset = BootstrapDataset([entry], root=root, crop_size=256, augment=False,
                                   deterministic=True, seed=seed, fixed_parameters=LEVELS[level],
                                   exposure_scale=exposure)
        item = dataset[0]
        # Reconstruct random_crop coordinates without changing the dataset's RNG stream.
        mosaic = read_mosaic(root/entry['clean'])
        rng = np.random.default_rng(seed)
        y = 2*int(rng.integers((mosaic.shape[0]-512)//2+1))
        x = 2*int(rng.integers((mosaic.shape[1]-512)//2+1))
        samples.append(dict(noisy=item['noisy'].numpy(), clean=item['clean'].numpy(),
                            provenance=dict(source_sha256=entry['source_sha256'], clean=entry['clean'],
                                            reference_sha256=digest(root/entry['clean']),
                                            sensor_crop_xy=[x, y], seed=seed, level=level,
                                            exposure_scale=exposure)))
    return samples, None


def latency(path, iterations, warmup, compute_units=None):
    unit = coreml_compute_units(compute_units=compute_units)
    rng = np.random.default_rng(20261104)
    tile = rng.normal(.05, .004, (1, 4, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)).astype(np.float32)
    try:
        model = Denoiser.load(path, backend='coreml', compute_units=unit)
        for _ in range(warmup):
            model._call(tile)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            output = model._call(tile)
            times.append((time.perf_counter()-start)*1000)
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite benchmark output')
        return {unit: dict(median_ms=statistics.median(times), samples_ms=times)}
    except Exception as exc:
        return {unit: dict(error=f'{type(exc).__name__}: {exc}')}


def quality(path, examples, samples, reference_examples, reference_dev, compute_units=None):
    model = Denoiser.load(path, backend='coreml', compute_units=compute_units)
    # Production normalization, prepass, reflected apron and feathering for demo DNGs.
    example_predictions = [model.denoise_packed(packed, **tiling_settings()) for _, packed in examples]
    dev_predictions = [model.denoise_packed(s['noisy'], prepass=False, **tiling_settings()) for s in samples]
    rows = []
    for i, (example, prediction) in enumerate(zip(examples, example_predictions)):
        rows.append(dict(path=str(example[0]), input_sha256=digest(example[0]),
                         **errors(prediction, reference_examples[i] if reference_examples is not None else prediction)))
    dev = []
    for i, (sample, prediction) in enumerate(zip(samples, dev_predictions)):
        target_psnr = errors(prediction, sample['clean'])['psnr_db']
        ref = reference_dev[i] if reference_dev is not None else prediction
        baseline_psnr = errors(ref, sample['clean'])['psnr_db']
        dev.append(dict(**sample['provenance'], target_psnr_db=target_psnr,
                        baseline_target_psnr_db=baseline_psnr,
                        loss_db=baseline_psnr-target_psnr,
                        **errors(prediction, ref)))
    result = dict(compute_units=model.compute_units, tiling=tiling_settings(), examples=rows, dev=dev,
                  dev_mean_loss_db=float(np.mean([r['loss_db'] for r in dev])) if dev else None)
    return result, example_predictions, dev_predictions


def write_report(out, report):
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Core ML compression experiment', '',
             'Development diagnostics only; output agreement is not ground-truth quality.', '',
             f"Environment: `{report['platform']}`; Python {report['python']}; Core ML tools: {report.get('coremltools', 'unavailable')}.", '',
             f"Development manifest: {report['development_manifest']}. Demo DNG count: {report['example_count']}.", '',
             'Package size is the sum of file bytes, not filesystem allocation. Latency includes Python predict overhead, excludes load/compile, '
             'and uses 5 warmups and at least 20 timed predictions by default. CPU_AND_NE permits CPU fallback; it does not prove ANE placement.', '',
             f"Compute units for timing, quality and identity: `{report['compute_units']}`. Tiling: `{report['tiling']}`.", '',
             'Quality uses the frozen fp16 package as reference and production Denoiser tiling, '
             'and no prepass for synthetic dev crops. Dev targets are imperfect binned low-ISO references; '
             'fixed added noise uses evaluate_dev_v2 LEVELS. Identity uses production tiling at 1100×1500 packed pixels, all kinds/levels, with prepass and highlight protection off.', '',
             '| Variant | Bytes | Selected units ms | Dev loss dB | Identity | Acceptable |',
             '|---|---:|---:|---:|---|---|']
    def number(value):
        return 'unavailable' if value is None else f'{value:.6g}'
    for name, row in report['variants'].items():
        ts = row.get('latency', {})
        values = [number(ts.get(report['compute_units'], {}).get('median_ms'))]
        gate = row.get('identity', {}).get('passed')
        lines.append(f"| {name} | {row.get('bytes', 'unavailable')} | {' | '.join(values)} | "
                     f"{number(row.get('quality', {}).get('dev_mean_loss_db'))} | "
                     f"{('PASS' if gate else 'FAIL') if gate is not None else 'unavailable'} | {row.get('acceptable', False)} |")
    lines += ['', '## Output agreement with frozen fp16', '',
              '| Variant | Input | PSNR dB (peak 1) | Max absolute error |', '|---|---|---:|---:|']
    for name, row in report['variants'].items():
        for item in row.get('quality', {}).get('examples', []):
            lines.append(f"| {name} | {Path(item['path']).name} | {'infinite (identical)' if item['identical'] else number(item['psnr_db'])} | {item['max_abs']:.6g} |")
        dev = row.get('quality', {}).get('dev', [])
        if dev:
            finite = [r['psnr_db'] for r in dev if r['psnr_db'] is not None]
            score = number(float(np.mean(finite))) if finite else 'infinite (identical)'
            lines.append(f"| {name} | dev ({len(dev)} crops; mean PSNR / worst max abs) | {score} | {max(r['max_abs'] for r in dev):.6g} |")
    accepted = [name for name, row in report['variants'].items() if row.get('acceptable') and name != 'reference_fp16']
    lines += ['', '## Recommendation', '',
              ('Candidates meeting the requested dev/identity gate: '+', '.join(accepted)+'. Compare measured latency/size before selection.'
               if accepted else 'Retain the frozen fp16 model. No compressed variant has demonstrated both dev PSNR loss ≤ 0.05 dB and an identity-gate pass.'), '',
              '## Errors and missing evidence', '']
    for error in report.get('errors', []):
        lines.append('- '+error)
    for name, row in report['variants'].items():
        for stage, error in row.get('errors', {}).items():
            lines.append(f'- {name}, {stage}: {error}')
    lines += ['', 'Raw timing samples, per-example/per-dev errors, crop provenance and every identity row are in `results.json`.', '',
              'Reproduce from the worktree with the required venv and `PYTHONPATH=$PWD/src`:', '',
              '```sh', report['command'], '```', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--data-root', type=Path)
    ap.add_argument('--examples', type=Path, default=Path('examples/raw'))
    ap.add_argument('--out', type=Path, default=Path('runs/coreml_compress'))
    ap.add_argument('--compute-units', choices=COREML_COMPUTE_UNITS, default=coreml_compute_units())
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--dev-crops', type=int, default=36)
    ap.add_argument('--export-python', default=sys.executable,
                    help='Interpreter for export_coreml.py (coremltools may lag the evaluation torch version)')
    ap.add_argument('--reuse-exports', action='store_true', help='Skip export when the package already exists')
    args = ap.parse_args()
    if args.iterations < 20 or args.warmup < 1 or args.dev_crops < 1:
        ap.error('Require iterations >= 20, warmup >= 1, dev-crops >= 1')
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = {'reference_fp16': args.reference,
             **{mode: args.out/f'r4_plain_{mode}.mlpackage' for mode in COMPRESSION_MODES}}
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  command='PYTHONPATH=$PWD/src '+__import__('shlex').join([sys.executable, *sys.argv]),
                  iterations=args.iterations, warmup=args.warmup, errors=[],
                  compute_units=args.compute_units, tiling=tiling_settings(),
                  checkpoint_sha256=digest(args.checkpoint), variants={})
    manifest = args.data_root/'bootstrap_binned/x1d2_bin2_val_r4.json' if args.data_root else None
    report['development_manifest'] = str(manifest) if manifest and manifest.exists() else 'unavailable'
    report['example_count'] = len(list(args.examples.glob('*.dng')))
    for name, path in paths.items():
        report['variants'][name] = dict(path=str(path), errors={})
        if path.exists():
            report['variants'][name]['bytes'] = size_bytes(path)
    if importlib.util.find_spec('coremltools') is None:
        report['errors'].append('ModuleNotFoundError: coremltools is absent from the required interpreter. No installation attempted (task prohibits pip install). Exports, timings, output quality and identity are unmeasured.')
        write_report(args.out, report)
        print(report['errors'][0], flush=True)
        return 1
    import coremltools as ct
    report['coremltools'] = ct.__version__
    examples = [(p, pack(read_mosaic(p))) for p in sorted(args.examples.glob('*.dng'))]
    if len(examples) != 3:
        report['errors'].append(f'Expected three demo DNGs, found {len(examples)}')
    samples, dev_error = dev_samples(args.data_root, args.dev_crops) if args.data_root else ([], 'No data root supplied')
    if dev_error:
        report['errors'].append(dev_error)
    ref_examples = ref_dev = None
    for name, path in paths.items():
        row = report['variants'][name]
        try:
            if name != 'reference_fp16' and not (args.reuse_exports and path.exists()):
                log = args.out/f'export_{name}.log'
                with log.open('w') as f:
                    subprocess.run([args.export_python, str(Path(__file__).with_name('export_coreml.py')),
                                    '--checkpoint', str(args.checkpoint), '--output', str(path), '--compress', name],
                                   stdout=f, stderr=subprocess.STDOUT, check=True)
            row['bytes'] = size_bytes(path)
        except Exception as exc:
            row['errors']['export'] = str(exc)
            write_report(args.out, report)
            continue
        row['latency'] = latency(path, args.iterations, args.warmup, args.compute_units)
        try:
            if name != 'reference_fp16' and ref_examples is None:
                raise RuntimeError('Frozen fp16 reference quality failed')
            measured, ep, dp = quality(path, examples, samples, ref_examples, ref_dev, args.compute_units)
            row['quality'] = measured
            if name == 'reference_fp16':
                ref_examples, ref_dev = ep, dp
        except Exception as exc:
            row['errors']['quality'] = f'{type(exc).__name__}: {exc}'
        try:
            row['identity'] = package_identity(path, compute_units=args.compute_units)
        except Exception as exc:
            row['errors']['identity'] = f'{type(exc).__name__}: {exc}'
        loss = row.get('quality', {}).get('dev_mean_loss_db')
        row['acceptable'] = loss is not None and loss <= .05 and row.get('identity', {}).get('passed', False)
        print(name, {k: v for k, v in row.items() if k not in ('quality', 'identity')}, flush=True)
        write_report(args.out, report)
    return int(any(r['errors'] for r in report['variants'].values()) or bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
