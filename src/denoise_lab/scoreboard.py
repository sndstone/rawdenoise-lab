"""Local evaluation orchestration, cached baselines and development regression gates.

Requires the research scripts in a source checkout. Private inputs are read-only;
all evaluator outputs and caches live beneath the caller's output directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .noref import bootstrap_crops, crop_metrics
from .checkpoints import resolve_weights

SCRIPTS = Path(__file__).resolve().parents[2]/"scripts"
SUITES = ("dev", "measured", "burst", "camera", "speed")


def _script(name):
    # The existing burst script imports its two sibling evaluation helpers.
    directory = str(SCRIPTS)
    sys.path.insert(0, directory)
    try:
        spec = importlib.util.spec_from_file_location("scoreboard_"+name, SCRIPTS/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(directory)


def checkpoint_path(path):
    """Resolve the frozen baseline directory, or accept an explicit .pt file."""
    path = Path(path)
    if path.is_dir():
        for name in ("r4_plain.pt", "r4_plain_weights.pt", "last.pt"):
            if (path/name).is_file():
                return path/name
        raise ValueError(f"No known checkpoint filename in {path}; pass a .pt file")
    if path.suffix != ".pt" or not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_finite(value), indent=2, allow_nan=False)+"\n")


def _score(suite, name, metric, value, *, gate=False, **extra):
    return dict(suite=suite, slice=name, metric=metric, value=value, gate=gate, **extra)


def compare_scores(candidate, baseline):
    """Align every baseline AND candidate slice; missing/nonfinite values stay visible."""
    def index(rows):
        result = {}
        for row in rows:
            key = (row['suite'], row['slice'], row['metric'])
            if key in result:
                raise ValueError(f"Duplicate score: {key}")
            result[key] = row
        return result
    c, b = index(candidate), index(baseline)
    comparisons = []
    for key in sorted(c.keys() | b.keys()):
        cv, bv = c.get(key, {}).get('value'), b.get(key, {}).get('value')
        valid = cv is not None and bv is not None and np.isfinite(cv) and np.isfinite(bv)
        comparisons.append(dict(suite=key[0], slice=key[1], metric=key[2], value=cv, baseline=bv,
                                delta=float(cv-bv) if valid else None,
                                gate=c.get(key, {}).get('gate', False) or b.get(key, {}).get('gate', False),
                                ci95=c.get(key, {}).get('ci95'), baseline_ci95=b.get(key, {}).get('ci95')))
        if key[0] == 'burst' and key[2] == 'psnr_out':
            cf = c.get(key, {}).get('frame_scores', {})
            bf = b.get(key, {}).get('frame_scores', {})
            comparisons[-1]['per_frame_deltas'] = [dict(frame=frame, candidate=cf.get(frame), baseline=bf.get(frame),
                delta_db=cf[frame]-bf[frame] if cf.get(frame) is not None and bf.get(frame) is not None else None)
                for frame in sorted(cf.keys() | bf.keys())]
    return comparisons


def regression_gate(comparisons, thresholds=None, *, skipped=(), errors=(), identical=False):
    """PSNR-only regression gate; incomplete selected suites fail unless explicitly allowed.

    min_mean_gain_db is inclusive; require_improvement makes it strict. The latter
    is waived only for identical checkpoint hashes, to allow baseline sanity runs.
    """
    thresholds = thresholds or {}
    floor = float(thresholds.get('min_slice_delta_db', -.1))
    mean_floor = float(thresholds.get('min_mean_gain_db', 0.))
    rows = [r for r in comparisons if r['gate']]
    reasons = []
    if errors:
        reasons.append('suite errors: '+', '.join(errors))
    if skipped and not thresholds.get('allow_skips', False):
        reasons.append('selected suites incomplete: '+', '.join(skipped))
    if not rows:
        reasons.append('no comparable PSNR slices')
    missing = [f"{r['suite']}/{r['slice']}/{r['metric']}" for r in rows
               if r['delta'] is None or not np.isfinite(r['delta'])]
    if missing:
        reasons.append('missing/nonfinite PSNR: '+', '.join(missing))
    deltas = [r['delta'] for r in rows if r['delta'] is not None and np.isfinite(r['delta'])]
    worst, mean = (min(deltas), float(np.mean(deltas))) if deltas else (None, None)
    if worst is not None and worst < floor-1e-12:
        reasons.append(f'worst slice {worst:.4f} dB < {floor:g} dB')
    if mean is not None:
        strict = thresholds.get('require_improvement', False) and not identical
        if mean < mean_floor-1e-12 or (strict and mean <= mean_floor):
            reasons.append(f'mean gain {mean:.4f} dB must be {">" if strict else ">="} {mean_floor:g} dB')
    return dict(status='FAIL' if reasons else 'PASS', reasons=reasons,
                mean_delta_db=mean, worst_delta_db=worst,
                slices=len({(r['suite'], r['slice']) for r in rows}), measurements=len(rows), thresholds=thresholds)


def _dev(path, cfg, output, device, quick, weights="model"):
    evaluator = _script('evaluate_dev_v2')
    scores, details = [], []
    for i, dataset in enumerate(cfg.get('datasets', [])):
        name = dataset['name']
        args = SimpleNamespace(checkpoint=[f'model={path}'], camera=dataset.get('camera', 'x1d2'),
            root=Path(cfg.get('root', 'data')), manifest=Path(dataset['manifest']),
            train_manifest=Path(dataset['train_manifest']) if dataset.get('train_manifest') else None,
            output=output/f'dev_{i}.json', device=device, crops_per_source=1 if quick else cfg.get('crops_per_source', 3),
            max_sources=cfg.get('quick_sources', 2) if quick else cfg.get('max_sources'),
            crop_size=cfg.get('crop_size', 256), slices=cfg.get('slices'), references='', previous_evaluation=None,
            threads=torch.get_num_threads(), weights=weights)
        result = evaluator.evaluate(args)
        details.append(dict(name=name, output=str(args.output), sources=result['sources'], crops=result['crops']))
        for row in result['summary']:
            # Conditioned DEV models use their oracle map for the matched scoreboard.
            # Other map robustness probes remain in the evaluator's raw JSON.
            if row['model'] not in ('model', 'model@oracle'):
                continue
            for metric in ('model_psnr', 'flat_psnr', 'texture_psnr', 'color_mse'):
                scores.append(_score('dev', name+'/'+row['slice'], metric, row[metric],
                                     gate=metric in cfg.get('gate_metrics', ['model_psnr']),
                                     sources=row['sources'], crops=row['crops']))
    return scores, details


def _burst(path, cfg, output, device, quick, weights="model"):
    evaluator = _script('eval_burst_loo')
    scores, details = [], []
    for i, spec_path in enumerate(cfg.get('specs', [])):
        spec_path = Path(spec_path)
        spec = json.loads(spec_path.read_text())
        # Spec roots are relative to configured project_root, never the output.
        spec['root'] = str(Path(cfg.get('project_root', '.'))/spec['root'])
        if quick:
            spec['groups'] = spec['groups'][:cfg.get('quick_groups', 1)]
            for group in spec['groups']:
                group['test'] = group['test'][:1]
        if any(len(g['files']) < 3 for g in spec['groups']):
            raise ValueError('LOO reference-noise correction requires at least three frames')
        rows = evaluator.evaluate(SimpleNamespace(spec=spec, model=[f'model={path}'], device=device,
                                  output=output/f'burst_{i}', block=cfg.get('block', 16), crops='{}',
                                  packed_crop_size=cfg.get('quick_crop_size', 512) if quick else None, weights=weights))
        details.append(dict(spec=str(spec_path), rows=rows))
        for group in dict.fromkeys(r['group'] for r in rows):
            rs = [r for r in rows if r['group'] == group]
            for metric in ('psnr_out', 'gain_db_flat', 'gain_db_tex'):
                values = [r[metric] for r in rs]
                value = float(np.mean(values)) if all(v is not None for v in values) else None
                scores.append(_score('burst', spec_path.stem+'/'+group, metric, value, gate=metric == 'psnr_out',
                                     **({'frame_scores': {r['frame']: r['psnr_out'] for r in rs}} if metric == 'psnr_out' else {})))
            scores.append(_score('burst', spec_path.stem+'/'+group, 'pooled_corrected_mse_psnr',
                                 evaluator.pooled_psnr(rs), gate=False))
    return scores, details


def _camera(path, cfg, output, device, quick, weights="model"):
    from .inference import Denoiser
    camera = _script('benchmark_cameras')
    root = Path(cfg.get('root', 'data/camera_bench'))
    files = [f for f in json.loads((root/'manifest.json').read_text())['files'] if f['set'] == 'real']
    den = Denoiser.load(path, device=device, **({"weights": weights} if weights != "model" else {}))
    size, context = cfg.get('roi', 512), cfg.get('context', 64)
    by_band, details, photos = defaultdict(list), [], defaultdict(int)
    for f in files:
        stratum = (f['camera'], f['band'])
        if quick and photos[stratum] >= cfg.get('quick_photos_per_band', 1):
            continue
        inp, _ = camera.load_packed(root/f['file'])
        limit = cfg.get('quick_rois', 1) if quick else None
        count = 0
        for y, x, noisy, pred in camera.denoise_rois(den, inp, size, context, limit=limit):
            count += 1
            metrics = crop_metrics(noisy, pred)
            by_band[(f['camera'], f['band'])].append(metrics)
            details.append(dict(camera=f['camera'], band=f['band'], source_sha256=f['sha256'],
                                roi_yx=[y, x], **metrics))
        if count:
            photos[stratum] += 1
        else:
            details.append(dict(camera=f['camera'], band=f['band'], source_sha256=f['sha256'],
                                status='SKIP', reason='no valid input ROI'))
    scores = []
    for (cam, band), rows in by_band.items():
        for metric, stats in bootstrap_crops(rows, samples=cfg.get('bootstrap_samples', 1000)).items():
            scores.append(_score('camera', cam+'/'+band, metric, stats['mean'], ci95=stats['ci95'], crops=stats['crops']))
    # A stratum with no valid ROI must not silently disappear from coverage.
    covered = set(by_band)
    if covered != {(f['camera'], f['band']) for f in files}:
        raise ValueError('camera strata have no valid input ROIs')
    return scores, details


def _speed(path, cfg, output, device, quick, weights="model"):
    from .arch_registry import build_model
    bench = _script('bench_tile')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['model_config'].get('cond', 'none') != 'none':
        raise ValueError('speed suite requires unconditioned models')
    model = build_model(saved['model_config']).to(device).eval()
    model.load_state_dict(resolve_weights(saved, weights))
    result = bench.benchmark_tile(model, device, precision=cfg.get('precision', 'fp32'),
                                 warmup=1 if quick else cfg.get('warmup', 20),
                                 iterations=3 if quick else cfg.get('iterations', 30), size=cfg.get('size', 512))
    return [_score('speed', f"tile{cfg.get('size', 512)}", 'median_ms', result['median_ms'])], result


def _measured(path, cfg, output, device, quick, weights="model"):
    result = _script('evaluate_measured_dev').evaluate(SimpleNamespace(
        checkpoint=[f'model={path}'], config=cfg, output=output/'measured.json',
        device=device, quick=quick, weights=weights))
    scores = [_score('measured', r['slice'], 'model_psnr', r['model_psnr'],
                     gate=cfg.get('gate', False), sources=r['sources'], crops=r['crops'])
              for r in result['summary']]
    return scores, result


RUNNERS = dict(dev=_dev, measured=_measured, burst=_burst, camera=_camera, speed=_speed)


def _input_fingerprint(name, cfg, quick=False):
    """Fingerprint manifests/specs and referenced file metadata, without hashing RAWs."""
    paths = []
    if name in ('dev', 'measured'):
        root = Path(cfg.get('root', 'data'))
        if name == 'measured':
            paths.append(Path(cfg['calibration']))
        for dataset in cfg.get('datasets', []):
            for key in ('manifest', 'train_manifest'):
                if dataset.get(key):
                    manifest = Path(dataset[key])
                    paths.append(manifest)
                    if manifest.exists() and (name == 'dev' or key == 'manifest'):
                        entries = json.loads(manifest.read_text())
                        if name == 'measured':
                            evaluator = _script('evaluate_measured_dev')
                            grouped = defaultdict(list)
                            for e in entries:
                                grouped[e['source_sha256']].append(e)
                            count = 1 if quick else cfg.get('crops_per_source', 3)
                            limit = cfg.get('quick_sources', 1) if quick else cfg.get('max_sources')
                            selected = []
                            for source, values in sorted(grouped.items())[:limit]:
                                values = sorted(values, key=lambda e: e['clean'])
                                for crop in range(count):
                                    seed = evaluator.seed_for(cfg.get('crop_seed', 20261109), dataset['name'], source, crop)
                                    selected.append(values[int(np.random.default_rng(seed).integers(len(values)))])
                            entries = selected
                        paths.extend(root/e['clean'] for e in entries)
    elif name == 'camera':
        root = Path(cfg.get('root', 'data/camera_bench'))
        manifest = root/'manifest.json'
        paths.append(manifest)
        if manifest.exists():
            paths.extend(root/e['file'] for e in json.loads(manifest.read_text())['files'] if e['set'] == 'real')
    elif name == 'burst':
        for p in cfg.get('specs', []):
            spec_path = Path(p)
            paths.append(spec_path)
            if spec_path.exists():
                spec = json.loads(spec_path.read_text())
                root = Path(cfg.get('project_root', '.'))/spec['root']
                paths.extend(root/f for g in spec['groups'] for f in g['files'])
    records = []
    missing = []
    for path in sorted(set(paths)):
        if not path.is_file():
            missing.append(str(path))
            records.append([str(path), None])
        else:
            stat = path.stat()
            records.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns,
                            sha256(path) if path.suffix in ('.json', '.yaml', '.yml') else None])
    return records, missing


def run_checkpoint(path, config, suites, output, device, quick, *, cache=False, weights="model"):
    """Cache successful suites only; key includes checkpoint, config, inputs, code and device."""
    digest = sha256(path)
    scores, results = [], {}
    code_hash = hashlib.sha256()
    for source in sorted((SCRIPTS.parent/'src/denoise_lab').glob('*.py'))+sorted(SCRIPTS.glob('*.py')):
        code_hash.update(source.name.encode())
        code_hash.update(source.read_bytes())
    for name in suites:
        cfg = config['suites'][name]
        inputs, missing = _input_fingerprint(name, cfg, quick)
        if missing:
            results[name] = dict(status='SKIP', reason=f'missing private input: {missing[0]}', missing_files=len(missing))
            continue
        key_spec = dict(checkpoint=digest, suite=name, config=cfg, quick=quick, device=device,
                        threads=torch.get_num_threads(), inputs=inputs, code=code_hash.hexdigest())
        if weights != 'model':
            key_spec['weights'] = weights
        key = hashlib.sha256(json.dumps(key_spec, sort_keys=True).encode()).hexdigest()
        cached = output/'cache'/f'{key}.json'
        # Timings are hardware/session dependent; never reuse a speed measurement.
        if cache and name != 'speed' and cached.exists():
            value = json.loads(cached.read_text())
            value['cached'] = True
        else:
            started = time.monotonic()
            try:
                rows, details = RUNNERS[name](path, cfg, output/(digest[:12]+('-ema' if weights == 'ema' else '')),
                                             device, quick, weights=weights)
                value = dict(status='OK' if rows else 'SKIP', scores=rows, details=details,
                             seconds=time.monotonic()-started, cached=False)
                if not rows:
                    value['reason'] = 'no evaluation rows (empty suite)'
                if cache and name != 'speed' and rows:
                    _write(cached, value)
            except FileNotFoundError as exc:
                value = dict(status='SKIP', reason=str(exc), seconds=time.monotonic()-started)
            except Exception as exc:
                value = dict(status='ERROR', reason=f'{type(exc).__name__}: {exc}', seconds=time.monotonic()-started)
        results[name] = value
        scores.extend(value.get('scores', []))
        print(f"{path.name}: {name} {value['status']}"+(' (cached)' if value.get('cached') else '')
              +(': '+value['reason'] if value.get('reason') else ''), flush=True)
    return dict(checkpoint=str(path), sha256=digest, weights=weights, suites=results, scores=scores)


def markdown_report(report):
    lines = ['# Development evaluation scoreboard', '',
             'Regression diagnostics only. PASS does not establish camera acceptance or replace blind review.', '',
             f"Device: {report['device']}; weights: {report.get('weights', 'model')}; quick: {report['quick']}; elapsed: {report['seconds']:.1f} s.", '']
    for name, result in report['candidates'].items():
        gate = result['gate']
        lines += [f"## {name}: {gate['status']}", '', f"Checkpoint SHA-256: `{result['sha256']}`", '']
        lines += [f"- {reason}" for reason in gate['reasons']]
        lines += ['', '| Suite | Slice | Metric | Baseline | Candidate | Delta | Candidate 95% CI |',
                  '|---|---|---|---:|---:|---:|---|']
        def fmt(v):
            return 'unavailable' if v is None else f'{v:.5g}'
        for row in result['comparisons']:
            if row['suite'] == 'measured':
                continue
            ci = row.get('ci95')
            lines.append(f"| {row['suite']} | {row['slice'].replace('|', ' / ')} | {row['metric']} | "
                         f"{fmt(row['baseline'])} | {fmt(row['value'])} | {fmt(row['delta'])} | "
                         f"{('['+fmt(ci[0])+', '+fmt(ci[1])+']') if ci else '—'} |")
        measured = [r for r in result['comparisons'] if r['suite'] == 'measured']
        if measured:
            gating = report['config']['suites']['measured'].get('gate', False)
            lines += ['', '### Measured development ('+('gating' if gating else 'non-gating')+')', '',
                      '| Slice | Baseline PSNR | Candidate PSNR | Delta dB |', '|---|---:|---:|---:|']
            for row in measured:
                lines.append(f"| {row['slice']} | {fmt(row['baseline'])} | {fmt(row['value'])} | {fmt(row['delta'])} |")
            for profile in result['suites']['measured'].get('details', {}).get('profiles', []):
                if profile['status'] == 'SKIP':
                    lines.append(f"- {profile['profile']}: unavailable ({profile['reason']})")
        bursts = [r for r in result['comparisons'] if r['suite'] == 'burst' and r['metric'] == 'psnr_out']
        if bursts:
            pooled = {r['slice']: r['delta'] for r in result['comparisons']
                      if r['suite'] == 'burst' and r['metric'] == 'pooled_corrected_mse_psnr'}
            lines += ['', '### Burst aggregation sensitivity (dB)', '',
                      '| Group | Mean frame PSNR delta (gating) | Pooled corrected-MSE delta (non-gating) | Frame deltas |',
                      '|---|---:|---:|---|']
            for row in bursts:
                frames = ', '.join(f"{f['frame']}: {fmt(f['delta_db'])}" for f in row.get('per_frame_deltas', []))
                lines.append(f"| {row['slice']} | {fmt(row['delta'])} | {fmt(pooled.get(row['slice']))} | {frames} |")
            lines += ['', 'Overlapping LOO frames are not independent scenes; no scene interval is inferred.']
        lines += ['', 'Suite coverage:', '']
        for suite in report['selected_suites']:
            b, c = report['baseline']['suites'][suite], result['suites'][suite]
            lines.append(f"- {suite}: baseline {b['status']}, candidate {c['status']}"
                         f"{': '+c['reason'] if c.get('reason') else ''}")
        lines.append('')
    lines += ['No-reference metrics: lower residual_noise and chroma_blotch mean smoother output; ',
              'texture_retention is a mid/high-frequency power ratio to the noisy input. These are scene-confounded.',
              'Camera CIs resample crops, not pixels; nearby crops/photos can be correlated. Speed deltas are noisy.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='[NAME=]PATH')
    parser.add_argument('--baseline', default='checkpoints/r4_plain_2026-10-30/')
    parser.add_argument('--config', type=Path, default=Path('configs/scoreboard/default.yaml'))
    parser.add_argument('--suites', nargs='+', help='dev measured burst camera speed (comma-separated also accepted)')
    parser.add_argument('--weights', choices=['model', 'ema'], default='model')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, default=Path('runs/scoreboard'))
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    suites = [s for part in args.suites for s in part.split(',')] if args.suites else list(config['suites'])
    if not suites or len(suites) != len(set(suites)) or set(suites)-set(SUITES) or set(suites)-set(config['suites']):
        parser.error('select distinct configured suites from: '+', '.join(SUITES))
    torch.set_num_threads(config.get('threads', 2))
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    baseline_path = checkpoint_path(args.baseline)
    candidates = {}
    for item in args.checkpoint:
        name, path = item.split('=', 1) if '=' in item else (Path(item).stem, item)
        if name in candidates:
            parser.error('duplicate checkpoint label: '+name)
        candidates[name] = checkpoint_path(path)
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = run_checkpoint(baseline_path, config, suites, args.output, device, args.quick, cache=not args.no_cache, weights=args.weights)
    report = dict(protocol='scoreboard_v1', device=device, quick=args.quick, weights=args.weights, config=config,
                  selected_suites=suites, baseline=baseline, candidates={})
    for name, path in candidates.items():
        # Actual inference runs for both sides, including self-comparisons. Only the baseline uses the cache.
        result = run_checkpoint(path, config, suites, args.output, device, args.quick, weights=args.weights)
        result['comparisons'] = compare_scores(result['scores'], baseline['scores'])
        gating_suites = [s for s in suites if s != 'measured' or config['suites'][s].get('gate', False)]
        incomplete = [s for s in gating_suites if result['suites'][s]['status'] == 'SKIP' or baseline['suites'][s]['status'] == 'SKIP']
        errors = [s for s in gating_suites if result['suites'][s]['status'] == 'ERROR' or baseline['suites'][s]['status'] == 'ERROR']
        result['gate'] = regression_gate(result['comparisons'], config.get('gate'), skipped=incomplete, errors=errors,
                                         identical=result['sha256'] == baseline['sha256'])
        if 'measured' in suites:
            b, c = baseline['suites']['measured'], result['suites']['measured']
            if b['status'] == 'OK' and c['status'] == 'OK':
                evaluator = _script('evaluate_measured_dev')
                c['details']['source_paired_deltas'] = evaluator.paired_deltas(
                    c['details']['source_scores'], b['details']['source_scores'])
            if not gating_suites:
                result['gate'] = dict(status='NOT_GATED', reasons=['measured suite is diagnostic only'],
                                      mean_delta_db=None, worst_delta_db=None, slices=0, measurements=0)
        report['candidates'][name] = result
    report['seconds'] = time.monotonic()-started
    statuses = {r['gate']['status'] for r in report['candidates'].values()}
    report['status'] = 'FAIL' if 'FAIL' in statuses else ('NOT_GATED' if 'NOT_GATED' in statuses else 'PASS')
    if report['status'] == 'NOT_GATED' and any(
            run['suites']['measured']['status'] != 'OK' for run in [baseline, *report['candidates'].values()]):
        report['status'] = 'ERROR'
    _write(args.output/'scoreboard.json', report)
    (args.output/'scoreboard.md').write_text(markdown_report(_finite(report)))
    print(f"{report['status']}: {args.output/'scoreboard.md'} ({report['seconds']:.1f}s)", flush=True)
    return 0 if report['status'] in ('PASS', 'NOT_GATED') else 1
