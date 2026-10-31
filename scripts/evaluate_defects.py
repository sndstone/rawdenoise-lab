#!/usr/bin/env python3
"""Development defect diagnostics; synthetic references are imperfect low-ISO crops.

Real mode reports detector metrics only, without model inference or rendering.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import random
import numpy as np
import torch
from denoise_lab.arch_registry import build_model
from denoise_lab.bayer import pack, unpack
from denoise_lab.data import BootstrapDataset
from denoise_lab.defects import inject_defects, correct_defects
from denoise_lab.diffusion import BridgeRefiner
from denoise_lab.metrics import psnr

REFERENCE = Path(__file__).with_name('evaluate_arch_study.py')
_spec = importlib.util.spec_from_file_location('defects_arch_study', REFERENCE)
arch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arch)
SLICES, LEVELS, STRESS, binned_item = arch.SLICES, arch.LEVELS, arch.STRESS, arch.binned_item


def selected_entries(manifest):
    """Exactly the architecture-study source grouping/order and sampling stream."""
    grouped = defaultdict(list)
    for entry in json.loads(Path(manifest).read_text()):
        grouped[entry['source_sha256']].append(entry)
    rng = random.Random(20261028)
    entries = []
    for _, values in sorted(grouped.items()):
        values = sorted(values, key=lambda e: e['clean'])
        entries += rng.sample(values, min(3, len(values)))
    return entries


def real_report(path, device):
    # Reuse precisely the normalization and packing imported by X2D diagnostics.
    from arch_study_x2d import normalize_raw, pack as pack_raw, CHANNELS
    import rawpy
    with rawpy.imread(str(path)) as raw:
        source = torch.from_numpy(pack_raw(normalize_raw(raw)))[None].to(device)
    corrected, mask = correct_defects(source, **DETECTOR)
    counts = {}
    top = []
    h, w = source.shape[-2:]
    for c, channel in enumerate(CHANNELS):
        flags = mask[0, c]
        hot = flags & (source[0, c] > corrected[0, c])
        dead = flags & ~hot
        counts[channel] = dict(flagged=int(flags.sum()), hot=int(hot.sum()), dead=int(dead.sum()),
                               per_MP=float(flags.sum())/(h*w/1e6))
        # Rank by correction magnitude; compute locations on CPU for stable ties.
        loc = flags.nonzero().cpu().numpy()
        if len(loc):
            values = source[0, c][flags].cpu().numpy()
            medians = corrected[0, c][flags].cpu().numpy()
            order = np.argsort(-np.abs(values-medians), kind='stable')[:20]
            for i in order:
                y, x = loc[i]
                top.append(dict(y=int(y), x=int(x), channel=channel, value=float(values[i]),
                                neighbour_median=float(medians[i]), kind='hot' if values[i] > medians[i] else 'dead'))
    top.sort(key=lambda r: -abs(r['value']-r['neighbour_median']))
    n = int(mask.sum())
    return dict(raw=str(path), packed_shape=list(source.shape), flagged=n,
                per_MP=n/(source.numel()/1e6), hot=sum(v['hot'] for v in counts.values()),
                dead=sum(v['dead'] for v in counts.values()), channels=counts, top20=top[:20],
                per_MP_definition="Total per million sensor photosites; channel rates per million channel photosites",
                top20_ranking="Descending absolute correction, stable channel/row/column ties")


DETECTOR = {}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH')
    ap.add_argument('--weights', choices=['model', 'ema'], default='model')
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--root', type=Path, default=Path('data'))
    ap.add_argument('--baseline', help='Checkpoint label; comparisons match variant and input')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--slices', nargs='+')
    ap.add_argument('--limit', type=int, help='Limit selected crops, after matched source selection')
    ap.add_argument('--raw', action='append', type=Path, default=[])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--detector', default='{}', help='JSON kwargs for correct_defects, e.g. {"adjacency_ratio": .25}')
    args = ap.parse_args()
    DETECTOR.update(json.loads(args.detector))
    if args.limit is not None and args.limit < 1: ap.error('--limit must be positive')
    if not args.raw and not args.checkpoint: ap.error('Provide --raw or --checkpoint')
    if args.checkpoint and args.manifest is None: ap.error('--checkpoint requires --manifest')
    torch.set_num_threads(2)
    selected = {s for part in args.slices or [] for s in part.split(',')}
    known = {arch.slice_name(s) for s in SLICES}
    if selected-known: ap.error('Unknown slices: '+', '.join(sorted(selected-known)))
    models, provenance = {}, {}
    for spec in args.checkpoint:
        label, sep, path = spec.partition('=')
        if not sep or not label or label in models: ap.error('Checkpoints require unique NAME=PATH')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        key = 'ema_model' if args.weights == 'ema' else 'model'
        if key not in saved: raise ValueError(f'{path}: requested {args.weights} weights but checkpoint has no {key}')
        model = build_model(saved['model_config']).to(args.device).eval()
        model.load_state_dict(saved[key])
        models[label] = model
        provenance[label] = dict(path=path, sha256=arch.sha(path), weights=args.weights, model_config=saved['model_config'])
    baseline = args.baseline or next(iter(models), None)
    if models and baseline not in models: ap.error('--baseline must name a checkpoint')
    entries = selected_entries(args.manifest) if models else []
    if models and not entries: raise ValueError('Empty validation manifest')
    if args.limit: entries = entries[:args.limit]
    rows = []
    for si, spec in enumerate(SLICES):
        name = arch.slice_name(spec)
        if not models or selected and name not in selected: continue
        exposure, level, ref = spec
        parameters = {**LEVELS, **STRESS}[level]
        dataset = BootstrapDataset(entries, root=args.root, crop_size=256, augment=False,
                                   deterministic=True, seed=20261028+1000*si,
                                   fixed_parameters=parameters, exposure_scale=exposure)
        for i in range(len(dataset)):
            item = dataset[i]
            if ref == 'binned': item = binned_item(item['clean'], parameters, 20261028+1000*si+i)
            target = item['clean'][None].to(args.device)
            noisy = item['noisy'][None].to(args.device)
            cond = item['noise_map'][None].to(args.device)
            # Oracle local noise std includes shot, read, row and column variance;
            # deliberate tail impulses are excluded from this threshold.
            local_std = (parameters['K']*np.maximum(unpack(item['clean'].numpy()), 0)
                         +parameters['read_std']**2+parameters['row_std']**2+parameters.get('col_std', 0)**2)**.5
            mosaic, injected = inject_defects(unpack(item['noisy'].numpy()),
                np.random.default_rng(20261029+1000*si+i), density_range=(3e-4, 3e-4), sigma=local_std)
            defect = torch.from_numpy(pack(mosaic))[None].to(args.device)
            sites = torch.from_numpy(pack(injected))[None].to(args.device)
            threshold = torch.from_numpy(pack(np.maximum(.02, 5*local_std)))[None].to(args.device)
            pure_pre, pure_mask = correct_defects(noisy, **DETECTOR)
            defect_pre, defect_mask = correct_defects(defect, **DETECTOR)
            for label, model in models.items():
                for input_kind, original, pre, flags in [('clean', noisy, pure_pre, pure_mask), ('defect', defect, defect_pre, defect_mask)]:
                    predictions = {}
                    for variant, image in [('raw', original), ('prepass', pre)]:
                        pred = model(image, steps=1, stochastic=False) if isinstance(model, BridgeRefiner) else model(image, cond if model.cond == 'noise_map' else None)
                        if not torch.isfinite(pred).all(): raise ValueError('Nonfinite prediction: '+label)
                        predictions[variant] = pred
                        error = (pred-target).abs()
                        rows.append(dict(slice=name, model=label, variant=variant, input=input_kind,
                            source_sha256=entries[i]['source_sha256'], crop=entries[i]['clean'],
                            model_psnr=psnr(pred, target), defect_sites=int(sites.sum()),
                            defect_survival=float((error[sites] > threshold[sites]).float().mean()) if sites.any() else None,
                            defect_mae=float(error[sites].mean()) if sites.any() else None,
                            flagged_per_million=float(flags.float().mean())*1e6 if variant == 'prepass' else 0.,
                            false_positive_per_million=float(pure_mask.float().mean())*1e6))
                    delta = psnr(predictions['prepass'], target)-psnr(predictions['raw'], target)
                    for row in rows[-2:]: row['prepass_psnr_delta'] = delta
        print('EVALUATED', name, len(dataset), flush=True)
    summary, comparisons = [], []
    for name in dict.fromkeys(r['slice'] for r in rows):
        for variant in ('raw', 'prepass'):
            for input_kind in ('clean', 'defect'):
                group = [r for r in rows if (r['slice'], r['variant'], r['input']) == (name, variant, input_kind)]
                for label in models:
                    sources = defaultdict(list)
                    for r in group:
                        if r['model'] == label: sources[r['source_sha256']].append(r)
                    result = dict(slice=name, model=label, variant=variant, input=input_kind, sources=len(sources))
                    for key in ('model_psnr', 'defect_survival', 'defect_mae', 'false_positive_per_million', 'prepass_psnr_delta', 'flagged_per_million'):
                        means = [np.mean([r[key] for r in values if r[key] is not None]) for values in sources.values() if any(r[key] is not None for r in values)]
                        result[key] = float(np.mean(means)) if means else None
                    summary.append(result)
                    if label != baseline:
                        c = arch.paired(group, label, baseline)
                        if c:
                            c.update(slice=name, variant=variant, input=input_kind, metric='model_psnr', half_width_95=1.96*c['sem'])
                            comparisons.append(c)
    real = [real_report(path, args.device) for path in args.raw]
    report = dict(protocol='defects_arch_study_dev_v2', status='DEVELOPMENT; imperfect references; not acceptance',
        reference_script_sha256=arch.sha(REFERENCE), manifest_sha256=arch.sha(args.manifest) if models else None,
        checkpoints=provenance, baseline=baseline, density=3e-4, defect_seed=20261029, crop_packed=256,
        selection_seed=20261028, crops=len(entries), device=args.device,
        survival_threshold='max(0.02, 5*sqrt(K*clean_positive + read^2 + row^2 + col^2)), linear units',
        clean_input_definition='Defect-free synthetic noisy input; clean target unchanged',
        clean_site_metrics='At the same sites as the paired injected input (counterfactual)',
        aggregation='Mean within source then across sources; paired source PSNR deltas with normal 95% CI',
        detector=dict(k=8., abs_floor=.01, cross_channel_guard=True, chunk_rows=512),
        summary=summary, comparisons=comparisons, rows=rows, real=real)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(arch.finite_json(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['Development defect diagnostics; linear sensor units. Clean input means noisy input without injected defects.', '',
             '| Model | Slice | Input | Variant | PSNR | Survival | Defect MAE | Clean FP / million | Prepass ΔPSNR |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(value): return '—' if value is None else f'{value:.6g}'
    for r in summary:
        lines.append('| '+' | '.join([r['model'], r['slice'].replace('|', ' / '), r['input'], r['variant']]+[fmt(r[k]) for k in ('model_psnr','defect_survival','defect_mae','false_positive_per_million','prepass_psnr_delta')])+' |')
    lines += ['', '| Model | Baseline | Slice | Input | Variant | Paired ΔPSNR | 95% CI |', '|---|---|---|---|---|---:|---|']
    for c in comparisons:
        lines.append(f"| {c['a']} | {c['b']} | {c['slice'].replace('|', ' / ')} | {c['input']} | {c['variant']} | {fmt(c['mean_delta_db'])} | {fmt(c['ci95'][0])}, {fmt(c['ci95'][1])} |")
    for r in real:
        lines += ['', f"RAW: {r['raw']}; {r['flagged']} flags, {r['per_MP']:.6g}/MP; hot={r['hot']}, dead={r['dead']}.", '',
                  '| Channel | Flags | Per MP | Hot | Dead |', '|---|---:|---:|---:|---:|']
        for c, d in r['channels'].items(): lines.append(f"| {c} | {d['flagged']} | {d['per_MP']:.6g} | {d['hot']} | {d['dead']} |")
        lines += ['', '| y | x | Channel | Kind | Value | Neighbour median |', '|---:|---:|---|---|---:|---:|']
        for d in r['top20']: lines.append(f"| {d['y']} | {d['x']} | {d['channel']} | {d['kind']} | {d['value']:.6g} | {d['neighbour_median']:.6g} |")
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('SAVED', args.out)


if __name__ == '__main__': main()
