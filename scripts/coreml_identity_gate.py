#!/usr/bin/env python3
"""Trained-weight Core ML fp16 (CPU_AND_NE) latency + flat-field identity gate through the production tiler.

Run: TMPDIR=/tmp/coreml_tmp PYTHONPATH=src:scripts /usr/bin/python3 scripts/coreml_identity_gate.py --checkpoint NAME=PATH --out OUT.json
Uses the same kinds/levels/thresholds as scripts/evaluate_identity.py, plus PyTorch fp32 parity on a noisy tile.
"""
from __future__ import annotations
import argparse, importlib.util, json, statistics, tempfile, time
from pathlib import Path
import numpy as np
import torch
from denoise_lab.arch_registry import build_model
from denoise_lab.tiling import tiled_inference

spec = importlib.util.spec_from_file_location('ident', Path(__file__).with_name('evaluate_identity.py'))
ident = importlib.util.module_from_spec(spec); spec.loader.exec_module(ident)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', action='append', required=True)
    ap.add_argument('--frame-size', type=int, nargs=2, default=(1100, 1500))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    import coremltools as ct
    report = dict(precision='fp16', compute_units='CPU_AND_NE', frame_size=a.frame_size,
                  thresholds=dict(abs=ident.THRESHOLD, rel=ident.RELATIVE_THRESHOLD, bias=ident.BIAS_THRESHOLD), models={})
    for item in a.checkpoint:
        name, path = item.split('=', 1)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = build_model(saved['model_config']).eval(); model.load_state_dict(saved['model'])
        x = torch.rand(1, 4, 512, 512)*.05+.005
        x = x+torch.randn_like(x)*.004
        with torch.no_grad():
            traced = torch.jit.trace(model, x, check_trace=False)
            ref = model(x).numpy()
        ml = ct.convert(traced, inputs=[ct.TensorType(name='inp', shape=x.shape)], convert_to='mlprogram',
                        compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS13,
                        compute_units=ct.ComputeUnit.CPU_AND_NE)
        def run(t):
            out = ml.predict({'inp': t.numpy().astype(np.float32)})
            return torch.from_numpy(np.asarray(next(iter(out.values())), np.float32))
        for _ in range(4): run(x)
        ts = []
        for _ in range(30):
            s = time.perf_counter(); y = run(x); ts.append((time.perf_counter()-s)*1000)
        res = dict(checkpoint=path, step=saved.get('step'), median_ms=statistics.median(ts),
                   frame_seconds_x2d=statistics.median(ts)*140/1000,
                   fp16_vs_fp32_max_abs=float(np.abs(y.numpy()-ref).max()), rows=[])
        for kind in ident.KINDS:
            for level in ident.LEVELS:
                img = ident.flat_frame(level, kind, *a.frame_size, 'cpu')
                pred = tiled_inference(lambda t: run(t), img, tile_size=512, overlap=58, feather_power=2.2)
                res['rows'].append(dict(kind=kind, level=level, **ident.error_metrics(pred-img, level)))
        res['summary'] = {k: dict(max_abs_error=max(r['max_abs_error'] for r in res['rows'] if r['kind'] == k),
                                  passed=all(r['passed'] for r in res['rows'] if r['kind'] == k)) for k in ident.KINDS}
        res['passed'] = all(v['passed'] for v in res['summary'].values())
        report['models'][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
        a.out.write_text(json.dumps(report, indent=1)+'\n')


if __name__ == '__main__':
    main()
