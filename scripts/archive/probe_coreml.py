#!/usr/bin/env python3
"""Independent random-weight export/runtime probe. Run with system Python 3.9."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
import numpy as np
import torch
import yaml
from denoise_lab.arch_registry import build_model
from denoise_lab.diffusion import BridgeRefiner, UnrolledRefiner


def compute_plan_fraction(ct,path):
    """Count operations whose preferred compute device is a Neural Engine."""
    try:
        cls=ct.models.compute_plan.MLComputePlan
        plan=cls.load_from_path(str(path),compute_units=ct.ComputeUnit.CPU_AND_NE)
        structure=plan.model_structure
        program=structure.program
        ops=[]
        def visit(block):
            for op in block.operations:
                ops.append(op)
                for child in getattr(op,'blocks',[]):visit(child)
        for function in program.functions.values():visit(function.block)
        assigned=[]
        for op in ops:
            usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred=usage.preferred_compute_device if usage else None
            if preferred is not None:assigned.append('NeuralEngine' in type(preferred).__name__)
        return (sum(assigned)/len(assigned) if assigned else 'unavailable'),None
    except Exception as exc:return 'unavailable',str(exc)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',action='append',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--iterations',type=int,default=30)
    ap.add_argument('--warmup',type=int,default=5)
    args=ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    configs=args.config or [Path('configs/arch_study')/(n+'.yaml') for n in ['nafnet_s1','nafnet_s2','mdta','win_mid','bridge']]
    axis=torch.linspace(.01,.09,512)
    x=(axis[None,None,None,:]+axis[None,None,:,None])*.5
    x=x.expand(1,4,512,512).contiguous()+torch.randn(1,4,512,512)*.003
    rows=[]
    for path in configs:
        cfg=yaml.safe_load(path.read_text())['model']
        for steps in ([1,4] if cfg.get('arch')=='bridge_refiner' else [None]):
            row=dict(arm=path.stem,steps=steps,ane_op_fraction='unavailable')
            try:
                import coremltools as ct
                model=build_model(cfg).eval()
                # Keep attention/FFN branches active through MIL optimization;
                # their training initialization is otherwise an exact zero skip.
                for module in model.modules():
                    if hasattr(module, 'beta') and hasattr(module, 'gamma'):
                        torch.nn.init.constant_(module.beta, .01)
                        torch.nn.init.constant_(module.gamma, .01)
                row['residual_scales'] = .01
                torch.nn.init.normal_(model.body.ending.weight,std=1e-4)
                torch.nn.init.normal_(model.body.ending.bias,std=1e-5)
                arm=UnrolledRefiner(model,steps) if isinstance(model,BridgeRefiner) else model
                row['parameters']=sum(p.numel() for p in model.parameters())
                with torch.no_grad():
                    reference=arm(x).numpy()
                    traced=torch.jit.trace(arm,x,check_trace=False)
                converted=ct.convert(traced,inputs=[ct.TensorType(name='inp_1',shape=x.shape)],
                    convert_to='mlprogram',compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS13,compute_units=ct.ComputeUnit.CPU_AND_NE)
                row["conversion"]="passed"
                with tempfile.TemporaryDirectory(prefix='arch_coreml_') as tmp:
                    package=Path(tmp)/'model.mlpackage'
                    converted.save(str(package))
                    # Compute plans accept compiled models rather than mlpackages.
                    try:
                        compiled=ct.models.utils.compile_model(str(package))
                        fraction,detail=compute_plan_fraction(ct,compiled)
                        row['ane_op_fraction']=fraction
                        if detail:row['compute_plan_error']=detail
                    except Exception as exc:row['compute_plan_error']=str(exc)
                    inputs={'inp_1':x.numpy()}
                    for _ in range(args.warmup):converted.predict(inputs)
                    times=[]
                    for _ in range(args.iterations):
                        start=time.perf_counter();pred=converted.predict(inputs)
                        times.append((time.perf_counter()-start)*1000)
                    value=next(iter(pred.values()))
                    if not np.isfinite(value).all():raise ValueError('nonfinite Core ML prediction')
                    row.update(median_ms=statistics.median(times),max_abs_diff=float(np.max(np.abs(value-reference))))
            except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
            rows.append(row)
            print(json.dumps(row),flush=True)
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(dict(shape=[1,4,512,512],precision='fp16',compute_units='CPU_AND_NE',iterations=args.iterations,results=rows),indent=2)+'\n')

if __name__=='__main__':main()
