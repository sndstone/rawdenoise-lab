#!/usr/bin/env python3
"""Batch-one fixed 512-packed tile benchmark; random weights are not quality results."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import statistics
import time
import torch
import yaml
from denoise_lab.arch_registry import build_model
from denoise_lab.diffusion import BridgeRefiner, UnrolledRefiner


def synchronize(device):
    if device.type == 'cuda':torch.cuda.synchronize(device)
    elif device.type == 'mps':torch.mps.synchronize()


def benchmark_tile(model, device, *, precision='fp32', warmup=20, iterations=30, size=512):
    """Batch-one random tile timing with synchronized accelerator measurements."""
    if iterations < 1 or warmup < 0 or size < 16 or size % 16:
        raise ValueError('positive iterations, nonnegative warmup and size multiple of 16 required')
    device = torch.device(device)
    dtype = {'fp32':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}[precision]
    generator = torch.Generator().manual_seed(20261101)
    x = (torch.rand(1, 4, size, size, generator=generator)*.1).to(device)
    context = torch.autocast(device.type, dtype=dtype) if precision != 'fp32' else contextlib.nullcontext()
    with torch.inference_mode(), context:
        for _ in range(warmup):
            model(x)
        synchronize(device)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            out = model(x)
            synchronize(device)
            times.append((time.perf_counter()-start)*1000)
        if not torch.isfinite(out).all():
            raise ValueError('nonfinite output')
    return dict(median_ms=statistics.median(times), shape=[1, 4, size, size],
                iterations=iterations, warmup=warmup, dtype=precision)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',action='append',type=Path)
    ap.add_argument('--checkpoint',action='append',default=[],metavar='NAME=PATH')
    ap.add_argument('--device',default='mps')
    ap.add_argument('--dtypes',default='fp32,fp16,bf16')
    ap.add_argument('--iterations',type=int,default=30)
    ap.add_argument('--warmup',type=int,default=20)
    ap.add_argument('--bridge-steps',default='1,2,4,8,16')
    ap.add_argument('--out',type=Path)
    args = ap.parse_args()
    if args.iterations < 1:ap.error('iterations must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    specs=[]
    for path in args.config or ([] if args.checkpoint else sorted(Path('configs/arch_study').glob('[!s]*.yaml'))):
        specs.append((path.stem,yaml.safe_load(path.read_text())['model'],None))
    for spec in args.checkpoint:
        name,path=spec.split('=',1)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        specs.append((name,saved['model_config'],saved['model']))
    results=[]
    device=torch.device(args.device)
    for name,cfg,state in specs:
        model=build_model(cfg).eval()
        if state is not None:model.load_state_dict(state)
        count=sum(p.numel() for p in model.parameters())
        steps=[int(k) for k in args.bridge_steps.split(',')] if isinstance(model,BridgeRefiner) else [None]
        for k in steps:
            arm=UnrolledRefiner(model,k) if k else model
            for precision in args.dtypes.split(','):
                row=dict(arm=name,steps=k,dtype=precision,device=str(device),parameters=count)
                try:
                    model=model.to(device)
                    timing = benchmark_tile(arm, device, precision=precision,
                                            warmup=args.warmup, iterations=args.iterations)
                    median=timing['median_ms']
                    row.update(median_ms=median,ms_per_step=median/(k or 1),seconds_400_tiles=median*.4)
                except Exception as exc:row['error']=str(exc)
                results.append(row)
                print(json.dumps(row),flush=True)
    result=dict(shape=[1,4,512,512],warmup=args.warmup,iterations=args.iterations,results=results)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
