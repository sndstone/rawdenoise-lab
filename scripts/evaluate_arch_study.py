#!/usr/bin/env python3
"""Matched DEV-v2 development diagnostics, including stochastic hallucination tests.

Validation references are imperfect, previously inspected low-ISO frames; these
scores are not an untouched acceptance test or a measured sensor-noise study.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from denoise_lab.arch_registry import build_model
from denoise_lab.bayer import pack, unpack
from denoise_lab.bootstrap import add_bootstrap_noise
from denoise_lab.data import BootstrapDataset
from denoise_lab.diffusion import BridgeRefiner
from denoise_lab.metrics import psnr
from denoise_lab.model import gamma

# Frozen reference script is imported by file path, never edited or duplicated.
REFERENCE = Path(__file__).with_name('evaluate_dev_v2.py')
_spec = importlib.util.spec_from_file_location('arch_study_dev_v2', REFERENCE)
dev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dev)
SLICES, LEVELS, STRESS = dev.SLICES, dev.LEVELS, dev.STRESS
masks, paired, binned_item, color_error = dev.masks, dev.paired, dev.binned_item, dev.color_error


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slice_name(spec):
    exposure,level,ref=spec
    return f'{ref}|{exposure:g}|{level}'


def high_frequency(g):
    axis=torch.arange(-3,4,device=g.device,dtype=g.dtype)
    kernel=torch.exp(-axis.square()/2);kernel=kernel/kernel.sum()
    kernel=(kernel[:,None]*kernel[None,:]).expand(g.shape[1],1,7,7)
    return g-F.conv2d(F.pad(g,(3,3,3,3),mode='reflect'),kernel,groups=g.shape[1])


def hf_metrics(pred,target):
    hp,hc=high_frequency(gamma(pred)).double(),high_frequency(gamma(target)).double()
    denom=hc.square().mean().clamp_min(1e-20)
    beta=(hp*hc).mean()/denom
    return dict(hf_energy_ratio=float(hp.square().mean()/denom),hf_beta=float(beta),
                hf_unexplained=float((hp-beta*hc).square().mean()/denom))


def gamma_std(outputs,inputs,mask):
    # Empirical independent-draw standard deviations at each pixel/channel;
    # ratio of spatial means on the same fixed clean-reference mask.
    p=torch.stack(outputs).std(0,unbiased=True).mean(1)[0][mask].mean()
    n=torch.stack(inputs).std(0,unbiased=True).mean(1)[0][mask].mean()
    return float(p),float(p/n.clamp_min(1e-12))


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',action='append',required=True,metavar='NAME=PATH')
    ap.add_argument('--weights',choices=['model','ema'],default='model')
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--root',type=Path,default=Path('data'))
    ap.add_argument('--slices',nargs='+',help='Exact names, e.g. native|1|low; comma-separated also accepted')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--bridge-steps',default='1,2,4,8,16')
    ap.add_argument('--bridge-stochastic-steps',default='4,16')
    ap.add_argument('--bridge-seeds',type=int,default=4)
    ap.add_argument('--realizations',type=int,default=4)
    args=ap.parse_args()
    if args.realizations<2 or args.bridge_seeds<2:ap.error('at least two realizations and bridge seeds are required')
    torch.set_num_threads(2)
    started=time.monotonic()
    grouped=defaultdict(list)
    for entry in json.loads(args.manifest.read_text()):grouped[entry['source_sha256']].append(entry)
    rng=random.Random(20261028)
    entries=[]
    for _,values in sorted(grouped.items()):
        values=sorted(values,key=lambda e:e['clean'])
        entries+=rng.sample(values,min(3,len(values)))
    if not entries:raise ValueError('empty validation manifest')
    selected={s for part in args.slices or [] for s in part.split(',')}
    known={slice_name(s) for s in SLICES}
    if selected-known:ap.error('unknown slice(s): '+', '.join(sorted(selected-known)))
    arms,provenance={},{}
    for spec in args.checkpoint:
        name,path=spec.split('=',1)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=build_model(saved['model_config']).to(args.device).eval()
        key = 'ema_model' if args.weights == 'ema' else 'model'
        if key not in saved:
            raise ValueError(f"{path}: requested {args.weights} weights but checkpoint has no {key}")
        model.load_state_dict(saved[key])
        if name in provenance:raise ValueError('duplicate checkpoint label: '+name)
        provenance[name]=dict(checkpoint=path,sha256=sha(path),model_config=saved['model_config'],step=saved.get('step'),weights=args.weights)
        if isinstance(model,BridgeRefiner):
            for k in map(int,filter(None,args.bridge_steps.split(','))):arms[f'{name}@K{k}']=(model,k,False)
            for k in map(int,filter(None,args.bridge_stochastic_steps.split(','))):arms[f'{name}@K{k}s']=(model,k,True)
        else:arms[name]=(model,1,False)
    if args.baseline not in arms:ap.error('--baseline must name an evaluated arm, including @K suffix for bridges')
    if any(k<1 for _,k,_ in arms.values()):ap.error('bridge steps must be positive')
    # CPU-generated noise moves to the input device and is reused by arm/seed;
    # full-frame fields may be sliced identically for future tiled evaluation.
    def predict(arm,noisy,cond,seed=20261028):
        model,k,stochastic=arm
        if isinstance(model,BridgeRefiner):
            eps=None
            if stochastic:
                eps=torch.randn((k,*noisy.shape),generator=torch.Generator().manual_seed(seed)).to(noisy.device)
            return model(noisy,steps=k,stochastic=stochastic,eps_fields=eps)
        return model(noisy,cond if model.cond=='noise_map' else None)
    rows=[]
    with torch.inference_mode():
        for si,spec in enumerate(SLICES):
            name=slice_name(spec)
            if selected and name not in selected:continue
            exposure,level,ref=spec
            parameters={**LEVELS,**STRESS}[level]
            dataset=BootstrapDataset(entries,root=args.root,crop_size=256,augment=False,deterministic=True,
                                     seed=20261028+1000*si,fixed_parameters=parameters,exposure_scale=exposure)
            for i in range(len(dataset)):
                item=dataset[i]
                if ref=='binned':item=binned_item(item['clean'],parameters,20261028+1000*si+i)
                target=item['clean'][None].to(args.device)
                cond=item['noise_map'][None].to(args.device)
                flat,texture=masks(target)
                noisy_draws=[item['noisy'][None].to(args.device)]
                # Preserve the DEV-v2 first draw exactly. Extra draws use the
                # identical selected clean crop, not a newly sampled crop.
                clean_mosaic=unpack(item['clean'].numpy())
                for r in range(1,args.realizations):
                    seed=20261028+1000000*r+1000*si+i
                    mosaic,_=add_bootstrap_noise(clean_mosaic,np.random.default_rng(seed),fixed_parameters=parameters)
                    noisy_draws.append(torch.from_numpy(np.ascontiguousarray(pack(mosaic),np.float32))[None].to(args.device))
                input_gamma=[gamma(n) for n in noisy_draws]
                for label,arm in arms.items():
                    predictions=[predict(arm,n,cond) for n in noisy_draws]
                    if not all(torch.isfinite(p).all() for p in predictions):raise ValueError('nonfinite output: '+label)
                    pred=predictions[0]
                    row=dict(slice=name,model=label,source_sha256=entries[i]['source_sha256'],crop=entries[i]['clean'],
                        input_psnr=psnr(noisy_draws[0],target),model_psnr=psnr(pred,target),
                        interior_psnr=psnr(pred[...,16:-16,16:-16],target[...,16:-16,16:-16]),
                        gamma_mse=float((gamma(pred)-gamma(target)).square().mean()),
                        channel_bias=(pred-target).mean((0,2,3)).cpu().tolist(),
                        flat_psnr=dev.masked_psnr(pred,target,flat),texture_psnr=dev.masked_psnr(pred,target,texture),
                        color_mse=color_error(pred,target),**hf_metrics(pred,target))
                    output_gamma=[gamma(p) for p in predictions]
                    for kind,mask in [('flat',flat),('texture',texture)]:
                        std,ratio=gamma_std(output_gamma,input_gamma,mask)
                        row[f'realization_std_{kind}']=std
                        row[f'realization_std_ratio_{kind}']=ratio
                    if arm[2]:
                        seeded=[gamma(pred)]+[gamma(predict(arm,noisy_draws[0],cond,20261028+s)) for s in range(1,args.bridge_seeds)]
                        for kind,mask in [('flat',flat),('texture',texture)]:
                            std,ratio=gamma_std(seeded,input_gamma,mask)
                            row[f'seed_std_{kind}']=std
                            row[f'seed_std_ratio_{kind}']=ratio
                    rows.append(row)
            print('EVALUATED',name,len(dataset),round(time.monotonic()-started),flush=True)
        identity=[]
        for label,arm in arms.items():
            for level in [.001,.01,.05,.18,.7]:
                x=torch.full((1,4,512,512),level,device=args.device)
                p=predict(arm,x,torch.zeros_like(x[:,:1]))
                identity.append(dict(model=label,level=level,max_abs_error=float((p-x).abs().max()),mean_bias=float((p-x).mean())))
    summary,comparisons=[],[]
    for name in dict.fromkeys(r['slice'] for r in rows):
        srows=[r for r in rows if r['slice']==name]
        for label in arms:
            mr=[r for r in srows if r['model']==label]
            if not mr:continue
            sources=defaultdict(list)
            for r in mr:sources[r['source_sha256']].append(r)
            keys=[k for k,v in mr[0].items() if isinstance(v,(int,float))]
            result=dict(slice=name,model=label,sources=len(sources),crops=len(mr))
            for key in keys:
                result[key]=float(np.mean([np.mean([r[key] for r in values]) for values in sources.values()]))
            result['channel_bias']=np.mean([np.mean([r['channel_bias'] for r in values],axis=0) for values in sources.values()],axis=0).tolist()
            summary.append(result)
            if label!=args.baseline:
                for key in ['model_psnr','interior_psnr','flat_psnr','texture_psnr']:
                    c=paired(srows,label,args.baseline,key)
                    if c:
                        c.update(slice=name,metric=key,half_width_95=1.96*c['sem'])
                        comparisons.append(c)
    output=dict(protocol='arch_study_dev_v2',status='DEVELOPMENT diagnostics; previously inspected imperfect references; not acceptance',
        reference_script_sha256=sha(REFERENCE),manifest_sha256=sha(args.manifest),checkpoints=provenance,
        sources=len(grouped),crops=len(entries),crop_packed=256,selection_seed=20261028,realizations=args.realizations,
        bridge_seeds=args.bridge_seeds,baseline=args.baseline,levels=LEVELS,stress=STRESS,
        hf_definition='sigma=1, 7x7 Gaussian, reflected boundary, energy and regression pooled across channels',
        std_definition='mean per-pixel sample std in gamma space; ratio to same-mask empirical sensor-input std',
        summary=summary,comparisons=comparisons,flat_field_identity=identity,rows=rows,seconds=time.monotonic()-started)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(finite_json(output),indent=2,allow_nan=False)+'\n')
    lines=['| Arm | Slice | RAW PSNR | Gamma MSE | Flat | Texture | HF ratio | HF U | Real. std flat/texture | Seed std flat/texture |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        seed=f"{r['seed_std_ratio_flat']:.3g}/{r['seed_std_ratio_texture']:.3g}" if 'seed_std_ratio_flat' in r else '—'
        lines.append(f"| {r['model']} | {r['slice'].replace('|',' / ')} | {r['model_psnr']:.3f} | {r['gamma_mse']:.4g} | {r['flat_psnr']:.3f} | {r['texture_psnr']:.3f} | {r['hf_energy_ratio']:.3g} | {r['hf_unexplained']:.3g} | {r['realization_std_ratio_flat']:.3g}/{r['realization_std_ratio_texture']:.3g} | {seed} |")
    args.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('SAVED',args.out,flush=True)

if __name__=='__main__':main()
