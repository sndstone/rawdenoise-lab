#!/usr/bin/env python3
"""Locate flat-field identity errors: border vs interior, per channel, per level."""
import argparse, json
import torch
from denoise_lab.arch_registry import build_model

p = argparse.ArgumentParser()
p.add_argument("checkpoints", nargs="+")
p.add_argument("--device", default="cuda:0")
p.add_argument("--size", type=int, default=512)
p.add_argument("--out")
a = p.parse_args()
report = []
for path in a.checkpoints:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    m = build_model(saved["model_config"]).to(a.device).eval()
    m.load_state_dict(saved["model"])
    for level in [.001, .01, .05, .1, .18, .35, .5, .7, .9]:
        for kind in ["gray", "rgb"]:
            x = torch.full((1, 4, a.size, a.size), level, device=a.device)
            if kind == "rgb":  # non-neutral flat: R,G1,B,G2 like a daylight patch
                x = x * torch.tensor([.55, 1., .7, 1.], device=a.device).view(1, 4, 1, 1)
            with torch.no_grad():
                e = (m(x, torch.zeros_like(x[:, :1])) - x).abs()[0]
            b = 32
            inner = e[:, b:-b, b:-b]
            ring = e.clone(); ring[:, b:-b, b:-b] = 0
            idx = int(e.flatten().argmax()); c, rem = divmod(idx, a.size*a.size); yy, xx = divmod(rem, a.size)
            report.append(dict(ckpt=path, level=level, kind=kind, max=float(e.max()),
                               interior_max=float(inner.max()), border_max=float(ring.max()),
                               interior_mean=float(inner.mean()), argmax=[c, yy, xx],
                               per_channel_interior_max=[float(v) for v in inner.amax((1, 2))]))
            r = report[-1]
            print(f"{path.split('/')[-2]:12s} {kind} L={level:<5} max={r['max']:.4g} int_max={r['interior_max']:.4g} "
                  f"border={r['border_max']:.4g} int_mean={r['interior_mean']:.3g} at={r['argmax']}", flush=True)
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
