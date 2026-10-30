#!/usr/bin/env python3
"""Random-weight Core ML / ANE latency sweep over NAFNet sizes and tile sizes.

Run with system Python 3.9 (coremltools):  TMPDIR=/tmp/coreml_tmp PYTHONPATH=src /usr/bin/python3 scripts/probe_ane_sweep.py --out runs/nafnet_round/ane_sweep.json
Per-frame estimate = tiles for the X2D II packed frame (4421x5952, overlap 58) x median tile latency.
"""
from __future__ import annotations
import argparse, json, math, statistics, tempfile, time
from pathlib import Path
import numpy as np
import torch
from denoise_lab.model import DenoiseNet

CANDIDATES = {
    "w32_s3":        dict(width=32, enc_blk_nums=[1, 1, 2], middle_blk_num=4, dec_blk_nums=[1, 1, 1]),
    "w48_s3":        dict(width=48, enc_blk_nums=[1, 1, 2], middle_blk_num=4, dec_blk_nums=[1, 1, 1]),
    "w64_s3":        dict(width=64, enc_blk_nums=[1, 1, 2], middle_blk_num=4, dec_blk_nums=[1, 1, 1]),
    "w32_s3_mid12":  dict(width=32, enc_blk_nums=[1, 1, 2], middle_blk_num=12, dec_blk_nums=[1, 1, 1]),
    "w32_s3_deep":   dict(width=32, enc_blk_nums=[2, 2, 4], middle_blk_num=8, dec_blk_nums=[2, 2, 2]),
    "w32_s4":        dict(width=32, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=8, dec_blk_nums=[1, 1, 1, 1]),
    "w32_s4_enc2248_mid12":    dict(width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]),
    "w48_s4":        dict(width=48, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=6, dec_blk_nums=[1, 1, 1, 1]),
    "w40_s4_enc2248_mid12":    dict(width=40, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2]),
    # Additional four-stage capacity candidates.
    "w40_s4":        dict(width=40, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=6, dec_blk_nums=[1, 1, 1, 1]),
    "w40_s4_deep":   dict(width=40, enc_blk_nums=[1, 1, 4, 8], middle_blk_num=8, dec_blk_nums=[1, 1, 2, 2]),
    "w48_s4_m2":     dict(width=48, enc_blk_nums=[1, 1, 2, 2], middle_blk_num=2, dec_blk_nums=[1, 1, 1, 1]),
    "w32_s4_deep":   dict(width=32, enc_blk_nums=[1, 2, 4, 8], middle_blk_num=8, dec_blk_nums=[1, 1, 2, 2]),
    # round 5: extra depth only at 1/8 and 1/16 resolution
    "w40_s4_low8":   dict(width=40, enc_blk_nums=[1, 1, 2, 8], middle_blk_num=8, dec_blk_nums=[1, 1, 1, 2]),
    "w40_s4_low12":  dict(width=40, enc_blk_nums=[1, 1, 3, 8], middle_blk_num=12, dec_blk_nums=[1, 1, 2, 2]),
    "w40_s5":        dict(width=40, enc_blk_nums=[1, 1, 2, 4, 4], middle_blk_num=6, dec_blk_nums=[1, 1, 1, 1, 1]),
    "w44_s4":        dict(width=44, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=6, dec_blk_nums=[1, 1, 1, 1]),
}
FRAME = (4421, 5952)


def tiles(n, t, overlap=58):
    step = t-overlap
    return max(0, math.ceil((n+2*overlap-t)/step))+1


def flops(model, size):
    total = 0
    def hook(m, i, o):
        nonlocal total
        k = m.kernel_size[0]*m.kernel_size[1]
        total += o.numel()*k*m.in_channels//m.groups
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    with torch.no_grad():
        model(torch.zeros(1, 4, size, size))
    for h in hs: h.remove()
    return total*2/1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tiles", default="512,1024")
    ap.add_argument("--only", default="")
    ap.add_argument("--iterations", type=int, default=20)
    a = ap.parse_args()
    import coremltools as ct
    torch.manual_seed(0)
    rows = json.loads(a.out.read_text())["results"] if a.out.exists() else []
    done = {(r["arm"], r["tile"]) for r in rows if "median_ms" in r}
    names = [n for n in CANDIDATES if not a.only or n in a.only.split(",")]
    for name in names:
        for t in map(int, a.tiles.split(",")):
            if (name, t) in done: continue
            model = DenoiseNet(**CANDIDATES[name]).eval()
            for mod in model.modules():
                if hasattr(mod, "beta") and hasattr(mod, "gamma") and isinstance(getattr(mod, "beta"), torch.nn.Parameter):
                    torch.nn.init.constant_(mod.beta, .01); torch.nn.init.constant_(mod.gamma, .01)
            torch.nn.init.normal_(model.body.ending.weight, std=1e-4)
            row = dict(arm=name, tile=t, config=CANDIDATES[name],
                       parameters=sum(p.numel() for p in model.parameters()),
                       gflop_tile=round(flops(model.body, t), 2),
                       frame_tiles=tiles(FRAME[0], t)*tiles(FRAME[1], t))
            x = (torch.rand(1, 4, t, t)*.08+.01)
            try:
                with torch.no_grad():
                    traced = torch.jit.trace(model, x, check_trace=False)
                conv = ct.convert(traced, inputs=[ct.TensorType(name="inp", shape=x.shape)], convert_to="mlprogram",
                                  compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS13,
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)
                with tempfile.TemporaryDirectory() as tmp:
                    pkg = Path(tmp)/"m.mlpackage"; conv.save(str(pkg))
                    try:
                        from probe_coreml import compute_plan_fraction
                    except Exception:
                        import importlib.util, sys
                        spec = importlib.util.spec_from_file_location("probe_coreml", Path(__file__).with_name("probe_coreml.py"))
                        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                        compute_plan_fraction = mod.compute_plan_fraction
                    frac, _ = compute_plan_fraction(ct, ct.models.utils.compile_model(str(pkg)))
                    row["ane_op_fraction"] = frac
                    inp = {"inp": x.numpy()}
                    for _ in range(4): conv.predict(inp)
                    ts = []
                    for _ in range(a.iterations):
                        s = time.perf_counter(); conv.predict(inp); ts.append((time.perf_counter()-s)*1000)
                    row["median_ms"] = statistics.median(ts)
                    row["frame_seconds"] = row["median_ms"]*row["frame_tiles"]/1000
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "config"}), flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(dict(frame_packed=FRAME, overlap=58, precision="fp16",
                                             compute_units="CPU_AND_NE", results=rows), indent=1)+"\n")


if __name__ == "__main__":
    main()
