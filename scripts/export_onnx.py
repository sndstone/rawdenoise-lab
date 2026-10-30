#!/usr/bin/env python3
"""Export unconditioned checkpoints. Default: fixed 512, float32 I/O, fp16 compute.

Use --precision fp32 for CPU/DirectML portability. With --dynamic, batch/H/W
are dynamic; H and W must be multiples of 16 for a four-stage network.
"""
import argparse

import numpy as np
import torch
from torch import nn

from denoise_lab.arch_registry import build_model


class FloatIO(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.half()

    def forward(self, inp_1):
        return self.model(inp_1.half()).float()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=["fp32", "fp16io"], default="fp16io")
    p.add_argument("--dynamic", action="store_true", help="Dynamic B/H/W; four-stage H/W must be multiples of 16")
    p.add_argument("--weights", choices=["model", "ema"], default="model")
    args = p.parse_args()
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(min(8, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_config"].get("cond", "none") != "none":
        p.error("Fixed 4-channel deployment contract requires cond=none")
    model = build_model(checkpoint["model_config"]).eval()
    if checkpoint.get(args.weights) is None:
        p.error(f"Checkpoint has no {args.weights} weights")
    model.load_state_dict(checkpoint[args.weights])
    wrapper = FloatIO(model).eval() if args.precision == "fp16io" else model
    example = torch.zeros(1, 4, 512, 512, dtype=torch.float32)
    axes = {"inp_1": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}} if args.dynamic else None
    with torch.inference_mode():
        # Export the module directly so dynamic dimensions survive tracing.
        torch.onnx.export(wrapper, example, args.output, input_names=["inp_1"], output_names=["output"],
                          dynamic_axes=axes, opset_version=17, dynamo=False)
    graph = onnx.load(args.output)
    onnx.helper.set_model_props(graph, {"cond": "none", "weights": args.weights, "precision": args.precision})
    onnx.checker.check_model(graph)
    onnx.save(graph, args.output)
    print(f"Saved {args.output}; ONNX checker passed; precision={args.precision}, dynamic={args.dynamic}", flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = min(8, torch.get_num_threads())
    session = ort.InferenceSession(args.output, sess_options=options, providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(42)
    example = torch.rand((1, 4, 512, 512), generator=generator)
    with torch.inference_mode():
        expected = wrapper(example).numpy()
    actual = session.run(None, {"inp_1": example.numpy()})[0]
    diff = float(np.max(np.abs(expected-actual)))
    print(f"CPU ONNX vs torch max abs diff: {diff:.9g}", flush=True)
    if not np.isfinite(diff) or (args.precision == "fp32" and diff > 1e-3):
        raise RuntimeError(f"ONNX parity failed: max abs diff {diff} (fp32 limit 1e-3)")


if __name__ == "__main__":
    main()
