#!/usr/bin/env python3
"""Export unconditioned checkpoints. Default: fixed 512, float32 I/O and compute.

Use --precision fp32 for CPU/DirectML portability. With --dynamic, batch/H/W
are dynamic; H and W must be multiples of 16 for a four-stage network.
"""
import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch import nn

from denoise_lab.arch_registry import build_model
from denoise_lab.checkpoint import checkpoint_weights
from denoise_lab.model import DenoiseNet, gamma, inverse_gamma


class FloatIO(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.half()

    def forward(self, inp_1):
        return self.model(inp_1.half()).float()


class MixedPrecisionIO(nn.Module):
    """Half-precision NAFNet body with float32 sensor/gamma/skip arithmetic.

    The unquantized gamma input travels around the body. A zero correction
    preserves every input code exactly, including deep-shadow subnormals.
    This is still an approximation: trained body errors need DN-based checks.
    """
    def __init__(self, model):
        super().__init__()
        if not isinstance(model, DenoiseNet) or model.cond != "none":
            raise ValueError("fp16mixed requires an unconditioned Bayer NAFNet")
        self.body = model.body.half()
        self.exponent = model.gamma_exponent

    def forward(self, inp_1):
        source = inp_1.float()
        g = gamma(source, self.exponent)
        correction = self.body(g.half()).float()
        # Express the result as a linear correction, preserving the exact source
        # when correction=0 rather than re-encoding the source through a power.
        residual = inverse_gamma(g + correction, self.exponent) - inverse_gamma(g, self.exponent)
        return source + residual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--precision", choices=["fp32", "fp16mixed", "fp16io"], default="fp32")
    p.add_argument("--dynamic", action="store_true", help="Dynamic B/H/W; four-stage H/W must be multiples of 16")
    p.add_argument("--weights", choices=["model", "ema"], default="model")
    args = p.parse_args()
    destination = Path(args.output)
    if destination.exists():
        p.error(f"Refusing to overwrite existing file: {destination}")
    import onnx
    import onnxruntime as ort
    torch.set_num_threads(min(8, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_config"].get("arch") == "rgb_nafnet" or "rgb_contract" in checkpoint:
        p.error("This ONNX exporter supports Bayer checkpoints only; RGB export is unsupported")
    if checkpoint["model_config"].get("cond", "none") != "none":
        p.error("Fixed 4-channel deployment contract requires cond=none")
    model = build_model(checkpoint["model_config"]).eval()
    try:
        model.load_state_dict(checkpoint_weights(checkpoint, args.weights))
    except ValueError as exc:
        p.error(str(exc))
    if args.precision == "fp16io":
        print("WARNING: fp16io quantizes the sensor input and skip path; use fp32 for 16-bit color fidelity.", flush=True)
        wrapper = FloatIO(model).eval()
    elif args.precision == "fp16mixed":
        wrapper = MixedPrecisionIO(model).eval()
    else:
        wrapper = model
    example = torch.zeros(1, 4, 512, 512, dtype=torch.float32)
    axes = {"inp_1": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"}} if args.dynamic else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onnx-export-", dir=destination.parent) as directory:
        staged = Path(directory)/"model.onnx"
        with torch.inference_mode():
            # Export the module directly so dynamic dimensions survive tracing.
            torch.onnx.export(wrapper, example, str(staged), input_names=["inp_1"], output_names=["output"],
                              dynamic_axes=axes, opset_version=17, dynamo=False)
        graph = onnx.load(staged)
        onnx.helper.set_model_props(graph, {"cond": "none", "weights": args.weights, "precision": args.precision})
        onnx.checker.check_model(graph)
        onnx.save(graph, staged)
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(8, torch.get_num_threads())
        session = ort.InferenceSession(str(staged), sess_options=options, providers=["CPUExecutionProvider"])
        generator = torch.Generator().manual_seed(42)
        example = torch.rand((1, 4, 512, 512), generator=generator)
        with torch.inference_mode():
            expected = wrapper(example).numpy()
        actual = session.run(None, {"inp_1": example.numpy()})[0]
        del session  # Release the model file before TemporaryDirectory cleanup on Windows.
        diff = float(np.max(np.abs(expected-actual)))
        limit = 1e-3 if args.precision == "fp32" else 5e-3
        print(f"CPU ONNX vs torch max abs diff: {diff:.9g}; runtime limit={limit:g}", flush=True)
        if not np.isfinite(diff) or diff > limit:
            raise RuntimeError(f"ONNX parity failed: max abs diff {diff} (runtime limit {limit:g})")
        # A failed validation never leaves a final artifact, or alters an old one.
        # Exclusive creation also protects against concurrent exports to this path.
        created = False
        try:
            with destination.open("xb") as output, staged.open("rb") as source:
                created = True
                shutil.copyfileobj(source, output)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
    print(f"Saved {destination}; ONNX checker and parity passed; precision={args.precision}, dynamic={args.dynamic}", flush=True)


if __name__ == "__main__":
    main()
