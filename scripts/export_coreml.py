#!/usr/bin/env python3
"""Export our own checkpoint at the fixed ShopPhoto 4-plane contract."""
import argparse
import fnmatch
import inspect
import numpy as np
import torch
from denoise_lab.arch_registry import build_model
from denoise_lab.checkpoints import load_checkpoint, model_config, resolve_weights

COMPRESSION_MODES = ("none", "palettize4", "palettize6", "palettize8",
                     "int8-linear", "prune-50+palettize8")


def compression_config(model, op_config, *, skip_sensitive=True, skip_ops=()):
    """Exclude first/last convolution weights and optional MIL name globs.

    Register exclusions for constants and consumers for optimizer API versions.
    Use serialized conv order (constant declarations may be hoisted); exclude
    shared constants entirely.
    """
    from coremltools.optimize.coreml import OptimizationConfig, get_weights_metadata
    metadata = get_weights_metadata(model, weight_threshold=0)
    order = []
    def visit(block):
        for op in block.operations:
            if op.type == "conv":
                order.append(op.attributes["name"].immediateValue.tensor.strings.values[0])
            for child in op.blocks:
                visit(child)
    for function in model.get_spec().mlProgram.functions.values():
        for block in function.block_specializations.values():
            visit(block)
    convs = []
    for name, value in metadata.items():
        for child in value.child_ops or []:
            if child.op_type == "conv" and child.params_name_mapping.get("weight") == name:
                convs.append((child.name, name))
    if skip_sensitive:
        positions = {name: i for i, name in enumerate(order)}
        if any(name not in positions for name, _ in convs):
            raise ValueError("Cannot order sensitive convolutions in serialized program")
        convs.sort(key=lambda pair: positions[pair[0]])
    if skip_sensitive and not convs:
        raise ValueError("Cannot identify sensitive convolution weights; use --compress-sensitive to override")
    excluded = {convs[0][1], convs[-1][1]} if skip_sensitive else set()
    # Dual terminal head (output_head="green_split"): protect both ending (3 ch) and ending_gdiff (1 ch).
    if skip_sensitive and len(convs) >= 3:
        last_val = getattr(metadata[convs[-1][1]], "val", None)
        prev_val = getattr(metadata[convs[-2][1]], "val", None)
        if last_val is not None and prev_val is not None and last_val.shape[0] == 1 and prev_val.shape[0] == 3:
            excluded.add(convs[-2][1])
    for name, value in metadata.items():
        names = [name] + [child.name for child in value.child_ops or []]
        if any(fnmatch.fnmatchcase(n, pattern) for n in names for pattern in skip_ops):
            excluded.add(name)
    print("Compression skip constants:", ", ".join(sorted(excluded)))
    excluded_ops = set(excluded)
    for name in excluded:
        excluded_ops.update(child.name for child in metadata[name].child_ops or [])
    return OptimizationConfig(global_config=op_config,
                              op_name_configs={name: None for name in excluded_ops})


def supports_grouped_palettization():
    from coremltools.optimize.coreml import OpPalettizerConfig
    return "granularity" in inspect.signature(OpPalettizerConfig).parameters


def compress_model(model, mode="none", *, skip_sensitive=True, skip_ops=()):
    """Post-training weight compression only; I/O and activations stay unchanged."""
    if mode not in COMPRESSION_MODES:
        raise ValueError(f"Unknown compression mode: {mode}")
    if mode == "none":
        return model
    from coremltools.optimize import coreml as opt
    def config(op):
        return compression_config(model, op, skip_sensitive=skip_sensitive, skip_ops=skip_ops)
    if mode == "int8-linear":
        return opt.linear_quantize_weights(model, config(opt.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8")))
    nbits = 8 if mode.startswith("prune-") else int(mode.removeprefix("palettize"))
    options = dict(mode="kmeans", nbits=nbits)
    if supports_grouped_palettization():
        # Eight divides the released width (40) and its expanded/decoder widths.
        options.update(granularity="per_grouped_channel", group_size=8, channel_axis=0)
    palette = config(opt.OpPalettizerConfig(**options))
    if mode.startswith("prune-"):
        pruning = config(opt.OpMagnitudePrunerConfig(target_sparsity=.5))
        model = opt.prune_weights(model, pruning)
        # Joint compression preserves the sparse representation and palettes its nonzeros.
        return opt.palettize_weights(model, palette, joint_compression=True)
    return opt.palettize_weights(model, palette)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Destination .mlpackage")
    p.add_argument("--weights", choices=["model", "ema"], default="model",
                   help="Checkpoint weights to export (default: model)")
    p.add_argument("--compress", choices=COMPRESSION_MODES, default="none")
    p.add_argument("--compress-skip-op", action="append", default=[], metavar="GLOB",
                   help="Skip weights matching a MIL constant or consumer name; repeatable")
    p.add_argument("--compress-sensitive", action="store_true",
                   help="Allow compression of first/last conv (protected by default)")
    return p


def model_from_checkpoint(checkpoint, weights="model"):
    """Build and load an unconditioned model without importing Core ML tools."""
    if weights not in {"model", "ema"}:
        raise ValueError("weights must be model or ema")
    config = model_config(checkpoint)
    if config.get("cond", "none") != "none":
        raise ValueError("Fixed 4-channel deployment contract requires cond=none")
    model = build_model(config).eval()
    model.load_state_dict(resolve_weights(checkpoint, weights))
    return model


def main():
    p = argument_parser()
    args = p.parse_args()
    try:
        checkpoint = load_checkpoint(args.checkpoint)
        model = model_from_checkpoint(checkpoint, args.weights)
    except ValueError as exc:
        p.error(str(exc))
    import coremltools as ct
    example = torch.zeros(1, 4, 512, 512, dtype=torch.float32)
    # Fixed 512 inputs skip padding; this trace is specialized to that contract.
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, check_trace=False)
    # Grouped LUTs and joint sparse/LUT compression require the newer MIL dialect.
    target = ct.target.macOS13
    if args.compress != "none" and supports_grouped_palettization():
        target = ct.target.macOS15
    converted = ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=target,
                           inputs=[ct.TensorType(name="inp_1", shape=example.shape, dtype=np.float32)],
                           outputs=[ct.TensorType(name="output", dtype=np.float32)])
    converted = compress_model(converted, args.compress,
                               skip_sensitive=not args.compress_sensitive, skip_ops=args.compress_skip_op)
    converted.save(args.output)
    print(f"Saved {args.output}; float32 I/O, fp16 compute, compression={args.compress}")


if __name__ == "__main__":
    main()
