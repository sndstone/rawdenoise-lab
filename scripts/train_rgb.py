#!/usr/bin/env python3
"""Experimental paired uint16 RGB TIFF training from random weights only."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from denoise_lab.arch_registry import build_model
from denoise_lab.rgb import RGB_ARCH
from denoise_lab.rgb_data import PairedRGBDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New .pt checkpoint; never overwrite")
    parser.add_argument("--device", default="cpu", help="Explicit torch device; default CPU")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--enc-blocks", type=int, nargs="+", default=[1, 1, 1])
    parser.add_argument("--middle-blocks", type=int, default=2)
    parser.add_argument("--dec-blocks", type=int, nargs="+", default=[1, 1, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--identity-init", action="store_true", help="Initialize the output correction to zero")
    parser.add_argument("--identity-weight", type=float, default=0., help="Optional clean-input identity loss")
    parser.add_argument("--flat-identity-weight", type=float, default=0., help="Optional independently generated constant RGB identity loss")
    parser.add_argument("--color-bias-weight", type=float, default=0., help="Optional per-channel spatial-mean error loss")
    parser.add_argument("--boundary-mode", choices=["zeros", "replicate"], default="zeros", help="Experimental RGB convolution boundary mode")
    parser.add_argument("--upsample-mode", choices=["pixelshuffle", "nearest"], default="pixelshuffle", help="Experimental decoder upsampling")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 0 < args.lr < float("inf"):
        parser.error("steps and batch-size must be positive; lr must be finite and positive")
    if args.width < 1 or min(args.enc_blocks + args.dec_blocks + [args.middle_blocks]) < 0:
        parser.error("width must be positive and block counts nonnegative")
    if any(not 0 <= value < float("inf") for value in (args.identity_weight, args.flat_identity_weight, args.color_bias_weight)):
        parser.error("Identity, flat and color-bias weights must be finite and nonnegative")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing file: {args.output}")
    if args.output.suffix.lower() != ".pt":
        parser.error("RGB training output must be a .pt checkpoint")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    training = PairedRGBDataset(args.manifest, split="train", crop_size=args.crop_size)
    validation = PairedRGBDataset(args.manifest, split="validation", crop_size=args.crop_size, augment=False)
    # Audit every selected TIFF before optimization, not only the sampled crop.
    # Final-test files are only hashed for leakage checks, never decoded,
    # trained on, scored or used for model selection.
    for dataset in (training, validation):
        for index in range(len(dataset)):
            dataset[index]
    fingerprints, sources = {}, []
    for entry in training.all_entries:
        split = entry["split"]
        pair_hashes = []
        for key in ("noisy_path", "clean_path"):
            path = entry[key]
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            previous = fingerprints.setdefault(digest, split)
            if previous != split:
                raise ValueError("RGB identical-file leakage across manifest splits")
            sources.append({"path": str(path), "sha256": digest, "scene": entry["scene"], "split": split})
            pair_hashes.append(digest)
        if pair_hashes[0] == pair_hashes[1]:
            raise ValueError("RGB noisy and independent clean target must not be identical files")
    config = dict(arch=RGB_ARCH, width=args.width, enc_blk_nums=args.enc_blocks,
                  middle_blk_num=args.middle_blocks, dec_blk_nums=args.dec_blocks,
                  boundary_mode=args.boundary_mode, upsample_mode=args.upsample_mode)
    # Preflight augmentation must not affect initialization reproducibility.
    torch.manual_seed(args.seed)
    model = build_model(config).to(args.device)  # Independent random initialization.
    if args.identity_init:
        torch.nn.init.zeros_(model.body.ending.weight)
        torch.nn.init.zeros_(model.body.ending.bias)
    # This generator never consumes the ordinary model/data random stream.
    flat_generator = torch.Generator().manual_seed(args.seed+1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    iterator = iter(loader)
    model.train()
    last_loss = None
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        noisy, clean = batch["noisy"].to(args.device), batch["clean"].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy)
        loss = torch.nn.functional.l1_loss(prediction, clean)
        if args.identity_weight:
            loss = loss+args.identity_weight*torch.nn.functional.l1_loss(model(clean), clean)
        if args.flat_identity_weight:
            flat = (.005+.975*torch.rand((clean.shape[0],3,1,1), generator=flat_generator)).to(args.device).expand_as(clean)
            loss = loss+args.flat_identity_weight*torch.nn.functional.l1_loss(model(flat), flat)
        if args.color_bias_weight:
            loss = loss+args.color_bias_weight*(prediction-clean).mean((2,3)).abs().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite RGB training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        last_loss = float(loss.detach())
    model.eval()
    errors = []
    with torch.inference_mode():
        for batch in DataLoader(validation, batch_size=1):
            prediction = model(batch["noisy"].to(args.device))
            mse = (prediction - batch["clean"].to(args.device)).square().mean()
            if not torch.isfinite(mse):
                raise ValueError("Nonfinite RGB validation output")
            errors.append(float(mse))
    checkpoint = dict(model={k: v.cpu() for k, v in model.state_dict().items()},
        model_config=config, rgb_contract=training.contract, step=args.steps,
        experimental_rgb=True, initialization="independent_random", seed=args.seed,
        data_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        data_manifest=json.loads(args.manifest.read_text(encoding="utf-8")), sources=sources,
        training_recipe={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        metrics={"train_last_l1": last_loss, "validation_center_crop_mse": sum(errors)/len(errors)},
        quality_status="Experimental; no held-out acceptance or pretrained RGB quality demonstrated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(checkpoint, handle)
    print(f"Saved experimental RGB checkpoint: {args.output}; validation is development evidence only")


if __name__ == "__main__":
    main()
