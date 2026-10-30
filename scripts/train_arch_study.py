#!/usr/bin/env python3
"""Config-driven training from scratch. Checkpoints contain our own model only."""
import argparse
import math
import hashlib
from pathlib import Path
import json
import random
import shutil
import time
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import trange
from denoise_lab.data import BootstrapDataset, PairedBurstDataset, SyntheticDataset, mixed_sampler
from denoise_lab.model import expand_unconditioned_state, parameter_count
from denoise_lab.arch_registry import build_model
from denoise_lab.losses import DenoiseLoss
from denoise_lab.flat_augment import FlatAugmentedBootstrapDataset
from denoise_lab.clean_targets import BinnedBootstrapDataset


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def dataset(config, require_conditioning=False):
    config = dict(config)
    kind = config.pop("kind")
    if kind == "synthetic":
        return SyntheticDataset(**config), None
    if kind == "bootstrap":
        cls = FlatAugmentedBootstrapDataset if any(key in config for key in ("flat_probability", "flat_options", "defect_probability", "defect_options")) else BootstrapDataset
        if any(key in config for key in ("bin_probability", "bin_factors", "channel_gain_probability", "channel_gain_range", "resite", "iso_coupled", "noise_map_jitter")):
            cls = BinnedBootstrapDataset
        return cls(**config), None
    if kind == "paired":
        if require_conditioning and not config.get("calibration"):
            raise ValueError("Conditioned paired training requires a measured calibration")
        return PairedBurstDataset(**config), None
    if kind == "mixed":
        sources = [dataset(c, require_conditioning)[0] for c in config.pop("sources")]
        return mixed_sampler(sources, **config)
    raise ValueError(f"Unknown dataset kind {kind}")


def make_ema_state(model):
    """Clone without initialization/RNG draws; floating state accumulates in fp32."""
    return {name: (value.detach().float().clone() if value.is_floating_point()
                   else value.detach().clone()) for name, value in model.state_dict().items()}


@torch.no_grad()
def update_ema(ema, model, decay, t):
    """t is the zero-based count of completed optimizer updates before this one."""
    decay_t = min(decay, (1+t)/(10+t))
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].mul_(decay_t).add_(value.detach().float(), alpha=1-decay_t)
        else:
            ema[name].copy_(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--steps", type=int)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--resume", type=Path, help="Resume an interrupted run, including optimizer state")
    p.add_argument("--init-checkpoint", type=Path, help="Initialize weights for a new training stage")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("threads", 4))
    device = choose_device(args.device or cfg.get("device", "auto"))
    cfg.setdefault("model", {}).setdefault("arch", "nafnet")
    model = build_model(cfg["model"]).to(device)
    train = cfg["training"]
    if train.get("identity_init", False) and not (args.resume or args.init_checkpoint):
        torch.nn.init.zeros_(model.body.ending.weight)
        torch.nn.init.zeros_(model.body.ending.bias)
    parameter_count(model)
    ds, sampler = dataset(cfg["data"], require_conditioning=model.cond == "noise_map")
    workers = train.get("workers", 0)
    loader = DataLoader(ds, batch_size=train.get("batch_size", 1), sampler=sampler,
                        shuffle=sampler is None, num_workers=workers, drop_last=False,
                        pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    if len(loader) == 0:
        raise ValueError("Empty training dataset")
    steps = args.steps or train["steps"]
    opt = torch.optim.AdamW(model.parameters(), lr=train.get("lr", 2e-4), weight_decay=train.get("weight_decay", 1e-4),
                            betas=tuple(train.get("betas", (.9, .999))))
    warmup = int(train.get("warmup_steps", 0))
    min_ratio = train.get("min_lr", 1e-6)/train.get("lr", 2e-4)
    def lr_factor(index):
        if warmup and index < warmup:
            return (index+1)/warmup
        fraction = min(1., max(0., (index-warmup)/max(1, steps-warmup)))
        return min_ratio + (1-min_ratio)*.5*(1+math.cos(math.pi*fraction))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    accum = int(train.get("grad_accum", 1))
    if accum < 1:
        raise ValueError("grad_accum must be positive")
    # BF16 avoids FP16 overflow; CPU/MPS retain fp32 operator support.
    amp = bool(train.get("amp", True) and device.type == "cuda")
    amp_dtype = torch.bfloat16 if train.get("amp_dtype", train.get("amp")) in ("bfloat16", "bf16") else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    criterion = DenoiseLoss(**cfg.get("loss", {}))
    run = args.run_dir or Path(train.get("run_dir", "runs/train"))
    train["run_dir"] = str(run)
    run.mkdir(parents=True, exist_ok=True)
    (run/"config.yaml").write_text(yaml.safe_dump(cfg))
    start = 0
    initial = args.resume or args.init_checkpoint
    init_record = None
    if initial:
        saved = torch.load(initial, map_location=device, weights_only=True)
        state = saved["model"]
        if args.init_checkpoint and model.cond == "noise_map":
            state = expand_unconditioned_state(state, model)
        model.load_state_dict(state)
        init_record = {"path": str(initial), "sha256": hashlib.sha256(Path(initial).read_bytes()).hexdigest(),
                       "mode": "resume" if args.resume else "init", "source_step": saved.get("step")}
        if args.resume:
            start = saved["step"]
            opt.load_state_dict(saved["optimizer"])
            sched.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            if "torch_rng_state" in saved:
                torch.set_rng_state(saved["torch_rng_state"].cpu())
    ema_decay = train.get("ema_decay")
    ema = None
    ema_updates = 0
    if ema_decay is not None and ema_decay != 0:
        if not 0 < ema_decay < 1:
            raise ValueError("ema_decay must be in (0, 1), or zero/off")
        ema = make_ema_state(model)
        if args.resume:
            if "ema_model" not in saved:
                raise ValueError("EMA resume requires ema_model in the checkpoint")
            # Strictly validate names/shapes before restoring fp32 accumulators.
            if saved["ema_model"].keys() != ema.keys():
                raise ValueError("EMA checkpoint state keys do not match model")
            for name, value in saved["ema_model"].items():
                if value.shape != ema[name].shape:
                    raise ValueError("EMA checkpoint shape mismatch: " + name)
                ema[name].copy_(value)
            ema_updates = saved.get("ema_updates", start)
    if start >= steps:
        print(f"Checkpoint already completed {start} steps (requested {steps})")
        return
    iterator = iter(loader)
    history = json.loads((run/"history.json").read_text()) if args.resume and (run/"history.json").exists() else []
    history = [row for row in history if row["step"] <= start]
    started = time.monotonic()
    manifest = cfg["data"].get("manifest")
    manifest_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest() if isinstance(manifest, str) else None

    def save_checkpoint(step):
        checkpoint = {"model": model.state_dict(), "model_config": cfg.get("model", {}), "config": cfg,
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                      "step": step, "torch_rng_state": torch.get_rng_state(),
                      "data_manifest_sha256": manifest_sha, "init": init_record,
                      "provisional_bootstrap": cfg["data"]["kind"] == "bootstrap"}
        if ema is not None:
            checkpoint["ema_model"] = ema
            checkpoint["ema_updates"] = ema_updates
        temporary = run/"last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(run/"last.pt")
        (run/"history.json.tmp").write_text(json.dumps(history, indent=2))
        (run/"history.json.tmp").replace(run/"history.json")
        keep = train.get("keep_checkpoints", 0)
        if keep:
            shutil.copyfile(run/"last.pt", run/f"step_{step:07d}.pt")
            for old in sorted(run.glob("step_*.pt"))[:-keep]:
                old.unlink()

    (run/"run_manifest.json").write_text(json.dumps({"config": cfg, "init": init_record, "steps": steps,
        "start": start, "data_manifest_sha256": manifest_sha, "device": str(device),
        "torch": torch.__version__, "started_unix": time.time()}, indent=2))
    model.train()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device} amp={amp} amp_dtype={amp_dtype} steps={steps} start={start}", flush=True)
    for step in trange(start, steps, disable=not train.get("progress", True)):
        opt.zero_grad(set_to_none=True)
        value = 0.
        for micro in range(accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
            cond = batch["noise_map"].to(device) if model.cond == "noise_map" else None
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                prediction = (model.training_prediction(noisy, clean)
                              if hasattr(model, "training_prediction") else model(noisy, cond))
                loss = criterion(prediction, clean)/accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            scaler.scale(loss).backward()
            value += float(loss.detach())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.get("grad_clip", 1.), error_if_nonfinite=True)
        previous_scale = scaler.get_scale() if ema is not None else None
        scaler.step(opt)
        scaler.update()
        if ema is not None and scaler.get_scale() >= previous_scale:
            update_ema(ema, model, ema_decay, ema_updates)
            ema_updates += 1
        sched.step()
        history.append({"step": step+1, "loss": value, "lr": sched.get_last_lr()[0],
                        "elapsed": time.monotonic()-started})
        if step == start or (step+1) % train.get("log_every", 1) == 0:
            elapsed = time.monotonic()-started
            status = {**history[-1], "total_steps": steps, "status": "running",
                      "elapsed_seconds": elapsed, "steps_per_second": (step+1-start)/max(elapsed, 1e-6)}
            (run/"progress.json").write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
        every = train.get("checkpoint_every", 0)
        if every and (step+1) % every == 0:
            save_checkpoint(step+1)
    save_checkpoint(steps)
    (run/"progress.json").write_text(json.dumps({**history[-1], "status": "completed",
                                                "elapsed_seconds": time.monotonic()-started}, indent=2))
    print(f"Completed {steps} steps; first_loss={history[0]['loss']:.7f} last_loss={history[-1]['loss']:.7f}; checkpoint={run/'last.pt'}")


if __name__ == "__main__":
    main()
