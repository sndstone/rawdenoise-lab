#!/usr/bin/env python3
"""Config-driven training from scratch. Checkpoints contain our own model only."""
import argparse
import hashlib
import inspect
import warnings
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
from denoise_lab.data import BootstrapDataset, PairedBurstDataset, SyntheticDataset, mixed_sampler, resumable_loader
from denoise_lab.model import expand_unconditioned_state, parameter_count
from denoise_lab.arch_registry import build_model
from denoise_lab.losses import DenoiseLoss


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
        return BootstrapDataset(**config), None
    if kind == "paired":
        if require_conditioning and not config.get("calibration"):
            raise ValueError("Conditioned paired training requires a measured calibration")
        return PairedBurstDataset(**config), None
    if kind == "mixed":
        sources = [dataset(c, require_conditioning)[0] for c in config.pop("sources")]
        return mixed_sampler(sources, **config)
    raise ValueError(f"Unknown dataset kind {kind}")


def _check_keys(options, allowed, path):
    unknown = options.keys()-set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {path}.{sorted(unknown)[0]}")


def _keyword_parameters(cls):
    parameters = {}
    for base in reversed(cls.__mro__):
        if "__init__" in base.__dict__:
            parameters.update({key: value for key, value in inspect.signature(base.__init__).parameters.items()
                               if key != "self" and value.kind in (value.POSITIONAL_OR_KEYWORD, value.KEYWORD_ONLY)})
    return parameters


def validate_dataset_config(config, arch_study=False, path="data"):
    """Validate constructor contracts without opening manifests or sensor data."""
    from denoise_lab.camera_noise import CameraNoise, CameraNoiseDataset
    from denoise_lab.clean_targets import BinnedBootstrapDataset, iso_coupled_parameters
    from denoise_lab.flat_augment import FlatAugmentedBootstrapDataset, synthetic_smooth_mosaic
    from denoise_lab.defects import inject_defects

    kind = config.get("kind")
    options = {k: v for k, v in config.items() if k != "kind"}
    if kind == "mixed":
        _check_keys(options, ("sources", "probabilities", "num_samples"), path)
        sources = options["sources"]
        weights = options.get("probabilities", [1]*len(sources))
        if len(weights) != len(sources) or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(f"{path}.probabilities needs one nonnegative weight per source and a positive sum")
        for index, source in enumerate(sources):
            validate_dataset_config(source, arch_study, f"{path}.sources[{index}]")
        return
    builders = {"synthetic": SyntheticDataset, "bootstrap": BootstrapDataset, "paired": PairedBurstDataset}
    if arch_study:
        builders.update(bootstrap=BinnedBootstrapDataset, camera=CameraNoiseDataset)
    if kind not in builders:
        raise ValueError(f"Unsupported {path}.kind: {kind!r}")
    cls = builders[kind]
    # Match the research trainer's bootstrap dispatch, including its inherited contracts.
    if arch_study and kind == "bootstrap":
        cls = BootstrapDataset
        if any(k in options for k in ("flat_probability", "flat_options", "defect_probability", "defect_options")):
            cls = FlatAugmentedBootstrapDataset
        if any(k in options for k in ("bin_probability", "bin_factors", "channel_gain_probability", "channel_gain_range",
                                     "resite", "iso_coupled", "noise_map_jitter",
                                     "clip_highlight_probability", "clip_exposure_range", "clip_ceiling_range")):
            cls = BinnedBootstrapDataset
    parameters = _keyword_parameters(cls)
    _check_keys(options, parameters, path)
    for key, parameter in parameters.items():
        if parameter.default is inspect.Parameter.empty and key not in options:
            raise ValueError(f"Missing {path}.{key}")
    for key, function in (("flat_options", synthetic_smooth_mosaic), ("defect_options", inject_defects),
                          ("iso_coupled", iso_coupled_parameters), ("camera_options", CameraNoise)):
        if options.get(key) is not None:
            allowed = inspect.signature(function).parameters.keys()-{"rng", "mosaic", "h", "w", "calibration", "cameras"}
            _check_keys(options[key], allowed, f"{path}.{key}")
    if kind == "camera":
        CameraNoiseDataset.validate_options(options)
        if not options.get("calibration") or not options.get("cameras"):
            raise ValueError(f"{path} camera noise needs calibration and cameras")
        physics = options.get("camera_options", {}).get("physics_options")
        if physics is not None:
            _check_keys(physics, ("banding", "quantization", "horizontal_correlation", "vertical_correlation",
                                  "moment_read", "fixed_seed"), f"{path}.camera_options.physics_options")
    elif options.get("iso_coupled") and options.get("augment", True):
        warnings.warn(f"{path}.iso_coupled overrides generic noise controls and exposure_range, "
                      "exposure_sampling, bright_exposure_probability; exposure_scale still applies", UserWarning)


def validate_config(config, arch_study=False):
    """Fail on unsupported training/dataset settings before model allocation or IO."""
    allowed = {"seed", "threads", "device", "model", "data", "training", "loss"}
    if arch_study:
        allowed.update(("selfsup", "distill"))
    _check_keys(config, allowed, "config")
    training = config["training"]
    allowed = {"steps", "batch_size", "workers", "lr", "min_lr", "weight_decay", "amp", "amp_dtype",
               "identity_init", "grad_clip", "run_dir", "checkpoint_every", "keep_checkpoints", "log_every", "progress"}
    if arch_study:
        allowed.update(("betas", "warmup_steps", "grad_accum", "ema_decay",
                        "transpose_consistency_weight", "transpose_consistency_fraction"))
    _check_keys(training, allowed, "training")
    if "transpose_consistency_weight" in training and float(training["transpose_consistency_weight"]) < 0:
        raise ValueError("training.transpose_consistency_weight must be nonnegative")
    if "transpose_consistency_fraction" in training and not 0.0 < float(training["transpose_consistency_fraction"]) <= 1.0:
        raise ValueError("training.transpose_consistency_fraction must be in (0, 1]")
    amp_dtypes = ("float16", "fp16", "bfloat16", "bf16")
    if not isinstance(training.get("amp", True), bool) and training.get("amp") not in amp_dtypes:
        raise ValueError("training.amp must be boolean or a supported precision: fp16/bf16")
    if "amp_dtype" in training and training["amp_dtype"] not in amp_dtypes:
        raise ValueError("Unsupported training.amp_dtype")
    if config.get("selfsup"):
        _check_keys(config["selfsup"], ("manifest", "root", "crop_size", "min_iso", "batch_size", "workers",
                                       "camera_weights", "weight", "gamma"), "selfsup")
    if config.get("distill"):
        distill = config["distill"]
        _check_keys(distill, ("checkpoint", "weights", "weight", "min_weight", "kind", "eps",
                              "linear_weight", "gamma_weight", "chroma_weight", "chroma_lowpass_weight",
                              "chroma_lowpass_kernel", "green_balance_weight",
                              "highlight_chroma_weight", "highlight_threshold",
                              "green_dc_weight", "green_dc_kernel",
                              "protect_teacher_highlights"), "distill")
        if not distill.get("checkpoint"):
            raise ValueError("distill.checkpoint is required")
        if distill.get("weights", "model") not in ("model", "ema"):
            raise ValueError("distill.weights must be model or ema")
        if not isinstance(distill.get("protect_teacher_highlights", False), bool):
            raise ValueError("distill.protect_teacher_highlights must be a boolean")
        if float(distill.get("weight", 0.5)) < 0 or float(distill.get("min_weight", distill.get("weight", 0.5))) < 0:
            raise ValueError("distill.weight and distill.min_weight must be nonnegative")
    validate_dataset_config(config["data"], arch_study)


def amp_dtype(training):
    return torch.bfloat16 if training.get("amp_dtype", training.get("amp")) in ("bfloat16", "bf16") else torch.float16


def capture_rng_state():
    numpy = np.random.get_state()
    return {"torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
            "numpy_rng_state": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng_state(saved):
    if "torch_rng_state" in saved:
        torch.set_rng_state(saved["torch_rng_state"].cpu())
    if "python_rng_state" in saved:
        random.setstate(saved["python_rng_state"])
    if "numpy_rng_state" in saved:
        name, keys, position, gaussian, cached = saved["numpy_rng_state"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    if saved.get("cuda_rng_state_all") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_state_all"]])


def training_stream(saved, seed):
    if saved is None:
        return {"version": 1, "seed": seed, "batches": 0, "selfsup_batches": 0}
    state = saved.get("data_stream")
    if state is None:
        print("Legacy checkpoint: no data_stream progress; using previous loader restart behavior. "
              "Exact data continuation is unavailable.", flush=True)
    elif state.get("version") != 1:
        raise ValueError("Unsupported checkpoint data_stream version")
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--steps", type=int)
    p.add_argument("--resume", type=Path, help="Resume an interrupted run, including optimizer state")
    p.add_argument("--init-checkpoint", type=Path, help="Initialize weights for a new training stage")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    seed = cfg.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("threads", 4))
    device = choose_device(args.device or cfg.get("device", "auto"))
    model = build_model(cfg.get("model", {})).to(device)
    train = cfg["training"]
    if train.get("identity_init", False) and not (args.resume or args.init_checkpoint):
        torch.nn.init.zeros_(model.body.ending.weight)
        if model.body.ending.bias is not None:
            torch.nn.init.zeros_(model.body.ending.bias)
        if getattr(model.body, "ending_gdiff", None) is not None:
            torch.nn.init.zeros_(model.body.ending_gdiff.weight)
    parameter_count(model)
    ds, sampler = dataset(cfg["data"], require_conditioning=model.cond == "noise_map")
    workers = train.get("workers", 0)
    steps = args.steps or train["steps"]
    opt = torch.optim.AdamW(model.parameters(), lr=train.get("lr", 2e-4), weight_decay=train.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=train.get("min_lr", 1e-6))
    # BF16 avoids FP16 overflow; CPU/MPS retain fp32 operator support.
    amp = bool(train.get("amp", True) and device.type == "cuda")
    precision = amp_dtype(train)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and precision == torch.float16)
    criterion = DenoiseLoss(**cfg.get("loss", {}))
    run = Path(train.get("run_dir", "runs/train"))
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
            restore_rng_state(saved)
    if start >= steps:
        print(f"Checkpoint already completed {start} steps (requested {steps})")
        return
    stream = training_stream(saved if args.resume else None, seed)
    loader_options = dict(batch_size=train.get("batch_size", 1), num_workers=workers,
                          pin_memory=device.type == "cuda")
    loader = (resumable_loader(ds, sampler, seed=stream["seed"], batches=stream["batches"], **loader_options)
              if stream is not None else DataLoader(ds, sampler=sampler, shuffle=sampler is None,
                                                    persistent_workers=workers > 0, **loader_options))
    if len(loader) == 0:
        raise ValueError("Empty training dataset")
    iterator = iter(loader)
    history = json.loads((run/"history.json").read_text()) if args.resume and (run/"history.json").exists() else []
    history = [row for row in history if row["step"] <= start]
    started = time.monotonic()
    manifest = cfg["data"].get("manifest")
    manifest_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest() if isinstance(manifest, str) else None

    def save_checkpoint(step):
        checkpoint = {"model": model.state_dict(), "model_config": cfg.get("model", {}), "config": cfg,
                      "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                      "step": step, **capture_rng_state(), "data_stream": stream,
                      "data_manifest_sha256": manifest_sha, "init": init_record,
                      "provisional_bootstrap": cfg["data"]["kind"] == "bootstrap"}
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
    print(f"device={device} amp={amp} amp_dtype={precision} steps={steps} start={start}", flush=True)
    for step in trange(start, steps, disable=not train.get("progress", True)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if stream is not None:
            stream["batches"] += 1
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
        cond = batch["noise_map"].to(device) if model.cond == "noise_map" else None
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=precision, enabled=amp):
            loss = criterion(model(noisy, cond), clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.get("grad_clip", 1.), error_if_nonfinite=True)
        scaler.step(opt)
        scaler.update()
        sched.step()
        value = float(loss.detach())
        history.append({"step": step+1, "loss": value, "lr": sched.get_last_lr()[0]})
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
