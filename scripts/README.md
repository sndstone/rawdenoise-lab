# Research scripts

Run commands from the repository root with `PYTHONPATH=$PWD/src` and the project Python environment. Arguments in uppercase and `model.pt` are placeholders; use `--help` for full options. Generated outputs belong under ignored `runs/` directories. No command below is run automatically.

## Classification

Every top-level Python script is **ACTIVE**: a current documented workflow, scoreboard dependency, training/export tool, CLI-adjacent utility, calibration/defect/burst tool or infrastructure entry point. **HISTORICAL** scripts live in `archive/`: superseded experiment-specific evaluators, previews, probes, summaries and one-off inventories. Classification was checked against `README.md`, `docs/`, `tests/`, `configs/`, `scripts/` and `src/`, including sibling imports and file-based test loads.

Start evaluation with `scoreboard.py`; its `dev`, `burst`, `camera` and `speed` suites reuse the retained evaluators. Lower-level commands remain available for targeted diagnostics. See [evaluation](../docs/05-evaluation.md#one-command-development-scoreboard).

## Inference/export

| ACTIVE script | Purpose | Main command |
|---|---|---|
| [denoise.py](denoise.py) | CLI inference wrapper; also available as `rawdenoise` | `python scripts/denoise.py photo.dng --model r4_plain` |
| [denoise_raw_preview.py](denoise_raw_preview.py) | Save normalized model output for an evaluation RAW | `python scripts/denoise_raw_preview.py --raw photo.dng --checkpoint model.pt --output runs/preview` |
| [render_raw_preview.py](render_raw_preview.py) | Render normalized evaluation mosaics with source RAW metadata | `python scripts/render_raw_preview.py --source photo.dng --normalized output.npy --output runs/render` |
| [make_examples.py](make_examples.py) | Build matched noisy/denoised illustration crops | `python scripts/make_examples.py --pair photo.dng photo_denoised.dng --out runs/examples/pair.jpg` |
| [export_onnx.py](export_onnx.py) | Export ONNX with CPU parity verification | `python scripts/export_onnx.py --checkpoint model.pt --output runs/export/model.onnx --precision fp32` |
| [check_np_parity.py](check_np_parity.py) | Compare NumPy and torch preprocessing/tiling with the same CPU ONNX model | `python scripts/check_np_parity.py --model runs/export/model.onnx --output runs/torchfree/parity.json` |
| [export_coreml.py](export_coreml.py) | Export fixed-shape Core ML, optionally compressed | `python scripts/export_coreml.py --checkpoint model.pt --output runs/export/model.mlpackage` |

## Training

| ACTIVE script | Purpose | Main command |
|---|---|---|
| [train.py](train.py) | Baseline trainer and synthetic CPU smoke fixture | `python scripts/train.py --config configs/smoke.yaml --device cpu` |
| [train_arch_study.py](train_arch_study.py) | Research trainer for binned targets, defects and EMA | `python scripts/train_arch_study.py --config CONFIG.yaml --run-dir runs/train` |
| [interpolate_checkpoints.py](interpolate_checkpoints.py) | Blend compatible model or EMA weights with parent provenance | `python scripts/interpolate_checkpoints.py --a A.pt --b B.pt --alpha 0.25 --alpha 0.5 --alpha 0.75 --output-dir runs/interpolation` |

## Evaluation

| ACTIVE script | Purpose | Main command |
|---|---|---|
| [scoreboard.py](scoreboard.py) | Main entry point: development, burst, camera and speed suites | `python scripts/scoreboard.py --checkpoint candidate=model.pt --quick --output runs/scoreboard` |
| [evaluate.py](evaluate.py) | Independent paired-manifest evaluation | `python scripts/evaluate.py --checkpoint model.pt --manifest pairs.json --output runs/paired.json` |
| [evaluate_dev_v2.py](evaluate_dev_v2.py) | Deterministic development slices; used by scoreboard and compression checks | `python scripts/evaluate_dev_v2.py --checkpoint candidate=model.pt --camera x1d2 --output runs/dev.json` |
| [evaluate_measured_dev.py](evaluate_measured_dev.py) | Measured physics-noise development profiles; non-gating scoreboard suite | `python scripts/evaluate_measured_dev.py --checkpoint candidate=model.pt --quick --output runs/measured.json` |
| [benchmark_cameras.py](benchmark_cameras.py) | Camera benchmark preparation, proxies and reports; scoreboard dependency | `python scripts/benchmark_cameras.py run --bench BENCH_DIR --model model.pt --output runs/camera_bench` |
| [bench_tile.py](bench_tile.py) | Tile timing and scoreboard speed-suite helper | `python scripts/bench_tile.py --checkpoint candidate=model.pt --device cpu --out runs/bench.json` |
| [eval_burst_loo.py](eval_burst_loo.py) | Frozen leave-one-out burst comparison; scoreboard dependency | `python scripts/eval_burst_loo.py --spec spec.json --model candidate=model.pt --output runs/loo` |
| [eval_real_burst.py](eval_real_burst.py) | Masked paired-burst evaluation and optional alignment | `python scripts/eval_real_burst.py --spec spec.json --model model.pt --output runs/burst` |
| [eval_camera_noise.py](eval_camera_noise.py) | Camera-calibrated synthetic-noise diagnostic | `python scripts/eval_camera_noise.py --model candidate=model.pt --output runs/camera_noise` |
| [audit_real_raw.py](audit_real_raw.py) | Unpaired RAW distributions, residual and impulse diagnostics | `python scripts/audit_real_raw.py --source photo.dng --label scene --output runs/audit` |
| [evaluate_identity.py](evaluate_identity.py) | Flat-field identity gate; shared Core ML helper and tested round tooling | `python scripts/evaluate_identity.py --checkpoint candidate=model.pt --device cpu --out runs/identity.json` |
| [coreml_identity_gate.py](coreml_identity_gate.py) | Core ML identity/parity gate; compression dependency | `python scripts/coreml_identity_gate.py --package candidate=model.mlpackage --out runs/coreml_identity.json` |
| [evaluate_coreml_compression.py](evaluate_coreml_compression.py) | Compare compressed exports, parity, quality and speed | `python scripts/evaluate_coreml_compression.py --checkpoint model.pt --reference reference.mlpackage --out runs/compression` |
| [evaluate_highlights.py](evaluate_highlights.py) | Highlight clipping and colour-fringe diagnostics | `python scripts/evaluate_highlights.py --checkpoint model.pt --output runs/highlights` |
| [evaluate_defects.py](evaluate_defects.py) | Defect injection/detector evaluator with tested historical helpers | `python scripts/evaluate_defects.py --checkpoint candidate=model.pt --out runs/defects.json` |
| [evaluate_defect_maps.py](evaluate_defect_maps.py) | Held-out dark-map repeatability and synthetic-star check | `python scripts/evaluate_defect_maps.py --dark-dir CAMERA=DARK_DIR --output-dir runs/defect_maps` |
| [measure_tile_seams.py](measure_tile_seams.py) | Tile-origin/overlap phase-consistency diagnostics | `python scripts/measure_tile_seams.py --checkpoint model.pt --output runs/tile_seams` |
| [validate_noise_synth.py](validate_noise_synth.py) | Compare synthetic noise with measured dark profiles | `python scripts/validate_noise_synth.py --strict --output runs/noise_synth/validation` |
| [blind_review.py](blind_review.py) | Prepare offline blind comparisons and score exported votes | `python scripts/blind_review.py prepare photo.dng --arm candidate model.pt 1 --arm input model.pt 0 --output runs/review` |

## Data & calibration

| ACTIVE script | Purpose | Main command |
|---|---|---|
| [calibrate_noise.py](calibrate_noise.py) | Fit scalar per-ISO flat/dark noise profiles | `python scripts/calibrate_noise.py --dark DARK_DIR --flat FLAT_DIR --camera BODY --quant-step STEP --output runs/calibration.yaml` |
| [build_dark_calibration.py](build_dark_calibration.py) | Fit temporal dark profiles from a capture specification | `python scripts/build_dark_calibration.py --spec spec.json --output runs/darks.yaml` |
| [build_defect_map.py](build_defect_map.py) | Build a body-specific map from repeated dark captures | `python scripts/build_defect_map.py DARK_DIR --output-dir runs/defect_maps` |
| [prepare_binned_cache.py](prepare_binned_cache.py) | Prepare same-colour binned training targets from read-only SMB; requires optional NAS packages | `python scripts/prepare_binned_cache.py --help` |
| [sample_smb_metadata.py](sample_smb_metadata.py) | Read-only SMB helper retained for binned-cache imports and tests; requires optional NAS packages | `python scripts/sample_smb_metadata.py --help` |
| [cache_real_noisy.py](cache_real_noisy.py) | Cache stratified real-noisy crops with capture-day exclusions | `python scripts/cache_real_noisy.py --scan scan.json --source-root RAW_DIR --output runs/noisy_cache` |

## Capture

| ACTIVE script | Purpose | Main command |
|---|---|---|
| [check_burst.py](check_burst.py) | Screen fresh RAW bursts for static usable regions | `python scripts/check_burst.py CAPTURE_DIR --output runs/check_burst` |
| [find_burst_candidates.py](find_burst_candidates.py) | Find repeated-capture candidates in a cached metadata catalog | `python scripts/find_burst_candidates.py --catalog catalog.jsonl --output runs/burst_candidates.json` |
| [make_burst_targets.py](make_burst_targets.py) | Build robust independent burst means and optional alignment masks | `python scripts/make_burst_targets.py frame*.npy --output runs/burst/clean.npy --manifest runs/burst/pairs.json` |
| [prepare_cross_iso_pairs.py](prepare_cross_iso_pairs.py) | Group capture metadata; prepare independent cross-ISO targets, masks, support and QA | `python scripts/prepare_cross_iso_pairs.py CAPTURE_DIR --dry-run` (then `--output runs/cross_iso`) |

## Infrastructure

| ACTIVE script | Purpose | Main command |
|---|---|---|
| [gpu_jobs.py](gpu_jobs.py) | Manage prepared LAN GPU jobs; sync is a dry run without `--yes` | `python scripts/gpu_jobs.py --help` |
| [run_cuda.py](run_cuda.py) | Run project scripts in the prepared offline CUDA container | `python scripts/run_cuda.py scripts/train.py --config configs/smoke.yaml --device cuda:0` |

## Archive

Run archived tools as `PYTHONPATH=$PWD/src python scripts/archive/NAME.py --help` from the repository root. Their original data assumptions, optional dependencies and experiment output defaults still apply. Archiving does not validate historical results. NAS tools need the existing optional `nas` dependencies; previews need their original private inputs.

The catalog below documents each archived tool, including utilities that previously had no prose reference. The linked research document provides experiment/data context; it does not necessarily contain an original command for every utility. `evaluate_defects.py` explicitly loads the archived architecture evaluator and imports the archived X2D helper; the X2D tests load its new path. The shared `sample_smb_metadata.py`, `evaluate_dev_v2.py` and identity gates remain active because current tools/tests need them.

| HISTORICAL script | Description | Research context |
|---|---|---|
| [analyze_raws.py](archive/analyze_raws.py) | Early sensor-value/crop survey; no clean-target certification | [03-data.md](../docs/03-data.md) |
| [arch_study_x2d.py](archive/arch_study_x2d.py) | X2D real-RAW architecture diagnostics; helper still loaded by defect evaluation/tests | [07-architecture-study.md](../docs/07-architecture-study.md) |
| [cache_real_examples.py](archive/cache_real_examples.py) | Early high-ISO qualitative-review cache over read-only SMB | [04-training.md](../docs/04-training.md) |
| [catalog_smb_raws.py](archive/catalog_smb_raws.py) | One-off complete read-only SMB metadata inventory | [03-data.md](../docs/03-data.md) |
| [diag_flat_identity.py](archive/diag_flat_identity.py) | Early border/interior identity-error localization | [08-nafnet-round2.md](../docs/08-nafnet-round2.md) |
| [evaluate_arch_study.py](archive/evaluate_arch_study.py) | Historical architecture slices; shared defect-evaluation helper | [07-architecture-study.md](../docs/07-architecture-study.md) |
| [evaluate_bootstrap.py](archive/evaluate_bootstrap.py) | Added-noise evaluation against noisy low-ISO references | [04-training.md](../docs/04-training.md) |
| [evaluate_shadow_refinement.py](archive/evaluate_shadow_refinement.py) | Controlled exposure-attenuation development experiment | [04-training.md](../docs/04-training.md) |
| [fetch_public_raws.py](archive/fetch_public_raws.py) | Early exact-camera public sample collection with provenance hashes | [03-data.md](../docs/03-data.md) |
| [fetch_real_test.py](archive/fetch_real_test.py) | Historical stratified high-ISO no-reference sample collection | [04-training.md](../docs/04-training.md) |
| [inventory_smb.py](archive/inventory_smb.py) | One-off read-only NAS directory inventory | [03-data.md](../docs/03-data.md) |
| [prepare_bootstrap_cache.py](archive/prepare_bootstrap_cache.py) | Original noisy-single-exposure training crop cache | [04-training.md](../docs/04-training.md) |
| [preview_bootstrap_real.py](archive/preview_bootstrap_real.py) | Provisional bootstrap-model qualitative previews | [04-training.md](../docs/04-training.md) |
| [probe_ane_sweep.py](archive/probe_ane_sweep.py) | Round-2 architecture/ANE sweep | [08-nafnet-round2.md](../docs/08-nafnet-round2.md) |
| [probe_coreml.py](archive/probe_coreml.py) | Early Core ML op-placement and latency probe | [07-architecture-study.md](../docs/07-architecture-study.md) |
| [real_examples_audit.py](archive/real_examples_audit.py) | Pilot-round high-ISO crop residual/noise-envelope audit | [05-evaluation.md](../docs/05-evaluation.md) |
| [summarize_nafnet_round.py](archive/summarize_nafnet_round.py) | Round-2 decision-table summarizer | [08-nafnet-round2.md](../docs/08-nafnet-round2.md) |
| [summarize_opus_round.py](archive/summarize_opus_round.py) | Pilot-round frozen decision-rule summarizer | [05-evaluation.md](../docs/05-evaluation.md) |
