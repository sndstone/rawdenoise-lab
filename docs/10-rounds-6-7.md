# 10. Rounds 5–7: calibration, wider noise and real-noise self-supervision

Status: experimental; none of these arms replaced `r4_plain`. Synthetic development
scores, fitted burst references and no-reference diagnostics do not establish the
acceptance gates in [05 — Evaluation](05-evaluation.md).

## Experiments and negative results

Round 5 tried a low-ISO tier, longer continuation, stronger gamma loss, about 75%
more X1D II frames and 440 A7 III frames. Matched 24–48k-step comparisons changed
development PSNR by at most about 0.05 dB (see [09](09-clean-targets-and-defects.md)).

Round 6 fine-tuned `r4_plain` (step 120000, hash prefix `0afcbc88`) for 24k steps at
learning rate 1e-4. `r5_ctrl24` continued the original recipe; `r6_wide` extended
shot K and read sigma to 0.02 and row sigma to 0.002; `r6_calib` used 80% camera
profiles and 20% broad-prior noise. The calibrated camera mix included leaf and
electronic X1D II, A7 III, A5100 and an estimated RX100 V profile. K remained an
interim burst/blind estimate: no flat fields were available.

Same-ISO leave-one-out (LOO) burst results, output PSNR in dB; experiment columns
are deltas against `r4_plain`:

| Group | r4_plain | r5_ctrl24 Δ | r6_wide Δ | r6_calib Δ |
|---|---:|---:|---:|---:|
| A7 III ISO 1250 | 38.05 | −0.01 | −0.01 | −0.01 |
| A7 III ISO 6400 | 35.18 | −0.01 | −0.01 | −0.02 |
| A7 III ISO 25600 | 38.99 | +0.00 | +0.01 | −0.01 |
| A7 III ISO 204800 | 26.64 | +0.07 | +1.03 | +1.04 |
| X1D II ISO 3200 | 49.66 | +0.01 | −0.48 | −0.04 |
| X1D II ISO 25600 f/8 | 40.59 | +0.00 | −0.03 | −0.07 |
| X1D II ISO 25600 f/7.1 | 43.67 | −0.00 | −0.06 | −0.06 |

The wide arm lost 1.32 dB on the X1D II ISO 3200 centre frame. Calibration lost
0.20–0.56 dB on dark generic synthetic development exposures and 2.3 dB on the
heavy-tail 1/64 slice. Its own calibrated synthetic test gained up to 1.5 dB for
A5100/processed-ISO shadows and about 5 dB for X1D II ISO 25600 at 1/32 exposure;
that last gain did not transfer to real bursts. It also lost 2.3–2.6 dB on the
ISO-12800-level white-noise ladder. The only clear real gain, about 1 dB at
A7 III ISO 204800, was equally available by widening the prior.

Round 7 kept the supervised control recipe (24k steps, 1e-4, seed 20261110) and
added neighbour-pair loss at weights 0.25 (`r7_ss025`) or 1 (`r7_ss1`). Its 6,195
real noisy crops came from 1,349 RAWs: A7 III 2,680, RX100 V 1,793, A5100 1,668,
X1D II 54. Per-camera sampling shares were 0.3/0.3/0.3/0.1, independent of crop
counts. Neighbour pairs share geometry across packed planes; their independent
noise assumption may fail for correlated or processed RAW noise.

- A7 III ISO 1250–25600: −0.00 to +0.02 dB versus `r4_plain`.
- A7 III ISO 204800: −0.01 dB (weight 0.25), +0.24 dB (weight 1).
- X1D II bursts: −0.01 to −0.05 dB and −0.01 to −0.12 dB, respectively.
- Every synthetic X1D II development slice regressed: 0.08–0.23 dB and
  0.23–0.54 dB, respectively.
- Low-SNR no-reference noise-left proxies rose from 0.012 to 0.024–0.027 for
  RX100 V and from 0.007 to 0.012 for A5100. These are scene-dependent proxies,
  not ground-truth error measurements.

Training excluded X1D II test days, benchmark files and all November 1 bursts.
Sibling captures from benchmark days could still be in self-supervised training,
so those no-reference scores may be optimistic. Neither round was promoted.

## What the measurements support

Dark-statistics snapshots from November 1 give the following mean per-channel
normalised read sigma (these snapshots and the rebuilt calibration need not have
identical frame pooling):

| Camera/shutter | ISO | Read sigma | Temporal row sigma | DN code step |
|---|---:|---:|---:|---:|
| X1D II leaf | 100 | 6.38e-5 | 1.23e-5 | 1 |
| X1D II leaf | 3200 | 9.71e-4 | 1.05e-4 | 1 |
| X1D II leaf | 25600 | 9.72e-4 | 1.07e-4 | 1 |
| X1D II electronic | 3200 | 6.18e-4 | 7.58e-5 | 1 |
| A7 III mechanical | 400 | 2.25e-4 | 2.81e-5 | 1 |
| A7 III mechanical | 800 | 1.18e-4 | 1.37e-5 | 1 |
| A7 III mechanical | 25600 | 2.46e-3 | 2.29e-4 | 1 |
| A5100 mechanical | 100 | 2.63e-4 | 2.37e-5 | 4 |
| A5100 mechanical | 6400 | 3.04e-3 | 2.64e-4 | 4 |

The X1D II read plateau and A7 III ISO 400→800 discontinuity argue against a single
read-noise slope. Electronic-shutter shot gain was assumed equal to leaf shutter;
RX100 V had no darks/flats and remains estimated.

Cross-ISO paired burst fitting was fragile. Only 17% of the X1D II ISO 25600
f/7.1 scene survived the static mask: real PSNR 15.42→28.31 dB versus synthetic
15.93→31.01 dB. ISO 3200 and ISO 25600 f/8 retained only 0.6% and 4.3%, and all
A7 III groups rounded to 0.0%; they were not scored. These are different reference
constructions from the same-ISO LOO table above. X1D II temporal K was roughly
3.0–3.2e-4 at ISO 3200 and 5.9–7.1e-4 at ISO 25600, still subject to scene motion.

The original camera benchmark had low-SNR noise-left proxies 0.00662 (A5100),
0.0119 (RX100 V), 0.0100 (A7 III) and 0.00587 (X1D II). At RX100 V ISO
6400–25600 all five blind estimates lay outside the original prior; synthetic
white-noise PSNR was −0.76→32.86 dB. This motivates checking noise range and
structure but does not validate real-photo quality across cameras.

Evidence: private main-checkout `runs/round6/RESULTS.md`, `runs/round7/RESULTS.md`,
`runs/{x1d2,a7m3,a5100}_darks_20261101/dark_stats.json`,
`runs/{x1d2,a7m3}_burst_20261101/summary.md` and `runs/camera_bench/summary.md`.
These data and outputs are not distributed with the repository. Historical scores
above have not been recomputed after the helper fixes described below.

## Calibration and training interfaces

`calib/darks_20261101.yaml` declares `units: normalized`, a provenance `source`,
and `cameras.<name>.iso.<ISO>` profiles. Signal and offsets use
`(DN-black)/(white-black)`; K satisfies shot variance `K * max(signal, 0)`.
Each profile contains:

| Fields | Meaning |
|---|---|
| `K`, `K_source` | Shot variance coefficient and its provenance/uncertainty |
| `read_std`, `tukey_lambda` | Robust MAD sigma of frame-pair differences / sqrt(2), and fitted read-tail shape |
| `row_std`, `col_std` | Temporal row/column offset sigma |
| `row_fixed_std`, `col_fixed_std` | Fixed-pattern row/column sigma; synthesis combines each with its temporal sigma in quadrature |
| `black_offset`, `hot_density`, `lag1_h` | Additive offset, hot-pixel fraction, same-colour horizontal correlation |
| `code_step`, `black_dn`, `white_dn` | Normalised quantisation step and DN bounds |
| `frames`, `ppcc`, `lag1_v` | Measurement diagnostics; vertical correlation is not synthesised |

Camera-level `processed_isos` excludes those profiles from normal ISO interpolation;
with `processed_probability` they are sampled directly and hot-pixel injection is
disabled. Optional `quantization: sony_curve` selects DN-dependent steps;
otherwise `code_step` is used. Negative black-subtracted values survive down to
DN zero, and output clips at `white_dn`.

Normal ISO is log-uniform; K and sigma fields interpolate logarithmically, while
tail shape, lag, density and offset interpolate linearly in log ISO. K/read jitter
is log-normal. `at(camera, iso, processed=False)` clamps to the normal ISO range
and disables jitter. Tail scale reproduces the robust pair-difference measurement;
heavy-tailed read RMS need not equal `read_std`. Horizontal correlation uses a
normalised three-tap filter on each packed plane.

`train_arch_study.py` accepts `data.kind: camera` (also within mixed sources),
`calibration`, camera weights, `camera_options` and optional `gain_cap_iso`.
The cap attenuates target signal by cap/ISO above the cap. Top-level `selfsup`
sets `manifest`, `root`, packed `crop_size` (multiple of 32), batch size, workers,
`weight`, `gamma`, `min_iso` and camera sampling shares. Each cached manifest row
contains `crop`, `camera`, `iso`, `sigma` and source/crop provenance. Cached arrays
are packed R/G1/B/G2 float16; training loads float32. The neighbour loss divides
squared errors by `2*sigma` (not sigma squared), with a detached full-resolution
consistency target. Identity predictions have zero consistency penalty but still
pay the noisy-neighbour reconstruction error.

## Running the tools

Run from the checkout with its `src` first on the import path. The commands below
are templates requiring private manifests/data and existing independently trained
checkpoints. Replace input paths as appropriate. Direct every generated artifact
to this checkout's `runs/rounds67-tools/`; do not overwrite source data or frozen
checkpoints.

```bash
export PYTHONPATH="$PWD/src"
PY=python
TASK_OUT="$PWD/runs/rounds67-tools"
BASE=/path/to/rawdenoise-lab
MODEL="$BASE/checkpoints/r4_plain_2026-10-30/r4_plain.pt"

# Dark profile construction (the existing private spec contains source paths).
"$PY" scripts/build_dark_calibration.py --spec "$BASE/runs/round6/calib_spec.json" \
  --output "$TASK_OUT/calibration.yaml"

# Cross-ISO pairs: spec {root, groups: [{name, noisy: [...], clean: [...], shift: [dy, dx]}]}.
# Shifts are integer packed pixels; registration must be measured beforehand.
"$PY" scripts/eval_real_burst.py --spec paired_spec.json --model "$MODEL" \
  --device cpu --output "$TASK_OUT/paired"

# Same-ISO LOO: spec {root, groups: [{name, files: [...], test: [0, 1]}]}.
# Use at least three independent frames with identical ISO/exposure/framing.
"$PY" scripts/eval_burst_loo.py --spec loo_spec.json --model "r4=$MODEL" \
  --device cpu --output "$TASK_OUT/loo"

# Profile-matched synthetic controls (validation manifest paths are fixed in CASES).
"$PY" scripts/eval_camera_noise.py --calibration calib/darks_20261101.yaml \
  --model "r4=$MODEL" --crops 24 --device cpu --output "$TASK_OUT/camera_noise"

# Benchmark selection writes copies, hashes and manifests; source archive stays read-only.
"$PY" scripts/benchmark_cameras.py select --scan scan.json --source-root /path/to/archive \
  --provenance "$BASE/data/PROVENANCE.csv" --output "$TASK_OUT/bench_inputs"
"$PY" scripts/benchmark_cameras.py run --bench "$TASK_OUT/bench_inputs" \
  --model "$MODEL" --device cpu --output "$TASK_OUT/bench"
"$PY" scripts/benchmark_cameras.py report --output "$TASK_OUT/bench"

# Real noisy crops only; exclusion manifests must cover evaluation capture days/files.
"$PY" scripts/cache_real_noisy.py --scan scan.json --source-root /path/to/archive \
  --exclude-days test_days.json burst_days.json --exclude-files benchmark_files.json \
  --output "$TASK_OUT/selfsup_real" --workers 2

# Fine-tuning hooks; choose the wide/calibrated or self-supervised config.
# For self-supervision, edit a local config copy to use the generated cache manifest/root.
"$PY" scripts/train_arch_study.py --config configs/round6/r6_calib.yaml \
  --init-checkpoint "$MODEL" --run-dir "$TASK_OUT/r6_calib" --device cpu
"$PY" scripts/train_arch_study.py --config configs/round7/r7_ss025.yaml \
  --init-checkpoint "$MODEL" --run-dir "$TASK_OUT/r7_ss025" --device cpu
```

The dark spec supplies each camera's `dir` (with `manifest.json` rows containing
`file`, `iso`, optionally `shutter`), optional shutter filter, `K_per_iso`, required
`K_source`, optional `K_cap_iso`, `K_floor_iso`, string-keyed `K_override`, manual
profiles, quantisation and processed ISOs. At least two dark frames are required
per measured profile. `fit_gain` requires at least two static blocks. Paired scores
subtract estimated reference noise; LOO scores subtract the other frames' mean
noise. Clipping, illumination fitting, finite burst size and motion can bias both.
Calibrated synthetic controls favour their own training prior.

Synthetic CPU tests cover distribution scale/tails/correlation, camera shares,
packed crops, neighbour loss and script math. Helper fixes prevent mask dilation
from wrapping at image edges, retain exact affine-fit blocks, support small gain-fit
and dark-frame fixtures, and report constant-dark correlations as zero. Camera crops
now report their actual bin factor and honour linear/log exposure sampling. Groups
without enough static blocks are skipped before fitting. Configured defect injection in
camera datasets is now applied; the historical implementation ignored it despite
the round 6 config requesting it. Historical training and private-data evaluations
were not rerun, so a new run of that config has changed augmentation behaviour.

Next measurements: flat fields for K, a review of dark-tail fits, real A5100 bursts,
and a narrow noise-range extension beyond the original prior. Real paired data and
blind acceptance remain necessary before promoting any replacement.

## Round 8 (2026-11-04): defect-fixed calibration and physics noise

Both arms fine-tuned from r4_plain for 24k steps on the GPU host (`scripts/gpu_jobs.py`) and were scored with `scripts/scoreboard.py --suites dev burst` (MPS) against the frozen r4_plain. Output: `runs/round8/scoreboard/full/`.

- **r6_calib_fix:** the `r6_calib` config rerun after fixing `CameraNoiseDataset`, which had silently skipped defect injection and `exposure_sampling`.
- **r8_physnoise:** `configs/round8/r8_physnoise.yaml` (65% broad prior + 35% measured physics noise: banding, quantisation, shutter profiles; see [13](13-physics-noise.md)), initialised from r4_plain.

**Real bursts (mean-of-frame leave-one-out PSNR delta vs r4_plain, dB):**

| Slice | r6_calib_fix | r8_physnoise | r6_calib | r5_ctrl24 |
|---|---:|---:|---:|---:|
| A7 III ISO 1250 | -0.02 | -0.01 | -0.01 | -0.01 |
| A7 III ISO 6400 | -0.03 | -0.01 | -0.02 | -0.01 |
| A7 III ISO 25600 | -0.01 | -0.00 | -0.01 | +0.00 |
| A7 III ISO 204800 | +1.04 | +1.22 | +1.05 | +0.07 |
| X1D II ISO 3200 | -0.00 | +0.21 | -0.04 | +0.00 |
| X1D II ISO 25600 f/8 | -0.07 | -0.05 | -0.07 | +0.00 |
| X1D II ISO 25600 f/7.1 | -0.06 | -0.05 | -0.06 | -0.00 |

**Development scope and combined gate (dB):**

| Arm | Dev model PSNR mean | Dev model PSNR worst | Combined gate mean | Combined gate worst |
|---|---:|---:|---:|---:|
| r6_calib_fix | -0.2976 | -2.3545 | -0.2887 | -3.5068 |
| r8_physnoise | -0.0907 | -0.6506 | -0.0764 | -0.8651 |
| r6_calib | -0.3072 | -2.3419 | -0.2986 | -3.5021 |
| r5_ctrl24 | +0.0228 | -0.0976 | +0.0251 | -0.1061 |

The dev-only columns average 28 model-PSNR measurements. The actual gate averages
91 measurements: dev model, flat and texture PSNR plus burst PSNR. The default
dev configuration covers two X1D II reference geometries and has no A7 III dev
manifest; the A7 III evidence here comes from bursts. The combined r8 worst
measurement is flat PSNR on `x1d2_bin4resite/native|0.015625|tails`.
The control also fails the combined gate, despite its positive mean.

The `tails` slice is Gaussian read noise plus sparse impulses, with shot noise,
row and column offsets. It is not a Tukey read-noise draw: see
[`STRESS` in evaluate_dev_v2.py](../scripts/evaluate_dev_v2.py) and
[`add_bootstrap_noise`](../src/denoise_lab/bootstrap.py).
It tests a specific uncalibrated robustness envelope. The largest calibrated-arm
and r8 flat-region losses in the table are on that shadow slice; the control's
worst measurement is instead flat PSNR on `x1d2_bin4resite/native|1|low`.

For r8 at X1D II ISO 3200, the three frame deltas are:

| Frame | PSNR delta dB |
|---|---:|
| B0002510.3FR | +0.031860 |
| B0002512.3FR | +0.575987 |
| B0002514.3FR | +0.025624 |

The mean-frame gain is **+0.211157 dB**; pooling corrected MSE before
converting to a delta gives **+0.041386 dB**. The mean-frame gain is dominated
by the middle frame, whose corrected output MSE is much smaller. These are
different aggregations, not interchangeable estimates. The overlapping LOO
frames are not independent scenes and provide no scene-level interval.
All metrics above were recomputed from `scoreboard.json` in the stored output;
the original `scoreboard.md` retains the unmodified historical report.

**Conclusions:**

- The defect fix did not recover the calibrated arm's development regression.
- r8 has a sensitive positive burst signal on X1D II ISO 3200 and a larger
  A7 III ISO 204800 gain, while its development losses are smaller than the
  calibrated arms. This does not establish independent real-photo benefit.
- r8 fails the unchanged combined gate, so **r4_plain stays**.
- Simulator mismatch is a diagnostic hypothesis. Add matched measured-noise
  development slices without relaxing `tails`; losses there would falsify that
  explanation. Training-matched simulation alone cannot confirm real benefit.

**Next:**

- Use `configs/scoreboard/round9.yaml` for A7 III dev coverage and the separate,
  initially non-gating measured suite; see [evaluation](05-evaluation.md).
- Retain unchanged historical LOO scoring with `scripts/eval_burst_loo.py`.
  That command has no `--align` argument. Alignment is available in the
  cross-ISO evaluator `scripts/eval_real_burst.py --align`; score it separately.
- Obtain independent real evidence and owner blind review against r4_plain.
