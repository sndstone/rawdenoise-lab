# Opt-in dark-frame physics (round 8 proposal)

The original `NoiseModel` and `CameraNoise` paths remain unchanged unless new
options are selected. In particular, the original camera path already includes
DN quantisation, same-colour horizontal correlation, processed-ISO handling and
separate named profiles for the two X1D II shutter modes. Its banding combines
fixed and temporal amplitudes and redraws both every frame.

`CameraNoise(..., physics_options={...})`, also available through the training
config's `camera_options`, accepts these independently selectable components:

- `banding: true`: separate temporal and repeatable fixed row/column offsets,
  independently for each CFA plane, using corrected estimates under `physics`.
  `fixed_seed` controls the pattern; camera/shutter and crop shape key it. It is
  a stationary band-power model, not a full-sensor shading map or a model of
  arbitrary crop coordinates. Fixed and temporal offsets have zero ensemble mean.
- `quantization: true`: use the profile's absolute black/white DN and
  `physics.code_step_dn` before normalization. Existing signal-dependent camera
  quantisation curves remain in use where declared. Signed black residuals are
  retained; saturation clamps to a valid code. The shared quantizer also accepts
  `effective_bits` with optional `sensor_bits` (otherwise inferred from white DN).
- `horizontal_correlation: true`: reflected same-colour three-tap filter fitted
  to packed-pixel lag-1. `vertical_correlation` applies the equivalent vertical
  filter. This reproduces covariance, not the camera's compression algorithm.
- `moment_read: true`: replace the legacy robust-std Tukey draw and randomly
  located hot-pixel injection with a zero-mean Gaussian scale mixture matching
  dark-pair variance and excess kurtosis. The mixture has finite fourth moments;
  legacy Tukey fits below lambda -0.25 have no finite theoretical fourth moment.
  Read jitter scales the fitted variance too. This option covers temporal tails;
  it does not model persistent pixel defects, shading or per-channel bias.

Omitted options are off. The legacy camera path already applies correlation and
quantisation: these flags select their corrected implementations rather than
remove the old components. `NoiseModel` supports `banding`,
`horizontal_correlation`, `quantization` and `fixed_seed` with normalized profile
stds and absolute DN metadata; its default uniform quantisation approximation
is preserved. `moment_read` and `vertical_correlation` are camera-path options.
Noise maps retain their historical shot/read approximation; the proposed model
is unconditioned.

The fitter retains existing ISO keys and adds `physics` with read variance,
read excess kurtosis, band stds, DN code step and the real reference statistics.
Every measured profile records shutter and crop size. `shutters.<mode>.iso`
keeps modes separate; `CameraNoise(..., shutter_modes={camera: mode})` selects a
mode explicitly. Old named `x1d2` / `x1d2_electronic` entries still work. A
multi-mode input defaults to one complete shutter series (`default_shutter` in
the spec), never an ISO series assembled across different modes. `NoiseModel`
also accepts a `shutter` selecting its calibration's `shutters.<mode>.iso`.

Use the existing JSON spec format, with one dark directory per camera, optional
shutter filter/default, interim `K_per_iso`/overrides and `K_source`:

```sh
PYTHONPATH=$PWD/src python scripts/build_dark_calibration.py --spec runs/noise_synth/spec.json --output calib/darks_physnoise_20261101.yaml --crop-size 1024
PYTHONPATH=$PWD/src python scripts/validate_noise_synth.py --strict
PYTHONPATH=$PWD/src python scripts/validate_noise_synth.py --legacy --model-calibration calib/darks_20261101.yaml --output runs/noise_synth/legacy_validation
```

The committed new calibration was measured from all available frames in each
shutter/ISO group, on a central 1024-sensor-pixel crop. The original calibration
is unchanged. RX100 V remains an estimated profile without measured dark
moments; validation reports it as unavailable and round 8 excludes it.

The validator compares disjoint pair differences / sqrt(2), detrends rows and
columns within each CFA plane, and reports variance, excess kurtosis, horizontal
and vertical lag-1, temporal/fixed row/column powers, and code step. Pair excess
kurtosis is half independent single-frame excess. It corrects band estimates
for finite-width white/correlated-pixel leakage and finite-burst temporal leakage.
Quantisation variance subtraction assumes approximately uniform rounding error;
this is least reliable for the A5100 at ISO 100. Dark frames cannot calibrate K,
shot noise, PRNU, bright-level quantisation, or signal-dependent correlations.

Targets are variance within 15%, excess kurtosis within max(2, 35%), lag-1 within
0.035 absolute, band power within 35% plus four estimated sampling errors, and
code step within 0.001 DN. A temporal band estimate's sampling floor includes
band variance and pixel leakage over four CFA planes and the frame pairs; a
fixed estimate uses the uncertainty of one temporal mean, including temporal
band power / frame count. These are practical fitting tolerances, not formal
confidence intervals for heavy-tailed distributions. `--strict` fails on any
miss or an empty measured reference set. Missing real data needs no access to
private paths: reference moments travel in the calibration YAML. Results are
in-sample fit diagnostics, not held-out validation or denoising-quality evidence.

`configs/round8/r8_physnoise.yaml` proposes a 24k-step random-initialisation
ablation at the existing w40 architecture/loss/batch settings, retaining 65%
broad-prior samples and adding 35% measured physics. Do a matched same-seed
broad-prior control and real-photo development checks before considering more
physics weight: round 6's calibrated prior regressed X1D II development slices.
No training was performed for this change. The detailed audit, tables and CPU
batch evidence live locally under `runs/noise_synth/` and remain gitignored.
