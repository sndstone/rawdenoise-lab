# Tripod burst alignment and reference masks

`scripts/make_burst_targets.py --align` and `scripts/eval_real_burst.py --align`
opt in to `denoise_lab.burst_align`. Without this flag integer-shift reference
preparation remains in use. This is reference preparation research;
it does not promote a model or establish an independent acceptance benchmark.

## Geometry and noise

Arrays are normalized `[N,4,H,W]` packed RAW (R, G1, B, G2). A shift `(dy,dx)`
means `aligned[c,y,x] = source[c,y+dy,x+dx]`. Units are packed pixels, twice the
sensor-pixel displacement. Registration uses partially whitened phase
correlation on green-plane averages, a local DFT peak fit at 0.01 packed pixel,
and the median of up to nine textured tiles. An integer overlap recentering
step reduces window bias. The evaluator's existing integer spec shifts supply
an initial displacement; each frame receives a separate residual translation.
A majority of the registration tiles must depict the same static scene.
Flat or wholly invalid registrations fail explicitly.

Each colour plane is translated independently with bilinear interpolation.
Integer packed shifts (even sensor shifts) are exact. Fractional shifts preserve
CFA identity but blur detail and spatially correlate noise. Unsupported edges
are NaN, never wrapped or filled into the valid mask. For independent source
noise of locally constant variance, the output variance multiplier is
`q = ((1-fy)^2+fy^2) * ((1-fx)^2+fx^2)`. Merging propagates this factor, and block
mean tests use source variance as a conservative bound on correlated noise.
Row noise, fixed pattern, registration uncertainty and interpolation bias are
not included in the target variance estimate. Rotation, lens distortion,
parallax and rolling-shutter motion are not fitted.

## Merge and masks

`robust_merge` takes only independent target frames, and a shot/read NLF dict
(or four per-channel dicts). It rejects samples relative to the temporal median
using `noise.noise_std` and the interpolation factors, then computes a clipped
mean, retained frame counts per CFA sample, estimated mean variance, a confidence
map and a static mask. Motion confidence uses the original temporal residual
energy, so an object removed from the mean remains excluded from scoring.
Masks reject temporal excess in blocks and expand failures by one block without
wrapping across edges. Counts are accepted independent frame counts; they are
not a claim that rejected-motion pixels are valid or that fixed pattern averages
away. The default requires at least 75% temporal support and two retained frames.

`estimate_burst_noise` fits a nonnegative shot/read curve to median temporal
block variances after registration, correcting average interpolation variance.
This is a diagnostic estimated from the capture, not an independent calibration.
Motion affecting most blocks, illumination drift and registration error can
inflate it. Prefer a measured calibration when available.

`match_cross_iso` reuses the evaluator's `fit_gain` callback for per-channel
exposure/offset and smooth local illumination matching. The final mask requires
clean-burst stability, complete noisy support, unclipped values, noisy-burst
stability, and noisy-mean agreement with the matched target under both a block
mean test and a residual-energy test. Target variance includes fitted local gain.
The statistical tests assume approximately Gaussian independent temporal noise;
the five-standard-deviation block cutoffs are conservative approximations.

## Commands and outputs

```bash
PYTHONPATH=$PWD/src python scripts/make_burst_targets.py frame*.npy \
  --output runs/burst_align/clean.npy --manifest runs/burst_align/pairs.json \
  --align --calibration calib/example.yaml

PYTHONPATH=$PWD/src python scripts/eval_real_burst.py --spec spec.json \
  --align --alignment-only --output runs/burst_align/after
```

The target builder still requires JSON capture sidecars and matching ISO,
exposure and aperture. Selected noisy inputs are excluded from the clean target.
It writes aligned noisy mosaics, the target mosaic, a sensor-coordinate
`*_valid.npy`, and a `*_support.npz` with packed-coordinate `effective_count`,
`variance`, `confidence`, `static`, and all packed shifts. The final mask also
tests held-out input agreement, assuming the identical exposure required by the
sidecars. Manifest entries point to aligned inputs and include `valid_mask` and
`support`. `PairedBurstDataset` now consumes both fields, requires at least two
retained target samples per CFA site, and excludes nonfinite samples, zero
confidence and nonstatic support. Named missing files or mismatched geometry
fail explicitly. Training selects fully valid finite windows at even sensor
origins; a deterministic exhaustive fallback finds rare windows or raises a
clear error. Crop and CFA-aware flip/transpose geometry also transform the mask.
Draws follow the explicit dataset seed and the seeded Torch worker stream.
Existing unmasked finite pairs keep their numerical values; crop draws are now
reproducible when Torch seeds are reset. Invalid border samples can be NaN in
stored aligned arrays; full-frame loading replaces nonfinite values with zero
and returns a packed four-plane `valid_mask`. **Full-frame consumers must use
that mask.** `scripts/evaluate.py` excludes masked errors and reports valid
fractions. Linear SSIM requires a fully valid Gaussian window; preview metrics
also exclude the demosaicer's neighbourhood of invalid samples. Unsupported
spatial metrics are null, rather than scores on filled pixels.

`scripts/prepare_cross_iso_pairs.py CAPTURE_DIR --dry-run` reads metadata only
and reports chronological ISO/exposure/aperture groups without decoding or
writing files. Run with `--output runs/cross_iso` to prepare one selected
low/high pair (`--low-group` and `--high-group` disambiguate multiple bursts).
Both bursts need at least two frames. It reuses `prepare_pair(..., align=True)`
for native packed-plane alignment, robust low-ISO merging and the frozen
photometric fit to the high-ISO burst. High-ISO samples never enter the clean
merge. Outputs are finite normalized RGGB `clean.npy`, per-frame `noisy_*.npy`,
sensor-coordinate `valid.npy`, packed per-CFA retained counts `support.npy`,
`pairs.json` and `qa.json`. The manifest is accepted directly by the paired
loader and evaluator. QA includes static fraction, gains, shifts, clipping,
frame counts, target noise and differences of disjoint interleaved half-burst
means under the same registration/photometric fit. This half comparison does
not independently validate those fitted transforms. Preparation rejects below
`--min-static` (default 0.1), nonpositive gain or empty supported masks; these
checks alone do not certify target quality. Inspect QA and native detail before
scoring. Newly inspected scenes remain development evaluation data; preparation
does not authorize training or establish an independent split.

The evaluator writes `alignment.json` (per-frame shifts, NLF fits, clean-only
static area, final usable area and target noise estimates) and per-group masks.
`--alignment-only` needs no checkpoint and runs no denoising. For model scoring,
models run on native noisy inputs before their outputs are translated. With
`--align`, real metrics describe resampled planes; the legacy synthetic control
is not matched to their interpolation/correlation and should not be interpreted
as a native-grid real-versus-synthetic gap.

The paired evaluator now records `metric_protocol` in both JSON outputs and its
summary. Default `paired-burst-v2` averages SSIM only over Gaussian windows fully
supported by the static mask (no supported windows yields JSON null / summary
`n/a`) and propagates `(global_gain * local_gain)^2` into non-aligned reference
variance. Outputs produced before this change used `paired-burst-v1`: excluded
pixels were replaced with the target before whole-image SSIM, and non-aligned
reference variance used only global gain squared. Use `--legacy-metrics` to
reproduce that protocol; aligned reference variance remains unchanged. Keep
protocol versions separate in historical comparisons. Leave-one-out evaluation
and scoreboard metrics retain their existing numerical behaviour.

Synthetic CPU tests cover <0.05 sensor-pixel translation errors, CFA identity,
exact integer shifts, invalid borders, interpolation variance, shot-noise-aware
merging, moving-object rejection (including motion with zero temporal mean),
and the expected sigma/sqrt(N) noise on static samples. The private capture
comparison and its limitations are recorded in `runs/burst_align/REPORT.md`.
