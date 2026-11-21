# 5. Evaluation

## Acceptance requirement

A model is accepted only for the cameras and ISO bands where it has been shown
to meet the owner's quality bar on real photos. This applies to every camera we
claim to support (Sony A7 III, X1D II, and any future X2D II model).
Intermediate checkpoints remain experiments.
- Cross-camera transfer has to be tested, not assumed.
- A shared Bayer layout or successful execution is not a quality measurement.
- A mismatch between training and test cameras explains an experiment's scope;
  it never lowers the bar.

## Test sets (held out at scene level and never trained on)

1. **Real pairs.** 10 of the 40–60 scenes per body, covering every ISO.
2. **Real-world set.** About 30 unpaired owner photos per body (landscape,
   portrait, night). Judged visually, blind.
3. **Synthetic sanity checks.**
   - Flat fields at 5 levels: output must equal input within 1e-3.
   - A seam test: a full frame tiled at two phase offsets must give the same
     result.
   - A mirrored-apron border test: no colour shift in the first or last tile
     row.

Keep development validation separate from a fixed final acceptance set.
Record source hashes, scene split, camera, ISO, exposure, processing settings,
checkpoint hash, and crop coordinates. Do not move a hard
benchmark photo into training after inspecting its comparison. Include the
X2D II ISO 12800 file `stress_iso12800.3fr` as a named unpaired stress case; it is
evaluation-only, not a clean reference.

## Metrics

- PSNR and SSIM on packed linear RAW against the burst-mean target, reported
  **at α = 1**.
- The same metrics on a fixed quick-demosaic + sRGB preview, to show what
  people actually see.
- Per-channel mean bias, which catches colour casts and wrong channel order.
- Error broken down by tile phase, to detect seams.
- Speed per 50 MP and 100 MP frame on the owner's Mac (Core ML / ANE).

## Tile-phase diagnostics

Run the offline diagnostic on the three approved RAW examples and a seeded
flat field with additive Gaussian noise:

```bash
PYTHONPATH=$PWD/src python scripts/measure_tile_seams.py \
  --checkpoint checkpoints/r4_plain_2026-10-30/r4_plain_weights.pt \
  --device cpu --threads 4 --output runs/tile_seams
```

The script writes cached predictions, timings and `results.json`. It compares
historical per-plane reflection, corrected sensor reflection, 16-pixel aligned
origins, cosine weights, overlap 128 and two-grid averaging. Standard phases
are (0,0), (64,64), (128,128), (200,200) packed pixels; aligned runs use
(192,192) for the last phase. A separate overlap-64 probe compares shifts
1, 8, 15 and 16 to isolate sensitivity to origin residue modulo 16.

`metrics.seam_energy` reports max/mean absolute differences and mean squared
difference energy near boundaries and in the interior. Boundary strips extend
8 packed pixels either side of the starts and ends of tiles in both compared
grids; the outer image edge is excluded as a boundary line. The boundary/interior
mean-energy ratio corrects for strip area. The boundary energy fraction is also
reported, along with the fraction of image area covered by strips. Empty-region
or zero-denominator ratios are undefined (`None`). These are phase-consistency
diagnostics, not errors against clean ground truth or proof of visible seams.

The API `metrics.seam_test(..., reference_offset=128)` compares two grids
directly, reflecting the source once. Without `reference_offset`, it retains
the tiled-versus-full diagnostic; full-image pooled statistics can differ.
`tiling.tiled_inference` accepts an integer or (y,x) `grid_offset`, plus
`grid_alignment` and `feather_mode`. Aligned offsets must be multiples of the
alignment. The CLI offers `--align-tiles`, `--feather-mode cosine` and
`--tta-shift N`. Shift averaging samples N evenly spread diagonal phases and
applies strength blending once, after averaging. It is not an exhaustive set
of two-dimensional offsets and does not guarantee phase independence.

Sensor-aware reflection and its regression fixtures were ported from local
commit `a930852`; visible-only normalization, optional tile batching and their
regressions came from `d25ed9b`. The input normalization and GPU backend changes
in those commits are outside this tiling change.

## Acceptance gates

- **Quantitative gate:** with independent clean ground truth, the model must meet
  the owner's quality bar on PSNR/SSIM in each camera/ISO band, with RAW-channel
  and rendered colour-error checks.
  - Report named hard-case slices separately.
  - An overall average cannot conceal a failing band or recurring artifact.
- **Visual gate:** the owner must accept blind A/B comparisons at matched display
  settings, at full-photo and native-pixel scales.
  - Include at least 20 random crops plus the named stress cases.
  - Check colour speckles and blotches, fine texture (stone, foliage, hair), edges,
    hot pixels, banding, seams and colour shifts.
  - Smoother output alone is not sufficient: lost or invented detail is a quality
    failure.
  - Report disagreements between metrics and visual results.
- **Decision:** a model is accepted only when both gates pass for the supported scope.
  - Missing ground truth, unreviewed visuals, or incomplete camera/ISO coverage mean
    **not demonstrated**, not a pass.
  - Real unpaired photos support visual findings but no ground-truth PSNR claim.
- Training starts from random initialisation. Training targets and model-selection
  evidence must have documented independent provenance.

## DEV evaluation v2 and real-RAW audit (opus55 pilot round, 2026-10-27)

- `scripts/evaluate_dev_v2.py` scores all validation sources with 3 deterministic
  256-packed crops each, adds structured-tail, out-of-prior and 2x2 binned-reference
  slices, and scores noise-map models with oracle, x0.5, x2 and blind-estimated maps.
  The split was inspected during earlier selection, so treat it as development only.
  `scripts/archive/evaluate_shadow_refinement.py` and its outputs are unchanged.
- `scripts/audit_real_raw.py` reports code
  distributions, optical-black statistics, green-pair NLF brackets, block-mean shifts,
  residual structure and impulse survival. Single-frame statistics are diagnostics,
  not calibration.
- LibRaw's "visible" X2D II array includes optical-black rows and columns. Pass an
  explicit `--image-box` and `--ob-region`.
- Results: `runs/opus55_round/results.json`.

## Offline blind visual review

Preparation uses optional Pillow, already included in the `dev` extra.
Prepare a new self-contained review folder with two or more labelled arms. Each
arm takes a checkpoint (`.pt`, `.onnx`, `.mlpackage`) and a strength in [0, 1];
strength 0 provides an identity baseline. For example, from the repository root:

```bash
PYTHONPATH=$PWD/src python scripts/blind_review.py prepare examples/raw/*.dng \
  --arm candidate models/r4_plain_weights.pt 1 \
  --arm gentler models/r4_plain_weights.pt 0.5 \
  --crops 7 --crop-size 256 --seed 42 --exposure-ev 0 \
  --output runs/blind_review
```

Crop coordinates are deterministic and chosen from the noisy render, preferring
windows containing both flatter regions and texture. Every crop is compared for
all model pairs with shuffled trial and left/right order. Rendering uses fixed
camera white balance, sRGB, no automatic brightening and the same exposure lift
in EV for all arms. `--no-prepass`, `--overlap`, and `--device` control inference.
A JSON `--manifest` can supplement or replace positional RAW paths: entries are
paths or objects such as `{"path":"photo.dng","iso":3200,"scene":"forest",
"split":"acceptance"}`. Paths are relative to the manifest. Optional `camera`,
`shutter` (seconds), and `aperture` fields are recorded too. Supply missing EXIF
metadata explicitly; absent ISO is reported as unknown, never inferred from a
filename. Source/checkpoint hashes and processing settings are recorded in the key.

Open `index.html` directly with a browser (`file://`, no server or network needed).
Keep `private/key.json` with the coordinator until voting finishes: it contains
model identities and is never loaded by the page. Opaque asset names prevent
casual disclosure; this is reviewer blinding, not protection against inspecting
the key. Use **1 / 2 / 0** for left / right / tie, **Space** to skip ahead,
**n** for noisy input, **z** for native-pixel / 2× nearest-neighbour views, and
**f** for a full-photo overview. Votes advance automatically; Previous lets you
revise them. Progress counts voted pairs. Votes persist in localStorage when the
browser permits it for local files. If storage is unavailable the page warns;
export before closing. Use **Export votes** to download `votes.json`.

```bash
PYTHONPATH=$PWD/src python scripts/blind_review.py score \
  --votes ~/Downloads/votes.json --key runs/blind_review/private/key.json \
  --output runs/blind_review/review_results.md --bootstrap 1000 --seed 42
```

The report includes per-pair wins/ties/losses, per-image and ISO breakdowns, and
Bradley–Terry mean-zero log strengths with image-cluster bootstrap 95% intervals.
Ties count as half wins; a small ridge keeps unanimous comparisons finite.
Partial reviews are supported and their completion count is reported. Small or
disconnected samples do not establish rankings; confidence intervals from one
image are degenerate. Use a fixed, independently sourced acceptance manifest
covering cameras/ISO bands and named stress cases, with at least 20 crops overall.
The demo images only illustrate the workflow. Human review and the other
acceptance gates above still determine acceptance.

## One-command development scoreboard

From a source checkout, run the frozen baseline and one or more candidates with
identical deterministic inputs:

```bash
PYTHONPATH=$PWD/src python scripts/scoreboard.py \
  --checkpoint candidate=runs/my_round/last.pt \
  --baseline checkpoints/r4_plain_2026-10-30/ \
  --config configs/scoreboard/default.yaml --device cpu \
  --output runs/scoreboard
```

Repeat `--checkpoint NAME=PATH` for multiple candidates. `--weights model|ema`
selects the same weights for the baseline and every evaluator (default `model`).
EMA uses the shared resolver: `ema_model` first, then legacy `ema`; missing selected
weights are an error, with no fallback to model weights. The selection is recorded
in provenance. `--suites dev burst`
(or `--suites dev,camera,speed`) selects suites. `--quick` keeps all configured
DEV slices but uses one crop from two sources per dataset; it uses one centre
512-packed crop/test frame from the first group of each burst spec, one usable photo
and ROI per camera/ISO band (trying the next input when none is valid), and one warmup plus three timed tiles. Quick scores
are smoke checks, not model-selection evidence. A self-check can use the same
`.pt` for `--checkpoint` and `--baseline`.

`scoreboard.json` contains checkpoint hashes, suite configuration, per-suite
coverage and timings, raw crop diagnostics, scores and deltas. `scoreboard.md`
shows per-slice deltas, crop-bootstrap 95% intervals for camera metrics, and the
PASS/FAIL gate. Exit status is 0 for PASS and 1 for FAIL. The default gate checks
RAW, flat and texture PSNR slices for drops below -0.1 dB and requires a strictly
positive mean delta; identical checkpoint hashes allow a zero mean for sanity
checks. YAML `gate` settings control `min_slice_delta_db`, `min_mean_gain_db`,
`require_improvement` and `allow_skips`. Missing private inputs are SKIP, runtime
failures are ERROR, and incomplete selected suites fail by default. A run with
no comparable PSNR slices fails even if diagnostic-only suites complete, except
a measured-only diagnostic run: it reports `NOT_GATED` and exits 0 on successful
execution, or `ERROR` and exits 1 for suite errors/missing inputs. `NOT_GATED` is
not a PASS or evidence for promotion.

Paths in the YAML are relative to the working directory. For worktrees, copy the
config under a temporary directory and point dataset roots, manifests, burst specs
and the burst `project_root` at the read-only main checkout. Keep `--output` inside
the worktree or the temporary directory. No data or weights are copied or modified.
Successful baseline suites
are cached under the output directory using checkpoint SHA-256, suite config,
quick/device/thread settings, evaluation code, manifest/spec hashes and input
file sizes/mtimes. EMA has a distinct cache key and output subdirectory. Default
model keys omit an extra weights field, preserving the original key structure;
this implementation still invalidates old keys through the existing code hash.
RAW contents are not rehashed; preserve file metadata integrity
or use `--no-cache` after replacing data. Missing inputs invalidate cache reuse;
speed measurements are always rerun because timing depends on the current machine.

The camera suite uses input-selected flat and textured regions and reports:

- `residual_noise`: robust MAD of high-pass output in flat regions, corrected for
  the filter's white-noise gain; units are normalized linear RAW sigma.
- `texture_retention`: radially averaged output/input power ratio over
  0.15–0.45 cycles per packed pixel in textured blocks; identity is 1.
- `chroma_blotch`: low-pass R-G/B-G spatial variance in flat regions divided by
  squared input green intensity (floor 1e-4).

Confidence intervals bootstrap whole crop measurements, using a fixed seed.
A single crop gives a degenerate interval; nearby crops can be correlated.
These scene-confounded proxies cannot distinguish removed noise from removed
fine detail or establish real-photo quality. They and tile timings are reported
with deltas but do not vote in the PSNR gate. A development PASS never replaces
the quantitative and blind visual acceptance requirements above. Burst scores
retain the existing static-mask, brightness-matching and reference-noise
correction; an invalid corrected MSE remains unavailable and fails the gate.

## Measured-noise development suite

`configs/scoreboard/round9.yaml` retains the historical suites and gate thresholds,
adds the A7 III bin4 validation and matching training manifest to `dev`, and adds
`measured`. `default.yaml` remains unchanged, with its two X1D II dev geometries.
For example:

```bash
PYTHONPATH=$PWD/src python scripts/scoreboard.py \
  --baseline checkpoints/r4_plain_2026-10-30/ \
  --checkpoint r8=runs/round8/r8_physnoise/last.pt \
  --config configs/scoreboard/round9.yaml --suites dev measured \
  --weights model --device cpu --output runs/round9/scoreboard/r8
```

The measured suite is **non-gating by default**, reported in its own table;
its scores, missing profiles, errors and skips do not change the historical
suites' PASS/FAIL result. Set `suites.measured.gate: true` only after an explicit
gate decision; missing profile scores then fail the gate. A measured-only run is
useful for smoke checks. Add `--quick` to use one crop from one source per dataset,
while retaining all profiles and exposures. Full runs use three deterministic
256-packed crops per source. Source selection, crop coordinates and noise draws
are identical across models. The complete train/validation manifests are checked
for source, sensor and group overlap before quick truncation.

`scripts/evaluate_measured_dev.py` also runs directly with repeatable
`--checkpoint NAME=PATH`, `--config`, `--weights`, `--quick`, `--device` and a JSON
`--output`. With multiple checkpoints the first is the paired reference.
It uses `CameraNoise.at()` with zero K/read jitter and the round-8 physics
options: banding, quantization, horizontal/vertical read correlation and
moment-matched read noise. Requested measured profiles are:

| Camera/shutter | ISO | Processing |
|---|---|---|
| X1D II leaf | 100, 3200, 25600 | Unprocessed |
| X1D II electronic | 100, 800, 3200 | Unprocessed |
| A7 III mechanical | 400, 800, 6400, 25600 | Unprocessed |
| A7 III mechanical | 65535 | Processed; separate slices |

All are present in `calib/darks_physnoise_20261101.yaml`. Missing exact
camera/shutter/ISO profiles are reported as unavailable with a reason, never
silently clamped or interpolated. Estimated profiles from other bodies are not
included. Exposures are 1, 1/16 and 1/64; above the X1D gain cap of ISO 6400,
actual exposure is additionally multiplied by 6400/ISO for both shutter modes.
PSNR has peak 1 on exposed packed linear RAW, matching the DEV convention;
there is no exposure normalization. Do not combine its absolute scores with
`eval_camera_noise.py`, which divides by exposure and uses the legacy simulator.

JSON records calibration and checkpoint hashes, complete evaluated profile
parameters, physics options, manifest hashes, crop/noise seeds and coordinates,
requested and actual exposures, and source-paired deltas. The fixed-pattern seed
is 20261111, distinct from training's 20261104. Crop/noise base seeds are 20261109
and 20261110; stable source/crop/profile keys derive each draw. Calibration contents
and selected clean-file metadata enter the baseline cache key; quick fingerprinting
reads only the selected crops' metadata, plus both manifest hashes. Source means
average crops before comparing models. No repeated crop/frame is treated as an
independent scene and no scene-level confidence interval is claimed.

The burst report retains the historical **mean-of-frame PSNR delta** as the gate.
It adds per-frame deltas and a **pooled corrected-MSE delta** sensitivity column,
which does not vote in the gate. Pool the corrected error and squared peak with
static-sample weights, convert once to PSNR, then subtract the baseline; within a
group the static mask and peak are shared across frames. Nonpositive corrected
MSE remains unavailable. The direct LOO evaluator reports deltas against its
first model. Overlapping LOO references do not constitute independent scenes.
`eval_burst_loo.py` has no `--align`; alignment belongs to the separate cross-ISO
`eval_real_burst.py` workflow.

Measured simulation is a development diagnostic using previously inspected,
imperfect low-ISO references. Gains here cannot establish independent real-photo
benefit, justify relaxing the robustness gate, or replace owner blind review.
