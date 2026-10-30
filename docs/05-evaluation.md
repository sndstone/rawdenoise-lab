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
  `evaluate_shadow_refinement.py` and its outputs are unchanged.
- `scripts/audit_real_raw.py` reports code
  distributions, optical-black statistics, green-pair NLF brackets, block-mean shifts,
  residual structure and impulse survival. Single-frame statistics are diagnostics,
  not calibration.
- LibRaw's "visible" X2D II array includes optical-black rows and columns. Pass an
  explicit `--image-box` and `--ob-region`.
- Results: `runs/opus55_round/results.json`.
