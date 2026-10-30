# 9. Clean targets and hot/dead pixels (rounds 3–4)

Status: DEVELOPMENT. `r4_plain` (model files in the GitHub release) is the current best checkpoint. It has not
passed the acceptance gates in `05-evaluation.md` yet.

## Cleaner training targets

The single biggest improvement came from the training targets, not from the architecture.

- Training input is built as *target + synthetic noise*, so a model learns to keep whatever noise sits in the
  target. Single ISO 100 frames still contain ISO 100 noise. Models trained on them barely denoised real
  low-ISO and long-exposure photos, and training longer made this worse.
- **Same-colour Bayer binning** (`src/denoise_lab/clean_targets.py`) averages k×k photosites of each colour
  plane. The result is still a valid RGGB mosaic with 1/k² of the reference noise variance: 2×2 gives 1/4,
  4×4 gives 1/16.
  - 4×4 binning needs full frames. `scripts/prepare_binned_cache.py` reads them read-only from the owner
    archive.
- Fine-tuning on 2×2-binned targets raised the fraction of expected noise removed on our ISO 100 dev photos
  from 0.22 to 0.34. A native-target control stayed at 0.22.
- **Colour-site geometry:** binning puts the R, G and B planes nearly on the same position.
  - `resite_binned()` can resample the planes back to native offsets.
  - Models trained with resampling scored very differently depending on reference geometry.
  - Plain binning was robust on both geometries, so the current model uses plain binning.
- **ISO-coupled noise:** one data source draws an ISO, then sets shot and read noise from our own X1D II
  noise measurement, with exposure about 100/ISO. Other sources use a broad independent prior. Row, column and
  impulse noise is included.
- **Loss:** Charbonnier eps 1e-5 (down from 1e-3, which under-weighted low-noise samples) plus 0.25 ×
  gamma-domain loss.
- **Colour-gain augmentation:** random R/G and B/G gains widen the colour-ratio prior. This targets flat-field
  failures on unusual raw colour ratios.
- **Separation:** training manifests exclude every capture day used for testing or dev evaluation (`_r4`
  manifests).

## Hot and dead pixels

Two parts are used together.

1. **Deterministic pre-pass:** `defects.correct_defects(k=8, abs_floor=0.01, adjacency_ratio=0.1)`.
   - It flags a single photosite that exceeds its 8 same-colour neighbours by 8 robust σ and by at least 0.01.
   - Its four adjacent photosites of other colours must show less than 10% of that excess. Real detail and point
     lights spread onto neighbours through the lens; a stuck photosite does not.
   - Flagged pixels are replaced by the median of their neighbours.
   - False positives: about 0–0.5 per megapixel on normal photos, and ≤ 0.06 per million on synthetic flats.
2. **Training with injected defects:** hot, warm and dead pixels, plus adjacent pairs, at 1e-5 to 3e-4
   density.

**Caveat:** isolated single-pixel stars in night skies can be removed by the pre-pass. Use `--no-prepass` in
`scripts/denoise.py` for astro work.

## Architecture and speed

| model | params | ANE fp16, ms per 512 tile |
|---|---:|---:|
| NAFNet w40, 4 stages (enc [1,1,2,4], mid 6, dec [1,1,1,1]) | 23.8M | 38.9 |
| NAFNet w48, 3 stages | 5.9M | 39.5 |

The 4-stage w40 has four times the parameters at the same speed. At equal steps and recipe it scored higher on
dev data, so it was chosen.

## What did not help (round 5, 24–48k-step fine-tunes vs matched controls)

Each changed dev PSNR by at most about 0.05 dB:
- a separate low-ISO tier;
- a longer continuation;
- a higher gamma-loss weight;
- about 75% more X1D II frames;
- adding 440 Sony A7 III frames.

Next ideas being tested:
- **Noise-level conditioning:** a noise map from ISO plus our noise measurement.
- **A 5-stage w40,** which adds capacity only at coarse resolution.
