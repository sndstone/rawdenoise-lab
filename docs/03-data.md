# 3. Data

## Licensing gate

A model we intend to **ship** may only train on rows marked **Ship** below.
Research-only data may be used for private experiments in a separate run
directory. Weights from those runs are never exported.

| Source | Sensors | License (as found Oct 2026) | Use |
|---|---|---|---|
| **Our own captures** | X2D II, X1D II, Sony body | Ours | **Ship**. This is the core dataset. |
| raw.pixls.us | Many, incl. Sony and some Hasselblad | CC0 | **Ship**. Clean base-ISO RAWs for synthetic noise. |
| SIDD | 5 smartphones | MIT | **Ship**. Useful only for pretraining (phone sensors). |
| RawNIND (2025) | 11 cameras incl. Sony ILCE-7C | CC BY-SA 4.0 | Maybe. Whether weights count as "adapted material" under share-alike is unsettled. Ask counsel before shipping. |
| SID, ELD, LRD, PolyU, RENOIR | various | No dataset license stated | Research only |
| LRID, MCR, LED code | various | CC BY-NC 4.0 | Research only |
| RAISE, MIT-Adobe FiveK | Nikon/Canon | Research-only terms | Research only |
| AIM 2025 challenge data | Sony A7R4/A6700/ZV-E10M2 | Research only | Research only (useful as a benchmark) |
| DND | 4 cameras | Online benchmark | Evaluation only |

No public paired dataset exists for IMX461, IMX161 or IMX861. Medium-format
data has to come from our own captures.

## Capture protocol, per body (X2D II, X1D II, Sony)

General settings:

- Uncompressed RAW. Lossy-compressed ARW changes the noise statistics.
- Long-exposure NR off.
- Electronic or leaf shutter, tethered or remote release.
- Heavy tripod, mirror/IS off.
- Flicker-free DC or daylight-balanced LED lighting.
- Log the sensor temperature (from EXIF where available).

### A. Planned calibration capture protocol (about 1 day per body)

1. **Flat-field pairs for the PTC.** Use a diffuser over the lens (or a light
   panel), defocused. For every full ISO stop from base to 25600, shoot
   **2 frames × about 20 exposure levels** spread from near black to near
   clipping. This gives the system gain K per ISO and per channel. A kink in
   the curve marks the dual-conversion-gain switch.
2. **Bias / dark frames.** Lens cap on, in a dark room. For every ISO stop,
   shoot 64 frames at the fastest shutter speed, plus 16 frames at 1 s and at
   30 s.
   - From these we get read noise, Tukey-λ shape, row-noise σ, the
     dark-shading map (the mean of the 64) and a hot-pixel map.
   - Repeat a subset warm (after 20 minutes of live view) to see how much the
     sensor temperature matters.
3. Run `scripts/calibrate_noise.py` → `calib/<body>.yaml`. The current fitter
   aggregates mosaic statistics into scalar per-ISO profiles. Per-channel
   calibration and dark-shading map export are planned, not implemented.

### B. Paired scenes (2–3 days per body; target 40–60 scenes)

Scenes should be static and textured, and should vary in colour, contrast
and dynamic range:

- Fabric, foliage indoors, print and text, wood, stone, skin-tone charts
- Mixed colours and dark shadow detail
- Specular highlights, so clipped regions get handled

For each scene:

1. **Clean target.** At base ISO (64 on the X2D II, 100 on the X1D II),
   expose well without clipping.
   - Average **32–64 frames** for scenes that will be paired with the ≤800
     tier, and 16 frames for the high-ISO tiers.
   - SNR improves with √N.
2. **Noisy inputs.** At ISO 200, 400, 800, 1600, 3200, 6400, 12800 and 25600,
   shoot **4 frames each**.
   - Use the **same exposure brightness** by shortening the shutter speed,
     which is how real high-ISO photos are taken.
   - Also shoot a ×2 underexposed set at 3200 and above.
3. `scripts/make_burst_targets.py` requires at least three pre-aligned,
   matching-exposure frames and JSON capture sidecars. It rejects gross scene
   mismatches and forms a robust float32 mean with per-pixel outlier rejection,
   using frames independent of the selected noisy inputs. It does not register
   frames, verify fine alignment, fit per-channel gains/offsets or normalise
   exposure differences; registration and exposure matching remain preparation
   requirements.

### Volume and storage

| Item | X2D II (≈210 MB/frame uncompressed) | X1D II (≈105 MB/frame) |
|---|---|---|
| Calibration | about 1,100 frames ≈ 230 GB (transient) | ≈ 115 GB |
| Paired scenes, 50 × (≈48 clean + 32 noisy) | ≈ 840 GB (transient) | ≈ 420 GB |
| **Kept after ingest** (clean mean fp32 + noisy frames) | Recalculate for retained frames | Recalculate for retained frames |
| Planned calibration outputs (YAML + dark maps) | < 2 GB | < 1 GB |

Plan on a 2 TB external SSD as the working drive. Retain original bursts until
registration, target-quality checks and retention needs have been addressed.
`data/` is git-ignored.

### Cache precision

`prepare_bootstrap_cache.py`, `prepare_binned_cache.py` and
`cache_real_examples.py` default to **float32** NPY caches, retaining signed
normalized values and the float32 fractions produced by Bayer binning. Their
`--cache-dtype float16` (alias `fp16`) option is an explicit lossy storage
approximation: float16 has only 10 fraction bits and merges many distinct
16-bit sensor levels. New journal and manifest rows record `cache_dtype`.

Before SMB access, preparation checks existing local NPY dtypes and metadata,
including legacy rows without dtype fields. A precision mismatch or an
unverifiable cached row requires a fresh `--output`, or explicit float16 for a
matching historical cache. Existing arrays are never overwritten; matching
resumes verify local precision, and file collisions require identical values.
Selection, capture-day splits, crop policy and RNG remain unchanged.
Legacy rows with zero accepted crops are valid empty results. Binned-cache
resumes keep all journalled sources, including recorded errors, without retrying
them. Preflight also scans nested output directories; a nested legacy float16
cache therefore requires a separate float32 output root. An interrupted write
may leave a truncated array: preflight refuses it, and a fresh output directory
allows regeneration without overwriting the damaged file.

Historical caches and experiments retain their original precision. Regenerating
at float32 requires the **original RAWs in a new output directory**; merely
casting old float16 arrays to float32 does not restore lost levels.

## Clean base-ISO pool for synthetic training

For a smaller first collection, use the [real-reference capture pilot](12-capture-pilot.md)
before the larger per-camera acquisition above. It separates preparation,
validation and sealed test sessions, with storage and RGB export guidance.

- Every base-ISO frame from the owner's existing archive.
- Clean means from the paired scenes.
- CC0 RAWs from raw.pixls.us for extra sensor diversity, normalised with each
  file's own black and white levels.
- Synthetic noise is added on the fly from `calib/<body>.yaml` (see
  `src/denoise_lab/noise.py`), so one clean image yields unlimited noisy
  samples.

## Bayer handling rules

- Before cropping or flipping, convert every file to RGGB phase with
  BayerUnify (crop one row or column). After a flip, shift by one pixel to
  restore RGGB (BayerAug).
- Crops start at even offsets.
- Pack to R, G1, B, G2 only after augmentation.
- Training tiles near the image border are padded by **mirroring**, exactly as
  at inference.
