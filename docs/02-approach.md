Historical (October 2026). Superseded by [docs/09](09-clean-targets-and-defects.md) for the current model; kept for the record.

# 2. Approach

## Summary

Train a NAFNet RAW denoiser for ShopPhoto’s deployment interface. The training data combines three things:

1. **Calibrated synthetic noise** applied to clean base-ISO RAWs.
2. **Real paired data** from our own tripod bursts, used for fine-tuning.
3. **A per-camera dark-frame library**, which captures the noise that physics
   models miss.

The best recent evidence is the AIM 2025 RAW denoising challenge. The
organisers concluded that the gains came mostly from data synthesis and
training strategy, not from architecture. The top entries were NAFNet
variants trained with dark-frame-based synthesis. So most of our effort goes
into the **noise model and the data**, not into designing a network.

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Backbone | NAFNet, width 32, [2,2,4,8] / 12 / [2,2,2,2] | Published baseline; measure quality and ANE latency across capacity variants. |
| High-ISO variant | Width 32–48, GELU gate, optionally more middle blocks | Evaluate additional capacity on high-ISO development slices. |
| Head / tail | Sign-preserving gamma 0.4545; direct output `outro + g(x)` → inverse gamma | Drop-in contract. Keeps negative values. |
| Conditioning, Phase A | None. 4-channel input with per-ISO-tier models. | Loads in `sp-ai` with no changes |
| Conditioning, Phase B | Add two constant or per-pixel planes: shot-noise level K·x and read-noise σ_r | One model per sensor (or across sensors) handles every ISO and intermediate values. About a third of the shipped weights. Needs a small `sp-ai` change. |
| Avoid | Restormer-style softmax attention, Mamba/RWKV scans | fp16 overflow risk, and they don't map well to the ANE |

## Noise model

The model follows ELD and adds what the recent literature found missing:

```
raw = K·Poisson(signal/K)                 # shot noise; K = system gain (DN/e-), per ISO
    + read noise (Tukey-λ, long-tailed)   # from bias frames
    + row noise  N(0, σ_row) per row      # banding
    + dark shading / FPN                  # per-pixel stable bias (PMN), from averaged darks
    + quantisation U(-½, ½)
    + black-level offset jitter
```

- **Phase A synthesis:** parametric ELD-style noise, using parameters fitted
  per ISO from the photon transfer curve (PTC) and bias frames
  (`scripts/calibrate_noise.py`).
- **Phase B synthesis:** replace the signal-independent part by **sampling
  real dark frames** from our library (SFRN/PNNP style), and keep the
  parametric shot noise. This is the technique the AIM 2025 winners used.
- **Dark shading:** subtract it before training and at inference, using a
  per-ISO dark map that ships as a small table. This follows PMN. Otherwise
  the network learns to reproduce the fixed pattern.
- **Dual conversion gain:** the noise parameters jump at the sensor's DCG
  switch ISO. Find it as a kink in the K-vs-ISO curve, and make sure both
  sides appear in training and in tier boundaries.

## Phases and validation milestones

Meeting the acceptance gates in `05-evaluation.md` is the release requirement for
every supported camera. Intermediate milestones below describe
research progress; none permits accepting a lower-quality replacement.

| Phase | Deliverable | Validation milestone |
|---|---|---|
| **0. Setup** (done in this repo) | Scaffold, contract tests, smoke training | `pytest` green; the smoke run converges |
| **1. Calibration** (about 1 day of shooting per body) | PTC flats, dark library, `calib/<body>.yaml` for X2D II, X1D II, Sony | Fitted variance-vs-mean is linear (R² > 0.99 per channel); read noise is stable across repeats |
| **2. Synthetic baseline** | NAFNet trained only on synthetic noise over clean base-ISO RAWs (own + raw.pixls.us CC0) | Report quality on real held-out data by camera/ISO; synthetic gains alone do not establish quality |
| **3. Real pairs** (2–3 shooting days) | 40–60 static scenes × ISO ladder, burst-averaged targets | Pass both the quantitative and the owner blind visual gates in `05-evaluation.md`; no weaker camera/ISO bands hidden by averages |
| **4. Drop-in release** | Core ML + ONNX, 4-channel contract, 3 tiers per body, report full-frame inference latency on supported hardware | Acceptance gates still pass after export; ShopPhoto `sp-ai` selection/inference tests, speed and seam checks pass |
| **5. Conditioned model** | One noise-map-conditioned model per body (then try across bodies) | ≥ the Phase 4 tiers at every ISO, including in-between ISOs; `sp-ai` updated |

Phase 4 integrates independently trained models into ShopPhoto.

## Risks

- **Dataset size.** About 50 scenes may be too few for real-pair training.
  Mitigation: synthetic-first, real pairs only for fine-tuning.
- **Motion in bursts** (wind, water, clouds) contaminates the targets.
  Mitigation: indoor or still subjects, per-frame residual checks, rejection.
- **Medium-format files are heavy.** About 210 MB per uncompressed X2D II
  frame. Mitigation: average the bursts on ingest and keep only the clean
  mosaic plus a few noisy frames per ISO (see `03-data.md`).
- **ANE fp16 overflow in LayerNorm.** Mitigation: compute mean-subtracted
  variance, keep activations bounded, and compare CPU fp32 against ANE
  outputs on high-ISO tiles before each release.

## Secondary investigation: RAW super-resolution

**Priority: denoising first.** Measured camera noise, burst-averaged references,
and real-camera denoising evaluation remain the next main milestones. RAW
super-resolution is a separate, bounded research experiment.

There are two relevant uses of "subpixel":

- **Subpixel convolution / PixelShuffle** rearranges learned feature channels
  into a larger output grid. It enables efficient learned upsampling; it does
  not itself provide additional sensor measurements. The current denoiser's
  internal PixelShuffle layers restore feature resolution and do not upscale
  the final RAW image.
- **Subpixel shifts between RAW frames** provide different scene samples.
  With accurate registration, useful sampling diversity, and motion rejection,
  these can support genuine multi-frame resolution recovery as well as noise
  reduction. Identical aligned repeats mainly improve noise, rather than
  providing the shifted observations needed for additional resolution.

The candidate experiments are:

1. **Single-frame, CFA-aware 2× reconstruction.** Compare RAW-domain processing
   and joint demosaicing/upscaling against ordinary demosaicing followed by
   Lanczos at the same output size. A Bayer photosite measures one colour;
   missing high-resolution detail must be inferred. Prefer reconstruction
   fidelity and inspect false colour, fine text, ringing, and invented texture.
2. **Multi-frame RAW reconstruction.** Register a small burst at subpixel
   precision, reject moving/occluded regions, and reconstruct linear RGB
   directly from the CFA samples. Joint reconstruction can replace a separate
   demosaicing pass. Compare its actual detail recovery and noise against the
   single-frame baselines, initially on small crops.

Coordinate handling is essential. An input packed as four Bayer planes of
size `h×w` corresponds to a native sensor image of `2h×2w`. Native-size RGB
reconstruction is therefore `2h×2w`; a 2× linear-resolution RGB output is
`4h×4w`, four times the native pixel count. Merely unpacking Bayer data does
not add optical resolution. Never resize a colour mosaic as if it were a
grayscale photograph: simulation and reconstruction must respect CFA phase,
sample positions, optical filtering/pixel integration, and noise.

Existing single-frame crops can support pipeline experiments, but do not
constitute noiseless, complete high-resolution RGB ground truth. Do not train
against the denoiser's own outputs as if they were independent clean references.
Keep originals for joint burst reconstruction, since aggressive per-frame
denoising may remove fine signal needed for registration and fusion.

### Initial archive screen, 27 October 2026

`scripts/find_burst_candidates.py` scans the already cached metadata catalog
without opening the NAS or using a GPU. It groups at least four frames with
matching camera/ISO/exposure/aperture metadata, gaps of at most two seconds,
and a total duration of at most ten seconds. Matching 64-KiB prefixes and file
sizes are collapsed as duplicate candidates; this is not full-pixel validation.

The screen found 324 possible A7 III sequences and eight possible X1D II
sequences. Of these, 102 Sony and seven Hasselblad sequences are at ISO ≤400.
These are **metadata candidates**, not verified same-scene or subpixel-shift
bursts. Results are local in (private-data experiment; artefacts not published).

The next SR feasibility check is registration of a few candidate crops to
measure shift diversity and scene stability. Neural SR training follows only
after that check and suitable target/degradation definitions; denoising keeps
the main training and evaluation effort.

References: [Wronski et al., Handheld Multi-Frame Super-Resolution](https://sites.google.com/view/handheld-super-res/),
[Shi et al., efficient subpixel convolution](https://openaccess.thecvf.com/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html),
and [Xu et al., RAW-domain single-image SR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Towards_Real_Scene_Super-Resolution_With_Raw_Images_CVPR_2019_paper.pdf).
