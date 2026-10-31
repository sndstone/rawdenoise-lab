# 1. Deployment contract

This page records ShopPhoto's intended RAW denoising integration interface.
Training starts from random initialisation; training targets and model-selection
evidence must have documented independent provenance. Export and runtime
compatibility must be checked for each model.

## Task definition

- Denoising of **raw Bayer CFA data, before demosaic**, between black-level
  subtraction and demosaic. It is not an RGB post-process.
- Models are per sensor and may be split into ISO tiers. The pipeline selects
  the tier by ISO metadata.

## Tensor contract (must match for a drop-in model)

| Item | Value |
|---|---|
| Input / output | Float32 `1×4×512×512`, input named `inp_1` |
| Channel order | **R, G1, B, G2** = top-left, top-right, **bottom-right**, bottom-left of a red-topped 2×2 quad. This is not row-major order. Getting it wrong gives a magenta cast of about +2% R/B. |
| Scaling | `(code − black) / (white − black)`, linear. Keep negatives; do not clip. |
| Black level | Comes from the file: 4096 on 16-bit IMX461/861 data, 256 on IMX161 |
| Graph head | Sign-preserving gamma `sign(x)·|x|^0.4545`, built inside the model |
| Output | **Direct**, not a residual, then the inverse gamma (power 2.2) |
| Identity | A flat field at any level must come back unchanged |

## Architecture

We use the published NAFNet (Chen et al., ECCV 2022) with a gamma head and a
4-channel packed Bayer interface. See `07-architecture-study.md` and
`08-nafnet-round2.md` for the sizes we studied.

## Runtime behaviour, which inference must reproduce

- **Tiles:** 512×512, with a historical overlap of 58 packed px; the current CLI defaults to 60.
- **Apron:** the area outside the image is **mirrored, not edge-replicated**.
  Mirror sensor coordinates before packing, or use CFA-site-aware packed indices.
  Identical reflection indices for R/G1/B/G2 are incorrect: odd sensor sites have
  different reflection coordinates from even sites. Current unconditioned Bayer
  inference uses site-aware reflection; RGB uses ordinary tensor reflection.
  SCA pools statistics over the whole tile. A noise-free replicated apron
  shifted the colour of the whole first tile row by 1.5–2.6%. This matters for
  training as well: the model must see tiles whose statistics look natural.
- **Merge:** the feather mask carries a 2.2 power. Normalising by the
  accumulated weights gives a partition of unity; this alone does not guarantee seamless predictions.
- **Blend:** `out = α·denoised + (1−α)·input`. ShopPhoto's "Cleanest" setting is
  α = 1, and "Preserve detail" is 0.75–0.80. Evaluation reports α = 1 so
  residual artefacts stay visible.

## Design notes

1. A single model fed a per-pixel noise-level map can cover every ISO. The
   4-channel drop-in path stays available as Phase A.
2. High ISO needs the most capacity, so spend it there first.
3. Per-sensor black level and padding are runtime parameters. We take them from
   file metadata instead of a fixed table, which matters for Sony bodies.
