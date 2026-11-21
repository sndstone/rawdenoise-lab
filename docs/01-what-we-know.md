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
  For unconditioned R/G1/B/G2 input, reflection matches unpacking the sensor
  mosaic, reflecting it, then repacking; odd CFA sites have different indices
  from even sites. RGB and conditioned research inputs use ordinary per-plane
  reflection; the latter needs a separate noise-map geometry contract.
  SCA pools statistics over the whole tile. A noise-free replicated apron
  shifted the colour of the whole first tile row by 1.5–2.6%. This matters for
  training as well: the model must see tiles whose statistics look natural.
- **Merge:** the feather mask carries a 2.2 power. Normalising by the
  accumulated weights gives a partition of unity; this alone does not guarantee seamless predictions.
- **Blend:** `out = α·denoised + (1−α)·input`. ShopPhoto's "Cleanest" setting is
  α = 1, and "Preserve detail" is 0.75–0.80. Evaluation reports α = 1 so
  residual artefacts stay visible.

Optional grid alignment rounds overlap up to a multiple of 16 and places every
tile origin on the four-stage network's stride lattice. The default grid still
uses the requested overlap exactly. Cosine feathering and diagonal shift
averaging are also optional; tile-wide pooled attention can retain phase
dependence under all of these settings.

## Design notes

1. A single model fed a per-pixel noise-level map can cover every ISO. The
   4-channel drop-in path stays available as Phase A.
2. High ISO needs the most capacity, so spend it there first.
3. Per-sensor black level and padding are runtime parameters. We take them from
   file metadata instead of a fixed table, which matters for Sony bodies.

## Runtime dependencies and NumPy parity (2026-11-04)

ONNX and Core ML deployment use NumPy for normalization, Bayer packing,
defect correction, CFA-aware reflection, feathered accumulation, strength
blending and highlight protection. Torch is imported only for `.pt` inference
or research/training modules. The base package no longer installs torch:
select `.[onnx]`, `.[onnx-gpu]`, `.[directml]` or `.[coreml]` for deployment;
`.[torch]`/`.[train]` include torch, as do `.[dev]` and `.[export]`.
The distribution name remains `denoise_lab`.

The inference-only NumPy detector mirrors the existing adjacency-guarded
heuristic: eight same-colour neighbours, lower-median MAD statistics in eight
intensity bins, and simultaneous correction. Mapped correction excludes other
mapped sites. Training and other experimental torch detector modes retain their
existing implementation. Tile-grid geometry is shared between both pipelines.

The frozen `r4_plain_weights.pt` was exported with `scripts/export_onnx.py`
(`--precision fp32`, fixed 512), writing only under `runs/torchfree/`. The
exporter's ONNX-versus-torch model check reported max absolute error
`1.84774399e-6`. `scripts/check_np_parity.py` then ran both complete pipelines
against the **same CPU ONNX session** on the three approved 1024×1024 demo
DNGs, with default overlap 60, heuristic prepass and highlight protection:

| Input | Max normalized absolute difference | NumPy time | Torch pipeline time | Corrected sites |
|---|---:|---:|---:|---:|
| ISO 100 night river | 5.96046448e-8 | 1.124 s | 1.083 s | 8 |
| ISO 25600 living room | 3.72529030e-9 | 0.984 s | 0.921 s | 6 |
| ISO 3200 mushroom | 5.96046448e-8 | 1.006 s | 0.959 s | 0 |

Defect masks and corrected packed inputs matched exactly on all three. These
are single warmed-session timings on the development Mac with eight runtime
threads; they exclude RAW decode/model load and are not a general performance
benchmark or image-quality measurement. Float32 feather arithmetic accounts for
the remaining sub-1e-7 differences.

Reproduce from the repository root (set `PYTHONPATH` to this checkout's `src`):

```bash
PYTHONPATH=$PWD/src python scripts/export_onnx.py \
  --checkpoint /path/to/r4_plain_2026-10-30/r4_plain_weights.pt \
  --output runs/torchfree/r4_plain_fp32.onnx --precision fp32
PYTHONPATH=$PWD/src python scripts/check_np_parity.py \
  --model runs/torchfree/r4_plain_fp32.onnx
PYTHONPATH=$PWD/src python -m pytest -q tests/test_np_pipeline.py
```

The tests launch a fresh subprocess whose import hook raises `ImportError` for
`torch` and all submodules. That process runs the NumPy tests, loads a tiny
exported NAFNet in ONNX Runtime, exercises mapped/heuristic correction and CLI
DNG/TIFF/NPY output, and checks the actionable missing-torch `.pt` error.
CI additionally installs `.[onnx]` in a separate job with no torch, using the
tiny model exported by the export job. Core ML's NumPy input/output interface
is covered with a runtime stub; actual Core ML execution was not verified
locally because coremltools is absent from the development venv. No dependency
installation or hosted CI run was performed during this local task.
