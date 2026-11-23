# Changelog

## 0.2.0 — denoiser_1.1 (2026-11-22)

### Model release (`denoiser_1.1`, snapshot `mix_r10_r9p20`)

- Promoted **`denoiser_1.1`** (snapshot ID `mix_r10_r9p20`, release
  `model-denoiser_1.1-2026-11-22`) as the new default model (`DEFAULT_MODEL`).
  `r4_plain` remains available in the model registry (`--name r4_plain`).
- Validated across 5 Bayer cameras (`X1D II 50C`, `X2D II 100C`, `ILCE-7M3`,
  `ILCE-5100`, `DSC-RX100M5`) and ISO `100–204800`:
  - **98.7% blind visual approval** (`75 / 76` wins + ties, `11–1` strict win
    rate vs `r4_plain`) across 55 strictly unseen real photographs (76 crops),
    including a `9 / 9` sweep on `ILCE-7M3` extreme ISO (`32000–204800`).
  - **+1.023 dB** mean frame PSNR on real `ILCE-7M3` ISO 204800 burst
    leave-one-out (`+1.442 dB` flat, `+0.860 dB` texture) and **+0.106 dB** on
    `X1D II 50C` ISO 3200 burst leave-one-out.
  - Lower residual noise (`-11.1%`) with matched/higher texture retention on
    `DSC-RX100M5` ISO 6400–25600.
- Distributed as ONNX FP32 (`denoiser_1.1_fp32.onnx`), PyTorch weights
  (`denoiser_1.1_weights.pt`), and macOS Core ML FP16
  (`denoiser_1.1_fp16.mlpackage.zip`, `17.93 ms` per `512×512` tile with `ALL`
  compute units on Apple Silicon).

### Foliage grid and clipped-highlight fixes

- `denoiser_1.1` reduces residual foliage `G1−G2` channel-split bias by **3×**
  (`2.94% → 0.97%`) and 2×2 Bayer grid energy by **2×** relative to `r4_plain`.
- Fixed `--highlights protect` in both PyTorch and NumPy runtimes to preserve
  `>1.0` linear sensor headroom instead of capping at `1.0`, eliminating
  purple/magenta shifts on clipped highlights (such as fireworks and bright
  specular sources).
- Added opt-in `--tta-transpose` (Bayer `G1↔G2` transpose test-time
  augmentation), further halving residual foliage `G1−G2` bias when enabled.
- Added `scripts/evaluate_foliage_highlights_gate.py` for automated foliage grid
  and clipped-highlight regression gating.

### Runtime, CLI, and packaging

- ONNX and Core ML inference now use a float32 NumPy pipeline, including defect
  correction, CFA-aware reflection, tiling, strength/chroma blending, and
  highlight protection. Importing the CLI no longer requires `torch`.
- Installation change: the base package no longer installs `torch`. Select
  `.[torch]` for `.pt` models or `.[train]` for training; `.[dev]` and
  `.[export]` still include `torch`. ONNX extras remain runtime-only, and the
  new `.[coreml]` extra installs Core ML without `torch`. Existing `torch`
  installations keep working. The distribution name remains `denoise_lab`.
- Added folder discovery (`--recursive`, `--glob`), `--skip-existing`,
  `--report PATH.json`, `--chroma-strength`, `rawdenoise fetch-model`
  (`rawdenoise-fetch`), Core ML `--compute-units`, body-specific dark-frame
  defect maps (`--prepass map`), DNG EXIF/GPS/profile metadata preservation
  (`--metadata full`), and opt-in tile-phase controls (`--align-tiles`,
  `--feather-mode cosine`, `--tta-shift`).

### Research and training additions

- Added optional `green_split` dual-head architecture (`G_avg`, `G_diff`),
  anti-aliased `aa_shallow2` resampling, highlight/chroma/green-DC/transpose-
  consistency loss terms, and Core ML dual-head compression protection.

## 0.1.0 — Research preview

- Experimental `r4_plain` model for Hasselblad X1D II Bayer RAW photographs, distributed as separate release assets.
- Cross-platform `rawdenoise` CLI with PyTorch and ONNX Runtime backends, plus optional Core ML support on macOS.
- DNG, rendered TIFF and normalized NPY outputs, strength blending and configurable defect correction.
- Training, evaluation and export research tools with CPU fixture tests.

Model quality remains experimental: independent real paired-ground-truth evaluation and blind acceptance review are incomplete. Other cameras are unvalidated. Licensing remains undecided.
