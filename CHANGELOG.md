# Changelog

## Unreleased

- Added isolated clean-room 2× Bayer RAW-to-RAW super-resolution research package
  (`src/denoise_lab/sr/` — `PhotoSRv2` with compressive `asinh` residual domain,
  shift-equivariant `LocalSCA`, 24-ch post-`PixelShuffle(2)` HR refinement tail,
  and exact 2-fold Bayer transpose TTA), benchmark/training scripts
  (`scripts/sr_v2_train.py`, `scripts/sr_benchmark.py`, `scripts/sr_track_a_eval.py`),
  CPU unit tests (`tests/test_raw_sr_v2.py`), and an interactive 12-photo comparison
  page with full-resolution samples in `examples/sr_comparison/`.
- ONNX and Core ML inference now use a float32 NumPy pipeline, including defect
  correction, CFA-aware reflection, tiling, strength and highlight protection.
  Importing the CLI no longer requires torch.
- Installation change: the base package no longer installs torch. Select
  `.[torch]` for `.pt` models or `.[train]` for training; `.[dev]` and `.[export]`
  still include torch. ONNX extras remain runtime-only, and the new `.[coreml]`
  extra installs Core ML without torch. Existing torch installations still work.
  The distribution name remains `denoise_lab` (`denoise_lab[onnx]`).

## 0.1.0 — Research preview

- Experimental `r4_plain` model for Hasselblad X1D II Bayer RAW photographs, distributed as separate release assets.
- Cross-platform `rawdenoise` CLI with PyTorch and ONNX Runtime backends, plus optional Core ML support on macOS.
- DNG, rendered TIFF and normalized NPY outputs, strength blending and configurable defect correction.
- Training, evaluation and export research tools with CPU fixture tests.

Model quality remains experimental: independent real paired-ground-truth evaluation and blind acceptance review are incomplete. Other cameras are unvalidated. Licensing remains undecided.
