# Changelog

## Unreleased

- Local browser demo for Bayer RAW import and DNG/16-bit TIFF export, with explicit
  CPU reference settings, isolated processing and optional persistent model reuse.
  Setup, GPU opt-in and metadata limitations are documented in `docs/13-demo-app.md`.
- Explicit experimental FP32 MPS body with CPU sensor transforms, numeric
  canaries, producing-backend provenance and whole-image CPU fallback for device
  failures raised to Python. This is not universal lossless or photographic
  acceptance; existing accelerator defaults and shipped weights are unchanged.
- Optional CPU tile batching and thread selection, with regression checks for
  merge arithmetic, batching, fallback and Bayer geometry. Crop-before-division
  normalization avoids computing discarded apron pixels.
- Corrected column-noise synthesis to match its reported variance. Nonzero
  column profiles now use a changed random stream and need separate cohorts;
  existing weights and zero-column random streams are unchanged.

- Corrected Bayer apron reflection to preserve CFA-site geometry. Identical packed
  plane reflection could introduce colored grids near sensor-frame borders, even
  with unchanged model weights. RGB reflection remains unchanged.

- Separate experimental three-channel RGB NAFNet training and uint16 TIFF inference,
  with explicit color-space/ICC contracts and Bayer/RGB mismatch checks. No trained
  RGB model or RGB quality acceptance is shipped.
- Explicit calibration for integer NPY/TIFF Bayer targets; signed float32 processing
  and all-code uint16 precision checks.
- Float32 defaults for new RAW caches, explicit approximate float16 storage, and
  local precision/overwrite checks before network access or resume.
- EMA compatibility for training and portable checkpoints; fp32 export defaults and
  experimental ONNX body-only half precision. ONNX artifacts are validated before
  writing the requested output; existing outputs are protected.
- Matched inference candidate diagnostics with sensor-aware self ensembles, source
  and capture-day uncertainty, and separated screening/confirmation reports.
  Ensembles remain experimental and do not change production inference.
- UTF-8 defect reports on Windows.

## 0.1.0 — Research preview

- Experimental `r4_plain` model for Hasselblad X1D II Bayer RAW photographs, distributed as separate release assets.
- Cross-platform `rawdenoise` CLI with PyTorch and ONNX Runtime backends, plus optional Core ML support on macOS.
- DNG, rendered TIFF and normalized NPY outputs, strength blending and configurable defect correction.
- Training, evaluation and export research tools with CPU fixture tests.

Model quality remains experimental: independent real paired-ground-truth evaluation and blind acceptance review are incomplete. Other cameras are unvalidated. Licensing remains undecided.
