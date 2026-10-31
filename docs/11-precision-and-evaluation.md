# Precision and inference experiments

## Input precision

Bayer sensor codes are converted to float32 before subtracting black levels.
Float32 represents every uint16 integer exactly. Signed values below black and
values above white remain available during inference; quantization happens when
writing integer output. The separate RGB path has the same float32 processing
requirement; see [RGB TIFF](10-rgb-tiff.md) for color-space/profile contracts.

`bayer.read_mosaic` treats floating NPY/TIFF targets as already normalized sensor
mosaics. Integer targets require explicit `black_level` and `white_level`
arguments, each a scalar or 2x2 source-CFA-site array. `phase` specifies their
Bayer phase. Integer arrays are never silently treated as normalized targets.
RGB TIFFs use their own loader and model, rather than this mosaic loader.

## Sensor-aware borders

Packed Bayer planes occupy different sensor positions. Reflecting each plane with
identical indices does not mirror the sensor: for example, the G1 site at sensor
column -1 reflects to +1, while the R site at -2 reflects to +2. The packed G1 and
R indices therefore differ at the apron. The old per-plane approach could change
tile statistics and introduce colored grids in low-ISO shadow detail.

Unconditioned four-plane tiling now uses `reflected_bayer_pad`, equivalent to
unpacking the sensor mosaic, reflecting at twice the packed extent, and repacking.
It also handles aprons larger than the image. RGB keeps ordinary spatial
reflection. The low-level `bayer_reflection=False` option reproduces historical
padding for diagnostics; it is not the corrected production path. Conditioned
research models retain historical padding until their scalar-map geometry has a
defined sensor-aware contract; portable inference still rejects those models.

The correction is in host preprocessing and benefits existing weights. A newly
exported CNN used with an older host tiler still has the old border behavior.
Check native border and deep-shadow crops, not only downscaled overviews. Existing
development reports predating the correction describe their recorded runtime.
Earlier identity, seam and equivariance numbers can therefore differ when rerun;
keep the historical runtime/hash with those reports. The defect detector's
one-pixel same-color neighbour halo still uses per-plane reflection at the outermost
packed row/column. That statistical boundary rule is separate from the corrected
model apron, and edge-impulse behavior remains a validation limit.

## Export precision

New ONNX and Core ML exports default to fp32 computation and float32 I/O.
The existing fp32 release remains the recommended portable Bayer artifact.
Already distributed fp16 packages retain their original behavior; changing an
export script does not change an existing artifact.
For Core ML, fp32 can increase package size and change GPU/ANE placement; no Mac
parity or placement claim follows from a successful conversion on another host.

ONNX `--precision fp16io` retains the legacy full-half-precision approximation.
Its float32 interface does not preserve the original sensor precision: the input,
gamma encoding, global skip and inverse encoding are computed in half precision.
Core ML `--precision fp16` is also an explicit approximation. Neither option
implies exact 16-bit fidelity or low-ISO shadow parity. Runtime speed and ANE
placement must be measured on the actual export and device.

ONNX `--precision fp16mixed` is experimental and Bayer NAFNet-only. Its body uses
half precision, while sensor input, gamma encoding, global skip and inverse
encoding use float32. A zero body correction preserves the exact input, including
signed subnormal shadow values. Learned body predictions still incur rounding
error; this option does not reproduce fp32 output exactly. All precisions receive
the ONNX Runtime sanity check. Passing that broad check is not a camera quality
acceptance gate.
For half precision, that sanity check compares ONNX against the matching half
PyTorch wrapper, not against the original fp32 network. It does not establish
fp32 fidelity.

Evaluate deployed artifacts on dark low-ISO real tiles as well as high-ISO ones.
Report per-channel mean bias, RMS error and maximum error in source sensor DN,
using each channel's white-minus-black range. Include flat fields near 1e-4 and
1e-3, adjacent sensor-code steps, strong highlights and signed values around zero.
An absolute normalized tolerance of 1e-3 can hide tens of 16-bit code steps.

Training checkpoints store EMA weights as `ema_model`. Portable artifacts may
use the legacy `ema` key. Inference/export resolve both formats, preferring
`ema_model`, and fail when requested EMA weights do not exist. No fallback to
ordinary model weights occurs.

## Optional CPU tile batching

CPU PyTorch Bayer inference accepts `--tile-batch-size 2` (API:
`denoise_packed(..., tile_batch_size=2)`). The default remains 1. Larger values
group tiles into a model call and use more memory; accumulated predictions retain
their original raster order. RGB, ONNX and Core ML do not expose this opt-in path.
Recheck numerical fidelity and peak memory for each model, backend and runtime
before using batching. A different batch shape can select different arithmetic;
batching alone does not guarantee identical predictions or a speed improvement.

`--cpu-threads 8` explicitly selects PyTorch intra-op threads for a reproducible
runtime configuration. Without this option, the existing runtime thread setting
is retained up to a cap of 8. Thread count is part of the numerical contract:
recheck fidelity when changing it, including when comparing tile batch sizes.

## Explicit experimental MPS body

`--experimental-mps-body --device mps` is an opt-in PyTorch Bayer path. The host
keeps signed gamma, the global input skip, inverse gamma, CFA-aware reflection
and tile accumulation on CPU in float32; only the NAFNet body runs on MPS with
float32 tensors. It supports tile batch sizes1 and2. It does not change `auto`,
the normal MPS path, shipped packages, model weights or default thread settings.
In particular, existing `auto` still may select the full-model MPS path for a
PyTorch checkpoint or `ALL` for Core ML. Those existing defaults are not covered
by this mode's checks and must not be described as quality-accepted or safe on
the strength of these tests. Select `--device cpu --cpu-threads 8` explicitly for
the CPU reference configuration.
Use `--cpu-threads 8` explicitly when reproducing that tested CPU configuration.

Export `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0` before starting
Python. The opt-in loader refuses other/missing values and never changes them.
An API cannot retrospectively prove when an existing process set environment
variables; callers must establish them before importing/initializing PyTorch.

At load, the mode compares fixed highlight-flat, texture and signed near-zero
ramp tiles with a CPU reference, covering batch1 and2 call shapes. The canary gates max/RMS/absolute
bias per sample and CFA at0.25/0.02/0.005 DN65535. Canary inference adds cold-load
cost. These conservative numerical checks are not sensor calibration, render
equivalence, proof of no quality loss, or photographic acceptance. This mode is
explicitly **not quality-accepted**; rare adaptive-render differences remain a
separate evaluation requirement even when its numerical canary passes.

Unavailable MPS, accelerator initialization/canary failure, or a device/body/copy/
synchronization exception raised to Python disable the accelerator. Invalid
accelerator shapes or nonfinite output also disable it. The entire image restarts from its
original input on CPU, including its prepass and tiling; completed accelerated
tiles are never mixed with CPU replacements. Subsequent images stay on CPU.
The CPU model is reconstructed from the frozen in-memory CPU state, not reread
from a possibly changed checkpoint path. `Denoiser.provenance` and the CLI's
completion description report the producing backend, selected checkpoint
weights and SHA256 of exactly the checkpoint bytes loaded. Fallback warns with
the reason. Dedicated `AcceleratorError` failures trigger that restart; CPU
preparation and host-tiler errors, including their allocation failures, propagate without disabling the
accelerator or attempting a larger CPU rerun. CPU failures still surface
normally. Fallback cannot recover driver hangs, process kills, failures that do
not raise to Python, or guarantee success if CPU resources/input/model are
invalid. Use one `Denoiser` instance per thread: its fallback/provenance state is
mutable and is not a concurrent shared-instance interface.

Example explicit research invocation (not quality acceptance):

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=src \
  .venv/bin/python scripts/denoise.py photo.dng --model own_weights.pt \
  --device mps --experimental-mps-body --cpu-threads 8 \
  --tile-batch-size 2 --no-prepass --output-dir private_outputs
```

The wrapper, failure handling and CLI CFA preservation are tested with small CPU
models and injected failures. Real hardware parity, stable full-frame memory and
owner acceptance remain separate from those portable plumbing tests. This is a
research option, not a recommendation to replace the safe CPU configuration.
Optional real-MPS tests separate two contracts. An arbitrary random NAFNet may
legitimately fail the strict canary and must then produce exact CPU fallback
output. A separate positive device/tiler fixture uses two center-diagonal dyadic
convolutions whose independently known nonzero correction is `gamma/64`; it
checks finite multi-tile output, batch/repeat bit equality and CPU numeric bounds.
This generated arithmetic fixture does not certify learned photographic quality.
The canary covers512-pixel packed tiles; changing context size requires separate
validation. Core ML's fixed512 tile-size check and contiguous input packing are
host-contract safeguards, not a claim of bitwise parity for Core ML kernels.

## Noise-generator versioning

`NoiseModel` now samples nonzero `col_std` as an offset shared down each column,
matching the column variance already reported by `noise_std`. Missing/zero
column noise consumes no extra draws and preserves the legacy random stream.
Nonzero column profiles now generate a different training/stress distribution
and random stream. Record the generator revision and use a separate cohort when
comparing those profiles with old experiments; do not silently merge or rescore
historical results. This correction does not modify saved weights or itself
demonstrate better photographic denoising.

## Matched development experiments

`scripts/evaluate_quality_candidates.py` compares a hash-frozen snapshot with
its EMA weights, sensor-aware self ensembles, and precision approximations. It
checks source, sensor and capture-day separation against a training manifest.
Noise draws, crops and candidate inputs are matched. Confidence intervals are
reported both by source and by capture day, using a fixed bootstrap seed.

Its existing cached low-ISO references plus synthetic noise are development
diagnostics, not independent paired real ground truth. Screening subsets remain
screening. Run all frozen development slices and sources before drawing a wider
conclusion, and preserve untouched acceptance scenes for the final review.

`scripts/summarize_quality_candidates.py` separates sources and capture days
already used in screening from additional development sources/days. Report the
additional sources as the confirmation group, and all sources only as secondary
evidence. Additional to one screen still does not mean an untouched acceptance
set. Orientation-isolation stress slices hold row/column variance fixed while
changing its direction; their magnitude remains explicitly uncalibrated.

Self ensembles operate on the sensor mosaic and CFA geometry together, including
the G1/G2 exchange under transpose. Flipped mosaics are cropped to a red origin;
discarded borders receive no weight from that member. The original prediction
covers those borders. Individual packed planes are never simply flipped while
assuming the original sensor geometry. This experimental helper does not change
production defaults. More predictions cost more time, and a synthetic gain does
not establish better real photographs or acceptable seam/border behavior.

Optional `training.flat_identity` settings add independently generated clean
Bayer flats and gentle ramps to training through a separately seeded generator.
They are disabled by default, use signed float32 loss, and record generator state
for resume. This targets stability on clean fields and clipped highlights, not
measured camera noise. Passing identity gates does not establish better texture,
color or real-photo denoising; retain the matched unregularized control.
The auxiliary path fails fast on nonfinite predictions; bf16 or fp32 is recommended
when fp16 intermediate overflow would otherwise occur.

Retain changes only with consistent detail, noise, color and border evidence.
Unpaired photographs support blind visual comparison, not ground-truth PSNR.
Training on independent paired references, camera-specific dark/flat calibration
and actual deployment-device checks remain the next steps when synthetic
experiments plateau. Follow the [acceptance requirements](05-evaluation.md).
