# Core ML weight compression experiment

`scripts/export_coreml.py` accepts `--compress none` (default), `palettize4`,
`palettize6`, `palettize8`, `int8-linear`, or `prune-50+palettize8`. These use
post-training Core ML weight compression; activations remain fp16 with float32
I/O. No third-party parameters or denoised references are used.

The first and last convolutions are protected by default. Repeat
`--compress-skip-op GLOB` to exclude further MIL constants or their consumer
operations (names of excluded constants are printed). `--compress-sensitive`
removes the automatic first/last protection. Optimizer weight-size thresholds
still apply, so a requested pruning fraction is not a model-wide sparsity claim.

When the installed palettizer exposes `granularity`, exports use grouped-channel
LUTs with eight output channels per group (divides the released width of 40),
and a macOS 15 deployment target. Older versions use per-tensor LUTs with the
macOS 13 target. Joint sparse/LUT compression requires a coremltools version
supporting `joint_compression`; unsupported APIs fail explicitly. Compression
configuration and runtime support need validation on the actual installed version.

Run the complete local sweep from the worktree, with its source on PYTHONPATH:

```sh
PYTHONPATH=$PWD/src python scripts/evaluate_coreml_compression.py \
  --checkpoint checkpoints/r4_plain_2026-10-30/r4_plain_weights.pt \
  --reference checkpoints/r4_plain_2026-10-30/r4_plain_fp16.mlpackage \
  --data-root data \
  --out runs/coreml_compress
```

The sweep exports all six variants into the output directory and writes export
logs, `results.json` and `REPORT.md`. It measures package file bytes and tile
prediction medians for CPU_AND_NE, ALL and CPU_AND_GPU after warmup (minimum 20
measured predictions). These compute-unit policies allow CPU fallback and do not
prove device placement. Compilation/loading is excluded from latency.

Quality compares complete `Denoiser` outputs on all three demo DNGs to the frozen
fp16 artifact. Development checks use 36 deterministic 256-packed crops by default
from the binned validation manifest, with the noise parameters from
`evaluate_dev_v2.py`, three exposure levels and three noise levels. Scene-group,
source and sensor separation from training is checked. Crop coordinates, seeds,
reference hashes and per-crop metrics are retained. The targets are imperfect
low-ISO references on a previously inspected development split. Output-agreement
PSNR (peak=1) is distinct from target PSNR loss. Development crops use the
production tiler with prepass disabled; examples retain the default prepass.

`coreml_identity_gate.py --package NAME=PATH --out OUTPUT.json` tests an existing
artifact without re-exporting. Its default frame size, levels, kinds and
thresholds match the checkpoint gate. The sweep invokes that same helper for
every package. A candidate requires mean target PSNR loss no greater than
0.05 dB and a passing identity gate; missing evidence cannot pass. This is an
experimental compression screen, not a replacement for photographic acceptance.

At implementation time the required venv did not contain coremltools. The local
report records the measured frozen-package size and missing evidence; no runtime
or compressed-quality claims have been verified. Re-run after the coordinator
provides an environment with the optional dependency. No dependencies are installed
by these scripts.
