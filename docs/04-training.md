Historical (October 2026). Superseded by [docs/09](09-clean-targets-and-defects.md) for the current model; kept for the record.

# 4. Training, compute and export

## Recipe (starting point)

| Item | Setting | Source |
|---|---|---|
| Model | NAFNet-w32, [2,2,4,8] / 12 / [2,2,2,2], gamma head | Published NAFNet paper and SIDD configuration (see references.md) |
| Crops | Packed 256² early → 512² late (progressive, as MR-CAS did at AIM 2025) | AIM 2025 report |
| Batch | 16–32 effective (gradient accumulation on one GPU) | NAFNet official: 32 |
| Optimiser | AdamW, lr 1e-3 → 1e-7 cosine, betas (0.9, 0.9), wd 0 | NAFNet official |
| Iterations | 200k–400k (synthetic stage) + 20k–50k (real-pair fine-tune, lr 1e-4 → 1e-6) | NAFNet; AIM entries |
| Loss | L1 or Charbonnier in the gamma domain + 0.1 × L1 in the linear domain; optional 0.05 × FFT-L1 | AIM 2025 entries; our contract |
| Precision | bf16 autocast on NVIDIA; LayerNorm stats and the loss in fp32 | — |
| Augmentation | BayerAug flips and transposes, even crops, brightness scaling ×[0.5, 2] before the noise is added, mirrored borders | Liu et al. 2019 |
| Tiers (Phase A) | Sample K from each tier's ISO range, slightly overlapping the neighbouring tiers | Planned camera calibration and held-out validation |

**Model selection.** Choose by PSNR and SSIM on the real held-out pairs, not
on synthetic validation data.

## Current archive prototype and shadow experiment

The archive prototypes are smaller than the full recipe above: width 32,
three encoder stages `[1,1,2]`, four middle blocks, decoder `[1,1,1]`, and
2,650,404 parameters. Their initial 7,500 updates per camera used the owner's
ISO 100 single frames plus an uncalibrated shot/read/row prior. These remain
imperfect references and provisional models, not accepted releases.

The 27 October 2026 controlled experiment in `configs/refinement/` tests one
data change: exposure coverage. The control continues with the legacy linear
scale in `[0.6,1]`; the shadow arm draws log-uniform scales in `[1/64,1]`,
covering zero to six stops of attenuation. Attenuation happens before new
noise is drawn, so read noise is not attenuated with the reference. The same
initial checkpoint, manifest, seed, architecture, loss, batch/crop size and
learning-rate schedule are used within each camera comparison. NVIDIA GPU
differences may still introduce small numerical differences.

The initial X1D II shadow arm improved the attenuated development slices but
regressed on bright low-noise crops. The follow-up `*_balanced.yaml` configs
sample the legacy bright range with probability 0.5 and the broad log range
otherwise. X1D II uses the same 5,000-update budget and initial weights as the
control; the Sony follow-up is a shorter 2,500-update development run against
its original checkpoint, not an equal-budget causal ablation.

`scripts/evaluate_shadow_refinement.py` compares the original model, extra-
training control and shadow arm on the same held-out sources/noise draws at
reference scales 1, 1/16 and 1/64. It reports RAW PSNR, gamma error, colour-
difference error, bias and a zero-output baseline. These are development
diagnostics against artificially attenuated single frames; the release gate
still requires independent real references and owner review.

The primary literature reinforces the next data step: Li et al.,
[Noise Modeling in One Hour](https://arxiv.org/abs/2505.00045), CVPR 2025,
samples signal-independent noise from sensor dark frames and subtracts
averaged dark shading. Its results motivate collecting our own dark frames;
they do not validate our present generic prior.
The experiment uses independently authored code and existing owner photos.

## Compute options

CUDA training can run on a Linux NVIDIA host using
`scripts/run_cuda.py` (expects a prepared container; adapt to your setup).

| Option | Speed for one NAFNet-w32 model | Practical use |
|---|---|---|
| Apple Silicon (MPS), owner's Mac | Several times slower than a 4090 (no published benchmark; measure it) | Debugging, small fine-tunes, export and ANE checks |
| 1× RTX 4090 (24 GB) | About 1–3 days per model | Full training for one sensor and tier |
| 1× A100 / H100 (cloud) | About 1.5–3× faster than a 4090 | Batch runs: 3 tiers × 3 bodies |

For reference, AIM 2025 teams reported training times from about 28 h on 2×
4090 up to about 3 weeks on 4–8× 4090 for the winning NAFNet.

Phase A covers 8 models: X2D II ×3, X1D II ×2, Sony ×3. Plan on about 8–20
GPU-days in total, or about 3 conditioned models in Phase B. Rented
H100/A100 time at a few dollars an hour puts a full Phase A sweep in the
**low hundreds of dollars**. Check current prices before booking.

## Export to ShopPhoto

1. `scripts/export_coreml.py`: trace with a fixed `1×4×512×512` input named
   `inp_1`, fp16 compute, ML program. Use a **fixed shape**, because dynamic
   shapes fall off the ANE.
2. Record the exported model path, input contract and camera support in an
   owner-controlled deployment manifest.
3. `scripts/export_onnx.py`: opset ≥ 17, so LayerNorm stays a native op. Used
   by tract on Windows and Linux.
4. Before each release:
   - Compare CPU fp32 against ANE fp16 on high-ISO tiles (max abs diff
     < 1e-3 in the linear domain).
   - Check op placement with `MLComputePlan` or the Xcode performance report.
   - Record full-frame latency and exported artifact size for each checkpoint.
5. Optional: weight palettisation or int8, if quality holds.

## ShopPhoto integration

- **Phase A:** configure an owner-controlled deployment manifest for our models,
  then validate ShopPhoto selection and inference against the exported contract.
- **Phase B:** add a conditioned model type to `sp-ai`:
  - 6 input channels.
  - The runtime computes the noise planes from ISO and the calibration YAML.
  - The selection path is keyed on make + sensor.
  - This also lets the "Generic Sony" path ship publicly.
