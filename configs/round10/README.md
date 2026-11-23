# Round 10: Small-Sensor, High-ISO Quality & Speed Recipes

## Motivation
Phase 1 diagnostics identified three structural bottlenecks in Round 8/9:
1. **Dropped defect injection on camera-physics sources:** Round 8/9 omitted `defect_probability` on all `kind: camera` sources, reducing total explicit defect injection from 50% (`r4_plain`) to 32.5–40% and causing shadow impulse/heavy-tail regressions (`tails 1/64`, `a5100/iso800/exp=1/64`).
2. **Unrepresented extreme small-sensor physics:** RX100 V (1-inch) and A5100 (APS-C) exceed `r4_plain`'s `K <= 2e-3, read <= 6e-3` prior by 3–9×, with 12-bit quantization (`code_step = 2.52e-4`), horizontal lag-1 compression correlation (`0.06–0.20`), and higher banding. Widening the main broad source globally (`r6_wide`) diluted low/mid-ISO sampling density; keeping the core broad source untouched (55%) and adding a dedicated 15% small-sensor extreme-ISO source avoids dilution.
3. **Chroma blotch & `G1-G2` imbalance:** Real native ISO 3200 pairs (`nov08_pairs/scene_01`) showed residual `G2 - G1` channel bias and chroma MSE; adding WB-normalized gamma-domain opponent `(Cb, Cr)` + 4×4 low-pass chroma blotch + `G1 - G2` balance losses directly penalizes chroma blotches and post-demosaic maze artifacts without over-smoothing luma texture.

## Configs
- `r10_small_sensor_w40.yaml`: Drop-in `w40` NAFNet (23.8M params) for 24k fine-tune from `r4_plain` (or 120k scratch confirmation).
- `r10_slim0_fast.yaml`: 4-stage Slim-Stage-0, Deep-Bottleneck NAFNet (`stage_widths: [24, 64, 160, 320, 640]`, `enc: [0, 1, 2, 4]`, `mid: 8`, `dec: [1, 1, 1, 0]`, 29.5M params; 19.5 ms ANE / 11.4 ms Core ML ALL on M5 Pro).
- `r10_slim0_5stage.yaml`: 5-stage Slim-Stage-0 Fast NAFNet (`stage_widths: [24, 64, 128, 256, 512, 512]`, `enc: [0, 1, 2, 3, 4]`, `mid: 6`, `dec: [1, 1, 1, 1, 0]`, 26.15M params; 15.9 ms ANE / 8.19 ms Core ML ALL — **3.05× faster than `r4_plain`** with 2× wider receptive field).

## Running
```sh
# 24k own-lineage fine-tune from frozen r4_plain:
PYTHONPATH=$PWD/src python scripts/train_arch_study.py \
  --config configs/round10/r10_small_sensor_w40.yaml \
  --init-checkpoint checkpoints/r4_plain_2026-10-30/r4_plain_weights.pt \
  --run-dir runs/photo_high_iso_speed/r10_small_sensor_w40 --device cuda:0

# 120k random-initialization slim-stage-0 fast architecture:
PYTHONPATH=$PWD/src python scripts/train_arch_study.py \
  --config configs/round10/r10_slim0_fast.yaml \
  --run-dir runs/photo_high_iso_speed/r10_slim0_fast --device cuda:0
```
