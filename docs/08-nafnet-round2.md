Historical (October 2026). Superseded by [docs/09](09-clean-targets-and-defects.md) for the current model; kept for the record.

# 8. NAFNet round 2: capacity, training length and flat-field identity

Status: DEVELOPMENT. Nothing promoted. Protocol frozen before training:
(private-data experiment; artefacts not published) (with `addendum_1`, frozen before Phase B). Same evaluation as
[07](07-architecture-study.md): X1D II validation, 79 sources × 3 crops, 14 slices, paired per-source deltas.
"Shadow slices" means 1/16 high, 1/64 high and 1/16 tails.

## Bottom line

1. **Training length is the biggest lever we have found so far.** The same w32 model trained for 48k steps
   instead of 12k gains **+3.7 dB** on the shadow slices, and +0.3 to +0.9 dB at 1×. Everything in 07 was
   measured with models that were badly undertrained.
2. **Width at full resolution is the right kind of capacity.** w48 beats w32 at 12k (+1.0 dB shadow mean,
   reproduced by a second seed at +1.04) and still at 48k (+0.87 to +1.28 dB in shadows). It wins on all 14
   slices, with 0 of 79 sources worse on 11 of them. Spending parameters at low resolution barely helps: a
   4-stage w32 with 19M parameters gains only +0.17 dB, and 12 middle blocks gain nothing.
3. **The strongest model in this round, w48 trained for 48k steps, met the round’s latency rule.** With trained weights in Core ML
   fp16 it takes 39.0 ms per tile, or an estimated 5.5 s per X2D II frame. That is 1.40× the w32 model, inside the
   1.5× rule. All ops run on the ANE, and fp16 matches fp32 within 3.6e-4.
4. **Flat-field identity is mostly solved, except for clipped highlights.** The w48 48k model now passes the
   gate for daylight and tungsten flats. It still fails for equal-channel flats (1.4–1.5% error above level
   0.35), which in practice means clipped highlights. Training with flat patches from the start fixes that but
   costs about 0.5 dB everywhere, so it was rejected. A short flat fine-tune costs nothing measurable and cuts
   the clipped-highlight error about 4×, but does not reach the gate.

## Neural Engine budget (measured before any training)

Random weights, fp16, `CPU_AND_NE`, 512 tile. One X2D II frame = 140 tiles. Source:
(private-data experiment; artefacts not published).

| config | params | ms/tile | estimated s/frame | vs w32 |
|---|---:|---:|---:|---:|
| w32, 3 stages (round baseline) | 2.65M | 26.1 | 3.65 | 1.00× |
| w32, 4 stages [1,1,2,4]/8 | 18.98M | 30.6 | 4.28 | 1.17× |
| w32, 3 stages, 12 middle blocks | 6.39M | 31.1 | 4.35 | 1.19× |
| **w48, 3 stages** | 5.92M | 36.4 | 5.09 | 1.39× |
| w48, 4 stages | 34.25M | 42.1 | 5.89 | 1.61× |
| w64, 3 stages | 10.49M | 48.0 | 6.71 | 1.84× |
| w32, 3 stages, deep [2,2,4]/8/[2,2,2] | 4.95M | 49.8 | 6.98 | 1.91× |
| w32, 4 stages [2,2,4,8]/12/[2,2,2,2] | 29.16M | 56.9 | 7.96 | 2.18× |

- Cost is driven by blocks at full resolution, not by parameter count.
- 1024 tiles are never faster per frame: 3.72 s for w32 and 5.37 s for w48.

## Phase A: 12k-step screen (same recipe as 07's `nafnet_s1`)

PSNR change vs `nafnet_s1`, in dB. `*` marks an effect beyond max(2F, 0.05) whose 95% CI excludes 0.

| arm | 1× high | 1/16 high | 1/64 high | 1/16 tails | 1/16 extreme | binned 1× high | shadow mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| w32, 4 stages | +0.047 | +0.143* | +0.269* | +0.100 | +0.491* | +0.089* | +0.17 |
| w32, 12 middle blocks | +0.021 | −0.068 | −0.069 | −0.045 | −0.024 | +0.036 | −0.06 |
| **w48, 3 stages** | +0.345* | +1.085* | +1.099* | +0.797* | +1.717* | +0.649* | **+0.99** |
| w48, seed 2 | +0.364* | +1.090* | +1.256* | +0.766* | +1.784* | +0.701* | +1.04 |
| w48, 4 stages (over budget) | +0.391* | +1.174* | +1.471* | +0.906* | +1.981* | +0.760* | +1.18 |
| w64 from 07 (over budget) | +0.575* | +1.931* | +1.940* | +1.452* | +2.372* | +1.152* | +1.77 |
| flat patches, p=0.05 | −0.159* | −0.569* | −0.587* | −0.497* | −0.539* | −0.308* | −0.55 |
| flat patches, p=0.15 | −0.186* | −0.525* | −0.399* | −0.433* | −0.392* | −0.368* | −0.45 |

Full 14-slice tables: (private-data experiment; artefacts not published). Under the pre-registered rule, w48 3-stage was the
capacity winner, and both flat-patch arms were rejected.

## Phase B: 48k steps, no flat patches

Change vs `nafnet_s1` (w32 at 12k), in dB. Raw weights are primary; EMA was within ±0.035 dB of raw on every
slice.

| arm | 1× low | 1× high | 1/16 high | 1/64 high | 1/16 extreme | binned 1× low | binned 1× high | shadow mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w32 @ 48k | +0.274* | +0.916* | +3.708* | +4.796* | +5.116* | −0.334 | +1.868* | +3.74 |
| **w48 @ 48k** | +0.301* | +1.052* | +4.578* | +5.876* | +6.392* | +0.307 | +2.161* | **+4.54** |

Directly, w48 @ 48k minus w32 @ 48k (± 95% CI):

| slice | delta | sources worse |
|---|---:|---:|
| 1× low | +0.027 ± 0.018 | 27/79 |
| 1× high | +0.135 ± 0.014 | 0/79 |
| 1/16 high | +0.870 ± 0.090 | 0/79 |
| 1/64 high | +1.080 ± 0.116 | 0/79 |
| 1/16 extreme | +1.275 ± 0.120 | 0/79 |
| binned 1× low | +0.641 ± 0.179 | 16/79 |

- **No sign of overfitting.** On a training-subset evaluation the gains are the same size as on validation,
  or slightly smaller: w48 +3.84 dB shadow mean on train vs +4.54 dB on validation.
- **Learning curve.** The 24k checkpoints are not annealed, so they can't be compared directly. Still, the
  w48 gain went from about +2.4 dB at 24k to +4.5 dB at 48k. Longer schedules are the obvious next test.
- **Real photo (X2D II, ISO 12800).** In the local stone/window crop
  (private-data experiment; artefacts not published), both 48k models show cleaner stone
  shapes and less blotchy colour than the 12k model. w48 is slightly crisper than w32. All models are still
  somewhat waxy, and the hot pixel survives in every one. Additional gravel and sky/roof visuals are local (private-data experiment; artefacts not published).
  These models were trained on X1D II only.

## Flat-field identity

The gate runs through the production tiler (reflected apron, overlap 58, feather 2.2). It checks every kind
at levels 0.001 to 1.0 and requires all three: max abs error ≤ 1e-3, max abs error ≤ 5% of the level, and
|mean bias| ≤ 2.5e-4. Kinds: equal (R=G1=B=G2), daylight, tungsten, a ramp, and partial clip.

| model | equal | daylight | tungsten | ramp | partial clip |
|---|---:|---:|---:|---:|---:|
| w32 @ 12k (07) | 0.042 ✗ | 0.0021 ✗ | 0.0027 ✗ | 0.0058 ✗ | 0.016 ✗ |
| w32 @ 12k + flat fine-tune p=0.02 | 0.009 ✗ | 0.0025 ✗ | 0.0028 ✗ | 0.0041 ✗ | 0.006 ✗ |
| w32 @ 48k | 0.020 ✗ | 0.0013 ✗ | 0.0009 ✓ | 0.0021 ✗ | 0.012 ✗ |
| **w48 @ 48k** | 0.015 ✗ | **0.00097 ✓** | **0.00046 ✓** | 0.0016 ✗ | 0.011 ✗ |

The gate result is the same when the w48 model runs as Core ML fp16.

What we learned:

- **The 07 failure was not mainly an edge effect.** The worst errors sat at the bare-tile border, but the
  real tiler crops that away. The real error is a gain error on equal-channel flats of about 1–4% at bright
  levels. Equal-channel flats are rare in real scenes, *except* in clipped highlights.
- **For w48 @ 48k, the ramp fails only at levels 0.9 and 1.0,** where the ramp pushes the input above the
  white level of 1.0. That is out of range, so it is mostly a test-design artifact.
- **Training with flat patches from the start works but costs about 0.5 dB on every slice.**
- **A 1.5k-step flat fine-tune at learning rate 5e-5 is free.** It stays within the noise floor of its own
  control fine-tune and cuts the equal-channel error about 4×. It was not yet tried on the 48k model.

## Recommendations, in priority order

1. **Adopt w48, 3 stages, with long training as the development baseline.** Next, run a 96k–150k-step
   schedule to find where it saturates. On the 5060 Ti, 48k steps took 1.6 h.
2. **Clipped highlights.** Two options. One is to pass clipped pixels through unchanged, by masking any pixel
   at the white level; this is deterministic and ANE-friendly. The other is to add the flat fine-tune to the
   48k model and re-gate it. The passthrough is the more reliable fix. An open research question is whether
   the absolute 1e-3 limit should become relative at bright levels. The remaining daylight errors are about
   0.2% of the level, below one 8-bit display step.
3. **Hot pixels survive every model.** The data pipeline already supports impulse noise (`impulse_range`);
   turn it on in the next long run.
4. **Measured dark frames and real pairs are still the main way forward** for real-photo quality. The
   synthetic-noise PSNR here only ranks models.

## Limitations

- Development data: synthetic noise on imperfect ISO 100 references, and a validation split inspected in
  earlier rounds. X1D II only.
- Phase B is one seed per arm. The noise floor comes from the 12k replicate pairs: w32 s1/s2 and w48 seeds
  1/2.
- Phase A and B were run 3–4 jobs per GPU. That affects wall-clock time only, not results.
- ANE latencies are single-tile medians. A full-frame timing, including tiling and blending, is still to do.
- Protocol deviation (recorded in `addendum_1`): the protocol had no rule for the case where every flat arm
  fails the no-loss rule. Phase B therefore ran without flat patches.

## Reproduce

The original GPU runs used `scripts/run_cuda.py` and local orchestration
(private-data experiment; artefacts not published):
- `phase_a.sh`: 12k screen.
- `eval_a.sh`: Phase A scoring.
- `phase_b.sh`, `side_eval.sh`, `eval_b.sh`: 48k runs, replicate, fine-tunes, scoring.

On the Mac:
- ANE sweep: `TMPDIR=/tmp/coreml_tmp PYTHONPATH=src:scripts /usr/bin/python3 scripts/probe_ane_sweep.py`.
- Trained-weight Core ML gate: `scripts/coreml_identity_gate.py`.

Decision tables: `scripts/summarize_nafnet_round.py --eval … --identity …`.
