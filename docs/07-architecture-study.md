Historical (October 2026). Superseded by [docs/09](09-clean-targets-and-defects.md) for the current model; kept for the record.

# 7. Architecture study: transformers and diffusion vs NAFNet

Status: **development pilot, completed 28 October 2026.** Nothing here is
promoted, and nothing here meets the acceptance bar. Raw numbers and the literature review are local
(private-data experiment; artefacts not published).

## Bottom line

- **Keep NAFNet.** At equal size, data and training steps, neither transformer
  variant beat it beyond the seed-to-seed noise floor on any shadow slice.
  Channel attention (MDTA) was significantly worse on three slices. Both
  transformer arms failed the pre-registered follow-up gate.
- **Diffusion as the denoiser is rejected.** Iterative refinement with four
  steps did gain in deep shadows (+0.65 dB at 1/64 exposure, high noise). But
  a plain NAFNet given the same compute, by doubling its width, beat it on all
  14 slices by +0.49 to +2.83 dB. The gain came from extra compute, not from
  diffusion.
- **True generative sampling invents texture.** At low noise, two stochastic
  runs on the same input differ by 0.89–1.12× the input noise. The samples
  differ from each other about as much as the noise we are trying to remove.
  Under `05-evaluation.md`, invented detail is a failure.
- **Capacity is a real lever.** The width-64 NAFNet (10.5M parameters) gained
  +1.45 to +1.94 dB over width 32 on the three pre-registered shadow slices.
  High ISO needs the most capacity.
- **On the Apple Neural Engine (ANE),** NAFNet runs 100% on the ANE. MDTA is
  1.61× slower per tile. Window attention falls off the ANE completely and is
  19× slower.
- **Recommendation:** spend effort on (1) NAFNet capacity, tested against the
  ANE budget, (2) the flat-field identity defect that every arm shares, and
  (3) measured dark-frame noise and real pairs. Use diffusion, if at all, as
  an offline *noise generator* for training data. Revisit attention for
  multi-frame bursts, where it has new information to work with.

## Research questions

### (a) "Diffusion starts from noise and removes it over passes"

**What is right.** Iterating can help. Each pass makes a small correction
instead of one large jump. Published methods such as
[InDI](https://arxiv.org/abs/2303.11435) and
[cold diffusion](https://arxiv.org/abs/2208.09392) do exactly this, and they
start from the degraded image rather than from pure noise. Learning the camera’s real noise patterns is also a key research priority. Challenge organisers say this repeatedly (see
the next section).

**Research context.** The noise a diffusion model "starts from" is not the
camera's noise. It is artificial Gaussian noise that the model injects
itself, following a fixed schedule. A generative diffusion model does not
find the particular noise in a photo and subtract it. It *draws a
plausible clean image* that is consistent with the input and with what it
has seen in training. Where the input is ambiguous, for example fine stone
texture buried in shadow noise, different draws invent different texture. The
image looks clean, but the detail may never have been there.
[Blau and Michaeli](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)
formalise this: an output that looks natural and an output that is faithful
to the original are different goals.

**What we measured.** We built the formulation that is best suited to real
sensor noise. It is a Brownian-bridge/InDI refiner that starts from the noisy
RAW and walks to a clean estimate in K steps.

- One step (K=1) is a weaker one-pass denoiser: 0.04–2.65 dB below NAFNet
  on every slice.
- Four deterministic steps help in the darkest, noisiest slices (+0.65 dB at
  1/64 high, +1.24 dB on the out-of-range "extreme" slice). They cost 1.1–1.8
  dB at low noise. More steps than 4–8 make it worse again.
- Stochastic sampling, which is real generative diffusion, was worse than the
  deterministic version on every slice tested. At low noise it lost 4.5 dB,
  and seed-to-seed differences were 0.89–1.12× the input noise. That is
  invented texture by our own definition.
- The decisive control: a NAFNet with the same compute as four bridge steps
  beat the bridge everywhere. "Many passes" helped only because four passes
  are four times the compute.

**Where "learning many noise patterns" belongs.** In the training data. The
most useful published diffusion work for RAW is
[Dark Noise Diffusion / NoiseDiff](https://arxiv.org/html/2503.11262v2). It
uses diffusion to *synthesize realistic sensor noise*, then trains an
ordinary U-Net on it. Even there, carefully sampling real dark frames is
competitive. [2-Shots in the Dark](https://arxiv.org/html/2512.03245v1)
reports 43.57 vs NoiseDiff's 43.92 dB on SID ×100, but 47.13 vs 46.95 dB on
ELD ×100.

### (b) "Transformers know the details from previous passes"

**Research context.** An image-restoration transformer has no memory of
previous passes. It processes one input in one forward pass, the same way
NAFNet does. "Attention" means that the weights used to combine features
are computed from the image content itself, inside the current input. Two
further points matter here.

- Our model never sees "the whole picture." It sees one 512×512 packed tile
  at a time (1024×1024 sensor pixels), because that is the deployment
  contract (`01-what-we-know.md`).
- NAFNet already has a global context mechanism inside each tile. Its
  simplified channel attention pools the whole tile and reweights channels.
  Restormer's MDTA is a richer version of the same *channel* idea. It is not
  pixel-to-pixel attention over the whole photo.

**Why "semantic understanding" is a double-edged sword.** Denoising is mostly
local statistics: how much of the local variation is signal, and how much is
noise. A model that "recognises a brick wall" is tempted to draw bricks where
the data cannot support them. That is the failure that our acceptance gate
forbids.

**What is right.** Long-range context can help find *repeated structure*, so
that similar patches elsewhere in the tile vote on what is real. This is the
idea behind non-local means and BM3D. Iterative processing can help with **bursts**. With several frames of the same scene, attention can
align and fuse genuinely new measurements
([Burstormer](https://arxiv.org/abs/2304.01194)). That recovers real signal
instead of guessing it.

**What we measured.** Parameter-matched MDTA and window-attention arms tied
NAFNet within the noise floor on most slices. MDTA was significantly worse in
the darkest high-noise slice, the out-of-range slice and the binned low-noise
slice. Neither arm learned faster. On the ANE, MDTA costs 1.61× the latency,
and window attention does not run on the ANE at all.

## What the literature says

| Approach | Representative evidence | Takeaway for us |
|---|---|---|
| CNN (NAFNet) | [NAFNet](https://arxiv.org/html/2204.04676v3): SIDD RGB 40.30 dB | Strong, ANE-friendly baseline |
| Channel transformer | [Restormer](https://www.openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf): SIDD 40.02 dB, about 26.1M params, 140–141 G at 256² | No advantage over NAFNet on SIDD |
| Window transformer | [Uformer, as compared in CAT](https://arxiv.org/pdf/2211.13654): SIDD 39.89 dB, 50.88M params | No advantage |
| RAW challenge, AIM 2025 | [Report](https://arxiv.org/pdf/2510.06601): winner MR-CAS is NAFNet, 41.90 dB; constrained comparison ranked NAFNet > Restormer > U-Net | Organisers credit synthesis and training strategy |
| RAW challenge, NTIRE 2025 | [Report](https://arxiv.org/html/2506.02197v1): Samsung's NAFNet-derived ERIRNet won both tracks (46.04/42.25/40.10 dB vs SwinFIR 45.72/41.73/39.50) | Here a transformer was used as a teacher, not as the deployed model |
| RAW diffusion restorer | [TS-Diff](https://arxiv.org/html/2505.04281v1): SID ×300 36.71 vs 36.30 dB comparator; 2,000-timestep schedule | Small gains, unverified latency |
| Diffusion noise synthesis | [NoiseDiff](https://arxiv.org/html/2503.11262v2): SID ×100 43.92 vs PMN 43.47 dB; 43.6 s per 512² patch on an A100 | Offline data tool, feed-forward deployment |
| Noise modelling | [Noise Modeling in One Hour](https://arxiv.org/html/2505.00045v1): same U-Net, gains from noise preparation | Data beats architecture |

The SIDD and DND results are on processed RGB, not RAW. They do not predict
results on our cameras directly. Apple shows that transformers *can* run
well on the ANE when they are designed for it
([guidance](https://machinelearning.apple.com/research/neural-engine-transformers),
[vision guidance](https://machinelearning.apple.com/research/vision-transformers)).
That supports feasibility, not suitability. The review found no cited result
that establishes the needed quality on these cameras.

## What we tested

The protocol was frozen in (private-data experiment; artefacts not published) before any
training or scoring. It records the arms, the decision rules and the SHA-256
of every config and script.

| Arm | What it is | Params |
|---|---|---:|
| `nafnet_s1` | NAFNet w32, enc [1,1,2] / 4 middle / dec [1,1,1]. **Control.** | 2,650,404 |
| `nafnet_s2` | Same, different seed. Defines the noise floor F | 2,650,404 |
| `mdta` | Every block uses Restormer MDTA channel attention + NAF feed-forward | 2,666,454 |
| `win_mid` | Middle blocks [naf, win, naf, win]; 8×8 window self-attention at 1/8 resolution | 2,655,540 |
| `bridge` | Brownian-bridge/InDI refiner, NAF body, κ = 0.02. Evaluated at K = 1, 2, 4, 8, 16 deterministic, and K = 4, 16 stochastic × 4 seeds | 2,651,844 |
| `nafnet_w64` | NAFNet width 64. **Added post hoc** (see below) | 10,490,436 |

**Held constant.** Every arm was trained from scratch for 12,000 steps on the
X1D II balanced recipe: batch 4, 256 packed crops, AdamW at lr 4e-4 with
cosine decay, bf16, identity init, and the same loss and noise synthesis.

**Evaluation.** We used 79 X1D II validation sources × 3 crops, with 4 noise
realizations per crop. Slices combine exposure (1, 1/16, 1/64) with a noise
level (low, medium, high). Three stress slices add to these:

- *tails*: row and column noise plus 0.5% impulses;
- *extreme*: shot and read noise above the training range;
- *binned*: scored against a 2×2-binned, cleaner reference.

Each comparison is paired per source, with a 95% CI. The noise floor is
F = |s2 − s1|. An effect counts only if |Δ| > max(2F, 0.05 dB) and the CI
excludes zero.

**Fairness and known biases.**

- Short training on small data (about 3.8k crops from 162 sources) favours
  CNN inductive bias. This was recorded in advance.
- The bridge spends only 25% of its updates on the first step.
- The PSNR numbers are **development data**: synthetic noise added to
  imperfect ISO 100 references, on a validation split that was inspected in
  earlier rounds.
- All models were trained on X1D II only. The X2D II photo is a
  cross-camera stress case.
- The w64 control was **added after the main results were seen**. Its
  addendum (`protocol_addendum_w64.json`) applies the rule from the original
  protocol: any diffusion gain counts only as "iteration helps" until a
  compute-matched NAFNet control has run. The rule was pre-registered; the
  specific arm was not.

**Two diagnostics.** The HF ratio is the high-frequency energy of the output
relative to the reference. U is the part of that energy that is not
explained by the reference, so it measures invented texture or leftover
noise. In the 1/16 and 1/64 slices the reference HF is tiny, and these
ratios blow up to values such as 85. Read them in the full-exposure (1×) and
binned slices only.

## Results

### Table A: PSNR change vs the NAFNet control (dB, 95% CI)

**Bold** marks a significant effect under the pre-registered rule.

| Slice | F | Threshold | MDTA | Window | Bridge K=1 | Bridge K=4 |
|---|---:|---:|---|---|---|---|
| 1× low | 0.043 | 0.086 | −0.054 [−0.098, −0.009] | −0.056 [−0.084, −0.029] | **−0.525 [−0.622, −0.429]** | **−1.277 [−1.496, −1.058]** |
| 1× medium | 0.003 | 0.050 | −0.001 [−0.014, +0.012] | −0.006 [−0.012, +0.000] | −0.036 [−0.051, −0.022] | **−0.294 [−0.339, −0.249]** |
| 1× high | 0.018 | 0.050 | **+0.053 [+0.040, +0.066]** | −0.005 [−0.011, +0.002] | **−0.137 [−0.162, −0.113]** | **+0.083 [+0.053, +0.114]** |
| 1/16 low | 0.150 | 0.300 | −0.043 [−0.073, −0.013] | −0.087 [−0.154, −0.019] | **−0.397 [−0.441, −0.352]** | **−1.116 [−1.229, −1.004]** |
| 1/16 medium | 0.071 | 0.142 | +0.075 [+0.055, +0.095] | −0.050 [−0.060, −0.039] | **−0.825 [−0.900, −0.751]** | **−0.691 [−0.763, −0.619]** |
| 1/16 high ★ | 0.044 | 0.088 | +0.013 [−0.019, +0.045] | −0.061 [−0.081, −0.041] | **−2.210 [−2.400, −2.019]** | +0.126 [−0.010, +0.262] |
| 1/64 low | 0.128 | 0.255 | −0.038 [−0.066, −0.009] | −0.044 [−0.074, −0.014] | **−0.932 [−1.015, −0.850]** | **−1.774 [−1.943, −1.606]** |
| 1/64 medium | 0.118 | 0.236 | +0.022 [+0.001, +0.044] | −0.119 [−0.138, −0.101] | **−1.641 [−1.761, −1.522]** | **−0.647 [−0.760, −0.533]** |
| 1/64 high ★ | 0.059 | 0.118 | **−0.145 [−0.160, −0.130]** | −0.029 [−0.052, −0.006] | **−2.230 [−2.342, −2.119]** | **+0.652 [+0.570, +0.734]** |
| 1/16 tails ★ | 0.068 | 0.136 | +0.026 [+0.004, +0.049] | −0.016 [−0.028, −0.004] | **−1.404 [−1.534, −1.273]** | **+0.270 [+0.205, +0.336]** |
| 1/64 tails | 0.118 | 0.236 | −0.023 [−0.036, −0.010] | −0.010 [−0.026, +0.007] | **−1.995 [−2.143, −1.848]** | −0.062 [−0.156, +0.032] |
| 1/16 extreme | 0.149 | 0.298 | **−0.394 [−0.452, −0.335]** | −0.115 [−0.149, −0.081] | **−2.652 [−2.797, −2.506]** | **+1.241 [+0.954, +1.528]** |
| binned low | 0.173 | 0.346 | **−0.574 [−0.718, −0.431]** | +0.142 [−0.002, +0.287] | **−0.917 [−1.073, −0.762]** | **−0.898 [−1.240, −0.556]** |
| binned high | 0.028 | 0.056 | **+0.119 [+0.092, +0.145]** | −0.029 [−0.040, −0.019] | **−0.281 [−0.328, −0.233]** | **+0.076 [+0.026, +0.125]** |

★ marks the pre-registered shadow slices.

- **MDTA** has no significant shadow gain. It is significantly worse on 1/64
  high, extreme and binned low, and slightly better on 1× high and binned
  high. It fails the follow-up gate on shadow gains, on the maximum loss and
  on ANE latency (1.61× > 1.5×).
- **Window attention** has no significant effect anywhere. It fails the gate
  on ANE placement.
- **Learning curves** (1/16 high, at 3k / 6k / 9k / 12k steps) are:
  - NAFNet 59.49 / 62.99 / 64.92 / 65.41 dB;
  - MDTA 59.55 / 63.54 / 64.78 / 65.42 dB;
  - Window 59.78 / 63.13 / 64.72 / 65.35 dB.

  Over the last 3k steps the transformers gained +0.64 and +0.63 dB, against
  NAFNet's +0.49 dB. That is below the pre-registered "still learning"
  threshold of 2× the control. On the 1× low and 1/16 tails slices they
  gained less than NAFNet. None is pulling away.

### Table B: bridge K sweep (PSNR change vs control, dB)

| Slice | Control | K1 | K2 | K4 | K8 | K16 | K4 stoch. | K16 stoch. | Seed std / input noise (K4 / K16) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1× low | 59.66 | −0.52 | −0.84 | −1.28 | −1.67 | −2.03 | −4.49 | −4.66 | 0.89 / 0.95 |
| 1× high | 52.83 | −0.14 | +0.10 | +0.08 | +0.04 | −0.01 | −0.32 | −0.33 | 0.08 / 0.08 |
| 1/16 high | 65.41 | −2.21 | −0.16 | +0.13 | −0.08 | −0.51 | −0.53 | −1.23 | 0.02 / 0.02 |
| 1/64 high | 69.08 | −2.23 | +0.26 | +0.65 | +0.43 | −0.02 | +0.31 | −0.43 | 0.02 / 0.02 |
| 1/16 tails | 69.97 | −1.40 | +0.04 | +0.27 | +0.29 | +0.23 | −0.68 | −0.77 | 0.04 / 0.04 |
| binned low | 59.34 | −0.92 | −0.81 | −0.90 | −1.14 | −1.47 | −3.62 | −4.00 | 1.09 / 1.12 |

- PSNR peaks at K = 2–8, usually K = 4, and falls after that. At 1× low it
  falls with *every* added step.
- At 1× low, U rises from 0.136 (K1, same as the control's 0.137) to 0.243
  (K4) and 0.249 (K16). The extra steps add high-frequency content that the
  reference does not contain. In the binned-low slice, U falls with K
  instead, so this signal is not universal.
- Stochastic sampling is worse than deterministic K4 on all six slices. It
  is below the control on five of them; the exception is 1/64 high at K4
  (+0.31 dB).
- Seed std is the spread between stochastic runs on the *same* input,
  divided by the spread of the input noise. At low noise it is 0.89–1.12. At
  high noise it is 0.02–0.08, where the noise dwarfs any invented texture.
- The pre-registered rule "worth pursuing" requires K=1 to be within
  max(2F, 0.1 dB) of the control. It fails on every slice.

### Table C: compute-matched control, NAFNet w64 vs bridge K=4

Cost per 256² packed pass: w64 43.1 GFLOP, bridge K4 45.2 GFLOP (4 × 11.3).

| Slice | w64 − bridge K4 [95% CI] | Sources where w64 is worse | w64 − w32 | Bridge K4 − w32 |
|---|---|---:|---:|---:|
| 1× low | +1.42 [+1.18, +1.66] | 0/79 | +0.15 | −1.28 |
| 1× medium | +0.53 [+0.48, +0.58] | 0/79 | +0.23 | −0.29 |
| 1× high | +0.49 [+0.46, +0.52] | 0/79 | +0.57 | +0.08 |
| 1/16 low | +1.44 [+1.34, +1.54] | 0/79 | +0.32 | −1.12 |
| 1/16 medium | +1.69 [+1.56, +1.81] | 0/79 | +1.00 | −0.69 |
| 1/16 high ★ | +1.81 [+1.59, +2.02] | 0/79 | +1.93 | +0.13 |
| 1/64 low | +2.83 [+2.62, +3.04] | 0/79 | +1.06 | −1.77 |
| 1/64 medium | +2.27 [+2.08, +2.46] | 0/79 | +1.62 | −0.65 |
| 1/64 high ★ | +1.29 [+1.15, +1.43] | 0/79 | +1.94 | +0.65 |
| 1/16 tails ★ | +1.18 [+1.11, +1.26] | 0/79 | +1.45 | +0.27 |
| 1/64 tails | +1.79 [+1.69, +1.90] | 0/79 | +1.73 | −0.06 |
| 1/16 extreme | +1.13 [+0.79, +1.47] | 25/79 | +2.37 | +1.24 |
| binned low | +1.54 [+1.22, +1.87] | 6/79 | +0.64 | −0.90 |
| binned high | +1.08 [+0.99, +1.17] | 0/79 | +1.15 | +0.08 |

- The w64 arm wins on all 14 slices, with 0/79 sources worse on 12 of them.
  Under the addendum rule, the bridge gain is attributed to compute.
- The w64 − w32 column is a difference of means from a single w64 seed; no
  paired CI was computed for it. Every value is above 2F for its slice.
- In the 1× slices, w64 also lowers U (1× high: 0.148 vs 0.249). It removes
  more noise without adding texture.

### Table D: deployment cost per 512² packed tile

| Arm | ANE ops | ANE ms/tile | × 140 tiles (s) | × 400 tiles (s) | MPS fp16 / fp32 ms | GPU host, full X2D II frame (s) |
|---|---:|---:|---:|---:|---|---:|
| NAFNet w32 | 100% | 26.1 | 3.7 | 10.4 | 52.4 / 58.8 | 3.41 |
| MDTA | 97.7% | 42.0 | 5.9 | 16.8 | 64.3 / 68.5 | 4.75 |
| Window | **0%** | 496.7 | 69.5 | 198.7 | 57.6 / 61.8 | 3.24 |
| Bridge K1 | 100% | 26.1 | 3.7 | 10.4 | 52.4 / 58.2 | 3.14 |
| Bridge K4 | 100% | 101.5 | 14.2 | 40.6 | 206.9 / 228.9 | 12.46 |
| NAFNet w64 | 100% | 47.9 | 6.7 | 19.1 | not measured | not measured |

How these were measured:

- Core ML used fp16 and `CPU_AND_NE`, with the median of 30 runs and random
  weights. Latency depends on the architecture, not on the weights. The fp16
  max abs difference was ≤ 6.9e-4 for every arm.
- **Tile counts.** The "×400" column is the benchmark script's convention.
  Our tiler (overlap 58, mirrored apron) actually used **140 tiles** for the
  full X2D II frame, so the ×140 column is the realistic estimate. Both
  columns exclude tiling and copy overhead.
- Bridge K16 took 49.77 s per frame on the GPU host.

### Shared defects (not architecture effects)

- **Flat-field identity fails for every trained arm.** The gate requires
  error within 1e-3 (`05-evaluation.md`). NAFNet s1 passes at levels 0.001
  and 0.01, then fails:
  - 6.5e-3 at 0.05;
  - 0.021 at 0.18;
  - 0.084 at 0.7.

  NAFNet s2, MDTA, window and w64 fail in the same way, starting at level
  0.05 or earlier: 0.014–0.020 at 0.18 and 0.084–0.091 at 0.7. Bridge K4 is
  about twice as bad: 0.047 at 0.18 and 0.177 at 0.7. The mean bias is much
  smaller than the max error (|bias| ≤ 5.6e-3), which suggests localised
  errors rather than a global shift. This is a training-recipe issue that
  predates this study.
- **Seams.** We compared the output with a second run on a tile grid shifted
  by (128, 128). The error near tile boundaries, relative to elsewhere, was
  1.26–1.40 for every arm. The maximum difference was 0.0037–0.0068. There is
  no architecture difference. Some tile-phase dependence remains, likely from
  per-tile pooled statistics.

### Real photo: X2D II, ISO 12800

Stone-wall, window-frame and tree-against-sky crops were rendered with the
same display lift for every arm (private-data experiment; artefacts not published).

The bottom-right panel of each is the bridge K4 stochastic seed-std map.

- **All arms look nearly identical.** All are smooth and somewhat waxy. Stone
  texture is largely lost, and edges are soft. None meets the acceptance bar
  in `05-evaluation.md`. These models are small, short-trained
  and trained on X1D II only. The w64 arm was not rendered.
- A **hot pixel** (red dot, stone wall) survives in NAFNet, MDTA, window and
  bridge K1. It is fainter at K4 and gone at K16.
- Bridge K4 and K16 show slightly more granular fine texture along the window
  mullions. Without a clean reference we cannot say whether it is real.
- **Where diffusion is uncertain.** In the tree crop, the seed-std map is
  brightest along branch edges and in the foliage gaps. It is dark in the
  flat sky. The generative sampler varies precisely where the true detail is
  ambiguous. The map is scaled to its own 99th percentile. In absolute terms
  the seed std is 2.4–4.4% of the estimated input noise across the three
  ROIs, which is small because the noise at ISO 12800 is very large.

## Recommendation and next steps (in priority order)

1. **Keep NAFNet** as the architecture. Ties go to NAFNet by the
   pre-registered rule. It is the deployment pipeline's model class, and it runs fully
   on the ANE.
2. **Spend compute on capacity, measured against the ANE budget.** Test w48
   and w64. Also test a deeper published NAFNet configuration
   ([2,2,4,8] / 12 / [2,2,2,2]). Doubling width costs
   about 3.8× the FLOPs (11.3 → 43.1 GFLOP). Adding blocks at low resolution
   adds parameters more cheaply. That is a hypothesis to measure, not a
   result. Use a seed replicate and longer training, and probe ANE latency
   per frame.
3. **Fix flat-field identity.** It is a release gate, and it affects every
   arm. Find out why errors grow with level above 0.01 before comparing
   further models.
4. **Measured dark-frame noise and real pairs remain the main lever.** The
   literature and the AIM 2025 organisers agree on this. Nothing in this
   study changes it.
5. **Diffusion only as an offline noise generator,** after a dark-frame
   library exists. Compare it with direct sampling of real dark frames
   (2-Shots in the Dark showed that the simpler approach can win). Do not
   use it as the shipped denoiser.
6. **Attention: a small MDTA bottleneck experiment, if at all.** It must
   pass the same gate: a shadow gain beyond 2F, no loss beyond
   max(2F, 0.10 dB), and ANE latency ≤ 1.5× the control. Do not pursue window
   attention unless it can be made to run on the ANE.
7. **Bursts** are where attention genuinely adds information. Compare learned
   fusion against robust aligned averaging first, and keep the clean
   references independent of the input bursts.

## Limitations

- Development data only. The synthetic noise is added to imperfect ISO 100
  references, and the validation split was inspected in earlier rounds. The
  results show relative differences between arms, not absolute quality.
- 12k steps, small models (about 2.65M parameters) and
  about 3.8k training crops. Transformers may close the gap with more data
  and longer training. The literature at full scale still shows no advantage
  over NAFNet.
- Training was on X1D II only. The X2D II photo is an unpaired visual stress
  case with no ground-truth PSNR.
- One seed per arm except the control. The noise floor comes from one
  replicate pair. The w64 arm is a single seed, added post hoc.
- The bridge is one diffusion formulation. Its training put 25% of updates
  at the first step. Other schedules or distillation were not tried, and
  latent or pretrained diffusion priors were not tested.
- ANE timings are single-tile probes with random weights. They are not
  full-frame measurements.

## Reproduce

Configs are in `configs/arch_study/`: `nafnet_s1`, `nafnet_s2`, `mdta`,
`win_mid`, `bridge`, `nafnet_w64` (plus `smoke_*`). Run the following on the
GPU host from the repository root.

```bash
# Train (one arm per command; pick a free GPU)
python3 scripts/run_cuda.py scripts/train_arch_study.py \
  --config configs/arch_study/nafnet_s1.yaml --device cuda:0

R=runs/arch_study; M="--manifest data/bootstrap/x1d2_val.json --root data"
CK="--checkpoint nafnet_s1=$R/nafnet_s1/last.pt --checkpoint nafnet_s2=$R/nafnet_s2/last.pt \
    --checkpoint mdta=$R/mdta/last.pt --checkpoint win_mid=$R/win_mid/last.pt"

# Main evaluation (Table A)
python3 scripts/run_cuda.py scripts/evaluate_arch_study.py $CK \
  --checkpoint bridge=$R/bridge/last.pt --bridge-steps 1,4 --bridge-stochastic-steps "" \
  $M --device cuda:0 --baseline nafnet_s1 --out $R/eval_main.json

# Bridge K sweep (Table B)
python3 scripts/run_cuda.py scripts/evaluate_arch_study.py \
  --checkpoint nafnet_s1=$R/nafnet_s1/last.pt --checkpoint bridge=$R/bridge/last.pt \
  --bridge-steps 1,2,4,8,16 --bridge-stochastic-steps 4,16 --bridge-seeds 4 \
  --slices "native|1|low" "native|1|high" "native|0.0625|high" "native|0.015625|high" \
           "native|0.0625|tails" "binned|1|low" \
  $M --device cuda:1 --baseline nafnet_s1 --out $R/eval_bridge.json

# Compute-matched control (Table C)
python3 scripts/run_cuda.py scripts/evaluate_arch_study.py \
  --checkpoint nafnet_s1=$R/nafnet_s1/last.pt --checkpoint nafnet_w64=$R/nafnet_w64/last.pt \
  --checkpoint bridge=$R/bridge/last.pt --bridge-steps 4 --bridge-stochastic-steps "" \
  $M --device cuda:0 --baseline bridge@K4 --out $R/eval_w64.json

# Learning curves: pass step_0003000/6000/9000.pt checkpoints per arm with
# --bridge-steps 1 --realizations 2 --bridge-seeds 2 (private-data experiment; artefacts not published)

# X2D II real-photo diagnostics, seams and seed-std maps
python3 scripts/run_cuda.py scripts/arch_study_x2d.py \
  --raw data/evaluation_x2d/stress_iso12800.3fr $CK --checkpoint bridge=$R/bridge/last.pt \
  --device cuda:1 --out-dir $R/x2d
```

Run these on the Mac for Table D:

```bash
PYTHONPATH=src .venv/bin/python scripts/bench_tile.py --device mps --dtypes fp32,fp16 \
  --out runs/arch_study/bench_mps.json            # defaults to the five main arms
PYTHONPATH=src python3 scripts/probe_coreml.py \
  --out runs/arch_study/coreml_probe_native.json  # system Python 3.9; add --config for w64
```

The original launch scripts and pre-registration hashes are local
(private-data experiment; artefacts not published).
