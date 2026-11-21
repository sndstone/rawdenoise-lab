# Round 9: matched camera-noise recipes

`r9_phys35` is the fresh 35% physics control on the fixed code. Compare it with
`r9_legacy35` to test the physics package at the same mixture, including its
changed read-tail and hot-pixel branches. `r9_phys20` and `r9_phys10` test whether
a smaller camera mixture preserves robustness while retaining real-image gains.
All arms use seed 20261104, 24,000 steps, the same loss and learning rate, and
initialization from round 4:

```sh
python scripts/train_arch_study.py --config configs/round9/NAME.yaml --init-checkpoint runs/round4/r4_w40s4_plain/last.pt --run-dir runs/round9/NAME --device cuda:N
```

Use a new run directory for each arm. `checkpoint_every: 24000` and
`keep_checkpoints: 0` write/replace `last.pt` without creating numbered checkpoint
copies; they do not delete old copies from reused directories. EMA remains
embedded in `last.pt`. The trainer also writes config, history and progress files.

This is a mixture-recipe sweep: the weights also change clean-source proportions
and explicit defect-injection probability (45% / 40% / 32.5% at 10% / 20% / 35%).
Only the broad source requests 0.5 defects; camera sources default to zero.
Compare the original development gate first, then diagnostics, and shortlist at
most one dose for real/blind evaluation. Repeat a winning dose with seed 20261105
before claiming reproducibility. Synthetic agreement cannot establish real gains
or separate the individual physics components.

The DN fix selects discrete metadata from the nearest normal profile in log-ISO
(lower ISO at midpoint ties), or the exact processed profile. Only the processed
mode at ISO 65535 differs in the current calibration: black 1024 instead of 512,
span 15359 instead of 15871, and normalized code step 1/15359 instead of 1/15871.
The legacy negative floor changes from -512/15871 to -1024/15359. The round-8
physics quantizer already used the correct black/white levels and DN steps, so
its synthesized values remain unchanged by this metadata correction. Use the
fresh control to keep code/config hashes matched across arms. This processed
profile occurs in 2.075% of all round-6 samples and 0.4875% of all round-8 samples;
ordinary ISOs and other cameras keep their existing DN metadata.

Camera datasets reject non-default inherited generic noise/resiting/coupling
settings that their sampling path does not use. For ordinary bootstrap sources,
`iso_coupled` overrides generic noise controls and exposure sampling settings;
`exposure_scale` still applies. The baseline trainer rejects research-only
optimizer settings; use the research trainer for these recipes.

Camera identity maps historically retain nominal camera noise. The opt-in
dataset setting `zero_identity_noise_map: true` matches the zero shot/read maps
of bootstrap identity samples, including samples with subsequent defect
injection. It changes conditioning only, adds no random draws, and defaults to
false to preserve past recipes. These four recipes retain that default and use
an unconditioned model.
