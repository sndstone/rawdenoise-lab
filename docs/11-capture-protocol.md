# 11. Tripod burst capture checklist

**Start with the A5100 indoors under constant light.** On 2026-11-01, wind and
dusk changed outdoor scenes between ISO groups. Only X1D II ISO 25600 f/7.1 vs
ISO 100 f/7.1 supplied a useful evaluation region: about 17.5% of the image.
A tripod alone did not provide paired ground truth.

## Set up

- [ ] Static fabric, print, wood and shadow detail; remove moving foliage,
  screens, people and reflections. Close curtains. Use stable continuous light;
  check mains/PWM flicker at the shortest shutter.
- [ ] Solid tripod, stabilization off, manual focus, fixed focal length,
  aperture and WB. Prefer uncompressed RAW. Disable long-exposure NR and automatic
  exposure adjustments. Leave headroom below RAW clipping in every channel.
- [ ] Remote release or timer. Do not touch, refocus, reframe or change aperture
  between ISO groups; allow settling after controls. Log body, lens, shutter
  mode, light, temperature/warm-up state and scene identifier.

## Shoot a complete scene before moving anything

- [ ] Base ISO: A5100 / A7 III / X1D II ISO 100; X2D II ISO 64. Shoot **32–64 clean
  frames** for noisy ISO ≤800, or at least **16** for higher-ISO references.
- [ ] Shoot **4–8 noisy frames per ISO**, 200–25600 where supported. Match RAW
  brightness using `effective_ISO_low × t_low / f_low² = effective_ISO_high × t_high / f_high²`.
  At fixed aperture, ISO 100 at 1 s pairs with ISO 3200 at about 1/32 s.
  **X1D II RAW gain stops increasing above ISO 6400** in the 2026-11-01 dark
  calibration: use `effective_ISO = min(ISO, 6400)` for its expected RAW signal.
  An ISO 25600 label does not supply another two stops of RAW amplification.
  Record actual shutter values and verify RAW signal; other cameras need their
  own gain calibration.
- [ ] Keep aperture fixed: changing it changes depth of field and vignetting.
  If shutter limits intervene, change lighting and restart the entire set.
  Label intentional underexposure separately: one stop under uses half the
  matched shutter time.
- [ ] Bracket noisy groups with short base-ISO check bursts to reveal light
  changes. Complete the clean set promptly; do not wait for dusk.
- [ ] Copy each scene to a separate folder and retain originals. Run immediately:
  `PYTHONPATH=$PWD/src python scripts/check_burst.py CAPTURE_DIR --output runs/check_scene`.
  Inspect `index.html` thumbnails, motion masks, clipping, exposure and shift;
  reshoot while the setup remains available.

## Darks and flats: per camera and shutter profile

- [ ] **A5100 / A7 III:** log the available mechanical/electronic-curtain mode and
  calibrate in the scene mode. **X1D II:** keep leaf-shutter and electronic-shutter
  darks/flats and calibration profiles separate. Log mode manually if EXIF omits
  it. Separate distinct modes on other bodies too; never pool them silently.
- [ ] Darks: lens cap, dark room, long-exposure NR off. Per ISO, collect **64 at
  the shortest supported shutter**, **16 at 1 s**, **16 at 30 s** where supported.
  Log actual limits and temperature; repeat a subset after 20 minutes of live view.
- [ ] Flats: stable uniform panel/diffuser, defocus, fixed aperture/WB. Per ISO
  and mode, shoot **two independent frames at each of about 20 exposure levels**,
  near black to below clipping, varying shutter. Avoid flicker/gradients; retain
  pairs separately. See [data](03-data.md) for scalar calibration limitations.

## Read the verdict

Defaults: ≥10% static and ≥20 complete static blocks **USABLE**, 3–10%
**MARGINAL**, <3% **RESHOOT**. Too few static blocks downgrade the pair (zero
blocks require reshooting). Configure these with `--min-static`,
`--marginal-static` and `--min-blocks`. These thresholds screen for a scorable
research region; they do not certify a whole-scene clean target or a model.
The evaluator's default scoring floor is 5%, so a 3–5% MARGINAL region requires
an explicit lower evaluator floor or a reshoot.

Both tools import the same per-channel gain/offset and smooth local illumination
fit, clean-burst temporal variance gate, two-block dilation and five-sigma
noisy-group-mean test. The checker defaults to **native packed planes and
16×16 packed-pixel blocks**, matching `eval_real_burst.py`. Native pair comparisons
took about 0.6–3 seconds on the November captures (decoding/group analysis excluded).
`--max-side N` explicitly selects an approximate binned preview; its masks are
not calibrated against native masks and must not replace native scoring.

The checker reports measured gain, offset and local illumination ranges, plus
nominal and analog-capped exposure discrepancies as **INFO only**. These cannot
distinguish illumination drift from gain calibration errors. Set per-camera
ceilings with `--highest-analog-iso 'X1D II=6400'` (repeat as needed; zero disables
a ceiling), or `cameras.CAMERA.highest_analog_iso` in `--calibration` YAML. The
optional default `calib/darks_20261101.yaml` has precedence over built-in values
when that explicit field is present; absent files/fields retain the X1D II 6400
default. Dark read noise alone is not used to infer a new ceiling.

Automatic phase correlation is rounded to an integer packed shift. A correction
>2 sensor pixels, aperture changes or within-burst light steps >0.1 EV require
reshooting; a 1–2 pixel correction, >1% clipping, uncertain registration or <3
frames downgrade the pair. Thumbnail-based within-burst shifts also downgrade
for native alignment review, rather than treating subpixel preview estimates
as proven camera movement.

Use `--alignment-spec PATH` with an evaluator spec to reuse independently measured
integer packed shifts (matching clean/noisy filename lists). On X1D II ISO 25600
f/7.1 vs ISO 100 f/7.1, the recorded zero shift yields **USABLE, 17.4573% static**,
matching the evaluator. Automatic registration chooses `[0, 1]` packed pixels
and gives **MARGINAL, 21.1496%**; verify alignment before scoring. Other November
pairings still require reshooting for motion, light steps or aperture changes.

Noise expectations use brightness-bin medians from the burst, not independent
profiles; pervasive motion within a brightness bin can contaminate them. A smooth
illumination fit can absorb slow scene changes. Retain masks and inspect the
actual static regions, verify registration and independent target noise, and
assign held-out scenes before model evaluation; see [evaluation](05-evaluation.md).
