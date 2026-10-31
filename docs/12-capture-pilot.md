# Small real-reference capture pilot

Start with the X1D II, matching the frozen Bayer model's training scope. Another
body can be substituted, but results then apply to that body only. This is a
pilot to validate preparation and expose failures, not enough for full camera
and ISO acceptance. The larger protocol remains in [03-data.md](03-data.md).

## Keep three capture sessions separate

Use eight static scenes across three capture days. Keep the days and scenes out
of all previous training and validation. Record their intended role before
looking at denoised comparisons:

| Session | Scenes | Role |
|---|---:|---|
| A | 4 | Training/preparation pilot |
| B | 2 | Development validation |
| C | 2 | Sealed test; no selection or recipe tuning |

Include fabric or fine print, wood or stone, stationary foliage, and dark neutral
surfaces with small highlights. Include texture and smooth areas in every frame
where possible. Avoid people, wind, running water, flickering screens, moving
shadows and changing daylight. Test captures remain closed until a recipe and
checkpoint are frozen. If another capture day is not practical yet, collect a
single-session preparation pilot first and do not call it an independent test.

## Per scene

1. Put the camera on a stable tripod. Use manual focus and a fixed aperture,
   framing and white balance. Follow the camera's tripod guidance for any
   stabilization setting. Do not touch focus or zoom between sequences.
2. At ISO 100, expose the scene well while retaining the highlights. Record
   **32 reference RAW frames**, or 64 if practical. These form the independent
   clean-reference pool. Take **4 additional ISO-100 RAWs** as noisy inputs;
   these four must never enter the reference average.
3. At ISO 800, 3200 and 12800, take **4 RAWs each**. Keep the aperture fixed and
   shorten the exposure approximately in proportion to ISO so scene brightness
   stays comparable. Record actual exposure times. Do not change lighting to
   compensate. If a camera does not offer one of these settings, use and record
   the nearest supported value.
4. Preserve original RAWs and metadata, including clipped highlights and dark
   values. Do not denoise, sharpen or resize before sharing them. For each
   sequence record camera/lens, ISO, shutter time, aperture, capture day, scene
   identifier, frame role and any observed movement or lighting change.

With 32 reference frames this is 48 RAWs per scene, or 384 RAWs total. With 64
references it is 80 per scene, or 640 total. Budget storage from the camera's
actual file sizes: at 105 MB per RAW, roughly 40 or 67 GB respectively, before
targets and previews. Store originals on the NAS or an external drive. The
NVIDIA host's approximately 60 GB free is insufficient for the larger capture
plus all derived artifacts; ingest bounded crops while retaining a disk reserve.

## Simple folder layout

```text
real-reference-pilot/
  session-A/scene-01/reference-iso100/
  session-A/scene-01/input-iso100/
  session-A/scene-01/input-iso800/
  session-A/scene-01/input-iso3200/
  session-A/scene-01/input-iso12800/
  session-A/scene-01/capture-notes.txt
  ...
  session-B/...
  session-C/...
```

A shared-folder path is enough to start ingest. Do not move an older benchmark
or comparison scene into training. Comparator outputs are never targets.

## Preparation performed before training

- Hash originals and freeze scene/day roles. Audit duplicate pixels and overlap
  with prior datasets before opening the sealed test data.
- Check motion, registration, exposure matching, clipping, saturation levels and
  CFA geometry. Reject unusable pairs. Exposure matching across ISO requires
  measured photometric normalization; the nominal ISO ratio is only a capture
  guide. Do not normalize every frame independently to hide mismatch.
- Form the robust reference from reference frames only. Quantify remaining target
  noise and uncertainty; a finite-frame mean is a cleaner proxy, not noise-free
  truth. Unregistered averaging must not be used as a clean target.
- Use float32 caches from original sensor data. Preserve full code precision;
  existing float16 caches cannot be repaired by upcasting.
- The current `make_burst_targets.py` does not perform fine registration or
  cross-ISO exposure normalization. Its coarse mismatch check cannot certify
  subpixel alignment. Those preparation steps must be validated separately.
- For RGB, render the noisy and averaged clean data through one fixed pipeline
  into uint16 TIFF with the same white balance, exposure, color profile and
  orientation. Average/register in sensor or linear space, before the nonlinear
  display transform. Keep noise reduction and sharpening disabled. If a specific
  editor/export preset is the intended input workflow, also supply a few original
  uint16 TIFFs exported with that preset and identify its color profile. The RGB
  checkpoint must match the declared encoding and ICC contract.

Calibration darks and flat pairs will be needed for defensible camera/ISO noise
calibration. They can follow this first pilot; their protocol is in 03-data.md.
Pilot metrics alone do not establish calibration or broad real-photo acceptance.
