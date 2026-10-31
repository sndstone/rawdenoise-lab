# Experimental 16-bit RGB TIFF path

RGB TIFF and Bayer RAW are separate model domains. Released four-plane Bayer
weights **cannot process RGB**. There are no shipped trained RGB weights or
demonstrated RGB denoising quality. This feature provides experimental training
and inference plumbing; random weights and a smoke training run are not useful
photo denoisers. Acceptance still requires independent held-out references and
the owner's blind visual review under [evaluation requirements](05-evaluation.md).

## Model and precision contract

`model_config.arch = "rgb_nafnet"` builds a distinct three-input/three-output
residual NAFNet using this repository's independently implemented NAFNet blocks.
It operates directly in the declared RGB encoding, without Bayer packing, the
Bayer gamma wrapper, demosaicing, or sensor defect correction. Initialization is
random; no external weights, distillation or third-party denoised targets are
allowed. Other architecture parameters are the NAFNet body parameters, for
example `width`, `enc_blk_nums`, `middle_blk_num`, `dec_blk_nums`.

Experimental checkpoint flags `boundary_mode="replicate"` and
`upsample_mode="nearest"` replace zero convolution boundaries and PixelShuffle
with replicated boundaries and nearest resize followed by a learned 1x1
convolution. They preserve spatially constant features across decoder pixel
phases; they do not guarantee zero color bias or denoising quality. Flags are
part of the architecture and must be recorded in the checkpoint. Existing models
keep `zeros` and `pixelshuffle` defaults and their original predictions.

A PyTorch `.pt` checkpoint must contain `model_config`, `model` state and this
top-level `rgb_contract` (example for unprofiled sRGB-encoded values):

```json
{
  "version": 1,
  "channels": "RGB",
  "dtype": "uint16",
  "normalization": "uint16/65535",
  "color_space": "srgb"
}
```

`color_space` is `srgb`, `linear-srgb`, or `icc`. An `icc` contract additionally
requires `icc_sha256`, the lowercase SHA-256 digest of the exact embedded profile
bytes. Profiled inputs must use `icc`; they must match the checkpoint's profile
exactly. No ICC or transfer-curve transforms are implemented. An explicit CLI
declaration conflicting with embedded metadata fails. EXIF-tagged sRGB is
recognized; other unprofiled EXIF color spaces are rejected. TIFF colorimetry and
transfer-function tags are rejected because their transforms are unsupported.

TIFFs must have one page, three RGB channels, uint16 samples and no alpha.
Contiguous and separate planar RGB are supported. Normalization is finite
float32 `uint16 / 65535`, keeping RGB order. Reflected tiles are feathered and
accumulated in float32. Output is clipped to [0,1], multiplied by 65535, rounded
to nearest (ties to even), and written as uint16 RGB TIFF. There is no 8-bit
intermediate. Tests preserve all 65,536 codes in every channel through tiled
identity-model inference and TIFF output; this is a precision check, not a
quality claim for learned predictions.

## Inference

After training your own RGB checkpoint:

```bash
rawdenoise photo.tif --model runs/rgb/experimental.pt --device cpu --rgb-color-space srgb
```

Auto detection routes RGB-photometric `.tif`/`.tiff` inputs to RGB; Bayer TIFFs
keep the Bayer path. `--input-kind rgb` explicitly selects TIFF parsing.
Untagged, unprofiled TIFFs require `--rgb-color-space srgb` or `linear-srgb` to
declare the actual encoding, never to request a conversion. Profiled TIFFs use
their embedded profile. Outputs carry the contract as ImageDescription JSON
under `denoise_lab_rgb`, so they can be read without another declaration.

RGB defaults to `<stem>_denoised.tiff`; `--format tiff` is also accepted. DNG,
Bayer mosaic and NPY outputs are rejected for RGB. Existing files and source
files are never overwritten. `--strength` and `--overlap` work in RGB pixels.
The Bayer defect prepass is never applied to RGB (`--no-prepass` is unnecessary).
RGB runtime support is PyTorch only; current ONNX/Core ML artifacts and export
recipes are Bayer-specific. Model shape or color-space/profile mismatch fails
even at strength zero. Python inference uses `Denoiser.denoise_rgb` with an
explicit `color_contract`; `denoise_packed` remains Bayer-only.

ICC bytes, orientation, pixel geometry and TIFF resolution are preserved.
Orientation is retained as a tag without rotating pixels. Arbitrary EXIF,
descriptive text, editing history, previews and other TIFF tags are not copied;
retain your source. Profile bytes are preserved opaquely, not interpreted or
validated as a working color transform.

## Independent paired training recipe

Prepare registered noisy/independent-clean uint16 TIFF pairs in the exact same
color space, profile, orientation, geometry and processing pipeline. No automatic
alignment, resizing, demosaicing or color conversion is done. Clean targets must
have documented independent provenance, such as aligned low-noise captures or
burst means; do not use outputs from other denoisers. Record permissions for
each source. Keep related scenes together across splits; validation is for
development, and `test` scenes are reserved for final acceptance.

Create `data/rgb/pairs.json`; all file paths are relative to that manifest:

```json
{
  "version": 1,
  "rgb_contract": {
    "version": 1, "channels": "RGB", "dtype": "uint16",
    "normalization": "uint16/65535", "color_space": "srgb"
  },
  "pairs": [
    {
      "noisy": "scene01/noisy.tif", "clean": "scene01/burst_mean.tif",
      "scene": "scene01", "split": "train", "camera": "your camera", "iso": 3200,
      "source": "owner original captures", "usage_terms": "owner permission for intended training use",
      "target_provenance": "independent aligned burst mean; capture and processing records stored alongside",
      "processing": "fixed sRGB encoding, identical WB/exposure/renderer for both files"
    },
    {
      "noisy": "scene02/noisy.tif", "clean": "scene02/burst_mean.tif",
      "scene": "scene02", "split": "validation", "camera": "your camera", "iso": 3200,
      "source": "owner original captures", "usage_terms": "owner permission for intended training use",
      "target_provenance": "independent aligned burst mean; capture and processing records stored alongside",
      "processing": "same fixed pipeline as training pairs"
    }
  ]
}
```

Keep capture hashes, exposure, processing settings and independent target records
with the dataset. Required provenance strings are declarations, not automatic
verification of permission or ground truth. The loader rejects scene/file-path
leakage, unequal geometry, incompatible color metadata and missing provenance.
Crops and flip/transpose augmentation are paired and preserve channel order.

```bash
python scripts/train_rgb.py --manifest data/rgb/pairs.json --output runs/rgb/experimental.pt --device cpu --steps 1000 --crop-size 128
```

The standalone script requires train and validation scenes, starts from random
weights, uses float32 L1 optimization and checks finite loss/gradients. It records
the manifest and source hashes, initialization, architecture, color contract,
recipe, step count and development center-crop MSE. The output path must be new.
The script does not load any initial checkpoint and does not select or evaluate
on final test scenes (their files are only hashed to reject cross-split duplicates).
It provides a simple baseline recipe, without resume,
EMA, scheduler, quality promotion or exhaustive evaluation. CPU is the default;
accelerators must be explicitly requested. Larger training runs and held-out,
per-camera/noise/color-space evaluation are still required before any quality
claim.

Optional experimental training controls include `--identity-init` (zero output
correction), `--identity-weight` (preserve clean inputs), `--flat-identity-weight`
(preserve generated constant RGB fields), and `--color-bias-weight` (penalize
channel-mean error). All extra loss weights default to zero. Generated flat
colors use a separate seeded generator. `--boundary-mode replicate` and
`--upsample-mode nearest` select the architecture flags above. Evaluate low-noise,
clean-input, color-bias and border behavior alongside noisy-image scores when
using these recipes; a higher average can conceal a regression.

Training checkpoints contain the private manifest, source hashes and local paths,
including split metadata. Keep them private under `runs/`. Before distributing any
future RGB artifact, create a separate portable checkpoint containing only the
model configuration, selected state and RGB contract, plus a reviewed model card.
Do not publish training checkpoints or their provenance payload as release assets.
