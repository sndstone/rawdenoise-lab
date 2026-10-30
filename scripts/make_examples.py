#!/usr/bin/env python3
"""Make before/after example images: noisy input RAW vs. our denoised DNG.

Both files are rendered identically with rawpy (camera white balance, sRGB, the same
exposure lift). Each example has an overview of the denoised frame with the crop marked,
and the crop at 2x (nearest neighbour), input on the left and denoised on the right.

  python scripts/make_examples.py --pair photo.3FR photo_denoised.dng --out examples/x.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1212   # final image width
CROP = 300     # crop size in sensor-rendered pixels, shown at 2x


def render_linear(path):
    with rawpy.imread(str(path)) as raw:
        flip = raw.sizes.flip   # read before postprocess: user_flip=0 resets it
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                              gamma=(1, 1), user_flip=0)
    return rgb.astype(np.float32)/65535., flip


def orient(img, flip):
    """Apply LibRaw's orientation code (3: 180°, 5: 90° CCW, 6: 90° CW)."""
    return {3: lambda a: np.rot90(a, 2), 5: lambda a: np.rot90(a, 1), 6: lambda a: np.rot90(a, -1)}.get(flip, lambda a: a)(img)


def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= .0031308, 12.92*lin, 1.055*np.power(lin, 1/2.4) - .055)


def pick_crop(img, size=CROP):
    """Most textured mid-tone window, chosen on the denoised render (8x downscaled search)."""
    small = img[::8, ::8].mean(-1)
    s = size//8
    gy, gx = np.abs(np.diff(small, axis=0))[:, :-1], np.abs(np.diff(small, axis=1))[:-1]
    energy = gy + gx
    best, pos = -1., (img.shape[0]//2 - size//2, img.shape[1]//2 - size//2)
    for y in range(0, energy.shape[0]-s, max(1, s//4)):
        for x in range(0, energy.shape[1]-s, max(1, s//4)):
            m = small[y:y+s, x:x+s].mean()
            if not .12 < m < .7:
                continue
            e = float(np.median(energy[y:y+s, x:x+s]))
            if e > best:
                best, pos = e, (y*8, x*8)
    return pos


def label(img, text, font):
    d = ImageDraw.Draw(img)
    w = d.textlength(text, font=font) + 16
    d.rectangle([0, 0, w, 26], fill=(0, 0, 0))
    d.text((8, 4), text, fill=(255, 255, 255), font=font)


def make(inp, den, out, title, crop=None, exposure=None):
    a, flip = render_linear(inp)   # the source RAW's orientation applies to both renders
    b, _ = render_linear(den)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = orient(a[:h, :w], flip), orient(b[:h, :w], flip)
    lum = b.mean(-1)
    k = exposure or float(np.clip(.8/max(np.percentile(lum, 99.5), 1e-4), 1, 32))
    A, B = to_srgb(a*k), to_srgb(b*k)
    y, x = crop or pick_crop(B)
    try:
        font = ImageFont.truetype("Helvetica.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    ov = Image.fromarray((B*255).astype(np.uint8))
    scale = WIDTH/ov.width
    ov = ov.resize((WIDTH, round(ov.height*scale)), Image.LANCZOS)
    d = ImageDraw.Draw(ov)
    d.rectangle([x*scale, y*scale, (x+CROP)*scale, (y+CROP)*scale], outline=(255, 210, 0), width=3)
    label(ov, title, font)
    half = (WIDTH-12)//2
    crops = []
    for img, name in ((A, "Input (noisy RAW)"), (B, "Denoised (r4_plain)")):
        c = Image.fromarray((img[y:y+CROP, x:x+CROP]*255).astype(np.uint8)).resize((half, half), Image.NEAREST)
        label(c, name, font)
        crops.append(c)
    sheet = Image.new("RGB", (WIDTH, ov.height + 12 + half), (24, 24, 24))
    sheet.paste(ov, (0, 0))
    sheet.paste(crops[0], (0, ov.height+12))
    sheet.paste(crops[1], (half+12, ov.height+12))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90, optimize=True, progressive=True)
    return dict(out=str(out), crop_yx=[int(y), int(x)], exposure=k)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, required=True, metavar=("INPUT_RAW", "DENOISED_DNG"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--crop", help="y,x of the crop's top-left in rendered pixels (default: automatic)")
    p.add_argument("--exposure", type=float, help="linear exposure multiplier (default: automatic)")
    a = p.parse_args()
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    print(json.dumps(make(Path(a.pair[0]), Path(a.pair[1]), a.out, a.title, crop, a.exposure)))


if __name__ == "__main__":
    main()
