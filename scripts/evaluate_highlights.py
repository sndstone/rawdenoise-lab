#!/usr/bin/env python3
"""Reproduce clipped-highlight diagnostics with local, independently trained weights.

Writes only under --output. Real photos are unpaired: their Lab chroma distance
to input measures processing-induced change, not ground-truth colour accuracy.
Private RAWs are optional, read-only, evaluation-only inputs.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rawpy
import torch
from torch.nn import functional as F

from denoise_lab.bayer import normalize_raw, pack, unpack
from denoise_lab.defects import correct_defects
from denoise_lab.dng import RawMetadata, write_dng
from denoise_lab.inference import Denoiser, protect_highlights


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def to_srgb(linear):
    a = np.clip(linear, 0, 1)
    return np.where(a <= .0031308, 12.92*a, 1.055*a**(1/2.4)-.055)


def lab(srgb):
    """CIELAB D65 from display sRGB, without external colour libraries."""
    linear = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
    matrix = np.array([[.4124564, .3575761, .1804375],
                       [.2126729, .7151522, .0721750],
                       [.0193339, .1191920, .9503041]])
    xyz = (linear @ matrix.T)/np.array([.95047, 1., 1.08883])
    f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz/(3*(6/29)**2)+4/29)
    return np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]),
                     200*(f[..., 1]-f[..., 2])], axis=-1)


def regions(packed):
    core = torch.from_numpy((packed >= .98).any(axis=0))[None, None].float()
    grown = F.max_pool2d(core, 9, stride=1, padding=4)
    return core[0, 0].bool().numpy(), ((grown > 0) & (core == 0))[0, 0].numpy()


def synthetic(sigma=.0005):
    """Smooth Gaussian light, green clips first; noise precedes sensor clipping."""
    y, x = np.mgrid[:128, :128]
    light = .03+1.5*np.exp(-((y-64)**2+(x-64)**2)/(2*20**2))
    scene = np.array([.55, 1., .4, 1.])[:, None, None]*light
    clean = np.minimum(scene, 1).astype(np.float32)
    noisy = np.minimum(scene+np.random.default_rng(7).normal(0, sigma, scene.shape), 1).astype(np.float32)
    return clean, noisy


def cell_render(packed):
    # Synthetic camera: linear RGB with known WB, one RGB sample per Bayer cell.
    rgb = np.stack([packed[0]*1.7, (packed[1]+packed[3])*.5, packed[2]*2.], axis=-1)
    return to_srgb(rgb*.6)


def real_render(packed, metadata, path, exposure):
    # Canonical RGGB, normalized codes and source colour matrix/WB for all variants.
    meta = replace(metadata, pattern=np.array([[0, 1], [1, 2]], dtype=np.uint8),
                   black=np.zeros((2, 2)), white=np.full((2, 2), 65535.), orientation=1)
    write_dng(path, unpack(packed), meta)
    with rawpy.imread(str(path)) as raw:
        linear = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                 output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1),
                                 user_flip=0, adjust_maximum_thr=0.).astype(np.float32)/65535.
    return to_srgb(linear*exposure)


def crop_raw(path):
    with rawpy.imread(str(path)) as raw:
        metadata = RawMetadata.from_raw(raw, path)
        packed = pack(normalize_raw(raw))
    counts = (packed >= .98).sum(axis=(1, 2)).tolist()
    h, w = packed.shape[-2:]
    # For private full frames, evaluate a 512-packed crop with the most near-clips
    # among a fixed grid of windows. Include the final edge in the search.
    size = min(512, h, w)
    mask = (packed >= .98).any(axis=0)
    # Sparse scenes can otherwise select an isolated hot pixel in preference to
    # a real light. Check small read-only neighbourhoods before choosing a crop.
    if 0 < mask.sum() <= 64:
        for y, x in np.argwhere(mask):
            top, left = max(0, y-16), max(0, x-16)
            patch = torch.from_numpy(packed[:, top:y+17, left:x+17])[None]
            corrected = correct_defects(patch, k=8, abs_floor=.01, adjacency_ratio=.1)[0]
            mask[y, x] = bool((corrected[0, :, y-top, x-left] >= .98).any())
    ys = sorted(set(list(range(0, h-size+1, 256))+[h-size]))
    xs = sorted(set(list(range(0, w-size+1, 256))+[w-size]))
    _, y, x = max((int(mask[y:y+size, x:x+size].sum()), y, x) for y in ys for x in xs)
    return packed[:, y:y+size, x:x+size].copy(), metadata, (y, x), counts


def comparison(path, images, ring):
    # Pick a crop around the strongest original boundary chroma change, not texture.
    delta = np.linalg.norm(lab(images[1])[..., 1:]-lab(images[0])[..., 1:], axis=-1)
    y, x = np.unravel_index(np.argmax(np.where(ring, delta, -1)), ring.shape)
    size = min(160, *ring.shape)
    y = int(np.clip(y-size//2, 0, ring.shape[0]-size))
    x = int(np.clip(x-size//2, 0, ring.shape[1]-size))
    sheet = Image.new('RGB', (3*size*3, size*3+28), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (image, title) in enumerate(zip(images, ('Input / reference', 'r4_plain: off', 'r4_plain: protect'))):
        crop = Image.fromarray(np.rint(image[y:y+size, x:x+size]*255).astype(np.uint8))
        sheet.paste(crop.resize((size*3, size*3), Image.Resampling.NEAREST), (i*size*3, 28))
        draw.text((i*size*3+8, 8), title, fill='white')
    sheet.save(path)
    return [y, x, size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/highlights'))
    parser.add_argument('--private-root', type=Path, help='Optional main checkout; sources read only')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    model = Denoiser.load(args.checkpoint, device='cpu')
    cases = [('synthetic', None, .6), ('synthetic_noisier', None, .6)]
    cases += [(p.stem, p, 1.) for p in sorted(Path('examples/raw').glob('*.dng'))]
    if args.private_root:
        root = args.private_root/'data/real_test'
        cases += [('dusk_lake', root/'iso6400plus_2b3c19c0674a.3fr', 16.),
                  ('fireworks', root/'iso3200_3b337968c7c7.3fr', 1.)]
    results = []
    for name, source, exposure in cases:
        if source is not None and not source.exists():
            results.append(dict(name=name, skipped=f'Missing optional input: {source}'))
            continue
        if source is None:
            clean, noisy = synthetic(.002 if name == 'synthetic_noisier' else .0005)
            original, reference, origin = noisy, clean, (0, 0)
            counts = (original >= .98).sum(axis=(1, 2)).tolist()
        else:
            original, meta, origin, counts = crop_raw(source)
            reference = original
        print(f'{name}: inference on {original.shape}, packed origin {origin}', flush=True)
        off = model.denoise_packed(original, highlights='off')
        # Reuse the exact network output; this is the shared wrapper's final step.
        corrected, defects = correct_defects(torch.from_numpy(original)[None], k=8, abs_floor=.01, adjacency_ratio=.1)
        on = protect_highlights(torch.from_numpy(off)[None], corrected)[0].numpy()
        # Exclude known defects when measuring the actual protection footprint.
        core, ring = regions(corrected[0].numpy())
        footprint = core | ring
        if source is None:
            images = [cell_render(a) for a in (reference, off, on)]
        else:
            images = [real_render(a, meta, args.output/f'{name}_{variant}.dng', exposure)
                      for variant, a in zip(('input', 'off', 'protect'), (reference, off, on))]
            ring = ring.repeat(2, axis=0).repeat(2, axis=1)
        target = lab(images[0])
        row = dict(name=name, source=str(source) if source else 'generated Gaussian light; RNG seed 7',
                   source_sha256=sha256(source) if source else None, packed_origin_yx=list(origin),
                   packed_shape=list(original.shape), full_source_near_white_counts=counts,
                   crop_near_white_cells=int(core.sum()), ring_samples=int(ring.sum()), exposure=exposure,
                   prepass_defect_count=int(defects.sum()),
                   synthetic_noise_sigma=(.002 if name == 'synthetic_noisier' else .0005) if source is None else None,
                   reference='clean clipped synthetic' if source is None else 'noisy input (unpaired)',
                   protect_off_bit_identical=bool(np.array_equal(on, off)), metrics={})
        for variant, image, packed in zip(('off', 'protect'), images[1:], (off, on)):
            delta = lab(image)-target
            row['metrics'][variant] = dict(
                mean_delta_ab=float(np.linalg.norm(delta[..., 1:], axis=-1)[ring].mean()) if ring.any() else None,
                mean_delta_e76=float(np.linalg.norm(delta, axis=-1)[ring].mean()) if ring.any() else None,
                ring_signed_ab=delta[..., 1:][ring].mean(axis=0).tolist() if ring.any() else None,
                chroma_bias_norm=float(np.linalg.norm(delta[..., 1:][ring].mean(axis=0))) if ring.any() else None,
                above_white_in_footprint=int((packed[:, footprint] > 1).sum()))
        comparison_path = args.output/f'{name}_before_after.png'
        if ring.any():
            row['render_crop_yx_size'] = comparison(comparison_path, images, ring)
        else:
            comparison_path.unlink(missing_ok=True)
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'metrics.json').write_text(json.dumps(dict(
            checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
            settings=dict(backend='torch', device='cpu', weights='model', prepass=True,
                          strength=1., tile_size=512, overlap=60, feather_power=2.2), results=results), indent=2)+'\n')

    lines = ['# Clipped-highlight diagnostic', '',
             'Frozen r4_plain; no training or third-party outputs. See `metrics.json` for hashes, '
             'settings, source paths, crop coordinates and signed a*/b* changes.', '',
             'The ring is the exterior Chebyshev-distance 1–4 packed-pixel band around any '
             'pre-pass-corrected input photosite >= .98. Synthetic: known clipped clean target, Gaussian noise '
             'sigma .0005 (and a noisier .002 case) before clipping, Bayer-cell RGB, WB (1.7, 1, 2), exposure .6, sRGB. '
             'Real: canonical cropped DNGs with source matrix/WB, identical LibRaw linear '
             'sRGB demosaicing with automatic brightening/maximum adjustment disabled, then '
             'fixed exposure and sRGB transfer. No orientation rotation. Real-photo distances '
             'are changes relative to noisy input, **not ground-truth colour accuracy**.', '',
             '| Case | Near-white cells in crop | Ring samples | Mean chroma Δab off → protect | '
             'Mean ΔE76 off → protect | Chroma bias norm off → protect |', '|---|---:|---:|---:|---:|---:|']
    for row in results:
        if 'skipped' in row:
            lines.append(f"| {row['name']} | skipped: {row['skipped']} | | | |")
            continue
        a, b = row['metrics']['off'], row['metrics']['protect']
        def pair(key):
            return 'N/A (no near-clips)' if a[key] is None else f'{a[key]:.4f} → {b[key]:.4f}'
        lines.append(f"| {row['name']} | {row['crop_near_white_cells']} | {row['ring_samples']} | "
                     f"{pair('mean_delta_ab')} | {pair('mean_delta_e76')} | {pair('chroma_bias_norm')} |")
    lines += ['', '## Interpretation and limits', '',
              'Protection preserves input in the clipping core and its one-packed-pixel border, '
              'then feathers with input weights .75, .5, .25 at distances 2, 3, 4. '
              'All four CFA planes share these weights. Only this footprint is capped at white; '
              'input-unclipped sites stay below white. Strength zero is an exact bypass. '
              'Unclipped images and pixels beyond the footprint retain legacy processing exactly.', '',
              'This is conservative colour preservation, not highlight reconstruction. '
              'It retains noise. The pre-pass-corrected input seeds the mask and supplies '
              'preserved values, so confirmed defects are not restored. With --no-prepass '
              'saturated defects can be preserved. '
              'The fixed radius cannot guarantee removal of wider neural halos. Real inputs '
              'have no clean references; lower Δab to input is expected from this design and '
              'does not independently prove better colour or detail. Private full scenes use '
              'one clipping-dense 1024-sensor-pixel crop rather than full-frame inference. '
              'Dusk-lake display clipping after +4 EV is distinct from sensor clipping; '
              'the protection only targets the latter. In sparse scenes, small local '
              'pre-pass checks exclude isolated hot pixels from crop selection. Dusk-lake '
              'has only six raw near-white sites, of which five are excluded by these '
              'local checks; the chosen crop contains the remaining green-clipped light. '
              'No blind acceptance review was performed.', '',
              'The noisier synthetic explicitly measures the preservation/noise tradeoff: '
              'systematic chroma bias (norm of mean signed a*/b* residual) can drop even '
              'when mean absolute colour error increases. Do not interpret that as an '
              'across-the-board colour-error improvement. The Gaussian reproduces '
              'overshoot and small colour changes, not a conspicuous visible cyan fringe. '
              'Fireworks retains some input/demosaicing fringe colour; this fix only '
              'limits changes introduced by denoising.', '',
              'PNG sheets show reference/input, legacy off, protect at 3× nearest-neighbour, '
              'with the crop centered on the largest legacy boundary chroma change. '
              'The synthetic is rendered at one RGB sample per Bayer cell; real crops are '
              'full demosaiced sensor resolution. Existing example JPEGs were not regenerated.', '',
              '## Reproduce', '', '```bash',
              'PYTHONPATH=$PWD/src python '
              'scripts/evaluate_highlights.py \\', f'  --checkpoint {args.checkpoint} \\',
              f'  --output {args.output}'+(' \\' if args.private_root else '')]
    if args.private_root:
        lines.append(f'  --private-root {args.private_root}')
    lines += ['```', '']
    (args.output/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
