"""Float32 NumPy inference for unconditioned R/G1/B/G2 runtime models.

Normalization and packing live in bayer.py and already use only NumPy. This
module mirrors the torch inference contract, including sensor-site reflection,
lower-median noise statistics, simultaneous defect replacement and raster-order
tile accumulation. No training or conditioned-model operations are provided.
"""
import math

import numpy as np

from .bayer import SITES
from .tile_geometry import tile_starts


# Other-colour photosites adjacent in the sensor mosaic, in packed coordinates.
_ADJACENT = {0: ((1, 0, 0), (1, 0, -1), (3, 0, 0), (3, -1, 0)),
             1: ((0, 0, 0), (0, 0, 1), (2, 0, 0), (2, -1, 0)),
             2: ((3, 0, 0), (3, 0, 1), (1, 0, 0), (1, 1, 0)),
             3: ((2, 0, 0), (2, 0, -1), (0, 0, 0), (0, 1, 0))}


def reflect_indices(length, before, after):
    """Ordinary plane reflection, including singleton and arbitrarily large halos."""
    idx = np.arange(-before, length+after)
    if length == 1:
        return np.zeros_like(idx)
    period = 2*(length-1)
    idx %= period
    return np.minimum(idx, period-idx)


def reflected_bayer_pad(image, left, right, top, bottom):
    """Reflect sensor coordinates, preserving each packed plane's CFA site."""
    if image.ndim != 4 or image.shape[1] != 4 or min(image.shape) < 1:
        raise ValueError('Bayer reflection requires nonempty Bx4xHxW R/G1/B/G2')
    if any(not isinstance(n, int) or n < 0 for n in (left, right, top, bottom)):
        raise ValueError('Bayer padding extents must be nonnegative integers')

    def indices(length, before, after, site):
        positions = 2*np.arange(-before, length+after)+site
        period = 2*(2*length-1)
        reflected = positions % period
        return (np.minimum(reflected, period-reflected)-site)//2

    h, w = image.shape[-2:]
    return np.stack([image[:, c].take(indices(h, top, bottom, y), axis=-2).take(
        indices(w, left, right, x), axis=-1) for c, (y, x) in enumerate(SITES)], axis=1)


def feather(tile_size, overlap, power=2.2, mode='linear'):
    if mode not in ('linear', 'cosine'):
        raise ValueError('feather_mode must be linear or cosine')
    axis = np.ones(tile_size, np.float32)
    if overlap:
        ramp = (np.arange(overlap, dtype=np.float32)+1)/(overlap+1)
        if mode == 'cosine':
            ramp = .5-.5*np.cos(np.float32(np.pi)*ramp)
        axis[:overlap] = ramp
        axis[-overlap:] = np.minimum(axis[-overlap:], ramp[::-1])
    return ((axis[:, None]*axis[None, :])**power)[None, None]


def tiled_inference(model, image, tile_size=512, overlap=60, feather_power=2.2,
                    grid_offset=0, grid_alignment=1, feather_mode='linear'):
    """One Bx4xHxW tile per call; accumulate float32 predictions in raster order."""
    if image.ndim != 4 or image.shape[1] != 4 or min(image.shape) < 1:
        raise ValueError('Expected nonempty Bx4xHxW packed Bayer')
    if feather_power <= 0:
        raise ValueError('Invalid feather exponent')
    h, w = image.shape[-2:]
    ys = tile_starts(h, tile_size, overlap, grid_offset, grid_alignment)
    xs = tile_starts(w, tile_size, overlap, grid_offset, grid_alignment)
    top, left = -ys[0], -xs[0]
    ph, pw = ys[-1]+top+tile_size, xs[-1]+left+tile_size
    effective_overlap = tile_size-((tile_size-overlap)//grid_alignment)*grid_alignment
    padded = reflected_bayer_pad(image, left, pw-w-left, top, ph-h-top)
    weight = feather(tile_size, effective_overlap, feather_power, feather_mode)
    total = np.zeros(padded.shape, np.float32)
    weights = np.zeros((1, 1, ph, pw), np.float32)
    for y0 in ys:
        for x0 in xs:
            y, x = y0+top, x0+left
            region = (..., slice(y, y+tile_size), slice(x, x+tile_size))
            tile = padded[region]
            prediction = np.asarray(model(tile), dtype=np.float32)
            if prediction.shape != tile.shape:
                raise ValueError(f'Model output shape {prediction.shape} does not match input {tile.shape}')
            total[region] += prediction*weight
            weights[region] += weight
    visible = (..., slice(top, top+h), slice(left, left+w))
    return total[visible]/weights[visible]


def _neighbours(plane, rows, columns):
    h, w = plane.shape

    def reflected(idx, length):
        if length == 1:
            return np.zeros_like(idx)
        period = 2*(length-1)
        idx = idx % period
        return np.minimum(idx, period-idx)

    values = np.stack([plane[reflected(rows+dy, h)[:, None], reflected(columns+dx, w)[None, :]]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    values.sort(axis=0)
    # Extrema must not keep the eight-plane sort workspace alive.
    return values[0].copy(), (values[3]+values[4])*.5, values[-1].copy()


def _sigma_model(plane):
    h, w = plane.shape
    stride = max(1, math.ceil(math.sqrt(h*w/65536)))
    rows, columns = np.arange(0, h, stride), np.arange(0, w, stride)
    _, med, _ = _neighbours(plane, rows, columns)
    residual = (plane[rows[:, None], columns[None, :]]-med).ravel()
    med = med.ravel()

    def lower_median(values):
        mid = (values.size-1)//2
        return np.partition(values, mid)[mid]

    def mad(values):
        # torch.median uses the lower middle element for even sample counts.
        return lower_median(np.abs(values-lower_median(values)))/np.float32(.6744897501960817)

    fallback = mad(residual)
    edges = med.min()+(med.max()-med.min())*np.linspace(0, 1, 9, dtype=plane.dtype)
    bins = np.searchsorted(edges[1:-1], med, side='left')
    scales = []
    for i in range(8):
        r = residual[bins == i]
        scales.append(mad(r) if r.size >= 32 else fallback)
    return edges[1:-1], np.maximum.accumulate(scales)


def correct_defects(packed, k=8., abs_floor=.01, chunk_rows=512, adjacency_ratio=.1):
    """Inference heuristic: same-colour extrema with the mosaic-adjacency guard.

    Statistics and medians always use the original input, never corrected sites.
    Float32 input is assumed. One row of residual halo supplies mosaic adjacency;
    only this band retains statistics, independent of the image height.
    """
    if chunk_rows < 1 or k < 0 or abs_floor < 0:
        raise ValueError('Invalid detector parameters')
    output, mask = packed.copy(), np.zeros(packed.shape, bool)
    h, w = packed.shape[-2:]
    columns = np.arange(w)
    for b in range(packed.shape[0]):
        models = [_sigma_model(packed[b, c]) for c in range(4)]
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            top, bottom = max(0, start-1), min(h, stop+1)
            inner = slice(start-top, stop-top)
            excess = np.empty((4, bottom-top, w), packed.dtype)
            stats = []
            for c in range(4):
                nmin, med, nmax = _neighbours(packed[b, c], np.arange(top, bottom), columns)
                excess[c] = packed[b, c, top:bottom]-med
                stats.append((nmin, med, nmax))
            for c in range(4):
                nmin, med, nmax = (stat[inner] for stat in stats[c])
                edges, scales = models[c]
                sigma = scales[np.searchsorted(edges, med, side='left')]
                margin = np.maximum(k*sigma, abs_floor)
                value, e = packed[b, c, start:stop], excess[c, inner]
                adj = np.stack([excess[a][(np.clip(np.arange(start, stop)+dy, 0, h-1)-top)[:, None],
                                                np.clip(columns+dx, 0, w-1)[None, :]]
                                for a, dy, dx in _ADJACENT[c]])
                hot = (value-nmax > margin) & (adj.max(axis=0) < adjacency_ratio*e)
                dead = (nmin-value > margin) & (adj.min(axis=0) > adjacency_ratio*e)
                flags = hot | dead
                mask[b, c, start:stop] = flags
                output[b, c, start:stop] = np.where(flags, med, value)
    return output, mask


def correct_mapped_defects(packed, mask):
    """Simultaneously replace mapped sites from unlisted, in-bounds neighbours."""
    if mask.shape != packed.shape or mask.dtype != np.bool_:
        raise ValueError('Map mask must be bool with the same shape')
    output = packed.copy()
    h, w = packed.shape[-2:]
    for b in range(packed.shape[0]):
        for c in range(4):
            y, x = np.nonzero(mask[b, c])
            if not len(y):
                continue
            values = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    yy, xx = y+dy, x+dx
                    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                    yy, xx = np.clip(yy, 0, h-1), np.clip(xx, 0, w-1)
                    valid &= ~mask[b, c, yy, xx]
                    values.append(np.where(valid, packed[b, c, yy, xx], np.nan))
            values = np.sort(values, axis=0)
            count = (~np.isnan(values)).sum(axis=0)
            if (count == 0).any():
                raise ValueError('Mapped cluster has no valid same-colour neighbours')
            cols = np.arange(len(y))
            output[b, c, y, x] = (values[(count-1)//2, cols]+values[count//2, cols])*.5
    return output, mask.copy()


def protect_highlights(output, original):
    """Shared Bayer-cell mask, four max-filter dilations and headroom-preserving cap."""
    mask = (original >= .98).any(axis=1, keepdims=True).astype(output.dtype)
    if not mask.any():
        return output
    weight = np.zeros_like(mask)
    h, w = mask.shape[-2:]
    for _ in range(4):
        padded = np.pad(mask, ((0, 0), (0, 0), (1, 1), (1, 1)))
        mask = np.zeros_like(mask)
        for y in range(3):
            for x in range(3):
                np.maximum(mask, padded[..., y:y+h, x:x+w], out=mask)
        weight += .25*mask
    blended = weight*original+(1-weight)*output
    blended = np.where(weight == 1, original, blended)
    ceiling = np.where(original < 1, np.float32(1.-2**-24), original)
    return np.where(weight > 0, np.minimum(blended, ceiling), output)


def chroma_blend(output, original, strength, chroma_strength, gains):
    """Blend Bx4xHxW Bayer cells with separate luma/chroma strength (NumPy).

    Matches the torch path: WB-scaled opponent Y=(R+2G+B)/4, Cb=B-Y, Cr=R-Y;
    G1-G2 follows luma strength. Equal strengths use the legacy linear blend.
    """
    if chroma_strength == strength:
        return strength*output+(1-strength)*original
    wb = np.asarray(gains, np.float32)[[0, 1, 2, 1]][None, :, None, None]
    delta = (output-original)*wb
    r, g1, b, g2 = (delta[:, i] for i in range(4))
    y = (r+(g1+g2)*.5*2+b)*.25
    cb, cr = b-y, r-y
    yy, cb, cr = strength*y, chroma_strength*cb, chroma_strength*cr
    g = yy-(cb+cr)*.5
    difference = strength*(g1-g2)*.5
    correction = np.stack((yy+cr, g+difference, yy+cb, g-difference), axis=1)
    return original+correction/wb


def _tiled_passes(model, image, *, tile_size, overlap, offsets, grid_alignment, feather_mode):
    output = None
    for offset in offsets:
        prediction = tiled_inference(model, image, tile_size=tile_size, overlap=overlap,
                                     grid_offset=offset, grid_alignment=grid_alignment,
                                     feather_mode=feather_mode)
        output = prediction if output is None else output+prediction
    if len(offsets) > 1:
        output /= len(offsets)
    return output


def denoise_packed(model, packed, *, mode, defect_mask, overlap, strength, tile_size,
                   highlights, offsets, grid_alignment, feather_mode, chroma_strength=None,
                   gains=(1., 1., 1.), tta_transpose=False):
    """Run a validated Denoiser request with NumPy arrays throughout."""
    original = packed[None].copy()
    image = original
    if mode in {'map', 'map+heuristic'}:
        image = correct_mapped_defects(image, defect_mask[None])[0]
    if mode in {'heuristic', 'map+heuristic'}:
        image = correct_defects(image)[0]
    output = _tiled_passes(model, image, tile_size=tile_size, overlap=overlap,
                           offsets=offsets, grid_alignment=grid_alignment, feather_mode=feather_mode)
    if tta_transpose:
        transposed = np.ascontiguousarray(image[:, [0, 3, 2, 1]].swapaxes(-2, -1))
        out_t = _tiled_passes(model, transposed, tile_size=tile_size, overlap=overlap,
                              offsets=offsets, grid_alignment=grid_alignment, feather_mode=feather_mode)
        output = .5*(output + np.ascontiguousarray(out_t[:, [0, 3, 2, 1]].swapaxes(-2, -1)))
    output = chroma_blend(output, original, strength,
                          strength if chroma_strength is None else chroma_strength, gains)
    if not np.isfinite(output).all():
        raise ValueError('Model produced nonfinite output')
    if highlights == 'protect':
        output = protect_highlights(output, image)
    return output[0]
