"""Torch defect correction for training and the .pt inference backend."""
import math

import torch

from .tiling import reflect_indices


def _neighbours(plane, y, x):
    """Sorted eight neighbours at explicit reflected index grids."""
    h, w = plane.shape
    def reflected(i, n):
        if n == 1: return torch.zeros_like(i)
        i = i.remainder(2*(n-1))
        return torch.minimum(i, 2*(n-1)-i)
    values = torch.stack([plane[reflected(y+dy, h)[:, None], reflected(x+dx, w)[None, :]]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    values = values.sort(dim=0).values
    return values[0], (values[3]+values[4])*.5, values[-1]



def _chunk_neighbours(plane, start, stop):
    """One reflected row halo, then eight shifted views of the same apron."""
    h, w = plane.shape
    rows = torch.arange(start-1, stop+1, device=plane.device)
    if h == 1:
        rows.zero_()
    else:
        rows = rows.remainder(2*(h-1))
        rows = torch.minimum(rows, 2*(h-1)-rows)
    columns = reflect_indices(w, 1, 1, plane.device)
    apron = plane.index_select(0, rows).index_select(1, columns)
    height = stop-start
    values = torch.stack([apron[dy:dy+height, dx:dx+w]
                          for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)])
    values = values.sort(dim=0).values
    # Extrema must not keep the eight-plane sort workspace alive.
    return values[0].clone(), (values[3]+values[4])*.5, values[-1].clone()


def _sigma_model(plane):
    # At most ~65k deterministic samples, independent of chunk_rows. Equal-width
    # intensity bins; sparse bins use the global MAD. Cumulative max enforces a
    # conservative nondecreasing shot/read noise envelope.
    h, w = plane.shape
    stride = max(1, math.ceil(math.sqrt(h*w/65536)))
    y = torch.arange(0, h, stride, device=plane.device)
    x = torch.arange(0, w, stride, device=plane.device)
    _, med, _ = _neighbours(plane, y, x)
    residual = (plane[y[:, None], x[None, :]]-med).flatten()
    med = med.flatten()
    def mad(v):
        return (v-v.median()).abs().median()/.6744897501960817
    fallback = mad(residual)
    edges = torch.linspace(0, 1, 9, device=plane.device, dtype=plane.dtype)
    edges = med.min()+(med.max()-med.min())*edges
    bins = torch.bucketize(med.contiguous(), edges[1:-1])
    scales = []
    for i in range(8):
        r = residual[bins == i]
        scales.append(mad(r) if r.numel() >= 32 else fallback)
    scales = torch.stack(scales).cummax(0).values
    return edges[1:-1].contiguous(), scales


# Mosaic-adjacent photosites of each packed plane (R, G1, B, G2), as (plane, dy, dx)
# offsets in packed coordinates for an RGGB quad: R(0,0) G1(0,1) B(1,1) G2(1,0).
_ADJACENT = {0: ((1, 0, 0), (1, 0, -1), (3, 0, 0), (3, -1, 0)),
             1: ((0, 0, 0), (0, 0, 1), (2, 0, 0), (2, -1, 0)),
             2: ((3, 0, 0), (3, 0, 1), (1, 0, 0), (1, 1, 0)),
             3: ((2, 0, 0), (2, 0, -1), (0, 0, 0), (0, 1, 0))}


def _shift(plane, dy, dx):
    """plane[y+dy, x+dx] with edge replication."""
    h, w = plane.shape
    ys = (torch.arange(h, device=plane.device)+dy).clamp(0, h-1)
    xs = (torch.arange(w, device=plane.device)+dx).clamp(0, w-1)
    return plane.index_select(0, ys).index_select(1, xs)


@torch.inference_mode()
def correct_defects(packed, k=8., abs_floor=.01, cross_channel_guard=True,
                    static_mask=None, chunk_rows=512, adjacency_ratio=None):
    """Correct isolated extrema in Bx4xHxW R/G1/B/G2, reflecting at edges.

    Noise is estimated per batch item and plane from residual MAD in eight
    neighbour-median intensity bins. Statistics always use the original input.
    Sorting is per plane and row chunk, never a full-frame unfold. Input dtype,
    device and values at unflagged sites are preserved. No RNG is used.
    """
    if packed.ndim != 4 or packed.shape[1] != 4 or min(packed.shape) < 1 or not packed.is_floating_point():
        raise ValueError('Expected nonempty floating Bx4xHxW')
    if chunk_rows < 1 or k < 0 or abs_floor < 0:
        raise ValueError('Invalid detector parameters')
    if static_mask is not None and (static_mask.shape != packed.shape or static_mask.dtype != torch.bool or static_mask.device != packed.device):
        raise ValueError('static_mask must be bool with the same shape and device')
    if adjacency_ratio is not None:
        return _correct_defects_adjacency(packed, k, abs_floor, static_mask, chunk_rows, float(adjacency_ratio))
    output, mask = packed.clone(), torch.zeros_like(packed, dtype=torch.bool)
    h, w = packed.shape[-2:]
    for b in range(packed.shape[0]):
        models = [_sigma_model(packed[b, c]) for c in range(4)]
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            hot, dead, weak, medians = [], [], [], []
            for c, (edges, scales) in enumerate(models):
                nmin, med, nmax = _chunk_neighbours(packed[b, c], start, stop)
                sigma = scales[torch.bucketize(med.contiguous(), edges)]
                margin = (k*sigma).clamp_min(abs_floor)
                value = packed[b, c, start:stop]
                hot.append(value-nmax > margin)
                dead.append(nmin-value > margin)
                weak.append(value-nmax > (k*.5*sigma).clamp_min(abs_floor*.5))
                medians.append(med)
            hot, dead, weak = map(torch.stack, (hot, dead, weak))
            if cross_channel_guard:
                hot &= (weak.sum(0, keepdim=True)-weak.to(torch.int64)) < 2
            flags = hot | dead
            if static_mask is not None: flags |= static_mask[b, :, start:stop]
            mask[b, :, start:stop] = flags
            output[b, :, start:stop] = torch.where(flags, torch.stack(medians), packed[b, :, start:stop])
    return output, mask


def _correct_defects_adjacency(packed, k, abs_floor, static_mask, chunk_rows, ratio):
    """Single-photosite test: a defect deviates from its same-colour neighbours
    while its four mosaic-adjacent (other-colour) photosites show at most
    `ratio` of that deviation relative to their own same-colour neighbours.
    Real detail, glints and point lights spread over adjacent photosites
    through the lens PSF; stuck photosites do not.
    Retain only one band's statistics and a one-row residual adjacency halo.
    """
    output, mask = packed.clone(), torch.zeros_like(packed, dtype=torch.bool)
    h, w = packed.shape[-2:]
    columns = torch.arange(w, device=packed.device)
    for b in range(packed.shape[0]):
        models = [_sigma_model(packed[b, c]) for c in range(4)]
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            top, bottom = max(0, start-1), min(h, stop+1)
            inner = slice(start-top, stop-top)
            excess = torch.empty((4, bottom-top, w), device=packed.device, dtype=packed.dtype)
            stats = []
            for c in range(4):
                nmin, med, nmax = _chunk_neighbours(packed[b, c], top, bottom)
                excess[c] = packed[b, c, top:bottom]-med
                stats.append((nmin, med, nmax))
            flags_all, medians = [], []
            rows = torch.arange(start, stop, device=packed.device)
            for c in range(4):
                nmin, med, nmax = (stat[inner] for stat in stats[c])
                edges, scales = models[c]
                sigma = scales[torch.bucketize(med.contiguous(), edges)]
                margin = (k*sigma).clamp_min(abs_floor)
                value, e = packed[b, c, start:stop], excess[c, inner]
                adj = torch.stack([excess[a].index_select(0, (rows+dy).clamp(0, h-1)-top)
                                   .index_select(1, (columns+dx).clamp(0, w-1))
                                   for a, dy, dx in _ADJACENT[c]])
                hot = (value-nmax > margin) & (adj.amax(0) < ratio*e)
                dead = (nmin-value > margin) & (adj.amin(0) > ratio*e)
                flags = hot | dead
                if static_mask is not None: flags |= static_mask[b, c, start:stop]
                flags_all.append(flags); medians.append(med)
            flags = torch.stack(flags_all)
            mask[b, :, start:stop] = flags
            output[b, :, start:stop] = torch.where(flags, torch.stack(medians), packed[b, :, start:stop])
    return output, mask


@torch.inference_mode()
def correct_mapped_defects(packed, mask):
    """Replace ONLY mapped sites, excluding mapped neighbours from same-colour median.

    Use original neighbours simultaneously. Clusters without a good immediate
    neighbour are rejected; do not invent corrections from defective neighbours.
    """
    if packed.ndim != 4 or packed.shape[1] != 4 or not packed.is_floating_point() or min(packed.shape) < 1:
        raise ValueError('Expected nonempty floating Bx4xHxW')
    if mask.shape != packed.shape or mask.dtype != torch.bool or mask.device != packed.device:
        raise ValueError('Map mask must be bool with the same shape and device')
    output = packed.clone()
    for b in range(packed.shape[0]):
        for c in range(4):
            y, x = mask[b, c].nonzero(as_tuple=True)
            if not len(y):
                continue
            h, w = packed.shape[-2:]
            values = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    yy, xx = y+dy, x+dx
                    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
                    yy, xx = yy.clamp(0, h-1), xx.clamp(0, w-1)
                    valid &= ~mask[b, c, yy, xx]
                    values.append(torch.where(valid, packed[b, c, yy, xx], torch.nan))
            values = torch.stack(values).sort(dim=0).values
            count = (~values.isnan()).sum(0)
            if (count == 0).any():
                raise ValueError('Mapped cluster has no valid same-colour neighbours')
            lo, hi = (count-1)//2, count//2
            median = (values.gather(0, lo[None])[0]+values.gather(0, hi[None])[0])*.5
            output[b, c, y, x] = median
    return output, mask.clone()


