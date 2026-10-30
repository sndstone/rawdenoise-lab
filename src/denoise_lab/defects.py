"""Deterministic photosite defects and conservative same-colour correction.

The detector is a single pass, not an iterative despiker. Heavy-tailed sensor
noise can still trigger it; report its masks on held-out real frames before use.
"""
import math
import numpy as np
import torch
from .tiling import reflect_indices


def inject_defects(mosaic, rng, density_range=(1e-5, 3e-4),
                   kind_probs=None, pair_probability=.05, hot_range=(.15, 1.),
                   warm_sigma_range=(8., 80.), sigma=None):
    """Return a copy and an injected-site mask; never mutate the input.

    Density counts primary events; pairs add one orthogonally adjacent site.
    Repeated locations are allowed by the Poisson process. A pair shares its
    event's kind, with independently drawn amplitudes. Sigma may be scalar or
    broadcastable to the mosaic. The fallback is first-difference MAD / sqrt(2).
    """
    a = np.asarray(mosaic)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.floating) or not a.size:
        raise ValueError('Expected a nonempty HxW float mosaic')
    lo, hi = density_range
    probs = kind_probs if kind_probs is not None else dict(hot=.45, warm=.3, dead=.25)
    if not 0 < lo <= hi or not 0 <= pair_probability <= 1:
        raise ValueError('Invalid density or pair probability')
    if set(probs) - {'hot', 'warm', 'dead'} or not probs:
        raise ValueError('Unknown or empty defect kinds')
    p = np.array(list(probs.values()), dtype=float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Kind probabilities must sum to one')
    for bounds in (hot_range, warm_sigma_range):
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError('Invalid amplitude range')
    if sigma is None:
        # Differences two sensor pixels apart preserve Bayer colour.
        d = np.concatenate([np.diff(a[sy::2, sx::2], axis=1).ravel()
                            for sy in range(2) for sx in range(2)])
        sigma = np.median(np.abs(d-np.median(d)))/(.6744897501960817*math.sqrt(2)) if d.size else 0.
    scales = np.broadcast_to(np.asarray(sigma), a.shape)
    if np.any(scales < 0) or not np.isfinite(scales).all():
        raise ValueError('Sigma must be finite and nonnegative')
    out, mask = a.copy(), np.zeros(a.shape, dtype=bool)
    count = rng.poisson(a.size*np.exp(rng.uniform(np.log(lo), np.log(hi))))
    kinds = list(probs)
    for _ in range(count):
        y, x = int(rng.integers(a.shape[0])), int(rng.integers(a.shape[1]))
        kind = kinds[int(rng.choice(len(kinds), p=p))]
        sites = [(y, x)]
        if rng.random() < pair_probability:
            adjacent = [(v, u) for v, u in ((y-1,x),(y+1,x),(y,x-1),(y,x+1))
                        if 0 <= v < a.shape[0] and 0 <= u < a.shape[1]]
            if adjacent:
                sites.append(adjacent[int(rng.integers(len(adjacent)))])
        for y, x in sites:
            if kind == 'hot': out[y, x] = rng.uniform(*hot_range)
            elif kind == 'dead': out[y, x] = rng.uniform(-.002, .002)
            else: out[y, x] += rng.uniform(*warm_sigma_range)*scales[y, x]
            mask[y, x] = True
    return out, mask


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
    rows = reflect_indices(h, 1, 1, plane.device)[start:stop+2]
    columns = reflect_indices(w, 1, 1, plane.device)
    apron = plane.index_select(0, rows).index_select(1, columns)
    height = stop-start
    values = torch.stack([apron[dy:dy+height, dx:dx+w]
                          for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)])
    values = values.sort(dim=0).values
    return values[0], (values[3]+values[4])*.5, values[-1]


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
    """
    output, mask = packed.clone(), torch.zeros_like(packed, dtype=torch.bool)
    h, w = packed.shape[-2:]
    for b in range(packed.shape[0]):
        models = [_sigma_model(packed[b, c]) for c in range(4)]
        excess = torch.empty((4, h, w), device=packed.device, dtype=packed.dtype)
        stats = {}
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            for c, (edges, scales) in enumerate(models):
                nmin, med, nmax = _chunk_neighbours(packed[b, c], start, stop)
                excess[c, start:stop] = packed[b, c, start:stop]-med
                stats[(c, start)] = (nmin, med, nmax, scales[torch.bucketize(med.contiguous(), edges)])
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            flags_all, medians = [], []
            for c in range(4):
                nmin, med, nmax, sigma = stats.pop((c, start))
                margin = (k*sigma).clamp_min(abs_floor)
                value, e = packed[b, c, start:stop], excess[c, start:stop]
                adj = torch.stack([_shift(excess[a], dy, dx)[start:stop] for a, dy, dx in _ADJACENT[c]])
                hot = (value-nmax > margin) & (adj.amax(0) < ratio*e)
                dead = (nmin-value > margin) & (adj.amin(0) > ratio*e)
                flags = hot | dead
                if static_mask is not None: flags |= static_mask[b, c, start:stop]
                flags_all.append(flags); medians.append(med)
            flags = torch.stack(flags_all)
            mask[b, :, start:stop] = flags
            output[b, :, start:stop] = torch.where(flags, torch.stack(medians), packed[b, :, start:stop])
    return output, mask
