"""Frozen pre-streaming references and bounded-workspace regression checks."""
import math
import tracemalloc

import numpy as np
import pytest

from denoise_lab import np_pipeline


_LEGACY_ADJACENT = {0: ((1, 0, 0), (1, 0, -1), (3, 0, 0), (3, -1, 0)),
                    1: ((0, 0, 0), (0, 0, 1), (2, 0, 0), (2, -1, 0)),
                    2: ((3, 0, 0), (3, 0, 1), (1, 0, 0), (1, 1, 0)),
                    3: ((2, 0, 0), (2, 0, -1), (0, 0, 0), (0, 1, 0))}


def _legacy_numpy(packed, k=8., abs_floor=.01, chunk_rows=512, adjacency_ratio=.1):
    """Inference heuristic: same-colour extrema with the mosaic-adjacency guard.

    Statistics and medians always use the original input, never corrected sites.
    Float32 input is assumed; row chunks bound the eight-neighbour sort workspace.
    """
    if chunk_rows < 1 or k < 0 or abs_floor < 0:
        raise ValueError('Invalid detector parameters')
    output, mask = packed.copy(), np.zeros(packed.shape, bool)
    h, w = packed.shape[-2:]
    columns = np.arange(w)
    for b in range(packed.shape[0]):
        models = [_legacy_numpy_sigma_model(packed[b, c]) for c in range(4)]
        excess = np.empty((4, h, w), packed.dtype)
        stats = {}
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            for c, (edges, scales) in enumerate(models):
                nmin, med, nmax = _legacy_numpy_neighbours(packed[b, c], np.arange(start, stop), columns)
                excess[c, start:stop] = packed[b, c, start:stop]-med
                stats[c, start] = nmin, med, nmax, scales[np.searchsorted(edges, med, side='left')]
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            for c in range(4):
                nmin, med, nmax, sigma = stats.pop((c, start))
                margin = np.maximum(k*sigma, abs_floor)
                value, e = packed[b, c, start:stop], excess[c, start:stop]
                adj = np.stack([excess[a][np.clip(np.arange(start, stop)+dy, 0, h-1)[:, None],
                                                np.clip(columns+dx, 0, w-1)[None, :]]
                                for a, dy, dx in _LEGACY_ADJACENT[c]])
                hot = (value-nmax > margin) & (adj.max(axis=0) < adjacency_ratio*e)
                dead = (nmin-value > margin) & (adj.min(axis=0) > adjacency_ratio*e)
                flags = hot | dead
                mask[b, c, start:stop] = flags
                output[b, c, start:stop] = np.where(flags, med, value)
    return output, mask


def _legacy_torch_chunk_neighbours(plane, start, stop):
    """One reflected row halo, then eight shifted views of the same apron."""
    import torch
    from denoise_lab.tiling import reflect_indices

    h, w = plane.shape
    rows = reflect_indices(h, 1, 1, plane.device)[start:stop+2]
    columns = reflect_indices(w, 1, 1, plane.device)
    apron = plane.index_select(0, rows).index_select(1, columns)
    height = stop-start
    values = torch.stack([apron[dy:dy+height, dx:dx+w]
                          for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)])
    values = values.sort(dim=0).values
    return values[0], (values[3]+values[4])*.5, values[-1]


def _legacy_shift(plane, dy, dx):
    """plane[y+dy, x+dx] with edge replication."""
    import torch

    h, w = plane.shape
    ys = (torch.arange(h, device=plane.device)+dy).clamp(0, h-1)
    xs = (torch.arange(w, device=plane.device)+dx).clamp(0, w-1)
    return plane.index_select(0, ys).index_select(1, xs)


def _legacy_torch_adjacency(packed, k, abs_floor, static_mask, chunk_rows, ratio):
    """Single-photosite test: a defect deviates from its same-colour neighbours
    while its four mosaic-adjacent (other-colour) photosites show at most
    `ratio` of that deviation relative to their own same-colour neighbours.
    Real detail, glints and point lights spread over adjacent photosites
    through the lens PSF; stuck photosites do not.
    """
    import torch

    output, mask = packed.clone(), torch.zeros_like(packed, dtype=torch.bool)
    h, w = packed.shape[-2:]
    for b in range(packed.shape[0]):
        models = [_legacy_torch_sigma_model(packed[b, c]) for c in range(4)]
        excess = torch.empty((4, h, w), device=packed.device, dtype=packed.dtype)
        stats = {}
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            for c, (edges, scales) in enumerate(models):
                nmin, med, nmax = _legacy_torch_chunk_neighbours(packed[b, c], start, stop)
                excess[c, start:stop] = packed[b, c, start:stop]-med
                stats[(c, start)] = (nmin, med, nmax, scales[torch.bucketize(med.contiguous(), edges)])
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            flags_all, medians = [], []
            for c in range(4):
                nmin, med, nmax, sigma = stats.pop((c, start))
                margin = (k*sigma).clamp_min(abs_floor)
                value, e = packed[b, c, start:stop], excess[c, start:stop]
                adj = torch.stack([_legacy_shift(excess[a], dy, dx)[start:stop] for a, dy, dx in _LEGACY_ADJACENT[c]])
                hot = (value-nmax > margin) & (adj.amax(0) < ratio*e)
                dead = (nmin-value > margin) & (adj.amin(0) > ratio*e)
                flags = hot | dead
                if static_mask is not None: flags |= static_mask[b, c, start:stop]
                flags_all.append(flags); medians.append(med)
            flags = torch.stack(flags_all)
            mask[b, :, start:stop] = flags
            output[b, :, start:stop] = torch.where(flags, torch.stack(medians), packed[b, :, start:stop])
    return output, mask


def _legacy_torch(packed, k=8., abs_floor=.01, cross_channel_guard=True,
                    static_mask=None, chunk_rows=512, adjacency_ratio=None):
    """Correct isolated extrema in Bx4xHxW R/G1/B/G2, reflecting at edges.

    Noise is estimated per batch item and plane from residual MAD in eight
    neighbour-median intensity bins. Statistics always use the original input.
    Sorting is per plane and row chunk, never a full-frame unfold. Input dtype,
    device and values at unflagged sites are preserved. No RNG is used.
    """
    import torch

    if packed.ndim != 4 or packed.shape[1] != 4 or min(packed.shape) < 1 or not packed.is_floating_point():
        raise ValueError('Expected nonempty floating Bx4xHxW')
    if chunk_rows < 1 or k < 0 or abs_floor < 0:
        raise ValueError('Invalid detector parameters')
    if static_mask is not None and (static_mask.shape != packed.shape or static_mask.dtype != torch.bool or static_mask.device != packed.device):
        raise ValueError('static_mask must be bool with the same shape and device')
    if adjacency_ratio is not None:
        return _legacy_torch_adjacency(packed, k, abs_floor, static_mask, chunk_rows, float(adjacency_ratio))
    output, mask = packed.clone(), torch.zeros_like(packed, dtype=torch.bool)
    h, w = packed.shape[-2:]
    for b in range(packed.shape[0]):
        models = [_legacy_torch_sigma_model(packed[b, c]) for c in range(4)]
        for start in range(0, h, chunk_rows):
            stop = min(h, start+chunk_rows)
            hot, dead, weak, medians = [], [], [], []
            for c, (edges, scales) in enumerate(models):
                nmin, med, nmax = _legacy_torch_chunk_neighbours(packed[b, c], start, stop)
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


def _legacy_numpy_neighbours(plane, rows, columns):
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
    return values[0], (values[3]+values[4])*.5, values[-1]


def _legacy_numpy_sigma_model(plane):
    h, w = plane.shape
    stride = max(1, math.ceil(math.sqrt(h*w/65536)))
    rows, columns = np.arange(0, h, stride), np.arange(0, w, stride)
    _, med, _ = _legacy_numpy_neighbours(plane, rows, columns)
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


def _legacy_torch_neighbours(plane, y, x):
    """Sorted eight neighbours at explicit reflected index grids."""
    import torch

    h, w = plane.shape
    def reflected(i, n):
        if n == 1: return torch.zeros_like(i)
        i = i.remainder(2*(n-1))
        return torch.minimum(i, 2*(n-1)-i)
    values = torch.stack([plane[reflected(y+dy, h)[:, None], reflected(x+dx, w)[None, :]]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    values = values.sort(dim=0).values
    return values[0], (values[3]+values[4])*.5, values[-1]


def _legacy_torch_sigma_model(plane):
    # At most ~65k deterministic samples, independent of chunk_rows. Equal-width
    # intensity bins; sparse bins use the global MAD. Cumulative max enforces a
    # conservative nondecreasing shot/read noise envelope.
    import torch

    h, w = plane.shape
    stride = max(1, math.ceil(math.sqrt(h*w/65536)))
    y = torch.arange(0, h, stride, device=plane.device)
    x = torch.arange(0, w, stride, device=plane.device)
    _, med, _ = _legacy_torch_neighbours(plane, y, x)
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


_SHAPES = [(1, 4, 1, 1), (2, 4, 1, 9), (1, 4, 9, 1), (2, 4, 5, 7),
           (1, 4, 19, 23), (1, 4, 65, 33), (1, 4, 258, 7)]
_CHUNKS = (1, 2, 7, 16, 64, 512)


def _sample(shape):
    image = np.random.default_rng(12).normal(.2, .005, shape).astype(np.float32)
    h, w = shape[-2:]
    rows = {0, h-1, h//2}
    for chunk in _CHUNKS[:-1]:
        rows.update(y for y in (chunk-1, chunk, chunk+1) if y < h)
    for b in range(shape[0]):
        for c in range(4):
            for i, y in enumerate(sorted(rows)):
                image[b, c, y, (i*5+c*3+b) % w] = .95 if (i+c) % 2 else -.3
    if min(h, w) > 10:
        image[:, :, 6:9, 8:11] = .8
        image[0, 0, -2, -2] = -0.
    return image


@pytest.mark.parametrize('shape', _SHAPES)
@pytest.mark.parametrize('k,floor,ratio', [(8., .01, .1), (.5, .0001, 0.), (2., .02, .5)])
def test_numpy_matches_legacy_bits(shape, k, floor, ratio):
    image = _sample(shape)
    original = image.copy()
    for chunk in _CHUNKS:
        options = dict(k=k, abs_floor=floor, adjacency_ratio=ratio, chunk_rows=chunk)
        expected, expected_mask = _legacy_numpy(image, **options)
        actual, actual_mask = np_pipeline.correct_defects(image, **options)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(actual_mask, expected_mask)
        np.testing.assert_array_equal(image.view(np.uint32), original.view(np.uint32))


@pytest.mark.parametrize('shape', _SHAPES)
@pytest.mark.parametrize('mode', ['adjacency', 'guard', 'unguarded', 'static-adjacency', 'static-guard'])
def test_torch_matches_legacy_bits(shape, mode):
    torch = pytest.importorskip('torch')
    from denoise_lab import _torch_defects as defects

    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        image = torch.from_numpy(_sample(shape))
        original = image.clone()
        options = dict(k=3., abs_floor=.01, cross_channel_guard=mode != 'unguarded')
        if 'adjacency' in mode:
            options['adjacency_ratio'] = .1
        if mode.startswith('static'):
            static = torch.zeros_like(image, dtype=torch.bool)
            static[..., ::3, ::2] = True
            options['static_mask'] = static
        for chunk in _CHUNKS:
            expected, expected_mask = _legacy_torch(image, chunk_rows=chunk, **options)
            actual, actual_mask = defects.correct_defects(image, chunk_rows=chunk, **options)
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
            assert torch.equal(actual_mask, expected_mask)
            assert torch.equal(image.view(torch.int32), original.view(torch.int32))
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('height', [512, 1024])
def test_numpy_workspace_is_bounded_by_band(height):
    image = _sample((1, 4, height, 1024))
    chunk_rows = 64
    tracemalloc.start()
    try:
        output, mask = np_pipeline.correct_defects(image, chunk_rows=chunk_rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    workspace = peak-output.nbytes-mask.nbytes
    band_bytes = 4*chunk_rows*image.shape[-1]*image.itemsize
    assert workspace < 12*band_bytes, (workspace, band_bytes)
