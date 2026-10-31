"""Overlapping inference with site-aware Bayer or ordinary RGB reflection."""
from itertools import islice

import torch


def reflect_indices(length, before, after, device):
    # Unlike F.pad, supports aprons larger than the image itself.
    idx = torch.arange(-before, length+after, device=device)
    if length == 1:
        return torch.zeros_like(idx)
    period = 2*(length-1)
    idx = idx.remainder(period)
    return torch.minimum(idx, period-idx)


def reflected_pad(x, left, right, top, bottom):
    return x.index_select(-2, reflect_indices(x.shape[-2], top, bottom, x.device)).index_select(
        -1, reflect_indices(x.shape[-1], left, right, x.device))


def reflected_bayer_pad(image, left, right, top, bottom):
    """Mirror the actual sensor mosaic, then repack R/G1/B/G2.

    Reflecting each packed plane identically moves the odd CFA sites by one
    sensor pair at the boundary. Site-aware indices are equivalent to unpacking,
    reflecting at twice the requested extent, and packing, without a full mosaic.
    """
    if image.ndim != 4 or image.shape[1] != 4 or min(image.shape[-2:]) < 1:
        raise ValueError("Bayer reflection requires nonempty Bx4xHxW R/G1/B/G2")
    if any(not isinstance(n, int) or n < 0 for n in (left, right, top, bottom)):
        raise ValueError("Bayer padding extents must be nonnegative integers")

    def indices(length, before, after, site):
        positions = 2*torch.arange(-before, length+after, device=image.device)+site
        period = 2*(2*length-1)
        reflected = positions.remainder(period)
        reflected = torch.minimum(reflected, period-reflected)
        return (reflected-site)//2

    h, w = image.shape[-2:]
    return torch.stack([image[:, c].index_select(-2, indices(h, top, bottom, y)).index_select(
        -1, indices(w, left, right, x)) for c, (y, x) in enumerate(((0,0),(0,1),(1,1),(1,0)))], 1)


def feather(tile_size, overlap, power, device):
    axis = torch.ones(tile_size, device=device, dtype=torch.float32)
    if overlap:
        ramp = (torch.arange(overlap, device=device, dtype=torch.float32)+1)/(overlap+1)
        axis[:overlap], axis[-overlap:] = ramp, ramp.flip(0)
    return (axis[:, None]*axis[None, :]).pow(power)[None, None]


@torch.inference_mode()
def tiled_inference(model, image, tile_size=512, overlap=58, feather_power=1., alpha=1., noise_map=None,
                    return_weights=False, bayer_reflection=None, tile_batch_size=1):
    """image is BxCxHxW (C=3 RGB or C=4 packed Bayer); overlap is in pixels.

    Tile starts are separated by tile_size-overlap. The original image lies
    inside an overlap-wide reflected apron on all four sides. Unconditioned
    four-plane inputs reflect the sensor mosaic with its CFA-site offsets;
    RGB uses ordinary spatial reflection. Set bayer_reflection=False only to
    reproduce historical per-plane padding. Conditioned research models retain
    historical padding until a spatial contract for their noise map is defined.
    Weighted sums
    are divided by their accumulated weights, including irregular edges.
    tile_batch_size optionally groups tiles along the model batch dimension;
    accumulation retains raster order. The backend/model must support that batch
    size and be checked for numerical parity; the default calls one tile at a time.
    """
    if image.ndim != 4 or image.shape[1] not in (3, 4) or min(image.shape) < 1:
        raise ValueError("Expected nonempty Bx3xHxW RGB or Bx4xHxW packed Bayer")
    if not 0 <= overlap < tile_size or not 0 <= alpha <= 1 or feather_power <= 0:
        raise ValueError("Invalid overlap, alpha or feather exponent")
    if isinstance(tile_batch_size, bool) or not isinstance(tile_batch_size, int) or tile_batch_size < 1:
        raise ValueError("tile_batch_size must be a positive integer")
    if bayer_reflection is None:
        bayer_reflection = image.shape[1] == 4 and noise_map is None
    if bayer_reflection and (image.shape[1] != 4 or noise_map is not None):
        raise ValueError("Site-aware Bayer reflection requires unconditioned R/G1/B/G2 input")
    h, w = image.shape[-2:]
    step, apron = tile_size-overlap, overlap
    def extent(n):
        return tile_size+max(0, (n+2*apron-tile_size+step-1)//step)*step
    ph, pw = extent(h), extent(w)
    pad = reflected_bayer_pad if bayer_reflection else reflected_pad
    padded = pad(image, apron, pw-w-apron, apron, ph-h-apron)
    conditioned = None if noise_map is None else reflected_pad(noise_map, apron, pw-w-apron, apron, ph-h-apron)
    weight = feather(tile_size, overlap, feather_power, image.device)
    total = torch.zeros_like(padded, dtype=torch.float32)
    weights = torch.zeros((1, 1, ph, pw), device=image.device)
    positions = ((y, x) for y in range(0, ph-tile_size+1, step) for x in range(0, pw-tile_size+1, step))
    while locations := list(islice(positions, tile_batch_size)):
        tiles = [padded[..., y:y+tile_size, x:x+tile_size] for y, x in locations]
        tile = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)
        if conditioned is None:
            predictions = model(tile)
        else:
            conditions = [conditioned[..., y:y+tile_size, x:x+tile_size] for y, x in locations]
            if len(conditions) > 1:
                # Preserve a shared single-image map's broadcast over B images.
                conditions = [c.expand(image.shape[0], -1, -1, -1) if c.shape[0] == 1 else c
                              for c in conditions]
            condition = conditions[0] if len(conditions) == 1 else torch.cat(conditions, dim=0)
            predictions = model(tile, condition)
        if predictions.shape != tile.shape:
            raise ValueError(f"Model output shape {tuple(predictions.shape)} does not match input {tuple(tile.shape)}")
        for (y, x), pred in zip(locations, predictions.split(image.shape[0], dim=0)):
            total[..., y:y+tile_size, x:x+tile_size] += pred.float()*weight
            weights[..., y:y+tile_size, x:x+tile_size] += weight
    # Only normalize the visible image. Dividing the reflected apron and the
    # rounded-up tile extent allocates a full padded result that is discarded.
    # The weighted sums and per-pixel arithmetic are otherwise unchanged.
    visible = (..., slice(apron, apron+h), slice(apron, apron+w))
    merged = total[visible]/weights[visible]
    out = alpha*merged+(1-alpha)*image
    if return_weights:
        # Effective normalized partition, useful for auditing coverage.
        return out, weights[visible]
    return out
