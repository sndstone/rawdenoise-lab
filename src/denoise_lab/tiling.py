"""Overlapping packed-plane inference with a true reflected sensor apron."""
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


def feather(tile_size, overlap, power, device):
    axis = torch.ones(tile_size, device=device, dtype=torch.float32)
    if overlap:
        ramp = (torch.arange(overlap, device=device, dtype=torch.float32)+1)/(overlap+1)
        axis[:overlap], axis[-overlap:] = ramp, ramp.flip(0)
    return (axis[:, None]*axis[None, :]).pow(power)[None, None]


@torch.inference_mode()
def tiled_inference(model, image, tile_size=512, overlap=58, feather_power=1., alpha=1., noise_map=None,
                    return_weights=False):
    """image is Bx4xHxW; overlap is shared tile width in packed pixels.

    Tile starts are separated by tile_size-overlap. The original image lies
    inside an overlap-wide reflected apron on all four sides. Weighted sums
    are divided by their accumulated weights, including irregular edges.
    """
    if image.ndim != 4 or image.shape[1] != 4:
        raise ValueError("Expected Bx4xHxW")
    if not 0 <= overlap < tile_size or not 0 <= alpha <= 1 or feather_power <= 0:
        raise ValueError("Invalid overlap, alpha or feather exponent")
    h, w = image.shape[-2:]
    step, apron = tile_size-overlap, overlap
    def extent(n):
        return tile_size+max(0, (n+2*apron-tile_size+step-1)//step)*step
    ph, pw = extent(h), extent(w)
    padded = reflected_pad(image, apron, pw-w-apron, apron, ph-h-apron)
    conditioned = None if noise_map is None else reflected_pad(noise_map, apron, pw-w-apron, apron, ph-h-apron)
    weight = feather(tile_size, overlap, feather_power, image.device)
    total = torch.zeros_like(padded, dtype=torch.float32)
    weights = torch.zeros((1, 1, ph, pw), device=image.device)
    for y in range(0, ph-tile_size+1, step):
        for x in range(0, pw-tile_size+1, step):
            tile = padded[..., y:y+tile_size, x:x+tile_size]
            pred = model(tile) if conditioned is None else model(tile, conditioned[..., y:y+tile_size, x:x+tile_size])
            total[..., y:y+tile_size, x:x+tile_size] += pred.float()*weight
            weights[..., y:y+tile_size, x:x+tile_size] += weight
    merged = (total/weights)[..., apron:apron+h, apron:apron+w]
    out = alpha*merged+(1-alpha)*image
    if return_weights:
        # Effective normalized partition, useful for auditing coverage.
        return out, weights[..., apron:apron+h, apron:apron+w]
    return out
