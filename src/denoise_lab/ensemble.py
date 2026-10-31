"""Experimental sensor-geometry-aware self ensemble for packed Bayer tensors."""
import torch

from .tiling import reflected_bayer_pad


def _transform(image, flip_y, flip_x, transpose):
    if flip_y:
        image = image.flip(-2)
    if flip_x:
        image = image.flip(-1)
    return image.transpose(-2, -1) if transpose else image


def _inverse(image, flip_y, flip_x, transpose):
    if transpose:
        image = image.transpose(-2, -1)
    if flip_x:
        image = image.flip(-1)
    return image.flip(-2) if flip_y else image


@torch.inference_mode()
def bayer_self_ensemble(model, packed, transforms=((False, False, False), (False, False, True))):
    """Average predictions after true mosaic transforms and red-origin crops.

    A simple flip of each packed plane changes the CFA-site geometry. Here the
    full sensor mosaic and CFA map move together, and the discarded border is
    excluded from that member's weight. The original member covers every pixel.
    Only square packed tiles are supported so fixed-size exports remain usable.
    """
    if packed.ndim != 4 or packed.shape[1] != 4 or packed.shape[-2] != packed.shape[-1]:
        raise ValueError("Expected square Bx4xHxH Bayer tiles")
    if packed.shape[-1] < 2:
        raise ValueError("Bayer ensemble needs at least 2x2 packed pixels")
    transforms = tuple(transforms)
    if not transforms or (False, False, False) not in transforms:
        raise ValueError("Ensemble must include the original prediction for border coverage")
    batch, _, height, width = packed.shape
    sites = ((0, 0), (0, 1), (1, 1), (1, 0))
    mosaic = torch.empty((batch, 1, height*2, width*2), device=packed.device, dtype=packed.dtype)
    cfa = torch.tensor([[0, 1], [3, 2]], device=packed.device).repeat(height, width)[None, None]
    for c, (y, x) in enumerate(sites):
        mosaic[:, 0, y::2, x::2] = packed[:, c]
    total, weights = torch.zeros_like(mosaic, dtype=torch.float32), torch.zeros_like(mosaic, dtype=torch.float32)
    for fy, fx, tr in transforms:
        image = _transform(mosaic, fy, fx, tr)
        colors = _transform(cfa, fy, fx, tr)
        # The unique red photosite defines phase; crop instead of rolling pixels.
        red = (colors[0, 0, :2, :2] == 0).nonzero()[0]
        y, x = int(red[0]), int(red[1])
        h, w = (image.shape[-2]-y)//2*2, (image.shape[-1]-x)//2*2
        crop = image[..., y:y+h, x:x+w]
        member = torch.stack([crop[:, 0, a::2, b::2] for a, b in sites], 1)
        # Restore the original tile extent, keeping fixed ONNX/Core ML shapes.
        padded = reflected_bayer_pad(member, 0, width-member.shape[-1], 0, height-member.shape[-2])
        prediction = model(padded)[..., :h//2, :w//2]
        if prediction.shape != member.shape or not torch.isfinite(prediction).all():
            raise ValueError("Ensemble model returned invalid Bayer predictions")
        canvas, coverage = torch.zeros_like(mosaic, dtype=torch.float32), torch.zeros_like(mosaic, dtype=torch.float32)
        for c, (a, b) in enumerate(sites):
            canvas[:, 0, y+a:y+h:2, x+b:x+w:2] = prediction[:, c]
        coverage[..., y:y+h, x:x+w] = 1
        total += _inverse(canvas, fy, fx, tr)
        weights += _inverse(coverage, fy, fx, tr)
    result = total/weights
    return torch.stack([result[:, 0, y::2, x::2] for y, x in sites], 1)
