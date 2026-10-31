"""Exact output regression against the pre-optimization padded normalization."""
import pytest
import torch

from denoise_lab.tiling import (
    feather, reflected_bayer_pad, reflected_pad, tiled_inference,
)


@torch.inference_mode()
def padded_normalization_reference(model, image, overlap, power, alpha, noise, bayer):
    # Frozen accumulation/normalization order from a930852, small tiles only.
    size = 16
    height, width = image.shape[-2:]
    step = size-overlap
    extent = lambda n: size+max(0, (n+2*overlap-size+step-1)//step)*step
    ph, pw = extent(height), extent(width)
    padding = (overlap, pw-width-overlap, overlap, ph-height-overlap)
    padded = (reflected_bayer_pad if bayer else reflected_pad)(image, *padding)
    condition = None if noise is None else reflected_pad(noise, *padding)
    weight = feather(size, overlap, power, image.device)
    total = torch.zeros_like(padded, dtype=torch.float32)
    weights = torch.zeros((1, 1, ph, pw), device=image.device)
    for y in range(0, ph-size+1, step):
        for x in range(0, pw-size+1, step):
            tile = padded[..., y:y+size, x:x+size]
            prediction = model(tile) if condition is None else model(
                tile, condition[..., y:y+size, x:x+size])
            total[..., y:y+size, x:x+size] += prediction.float()*weight
            weights[..., y:y+size, x:x+size] += weight
    merged = (total/weights)[..., overlap:overlap+height, overlap:overlap+width]
    return alpha*merged+(1-alpha)*image, weights[..., overlap:overlap+height, overlap:overlap+width]


@pytest.mark.parametrize("channels,bayer,conditioned", [(4, True, False), (4, False, False),
                                                        (4, False, True), (3, False, False)])
@pytest.mark.parametrize("shape", [(1, 1, 3), (2, 19, 27)])
@pytest.mark.parametrize("overlap,power,alpha", [(0, 1., 1.), (3, 2.2, .3), (12, .8, 0.)])
def test_visible_normalization_is_bit_exact(channels, bayer, conditioned, shape, overlap, power, alpha):
    batch, height, width = shape
    generator = torch.Generator().manual_seed(5321)
    image = torch.randn(batch, channels, height, width, generator=generator)*.01
    image.flatten()[0] = -0.
    noise = torch.rand(batch, 1, height, width, generator=generator) if conditioned else None

    def model(tile, condition=None):
        # Nonlinear, channel-dependent prediction; avoid identity-only coverage.
        offset = torch.arange(channels).reshape(1, channels, 1, 1)*.001
        return tile.square()*.7+tile*.83+offset+(0 if condition is None else condition*.02)

    expected, weights = padded_normalization_reference(model, image, overlap, power, alpha, noise, bayer)
    actual, actual_weights = tiled_inference(model, image, tile_size=16, overlap=overlap,
        feather_power=power, alpha=alpha, noise_map=noise, bayer_reflection=bayer, return_weights=True)
    # Compare bit patterns as well as values (including signed zero).
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(actual_weights.view(torch.int32), weights.view(torch.int32))
    assert torch.all(actual_weights > 0)


@pytest.mark.parametrize("channels,condition_batch", [(3, None), (4, None), (4, 1), (4, 2)])
@pytest.mark.parametrize("tile_batch_size", [2, 3])
def test_tile_batching_preserves_order_and_partial_batch(channels, condition_batch, tile_batch_size):
    image = torch.randn(2, channels, 19, 27, generator=torch.Generator().manual_seed(291))
    noise = None if condition_batch is None else torch.linspace(0, 1, condition_batch*19*27).reshape(
        condition_batch, 1, 19, 27)
    calls = []

    def model(tile, condition=None):
        calls.append(tile.shape[0])
        return tile.square()*.1+tile*.87+(0 if condition is None else condition*.03)

    kwargs = dict(tile_size=16, overlap=3, feather_power=2.2, alpha=.7,
                  noise_map=noise, return_weights=True)
    expected, expected_weights = tiled_inference(model, image, **kwargs)
    tile_count = len(calls)
    assert tile_count == 6
    calls.clear()
    actual, weights = tiled_inference(model, image, tile_batch_size=tile_batch_size, **kwargs)
    assert calls == [2*tile_batch_size]*(tile_count//tile_batch_size)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(weights.view(torch.int32), expected_weights.view(torch.int32))

    # Seven tiles ensures both batch sizes have a final partial model call.
    image = image[..., :1, :1].expand(2, channels, 1, 85)
    noise = None if condition_batch is None else torch.linspace(0, 1, condition_batch*85).reshape(
        condition_batch, 1, 1, 85)
    calls.clear()
    kwargs.update(noise_map=noise, overlap=3)
    expected, _ = tiled_inference(model, image, **kwargs)
    tile_count = len(calls)
    calls.clear()
    actual, _ = tiled_inference(model, image, tile_batch_size=tile_batch_size, **kwargs)
    assert tile_count % tile_batch_size != 0
    assert calls[-1] == 2*(tile_count % tile_batch_size)
    assert sum(calls) == 2*tile_count
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, False, "2"])
def test_tile_batch_size_validation(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        tiled_inference(torch.nn.Identity(), torch.zeros(1, 4, 1, 1), tile_batch_size=invalid)


def test_batched_model_shape_is_checked():
    with pytest.raises(ValueError, match="Model output shape"):
        tiled_inference(lambda tile: tile[:1], torch.zeros(2, 4, 19, 27),
                        tile_size=16, overlap=3, tile_batch_size=2)
