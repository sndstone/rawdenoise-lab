"""Directional checks on synthetic noise/texture; no private captures required."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from denoise_lab.noref import bootstrap_crops, crop_metrics, gauss


class Blur(torch.nn.Module):
    def forward(self, x):
        return F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), 5, stride=1)


def predict(model, a):
    with torch.inference_mode():
        return model(torch.from_numpy(a)[None])[0].numpy()


def test_identity_and_blur_noise_texture_tradeoff():
    rng = np.random.default_rng(7)
    y, x = np.mgrid[:96, :96]
    scene = np.full((96, 96), .3)
    scene[:, 48:] += .08*np.sin(2*np.pi*x[:, 48:]/4)*np.cos(2*np.pi*y[:, 48:]/5)
    noisy = (scene[None]+rng.normal(0, .01, (4, 96, 96))).astype(np.float32)
    identity = crop_metrics(noisy, predict(torch.nn.Identity(), noisy))
    blur = crop_metrics(noisy, predict(Blur(), noisy))
    assert identity['texture_retention'] == pytest.approx(1.)
    assert identity['residual_noise'] == pytest.approx(.01, rel=.15)
    assert blur['residual_noise'] < identity['residual_noise']*.3
    assert blur['texture_retention'] < identity['texture_retention']*.15


def test_low_frequency_chroma_blotches_and_luma_normalization():
    y, x = np.mgrid[:96, :96]
    clean = np.full((4, 96, 96), .25)
    blotched = clean.copy()
    blotched[0] += .03*np.sin(x/9)*np.cos(y/11)
    baseline = crop_metrics(clean, clean)['chroma_blotch']
    value = crop_metrics(clean, blotched)['chroma_blotch']
    assert value > baseline+1e-4
    assert crop_metrics(clean*2, blotched*2)['chroma_blotch'] == pytest.approx(value)
    # Input region selection is fixed, so a smoothed output cannot select easier regions.
    assert crop_metrics(clean, gauss(blotched, 12))['chroma_blotch'] < value


def test_bootstrap_is_over_crops_reproducible_and_handles_missing_texture():
    rows = [dict(residual_noise=v, texture_retention=None, chroma_blotch=v/10) for v in (1., 2., 4., 5.)]
    a = bootstrap_crops(rows, samples=300, seed=2)
    assert a == bootstrap_crops(rows, samples=300, seed=2)
    assert a['residual_noise']['mean'] == 3.
    assert a['residual_noise']['ci95'][0] < 3 < a['residual_noise']['ci95'][1]
    assert a['texture_retention'] == dict(mean=None, ci95=None, crops=0)
    one = bootstrap_crops(rows[:1])['residual_noise']
    assert one['ci95'] == [1., 1.] and one['crops'] == 1
    assert bootstrap_crops([])['chroma_blotch']['mean'] is None


def test_flat_identity_and_validation():
    a = np.full((4, 32, 32), .2)
    metrics = crop_metrics(a, a)
    assert metrics['residual_noise'] < 1e-14
    assert metrics['texture_retention'] is None
    assert metrics['chroma_blotch'] == 0
    with pytest.raises(ValueError):
        crop_metrics(a, a[:, :16])
    with pytest.raises(ValueError):
        crop_metrics(a, a*np.nan)
