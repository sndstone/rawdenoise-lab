"""Shared native packed-RAW exposure and static-mask logic for burst evaluation.

Noise is estimated from each burst, not independent sensor calibration. Default
blocks are 16 packed pixels (32 sensor pixels); excluded blocks dilate by two.
"""
import numpy as np
import torch


def block_view(a, b):
    h, w = (a.shape[-2]//b)*b, (a.shape[-1]//b)*b
    return a[..., :h, :w].reshape(*a.shape[:-2], h//b, b, w//b, b)


def upsample(mask_blocks, b, shape):
    m = np.repeat(np.repeat(mask_blocks, b, -2), b, -1)
    out = np.zeros(shape, bool)
    out[..., :m.shape[-2], :m.shape[-1]] = m
    return out


def dilate(m, r):
    """Square-radius dilation with false borders (image edges do not wrap)."""
    out = m.copy()
    padded = np.pad(m, r, constant_values=False)
    h, w = m.shape
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            out |= padded[r+dy:r+dy+h, r+dx:r+dx+w]
    return out


def shifted(a, dy, dx):
    """aligned[:, y, x] = a[:, y+dy, x+dx]; uncovered borders are NaN."""
    out = np.full_like(a, np.nan)
    h, w = a.shape[-2:]
    ys, yd = (slice(dy, h), slice(0, h-dy)) if dy >= 0 else (slice(0, h+dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w-dx)) if dx >= 0 else (slice(0, w+dx), slice(-dx, w))
    out[..., yd, xd] = a[..., ys, xs]
    return out


def gauss(a, sigma):
    """Separable Gaussian blur over the last two axes, reflect padding."""
    r = int(np.ceil(3*sigma))
    k = np.exp(-.5*(np.arange(-r, r+1)/sigma)**2)
    k /= k.sum()
    a = np.asarray(a, np.float64)
    for axis in (-2, -1):
        pad = [(0, 0)]*a.ndim
        pad[axis] = (r, r)
        p = np.pad(a, pad, mode="reflect")
        n = a.shape[axis]
        a = sum(w*np.take(p, range(i, i+n), axis=axis) for i, w in enumerate(k))
    return a


def fit_gain(noisy_mean, clean, static, block=16, local=8, return_gain=False):
    """Per-channel noisy ~ (a*clean + b) * L(x, y), fitted on static block means.

    a, b are global (exposure, black offset). L is a smooth local brightness map from static
    super-blocks of local*block pixels (normalised Gaussian fill), absorbing slow illumination
    drift between groups; it cannot absorb motion at the block scale.
    return_gain additionally returns the per-pixel derivative a*L for variance propagation.
    """
    bn, bc = block_view(noisy_mean, block).mean((-3, -1)), block_view(clean, block).mean((-3, -1))
    ok = block_view(static, block).all((-3, -1)) & np.isfinite(bn).all(0)
    if ok.sum() < 2:
        raise ValueError("fit_gain needs at least two static blocks")
    target = np.empty_like(clean)
    gain_map = np.empty_like(clean) if return_gain else None
    coeffs = []
    for c in range(4):
        keep = ok.copy()
        for _ in range(3):  # trim outliers
            A = np.stack([bc[c][keep], np.ones(keep.sum())], 1)
            a, off = np.linalg.lstsq(A, bn[c][keep], rcond=None)[0]
            r = bn[c]-(a*bc[c]+off)
            tol = max(4*1.4826*np.median(np.abs(r[ok])), 1e-7)
            trimmed = ok & (np.abs(r) <= tol)
            if trimmed.sum() >= 2:
                keep = trimmed
        base_b = a*bc[c]+off
        # Local multiplicative drift on super-blocks, filled across excluded blocks.
        sb = min(local, *keep.shape)
        num = block_view(np.where(keep, bn[c], 0.), sb).sum((-3, -1))
        den = block_view(np.where(keep, base_b, 0.), sb).sum((-3, -1))
        w = block_view(keep.astype(float), sb).sum((-3, -1))
        ratio = np.where(den > 0, num/np.maximum(den, 1e-12), 1.)
        sw = gauss(w[None], 1.5)[0]
        L = gauss((ratio*w)[None], 1.5)[0]/np.maximum(sw, 1e-9)
        L = np.where(sw > 1e-6, L, 1.)
        # Bilinear upsample from super-block centres to pixels.
        H, W = clean.shape[-2:]
        Lt = torch.from_numpy(L.astype(np.float32))[None, None]
        size = (L.shape[0]*sb*block, L.shape[1]*sb*block)
        up = torch.nn.functional.interpolate(Lt, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
        Lf = np.pad(up[:H, :W], ((0, max(0, H-up.shape[0])), (0, max(0, W-up.shape[1]))), mode="edge")
        target[c] = (a*clean[c]+off)*Lf
        if return_gain:
            gain_map[c] = a*Lf
        coeffs.append([float(a), float(off), float(L.min()), float(L.max())])
    return (target, coeffs, gain_map) if return_gain else (target, coeffs)


def temporal_mask(mean, variance, block=16):
    """Match evaluator motion: variance >3x brightness-bin median, any channel."""
    tv = block_view(variance, block).mean((-3, -1))
    mu = block_view(mean, block).mean((-3, -1))
    moving = np.zeros(mu.shape[1:], bool)
    expected = np.empty_like(tv)
    for c in range(4):
        edges = np.quantile(mu[c], np.linspace(0, 1, 41))
        idx = np.clip(np.searchsorted(edges, mu[c], side='right')-1, 0, 39)
        medians = np.array([np.median(tv[c][idx == k]) if (idx == k).any()
                            else np.inf for k in range(40)])
        expected[c] = medians[idx]
        moving |= tv[c] > 3*expected[c]
    return ~moving, expected


def static_mask(clean, clean_variance, noisy_mean, noisy_variance, n_noisy, valid, block=16):
    """Evaluator's clean temporal gate, affine/local fit, then group-mean gate.

    No noisy temporal mask or pixel residual-energy gate is added. valid excludes
    uncovered/clipped pixels. The returned mask is at packed-pixel resolution.
    """
    temporal, _ = temporal_mask(clean, clean_variance, block)
    static = upsample(~dilate(~temporal, 2), block, valid.shape) & valid
    before = float(static.mean())
    if not block_view(static, block).all((-3, -1)).any():
        return np.zeros_like(valid), clean.copy(), None, before
    target, coeffs = fit_gain(noisy_mean, clean, static, block)
    diff = block_view(noisy_mean-target, block).mean((-3, -1))
    variance = block_view(noisy_variance, block).mean((-3, -1))
    z = np.abs(diff)/np.sqrt(np.maximum(variance/(n_noisy*block*block), 1e-16))
    moved = (np.nan_to_num(z, nan=99) > 5).any(0)
    static &= upsample(~dilate(moved, 2), block, valid.shape)
    return static, target, coeffs, before
