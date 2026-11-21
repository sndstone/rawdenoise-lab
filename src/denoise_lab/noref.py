"""Scene-confounded diagnostics for linear R/G1/B/G2 crops, never quality gates.

Regions are chosen from input alone. Confidence intervals resample whole crops,
not correlated pixels; one crop gives a degenerate interval, not strong evidence.
"""
from __future__ import annotations

import numpy as np


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


def _mad(a):
    return 1.4826*np.median(np.abs(a-np.median(a)))


def _texture_ratio(inp, out, blocks):
    ratios = []
    for y, x, size in blocks:
        window = np.outer(np.hanning(size), np.hanning(size))
        powers = []
        for a in (inp, out):
            tile = a[:, y:y+size, x:x+size]
            tile = (tile-tile.mean((-2, -1), keepdims=True))*window
            powers.append(np.abs(np.fft.fft2(tile))**2)
        freq = np.fft.fftfreq(size)
        radius = np.hypot(freq[:, None], freq[None, :])
        # Equal weighting of radial annuli, rather than of individual FFT bins.
        for lo, hi in zip(np.linspace(.15, .45, 7)[:-1], np.linspace(.15, .45, 7)[1:]):
            mask = (radius >= lo) & (radius < hi)
            if mask.any():
                p, q = (a[:, mask].mean() for a in powers)
                if p > 1e-18:
                    ratios.append(q/p)
    return float(np.mean(ratios)) if ratios else None


def crop_metrics(inp, out, block=16):
    """MAD noise sigma, mid/high radial power ratio, and luma-normalized blotch variance.

    Flat = bottom quartile of input smoothed-green gradient block scores; texture
    = top quartile with nonzero gradient. Noise sigma assumes locally white noise
    and corrects the sigma=1 high-pass filter gain. Blotch uses sigma=4 low-pass
    R-G/B-G variance divided by squared median absolute INPUT green (floor 1e-4).
    Texture power also contains input noise, so attenuation cannot prove detail loss.
    """
    inp, out = np.asarray(inp, np.float64), np.asarray(out, np.float64)
    if inp.shape != out.shape or inp.ndim != 3 or inp.shape[0] != 4 or min(inp.shape[1:]) < block:
        raise ValueError("Expected matching [4,H,W] crops at least block pixels wide")
    if block < 8 or not np.isfinite(inp).all() or not np.isfinite(out).all():
        raise ValueError("Expected finite crops and block >= 8")
    green = (inp[1]+inp[3])/2
    grad = np.hypot(*np.gradient(gauss(green, 1)))
    tiles = [(float(grad[y:y+block, x:x+block].mean()), y, x, block)
             for y in range(0, inp.shape[1]-block+1, block)
             for x in range(0, inp.shape[2]-block+1, block)]
    lo, hi = np.quantile([t[0] for t in tiles], [.25, .75])
    flat = np.zeros(green.shape, bool)
    texture = []
    for score, y, x, size in tiles:
        if score <= lo:
            flat[y:y+size, x:x+size] = True
        if score >= hi and score > 1e-10:
            texture.append((y, x, size))
    # Exclude filter-border pixels from noise statistics.
    interior = flat.copy()
    interior[:3] = interior[-3:] = False
    interior[:, :3] = interior[:, -3:] = False
    if not interior.any():
        interior = flat
    k = np.exp(-.5*np.arange(-3, 4)**2)
    k /= k.sum()
    kernel = np.outer(k, k)
    kernel[3, 3] -= 1
    gain = np.sqrt((kernel**2).sum())
    high = out-gauss(out, 1)
    noise = float(np.median([_mad(c[interior])/gain for c in high]))
    g = (out[1]+out[3])/2
    chroma = gauss(np.stack([out[0]-g, out[2]-g]), 4)
    luma = max(float(np.median(np.abs(green[flat]))), 1e-4)
    blotch = float(np.mean(np.var(chroma[:, flat], axis=1))/luma**2)
    return {"residual_noise": noise, "texture_retention": _texture_ratio(inp, out, texture),
            "chroma_blotch": blotch}


def bootstrap_crops(rows, *, samples=1000, seed=20261104, confidence=.95):
    """Mean and percentile bootstrap CI over finite crop measurements (fixed seed)."""
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence in (0, 1)")
    result = {}
    rng = np.random.default_rng(seed)
    for key in ("residual_noise", "texture_retention", "chroma_blotch"):
        values = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])])
        if not len(values):
            result[key] = {"mean": None, "ci95": None, "crops": 0}
            continue
        means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
        tail = (1-confidence)/2
        result[key] = {"mean": float(values.mean()),
                       "ci95": np.quantile(means, [tail, 1-tail]).tolist(), "crops": len(values)}
    return result
