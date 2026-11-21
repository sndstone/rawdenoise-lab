"""Tripod burst registration and independent clean references in packed RAW units.

Shifts are (dy, dx) in PACKED pixels: out[c,y,x] = in[c,y+dy,x+dx].
One packed pixel is two sensor pixels. Bilinear sampling never mixes CFA planes;
integer packed shifts are exact. Fractional shifts blur detail and correlate
neighbouring noise. Their white-noise variance multiplier is sum(kernel**2),
which is propagated here; this is not a model for fixed-pattern/row noise.
Only translation is fitted. Rotation, parallax and local motion remain masked.
"""
from dataclasses import dataclass
import warnings

import numpy as np

from .noise import noise_std
from .rawstats import _blocks, fit_nlf


def _packed(a):
    a = np.asarray(a, np.float32)
    if a.ndim != 3 or a.shape[0] != 4 or min(a.shape[-2:]) < 8:
        raise ValueError("Expected packed [4,H,W], H,W >= 8")
    return a


def interpolation_variance(shift):
    """Variance ratio for independent, locally constant-variance source noise."""
    f = np.asarray(shift, float) % 1
    return float(np.prod((1-f)**2+f**2))


def warp_packed(frame, shift):
    """Per-plane bilinear translation; unsupported pixels are NaN, never wrapped."""
    frame = _packed(frame)
    shift = np.asarray(shift, float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError("shift must contain finite dy, dx")
    h, w = frame.shape[-2:]
    iy, ix = np.floor(shift).astype(int)
    fy, fx = shift-[iy, ix]
    out = np.zeros_like(frame)
    valid = np.ones((h, w), bool)
    for oy, wy in ((0, 1-fy), (1, fy)):
        for ox, wx in ((0, 1-fx), (1, fx)):
            weight = wy*wx
            if weight == 0:
                continue
            yy, xx = np.arange(h)+iy+oy, np.arange(w)+ix+ox
            valid &= (yy[:, None] >= 0) & (yy[:, None] < h) & (xx >= 0) & (xx < w)
            # Row chunks bound advanced-indexing temporaries on full sensor frames.
            for y in range(0, h, 128):
                out[:, y:y+128] += weight*frame[:, np.clip(yy[y:y+128], 0, h-1)[:, None],
                                                          np.clip(xx, 0, w-1)]
    out[:, ~valid] = np.nan
    return out


def _phase_shift(ref, moving, max_shift):
    """Windowed, low-frequency phase correlation with local 100x DFT peak fit."""
    h, w = ref.shape
    window = np.hanning(h)[:, None]*np.hanning(w)[None]
    ref = (ref-ref.mean())*window
    moving = (moving-moving.mean())*window
    if min(float(np.std(ref)), float(np.std(moving))) < 1e-8:
        raise ValueError("Registration needs textured finite image regions")
    cross = np.fft.fft2(moving)*np.conj(np.fft.fft2(ref))
    fy, fx = np.fft.fftfreq(h), np.fft.fftfreq(w)
    # Partial whitening suppresses dominant DC structure without amplifying noisy bins.
    cross /= np.maximum(np.abs(cross), np.max(np.abs(cross))*1e-6)**.7
    cross *= np.exp(-((fy[:, None]**2+fx**2)/.22**2)**2)
    corr = np.fft.ifft2(cross).real
    yy, xx = np.fft.fftfreq(h)*h, np.fft.fftfreq(w)*w
    corr[(np.abs(yy[:, None]) > max_shift) | (np.abs(xx) > max_shift)] = -np.inf
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    peak = np.array([yy[y], xx[x]])
    for step in (.1, .01):
        ys = peak[0]+np.arange(-10, 11)*step
        xs = peak[1]+np.arange(-10, 11)*step
        local = (np.exp(2j*np.pi*ys[:, None]*fy) @ cross @
                 np.exp(2j*np.pi*fx[:, None]*xs)).real
        local[(np.abs(ys[:, None]) > max_shift) | (np.abs(xs) > max_shift)] = -np.inf
        y, x = np.unravel_index(np.argmax(local), local.shape)
        peak = np.array([ys[y], xs[x]])
    return peak


def estimate_shift(reference, moving, max_shift=128., tile=512):
    """Median translation of up to nine green tiles, in packed pixels.

    Input grids must coincide before registration (use a known integer initial
    shift first for larger displacements). A majority of tiles must be static.
    Flat/invalid tiles are ignored; an unregistrable frame raises ValueError.
    """
    reference, moving = _packed(reference), _packed(moving)
    if reference.shape != moving.shape or max_shift <= 0 or tile < 16:
        raise ValueError("Registration requires matching shapes and positive search extent")
    h, w = reference.shape[-2:]
    th, tw = min(tile, h), min(tile, w)
    estimates = []
    for y in np.unique(np.linspace(0, h-th, 3).astype(int)):
        for x in np.unique(np.linspace(0, w-tw, 3).astype(int)):
            sl = (slice(y, y+th), slice(x, x+tw))
            r = (reference[1][sl]+reference[3][sl])*.5
            m = (moving[1][sl]+moving[3][sl])*.5
            if not (np.isfinite(r).all() and np.isfinite(m).all()):
                continue
            try:
                extent = min(max_shift, min(th, tw)/4)
                coarse = _phase_shift(r, m, extent)
                # Recenter on the integer overlap before the fine fit. Applying
                # the same window to displaced structure otherwise biases the
                # subpixel peak towards zero, especially on small images.
                iy, ix = np.rint(coarse).astype(int)
                ry = slice(max(0, -iy), min(th, th-iy))
                rx = slice(max(0, -ix), min(tw, tw-ix))
                my = slice(max(0, iy), min(th, th+iy))
                mx = slice(max(0, ix), min(tw, tw+ix))
                estimates.append([iy, ix]+_phase_shift(r[ry, rx], m[my, mx], 1.))
            except ValueError:
                pass
    if not estimates:
        raise ValueError("No finite textured tiles available for registration")
    return np.median(estimates, axis=0)


def align_burst(frames, reference=None, initial=(0., 0.), max_shift=128.):
    """Return aligned frames and absolute shifts; reference defaults to frame 0."""
    frames = np.asarray(frames, np.float32)
    if frames.ndim != 4 or not len(frames):
        raise ValueError("Expected nonempty [N,4,H,W] burst")
    ref = _packed(frames[0] if reference is None else reference)
    initial = np.asarray(initial, float)
    shifts = []
    aligned = np.empty_like(frames)
    for i, frame in enumerate(frames):
        shift = np.zeros(2) if reference is None and i == 0 else initial+estimate_shift(
            ref, warp_packed(frame, initial), max_shift=max_shift)
        shifts.append(shift)
        aligned[i] = warp_packed(frame, shift)
    return aligned, np.asarray(shifts)


def _variance(parameters, signal):
    if isinstance(parameters, dict):
        return noise_std(parameters, signal)**2
    if len(parameters) != 4:
        raise ValueError("Need one NLF dict or four per-channel NLF dicts")
    return np.stack([noise_std(p, signal[c])**2 for c, p in enumerate(parameters)])


def estimate_burst_noise(frames, factors=None, block=16):
    """Temporal shot+read NLF diagnostic, robust to a minority of moving blocks.

    Estimate after registration. Correct the average interpolation variance
    factor to report source-noise parameters. Texture/motion and illumination
    drift can still inflate this estimate; independent calibration is preferred.
    """
    frames = np.asarray(frames, np.float32)
    if len(frames) < 2:
        raise ValueError("At least two independent frames needed for temporal NLF")
    factors = np.ones(len(frames)) if factors is None else np.asarray(factors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(frames, axis=0)
        var = np.nanvar(frames, axis=0, ddof=1)/np.mean(factors)
    mu = _blocks(mean, block).mean(-1)
    tv = _blocks(var, block).mean(-1)
    result = []
    for c in range(4):
        fit = fit_nlf(mu[c].ravel(), tv[c].ravel(), quantile=.5,
                      min_blocks=5, bins=20)
        result.append({"K": fit["K"], "read_std": float(np.sqrt(fit["c"]))})
    return result


def _block_mask(bad, block, radius=1):
    """Dilate block failures without wraparound; incomplete edge blocks invalid."""
    h, w = bad.shape
    hb, wb = h//block, w//block
    blocks = bad[:hb*block, :wb*block].reshape(hb, block, wb, block).any((1, 3))
    p = np.pad(blocks, radius, constant_values=False)
    expanded = np.zeros_like(blocks)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            expanded |= p[dy:dy+hb, dx:dx+wb]
    out = np.zeros((h, w), bool)
    out[:hb*block, :wb*block] = np.repeat(np.repeat(~expanded, block, 0), block, 1)
    return out


@dataclass
class BurstTarget:
    clean: np.ndarray
    effective_count: np.ndarray
    variance: np.ndarray
    confidence: np.ndarray
    static: np.ndarray


def robust_merge(frames, parameters, factors=None, sigma=4.5, block=16):
    """Clipped temporal mean, NLF variance and motion confidence in [0,1].

    Inputs are independent target frames only, already registered/exposure
    matched. parameters describes their shared source shot+read NLF. Counts
    are per CFA sample; zero support produces NaN. Motion uses temporal
    chi-square excess at block scale (5 standard deviations), with a one-block
    apron. Counts do not imply static confidence: rejected objects stay masked.
    """
    frames = np.asarray(frames, np.float32)
    if frames.ndim != 4 or len(frames) < 2 or sigma <= 0 or block < 1:
        raise ValueError("Need >=2 packed frames, positive sigma and block")
    _packed(frames[0])
    n, _, h, w = frames.shape
    factors = np.ones(n) if factors is None else np.asarray(factors, float)
    if factors.shape != (n,) or np.any(factors <= 0) or not np.isfinite(factors).all():
        raise ValueError("Need one positive finite variance factor per frame")
    clean = np.empty_like(frames[0])
    count = np.zeros_like(clean)
    variance = np.empty_like(clean)
    score = np.empty_like(clean)
    # Bound the median and rejection temporaries to a narrow row strip.
    for y in range(0, h, 64):
        sl = (slice(None), slice(y, y+64), slice(None))
        a = frames[:, :, y:y+64]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            center = np.nanmedian(a, axis=0)
        v = np.maximum(_variance(parameters, center), 1e-16)
        vi = factors[:, None, None, None]*v
        finite = np.isfinite(a)
        keep = finite & ((a-center)**2 <= sigma**2*vi*(1+np.pi/(2*n)))
        k = keep.sum(0)
        total = np.where(keep, a, 0).sum(0)
        clean[sl] = np.divide(total, k, out=np.full_like(center, np.nan), where=k > 0)
        count[sl] = k
        variance[sl] = np.divide((keep*vi).sum(0), k*k, out=np.full_like(v, np.nan), where=k > 0)
        # Standard sample-variance statistic, conservative when interpolation
        # factors differ. Use original (unclipped) samples to retain motion evidence.
        k0 = finite.sum(0)
        mean = np.divide(np.where(finite, a, 0).sum(0), k0,
                         out=np.zeros_like(center), where=k0 > 0)
        score[sl] = np.where(finite, (a-mean)**2/vi, 0).sum(0)/np.maximum(k0-1, 1)
    energy = _blocks(score, block).mean(-1).max(0)
    # Four correlated bilinear taps reduce the effective spatial degrees of freedom.
    threshold = 1+5*np.sqrt(8/((n-1)*block*block))
    bad_b = energy > threshold
    bad = np.ones((h, w), bool)
    bh, bw = bad_b.shape
    bad[:bh*block, :bw*block] = np.repeat(np.repeat(bad_b, block, 0), block, 1)
    valid = (count >= max(2, int(np.ceil(n*.75)))).all(0) & np.isfinite(clean).all(0)
    static = _block_mask(bad, block) & valid & (clean < .95).all(0)
    confidence = np.where(static, count.min(0)/n, 0).astype(np.float32)
    return BurstTarget(clean, count, variance, confidence, static)


def match_cross_iso(noisy, target, parameters, fit_gain, factors=None, block=16):
    """Reuse a supplied exposure fitter; return fitted target, variance, mask, fit.

    fit_gain is eval_real_burst.fit_gain (noisy mean, clean, static, block).
    Final static blocks pass both a signed mean test and a residual-energy test
    under the noisy NLF plus propagated target variance. A smooth gain fit cannot
    certify local scene changes. No noisy sample is included in the clean merge.
    """
    noisy = np.asarray(noisy, np.float32)
    n = len(noisy)
    if noisy.ndim != 4 or n < 1 or noisy.shape[1:] != target.clean.shape:
        raise ValueError("Cross-ISO matching needs independent, matching packed frames")
    factors = np.ones(n) if factors is None else np.asarray(factors)
    if factors.shape != (n,) or np.any(factors <= 0) or not np.isfinite(factors).all():
        raise ValueError("Need one positive finite variance factor per noisy frame")
    valid = target.static & np.isfinite(noisy).all((0, 1)) & (noisy < .95).all((0, 1))
    # Fit only fully supported blocks; NaNs outside valid still need finite fill
    # because the existing fitter constructs its smooth map over the full grid.
    mean = np.zeros_like(noisy[0])
    for frame in noisy:
        mean += np.nan_to_num(frame)/n
    clean = np.nan_to_num(target.clean)
    if _blocks(valid, block).all(-1).sum() < 4:
        raise ValueError("Too few static blocks for cross-ISO exposure fit")
    fitted, coeffs = fit_gain(mean, clean, valid, block)
    # Include local multiplicative illumination scaling in reference variance.
    gain = np.asarray([c[0] for c in coeffs])[:, None, None]
    offset = np.asarray([c[1] for c in coeffs])[:, None, None]
    base = gain*clean+offset
    local = np.divide(fitted, base, out=np.ones_like(fitted), where=np.abs(base) > 1e-8)
    tvar = target.variance*(gain*local)**2
    source_var = np.maximum(_variance(parameters, fitted), 1e-16)
    diff = mean-fitted
    # Original source variance bounds block-mean variance after interpolation:
    # adjacent output noise samples are correlated, so do not divide q by b².
    bv = _blocks(source_var/n+tvar/np.min(factors), block).mean(-1)/(block*block)
    bias = _blocks(diff, block).mean(-1)**2/np.maximum(bv, 1e-16)
    pv = source_var*np.mean(factors)/n+tvar
    energy = _blocks(diff**2/np.maximum(pv, 1e-16), block).mean(-1)
    bad_b = ((bias > 25) | (energy > 1+5*np.sqrt(8/(block*block)))).any(0)
    if n > 1:
        temporal = np.zeros_like(mean)
        for frame, q in zip(noisy, factors):
            temporal += (np.nan_to_num(frame)-mean)**2/(source_var*q*(n-1))
        bad_b |= (_blocks(temporal, block).mean(-1) >
                  1+5*np.sqrt(8/((n-1)*block*block))).any(0)
    bad = np.ones(valid.shape, bool)
    bh, bw = bad_b.shape
    bad[:bh*block, :bw*block] = np.repeat(np.repeat(bad_b, block, 0), block, 1)
    valid &= _block_mask(bad, block)
    return fitted, tvar, valid, coeffs
