"""Cleaner bootstrap targets: same-channel Bayer binning of the low-ISO reference.

A single ISO 100 frame still carries ISO 100 noise. Because the training input is
built as target + fresh noise, a model learns to preserve whatever noise sits in
the target. Averaging k x k same-colour photosites divides that residual noise
variance by k^2 while keeping a valid RGGB mosaic (at 1/k resolution).
"""
import numpy as np
import torch
from .bayer import read_mosaic, augment_mosaic, random_crop, pack, unpack
from .bootstrap import add_bootstrap_noise
from .data import _tensor
from .flat_augment import FlatAugmentedBootstrapDataset, synthetic_smooth_mosaic
from .noise import noise_std
from .defects import inject_defects


def bin_mosaic(mosaic, factor=2):
    """Average factor x factor same-channel photosites; returns an RGGB mosaic."""
    if factor == 1:
        return mosaic
    packed = pack(mosaic)
    h, w = packed.shape[-2]//factor*factor, packed.shape[-1]//factor*factor
    binned = packed[:, :h, :w].reshape(4, h//factor, factor, w//factor, factor).mean((2, 4))
    return unpack(binned.astype(np.float32))


def apply_channel_gains(mosaic, rng, gain_range=(.4, 2.5)):
    """White-balance-like augmentation: log-uniform R/G and B/G gains (G1 = G2).

    Gains are normalised so the largest is 1 (attenuation only, no new clipping), and the
    target and input see the same scene, so this only widens the colour-ratio prior.
    """
    r, b = np.exp(rng.uniform(np.log(gain_range[0]), np.log(gain_range[1]), 2))
    gains = np.array([r, 1., b])/max(r, 1., b)
    out = np.array(mosaic, dtype=np.float32, copy=True)
    out[0::2, 0::2] *= gains[0]
    out[0::2, 1::2] *= gains[1]
    out[1::2, 0::2] *= gains[1]
    out[1::2, 1::2] *= gains[2]
    return out


# Native packed-plane offsets (y, x) in packed pixels for R, G1, B, G2 (bayer.SITES order).
_NATIVE_OFFSETS = np.array([(0., 0.), (0., .5), (.5, .5), (.5, 0.)])


def _shift_axis(a, d, axis):
    """Catmull-Rom resample: out[i] = a(i + d) along axis, reflected edges."""
    if abs(d) < 1e-9:
        return a
    i0 = int(np.floor(d)); t = d - i0
    w = np.array([(-t**3 + 2*t**2 - t)/2, (3*t**3 - 5*t**2 + 2)/2, (-3*t**3 + 4*t**2 + t)/2, (t**3 - t**2)/2])
    pad = [(0, 0)]*a.ndim; pad[axis] = (3, 3)
    b = np.pad(a, pad, mode="reflect")
    n = a.shape[axis]
    out = 0.
    for k, wk in zip(range(-1, 3), w):
        out = out + wk*np.take(b, np.arange(n) + 3 + i0 + k, axis=axis)
    return out.astype(a.dtype)


def resite_binned(mosaic, factor):
    """Restore native colour-site geometry after k x k same-channel binning.

    Binning puts the four colour planes 1/k of their native offset apart (nearly co-sited).
    Each packed plane is resampled so the R/G1/B/G2 offsets are again 0 or 0.5 packed
    pixels, centred so the image does not move overall.
    """
    if factor <= 1:
        return mosaic
    packed = pack(np.asarray(mosaic, np.float32))
    shifts = (_NATIVE_OFFSETS - _NATIVE_OFFSETS.mean(0))*(1 - 1/factor)
    out = np.stack([_shift_axis(_shift_axis(packed[c], dy, 0), dx, 1) for c, (dy, dx) in enumerate(shifts)])
    return unpack(out)


def iso_coupled_parameters(rng, iso_range=(100., 25600.), K_6400=6.7e-4, read_6400=6e-4,
                           jitter_log=.3, row_relative_max=.1, exposure_jitter=(.5, 2.)):
    """Sample one camera-like operating point: noise scales with ISO, exposure ~ 100/ISO.

    Anchored on our own X1D II ISO 6400 NLF measurement (docs/09), with log-normal jitter.
    """
    iso = float(np.exp(rng.uniform(*np.log(iso_range))))
    K = K_6400*iso/6400*float(np.exp(rng.normal(0, jitter_log)))
    read = read_6400*iso/6400*float(np.exp(rng.normal(0, jitter_log)))
    exposure = min(1., 100/iso*float(np.exp(rng.uniform(*np.log(exposure_jitter)))))
    return {"K": K, "read_std": read, "row_std": float(rng.uniform(0, row_relative_max))*read}, exposure, iso


class BinnedBootstrapDataset(FlatAugmentedBootstrapDataset):
    """FlatAugmentedBootstrapDataset whose reference is binned with bin_probability.

    bin_factors lists candidate factors (one drawn uniformly per binned sample).
    The cached crop must stay at least 2*crop_size sensor pixels after binning.
    """
    def __init__(self, *args, bin_probability=0., bin_factors=(2,), channel_gain_probability=0.,
                 channel_gain_range=(.4, 2.5), resite=False, iso_coupled=None, noise_map_jitter=None, **kwargs):
        super().__init__(*args, **kwargs)
        # If set, the conditioning map uses only shot/read (what EXIF ISO + our NLF can give at
        # inference) with independent log-normal errors of this sigma on K and read.
        self.noise_map_jitter = None if noise_map_jitter is None else float(noise_map_jitter)
        self.resite = bool(resite)
        self.iso_coupled = dict(iso_coupled) if iso_coupled else None
        if not 0 <= channel_gain_probability <= 1:
            raise ValueError("channel_gain_probability must be in [0, 1]")
        lo, hi = channel_gain_range
        if not 0 < lo <= hi:
            raise ValueError("channel_gain_range must be positive and ordered")
        self.channel_gain_probability = float(channel_gain_probability)
        self.channel_gain_range = (float(lo), float(hi))
        if not 0 <= bin_probability <= 1:
            raise ValueError("bin_probability must be in [0, 1]")
        self.bin_probability = float(bin_probability)
        self.bin_factors = tuple(int(f) for f in bin_factors)
        if not self.bin_factors or min(self.bin_factors) < 1:
            raise ValueError("bin_factors must be positive integers")

    def __getitem__(self, index):
        seed = self.seed+index if self.deterministic else int(torch.randint(0, 2**62, ()).item())
        rng = np.random.default_rng(seed)
        clean = read_mosaic(self.root/self.entries[index]["clean"])
        factor = 1
        if self.bin_probability > 0 and rng.random() < self.bin_probability:
            factor = self.bin_factors[int(rng.integers(len(self.bin_factors)))]
            clean = bin_mosaic(clean, factor)
        if self.resite:
            clean = resite_binned(clean, factor*int(self.entries[index].get("bin_factor", 1)))
        if self.augment:
            clean = augment_mosaic(clean, *rng.integers(0, 2, 3).astype(bool))
        clean = random_crop(clean, self.crop_size, rng)
        if self.augment and self.channel_gain_probability > 0 and rng.random() < self.channel_gain_probability:
            clean = apply_channel_gains(clean, rng, self.channel_gain_range)
        if self.augment and self.flat_probability > 0 and rng.random() < self.flat_probability:
            clean = synthetic_smooth_mosaic(*clean.shape, rng, **self.flat_options)
        exposure = self.exposure_scale
        if self.augment:
            lo, hi = self.exposure_range
            bright = self.bright_exposure_probability > 0 and rng.random() < self.bright_exposure_probability
            if bright:
                exposure *= float(rng.uniform(.6, 1.))
            else:
                exposure *= float(np.exp(rng.uniform(np.log(lo), np.log(hi)))
                                  if self.exposure_sampling == "log" else rng.uniform(lo, hi))
        coupled = None
        if self.augment and self.iso_coupled is not None:
            coupled, exposure, _ = iso_coupled_parameters(rng, **self.iso_coupled)
            exposure *= self.exposure_scale
        clean = clean*exposure
        if self.augment and rng.random() < self.identity_probability:
            noisy, parameters = clean.copy(), {"K": 0., "read_std": 0., "row_std": 0.}
        elif coupled is not None:
            noisy, parameters = add_bootstrap_noise(clean, rng, fixed_parameters=coupled)
        else:
            noisy, parameters = add_bootstrap_noise(clean, rng, **self.parameters)
        if self.augment and self.defect_probability > 0 and rng.random() < self.defect_probability:
            noisy, _ = inject_defects(noisy, rng, **self.defect_options)
        inp, target = pack(noisy), pack(clean)
        if self.noise_map_jitter is not None:
            j = np.exp(rng.normal(0, self.noise_map_jitter, 2))
            parameters = {"K": parameters["K"]*float(j[0]), "read_std": parameters["read_std"]*float(j[1])}
        std = noise_std(parameters, np.maximum(inp, 0).mean(0, keepdims=True))
        return {"noisy": _tensor(inp), "clean": _tensor(target), "noise_map": _tensor(std),
                "exposure_scale": exposure, "bin_factor": factor}
