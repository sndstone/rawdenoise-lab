"""Optional clean-room smooth Bayer targets for bootstrap training."""
import numpy as np
import torch
from .bayer import SITES, read_mosaic, augment_mosaic, random_crop, pack
from .bootstrap import add_bootstrap_noise
from .data import BootstrapDataset, _tensor
from .noise import noise_std
from .defects import inject_defects


def synthetic_smooth_mosaic(h, w, rng, level_range=(1e-3, .95),
                            ratio_range=(.3, 1.6), equal_probability=.2,
                            gradient_probability=.5, gradient_max=.3):
    """Return float32 RGGB sensor pixels using only the supplied numpy RNG.

    R/G and B/G vary independently; both green sites have unit gain. The
    planar ramp uses coordinates in [-1, 1] and an L1-normalized direction,
    keeping its relative excursion bounded by gradient_max.
    """
    if h < 1 or w < 1:
        raise ValueError("Mosaic dimensions must be positive")
    lo, hi = level_range
    rlo, rhi = ratio_range
    if not 0 < lo <= hi <= 1 or not 0 < rlo <= rhi:
        raise ValueError("Invalid level_range or ratio_range")
    if not (0 <= equal_probability <= 1 and 0 <= gradient_probability <= 1):
        raise ValueError("Probabilities must be in [0, 1]")
    if not 0 <= gradient_max <= 1:
        raise ValueError("gradient_max must be in [0, 1]")
    level = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    ratios = np.ones(4)
    if rng.random() >= equal_probability:
        ratios[0], ratios[2] = rng.uniform(rlo, rhi, 2)
    field = np.full((h, w), level, dtype=np.float64)
    if rng.random() < gradient_probability:
        angle = rng.uniform(0, 2*np.pi)
        a, b = np.cos(angle), np.sin(angle)
        norm = abs(a)+abs(b)
        g = rng.uniform(-gradient_max, gradient_max)
        u = np.linspace(-1, 1, w)[None, :]
        v = np.linspace(-1, 1, h)[:, None]
        field *= 1+g*(a*u+b*v)/norm
    for ratio, (y, x) in zip(ratios, SITES):
        field[y::2, x::2] *= ratio
    return np.clip(field, 0, 1).astype(np.float32)


class FlatAugmentedBootstrapDataset(BootstrapDataset):
    """BootstrapDataset with optional smooth targets before exposure/noise.

    The parent item path is deliberately duplicated to leave data.py frozen.
    """
    def __init__(self, *args, flat_probability=0., flat_options=None,
                 defect_probability=0., defect_options=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= flat_probability <= 1:
            raise ValueError("flat_probability must be in [0, 1]")
        self.flat_probability = float(flat_probability)
        self.flat_options = dict(flat_options or {})
        if not 0 <= defect_probability <= 1:
            raise ValueError("defect_probability must be in [0, 1]")
        self.defect_probability = float(defect_probability)
        self.defect_options = dict(defect_options or {})

    def __getitem__(self, index):
        seed = self.seed+index if self.deterministic else int(torch.randint(0, 2**62, ()).item())
        rng = np.random.default_rng(seed)
        clean = read_mosaic(self.root/self.entries[index]["clean"])
        if self.augment:
            clean = augment_mosaic(clean, *rng.integers(0, 2, 3).astype(bool))
        clean = random_crop(clean, self.crop_size, rng)
        # No extra draw when disabled: preserve the parent's random stream.
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
        # Attenuate the imperfect low-ISO reference BEFORE drawing new shot/read
        # noise. This is exposure augmentation, not a measured camera ISO model.
        clean = clean*exposure
        if self.augment and rng.random() < self.identity_probability:
            noisy, parameters = clean.copy(), {"K": 0., "read_std": 0., "row_std": 0.}
        else:
            noisy, parameters = add_bootstrap_noise(clean, rng, **self.parameters)
        if self.augment and self.defect_probability > 0 and rng.random() < self.defect_probability:
            noisy, _ = inject_defects(noisy, rng, **self.defect_options)
        inp, target = pack(noisy), pack(clean)
        std = noise_std(parameters, np.maximum(inp, 0).mean(0, keepdims=True))
        return {"noisy": _tensor(inp), "clean": _tensor(target), "noise_map": _tensor(std),
                "exposure_scale": exposure}

