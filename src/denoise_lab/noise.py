"""ELD-style shot/read/row/quantization noise in normalized linear units.

K is DN/electron divided by (white-black). All std/offset/step values use
normalized sensor units. Tukey read_std means *standard deviation*, not scale.
"""
from pathlib import Path
import numpy as np
import yaml


def tukey_quantile(u, shape):
    u = np.clip(u, 1e-7, 1-1e-7)
    if abs(shape) < 1e-6:
        return np.log(u/(1-u))
    return (u**shape - (1-u)**shape)/shape


def tukey_std(shape):
    if shape <= -0.5:
        raise ValueError("Tukey-lambda shape must exceed -0.5 for finite variance")
    # Deterministic midpoint quadrature, also used by the calibration fitter.
    u = (np.arange(65536)+0.5)/65536
    return float(np.sqrt(np.mean(tukey_quantile(u, shape)**2)))


class NoiseModel:
    def __init__(self, calibration, seed=None):
        if isinstance(calibration, (str, Path)):
            with open(calibration) as f:
                calibration = yaml.safe_load(f)
        if calibration.get("units") != "normalized":
            raise ValueError("Calibration must declare units: normalized")
        self.calibration = calibration
        self.profiles = {int(k): v for k, v in calibration["iso"].items()}
        self.rng = np.random.default_rng(seed)

    def parameters(self, iso, sample=False, iso_range=None):
        keys = sorted(self.profiles)
        if iso_range is not None:
            lo, hi = iso_range
            selected = [k for k in keys if lo <= k <= hi]
            if not selected:
                raise ValueError("ISO range contains no calibrated ISO")
            k0, k1 = self.profiles[selected[0]]["K"], self.profiles[selected[-1]]["K"]
            K = float(np.exp(self.rng.uniform(np.log(k0), np.log(k1))))
            iso = min(selected, key=lambda i: abs(np.log(self.profiles[i]["K"]/K)))
        else:
            if int(iso) not in self.profiles:
                raise ValueError(f"ISO {iso} is not calibrated; available: {keys}")
            K = self.profiles[int(iso)]["K"]
            if sample:
                bounds = self.profiles[int(iso)].get("K_range", [K, K])
                K = float(np.exp(self.rng.uniform(*np.log(bounds))))
        p = dict(self.profiles[int(iso)])
        p["K"] = K
        fit = self.calibration.get("read_log_linear")
        if sample and fit:
            p["read_std"] = float(np.exp(fit["slope"]*np.log(K)+fit["intercept"]
                                         + self.rng.normal(0, fit.get("scatter", 0))))
        return p

    def __call__(self, clean, iso, sample=True, iso_range=None, return_params=False):
        p = self.parameters(iso, sample=sample, iso_range=iso_range)
        clean = np.asarray(clean, dtype=np.float32)
        if clean.ndim != 2:
            raise ValueError("Apply noise to the unpacked sensor mosaic to preserve row correlations")
        K = p["K"]
        if K <= 0:
            raise ValueError("K must be positive")
        positive = np.maximum(clean, 0)
        noisy = self.rng.poisson(positive/K)*K + (clean-positive)
        std = p.get("read_std", 0.)
        if p.get("read_distribution", "gaussian") == "tukey":
            shape = p.get("tukey_lambda", 0.)
            read = tukey_quantile(self.rng.uniform(size=clean.shape), shape)/tukey_std(shape)
        else:
            read = self.rng.normal(size=clean.shape)
        noisy += std*read
        noisy += self.rng.normal(0, p.get("row_std", 0.), (clean.shape[0], 1))
        q = p.get("quant_step", 0.)
        noisy += self.rng.uniform(-q/2, q/2, clean.shape)
        # Optional pixel-fixed PRNU map is supplied explicitly as an array;
        # random per-image PRNU is deliberately not mislabeled fixed-pattern.
        if "fixed_pattern" in p:
            pattern = np.load(p["fixed_pattern"], allow_pickle=False)
            if pattern.shape != clean.shape:
                raise ValueError("Fixed pattern must match sensor crop shape")
            noisy += pattern
        noisy += p.get("black_offset", 0.)
        noisy = noisy.astype(np.float32)
        return (noisy, p) if return_params else noisy

    def noise_map(self, iso, signal=0.18, shape=None):
        p = self.parameters(iso)
        return noise_std(p, signal, shape)


def noise_std(p, signal=0.18, shape=None):
    var = p["K"]*np.maximum(signal, 0)+p.get("read_std", 0.)**2
    var = var+p.get("row_std", 0.)**2+p.get("col_std", 0.)**2+p.get("quant_step", 0.)**2/12
    std = np.sqrt(var).astype(np.float32)
    return np.broadcast_to(std, shape).copy() if shape is not None else std


def noise_map(iso, calibration, signal=0.18, shape=None):
    """Expected linear std, at 18% gray unless a signal image is supplied."""
    return NoiseModel(calibration).noise_map(iso, signal, shape)
