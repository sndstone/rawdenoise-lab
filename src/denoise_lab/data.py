"""Data manifests contain only paths/ISO; camera data stays outside this repo."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from .bayer import read_mosaic, pack, augment_mosaic, random_crop
from .noise import NoiseModel, noise_std
from .bootstrap import add_bootstrap_noise


def load_manifest(path):
    with open(path) as f:
        entries = json.load(f)
    return entries


def _tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


class PairedBurstDataset(Dataset):
    """JSON list: {noisy: relative/path.raw, clean: target.npy, iso: 1600}.

    Targets are normalized RGGB mosaics; noisy and clean must be registered and
    have matching geometry after Bayer unification. No alignment is performed.
    """
    def __init__(self, manifest, root=".", crop_size=512, augment=True, calibration=None):
        self.entries = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.root, self.crop_size, self.augment = Path(root), crop_size, augment
        self.noise = NoiseModel(calibration) if calibration else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        e = self.entries[index]
        noisy, clean = [read_mosaic(self.root/e[k]) for k in ("noisy", "clean")]
        if noisy.shape != clean.shape:
            raise ValueError("Paired mosaics must have equal registered dimensions")
        pair = np.stack((noisy, clean))
        rng = np.random.default_rng()
        if self.augment:
            pair = augment_mosaic(pair, *rng.integers(0, 2, 3).astype(bool))
        if self.crop_size:
            pair = random_crop(pair, self.crop_size, rng)
        inp, target = pack(pair)
        iso = int(e.get("iso", 100))
        out = {"noisy": _tensor(inp), "clean": _tensor(target), "iso": iso,
               "noise_map": torch.zeros((1, *inp.shape[-2:]), dtype=torch.float32)}
        if self.noise:
            out["noise_map"] = _tensor(self.noise.noise_map(iso, np.maximum(inp, 0).mean(0, keepdims=True)))
        return out


class SyntheticDataset(Dataset):
    """Clean manifest: [{clean: low_iso.raw, iso: target_noise_iso}, ...].

    random_data is a pipeline smoke fixture, never a camera-quality training set.
    """
    def __init__(self, calibration, manifest=None, root=".", crop_size=512,
                 augment=True, length=100, random_data=False, iso=1600, iso_range=None, seed=0):
        self.entries = load_manifest(manifest) if manifest else None
        if not random_data and not self.entries:
            raise ValueError("A clean raw manifest is required unless random_data is true")
        self.root, self.crop_size, self.augment = Path(root), crop_size, augment
        self.length, self.random_data, self.iso, self.iso_range = length, random_data, iso, iso_range
        self.noise, self.seed = NoiseModel(calibration), seed

    def __len__(self):
        return len(self.entries) if self.entries else self.length

    def __getitem__(self, index):
        # Torch worker seed gives independent reproducible streams with workers.
        rng = np.random.default_rng((int(torch.randint(0, 2**62, ()).item())+self.seed+index) % (2**63))
        self.noise.rng = rng
        if self.random_data:
            small = rng.uniform(0.01, 0.8, (8, 8)).astype(np.float32)
            size = self.crop_size*2
            clean = np.repeat(np.repeat(small, (size+7)//8, 0), (size+7)//8, 1)[:size, :size]
            iso = self.iso
        else:
            entry = self.entries[index]
            clean, iso = read_mosaic(self.root/entry["clean"]), int(entry.get("iso", self.iso))
            if self.augment:
                clean = augment_mosaic(clean, *rng.integers(0, 2, 3).astype(bool))
            clean = random_crop(clean, self.crop_size, rng)
        noisy, p = self.noise(clean, iso, iso_range=self.iso_range, return_params=True)
        inp, target = pack(noisy), pack(clean)
        std = noise_std(p, np.maximum(inp, 0).mean(0, keepdims=True))
        return {"noisy": _tensor(inp), "clean": _tensor(target), "noise_map": _tensor(std), "iso": iso}


class BootstrapDataset(Dataset):
    """Low-ISO single-frame targets plus deliberately uncalibrated simulated noise.

    Suitable for provisional pretraining. These are not verified clean references,
    and the generic shot/read ranges are not assigned to real camera ISO settings.
    """
    def __init__(self, manifest, root="data", crop_size=128, augment=True, seed=0,
                 deterministic=False, shot_range=(1e-5, 1e-3), read_range=(1e-4, .004),
                 row_max=.0005, fixed_parameters=None, identity_probability=0.,
                 exposure_range=(.6, 1.), exposure_sampling="linear", exposure_scale=1.,
                 bright_exposure_probability=0., column_relative_max=0., impulse_range=None,
                 impulse_sigma_range=(3., 30.), impulse_positive=.6):
        self.entries = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        if not self.entries:
            raise ValueError("Bootstrap training requires cached source crops")
        self.root, self.crop_size, self.augment = Path(root), crop_size, augment
        self.seed, self.deterministic = seed, deterministic
        if not 0 <= identity_probability <= 1:
            raise ValueError("identity_probability must be between zero and one")
        self.identity_probability = identity_probability
        lo, hi = exposure_range
        if not 0 < lo <= hi <= 1 or not 0 < exposure_scale <= 1:
            raise ValueError("Exposure attenuation must be in (0, 1]")
        if exposure_sampling not in ("linear", "log"):
            raise ValueError("exposure_sampling must be linear or log")
        self.exposure_range = (float(lo), float(hi))
        self.exposure_sampling, self.exposure_scale = exposure_sampling, float(exposure_scale)
        if not 0 <= bright_exposure_probability <= 1:
            raise ValueError("bright_exposure_probability must be between zero and one")
        self.bright_exposure_probability = float(bright_exposure_probability)
        self.parameters = dict(shot_range=shot_range, read_range=read_range,
                               row_max=row_max, fixed_parameters=fixed_parameters)
        # Structural components stay out of the call unless enabled, so legacy
        # configurations draw exactly the original random stream.
        if column_relative_max:
            self.parameters["column_relative_max"] = float(column_relative_max)
        if impulse_range:
            self.parameters.update(impulse_range=tuple(impulse_range),
                                   impulse_sigma_range=tuple(impulse_sigma_range),
                                   impulse_positive=float(impulse_positive))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        seed = self.seed+index if self.deterministic else int(torch.randint(0, 2**62, ()).item())
        rng = np.random.default_rng(seed)
        clean = read_mosaic(self.root/self.entries[index]["clean"])
        if self.augment:
            clean = augment_mosaic(clean, *rng.integers(0, 2, 3).astype(bool))
        clean = random_crop(clean, self.crop_size, rng)
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
        inp, target = pack(noisy), pack(clean)
        std = noise_std(parameters, np.maximum(inp, 0).mean(0, keepdims=True))
        return {"noisy": _tensor(inp), "clean": _tensor(target), "noise_map": _tensor(std),
                "exposure_scale": exposure}


def mixed_sampler(datasets, probabilities=None, num_samples=None):
    """Return concatenated dataset and weighted replacement sampler.

    Paired datasets should supply calibration when training a conditioned model;
    uncalibrated paired records carry a zero (unknown) conditioning map.
    """
    probabilities = np.asarray(probabilities if probabilities is not None else np.ones(len(datasets)), float)
    if len(probabilities) != len(datasets) or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError("One nonnegative mixing weight per dataset is required")
    if any(len(d) == 0 for d in datasets):
        raise ValueError("Cannot mix empty datasets")
    weights = np.concatenate([np.full(len(d), p/len(d)) for d, p in zip(datasets, probabilities)])
    combined = ConcatDataset(datasets)
    return combined, WeightedRandomSampler(torch.as_tensor(weights), num_samples or len(combined), replacement=True)
