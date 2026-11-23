"""Data manifests contain only paths/ISO; camera data stays outside this repo."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from .bayer import read_mosaic, pack, unpack, augment_mosaic, random_crop
from .noise import NoiseModel, noise_std
from .bootstrap import add_bootstrap_noise


def _stream_seed(*key):
    return int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint64)[0]) % (2**63)


class SeededSamples(Dataset):
    """Isolate existing dataset RNG draws, keyed by absolute sample position.

    Dataset noise/augmentation formulas and direct callers retain their contracts.
    Only the CPU generator is seeded here; workers must never initialize CUDA.
    """
    def __init__(self, dataset, seed, stream):
        self.dataset, self.seed, self.stream = dataset, seed, stream

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        index, position = key
        seed = _stream_seed(self.seed, self.stream, 1, position)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


class ResumableBatches:
    """Infinite epoch stream; only *consumed* batches belong in checkpoints.

    Weighted draws use fixed-size blocks so restarting a long replacement sampler
    needs neither replaying its prefix nor materializing millions of indices.
    """
    def __init__(self, dataset, sampler, batch_size, seed, batches=0, stream=0, drop_last=False):
        if sampler is not None and not isinstance(sampler, WeightedRandomSampler):
            raise ValueError("Resumable training requires shuffle or weighted replacement sampling")
        if sampler is not None and not sampler.replacement:
            raise ValueError("Resumable weighted sampling requires replacement=True")
        self.size = sampler.num_samples if sampler is not None else len(dataset)
        self.weights = sampler.weights if sampler is not None else None
        self.batch_size, self.seed, self.batches, self.stream = batch_size, seed, batches, stream
        self.epoch_batches = self.size//batch_size if drop_last else (self.size+batch_size-1)//batch_size
        if not self.epoch_batches:
            raise ValueError("Empty training dataset or no complete training batch")

    def __len__(self):
        return self.epoch_batches

    def __iter__(self):
        batch = self.batches
        cached_epoch, cached_block, indices = None, None, None
        while True:
            epoch, within = divmod(batch, self.epoch_batches)
            start = within*self.batch_size
            keys = []
            for position in range(start, min(start+self.batch_size, self.size)):
                block = position//1024 if self.weights is not None else 0
                if epoch != cached_epoch or block != cached_block:
                    generator = torch.Generator().manual_seed(_stream_seed(self.seed, self.stream, 0, epoch, block))
                    indices = (torch.multinomial(self.weights, min(1024, self.size-block*1024), True,
                                                 generator=generator) if self.weights is not None else
                               torch.randperm(self.size, generator=generator))
                    cached_epoch, cached_block = epoch, block
                index = int(indices[position-block*1024 if self.weights is not None else position])
                keys.append((index, epoch*self.size+position))
            yield keys
            batch += 1


def resumable_loader(dataset, sampler, *, seed, batches=0, stream=0, batch_size=1,
                     num_workers=0, drop_last=False, pin_memory=False):
    return DataLoader(SeededSamples(dataset, seed, stream),
                      batch_sampler=ResumableBatches(dataset, sampler, batch_size, seed, batches, stream, drop_last),
                      num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
                      generator=torch.Generator().manual_seed(_stream_seed(seed, stream, 2)))


def load_manifest(path):
    with open(path) as f:
        entries = json.load(f)
    return entries


def _tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _support_mosaic(a, shape):
    a = np.asarray(a)
    if a.shape == (4, shape[0]//2, shape[1]//2):
        return unpack(a)
    if a.shape == (shape[0]//2, shape[1]//2):
        return a.repeat(2, 0).repeat(2, 1)
    if a.shape != shape:
        raise ValueError("Mask/support geometry must match the registered mosaics")
    return a


def _paired_valid(entry, root, noisy, clean, min_support):
    valid = np.isfinite(noisy) & np.isfinite(clean)
    if entry.get("valid_mask"):
        mask = np.load(root/entry["valid_mask"], allow_pickle=False)
        if mask.shape != noisy.shape:
            raise ValueError("valid_mask must have sensor-coordinate mosaic geometry")
        valid &= np.isfinite(mask) & (mask > 0)
    if entry.get("support"):
        support = np.load(root/entry["support"], allow_pickle=False)
        if isinstance(support, np.lib.npyio.NpzFile):
            with support:
                count = _support_mosaic(support["effective_count"], noisy.shape)
                valid &= np.isfinite(count) & (count >= min_support)
                for key in ("static", "confidence", "variance"):
                    if key in support:
                        a = _support_mosaic(support[key], noisy.shape)
                        valid &= np.isfinite(a) & (a >= 0 if key == "variance" else a > 0)
        else:
            count = _support_mosaic(support, noisy.shape)
            valid &= np.isfinite(count) & (count >= min_support)
    # A packed site is usable only if all four CFA samples are supported.
    packed = pack(valid).all(0)
    return unpack(np.broadcast_to(packed, (4, *packed.shape)))


def _valid_crop(pair, size, rng):
    valid = pack(pair[2]).all(0)
    h, w = valid.shape
    if min(h, w) < size:
        raise ValueError(f"Need at least {2*size}x{2*size} mosaic for a fully valid crop")
    for _ in range(64):
        y, x = int(rng.integers(h-size+1)), int(rng.integers(w-size+1))
        if valid[y:y+size, x:x+size].all():
            return pair[:, 2*y:2*(y+size), 2*x:2*(x+size)]
    # Exhaustive fallback avoids rejecting a rare but usable window by chance.
    integral = np.pad(valid, ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1)
    counts = integral[size:, size:]-integral[:-size, size:]-integral[size:, :-size]+integral[:-size, :-size]
    windows = np.flatnonzero(counts == size*size)
    if not len(windows):
        raise ValueError(f"No fully valid finite {size}x{size} packed training window")
    y, x = np.unravel_index(windows[int(rng.integers(len(windows)))], counts.shape)
    return pair[:, 2*y:2*(y+size), 2*x:2*(x+size)]


class PairedBurstDataset(Dataset):
    """JSON list: {noisy: relative/path.raw, clean: target.npy, iso: 1600}.

    Targets are normalized RGGB mosaics; noisy and clean must be registered and
    have matching geometry after Bayer unification. No alignment is performed.
    Optional valid_mask is a sensor-coordinate NPY; support is retained counts
    (sensor/packed NPY), or the aligned builder's packed support NPZ. Training
    requires fully valid crops. Full-frame evaluation fills nonfinite samples
    with zero and returns a four-plane valid_mask; consumers must use it.
    """
    def __init__(self, manifest, root=".", crop_size=512, augment=True, calibration=None,
                 seed=0, min_support=2):
        self.entries = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.root, self.crop_size, self.augment = Path(root), crop_size, augment
        self.noise = NoiseModel(calibration) if calibration else None
        if crop_size is not None and (isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size < 1):
            raise ValueError("crop_size must be a positive packed size or None")
        if not np.isfinite(min_support) or min_support < 2:
            raise ValueError("min_support must be at least two independent target frames")
        self.seed, self.min_support = seed, min_support

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        e = self.entries[index]
        noisy, clean = [read_mosaic(self.root/e[k]) for k in ("noisy", "clean")]
        if noisy.shape != clean.shape:
            raise ValueError("Paired mosaics must have equal registered dimensions")
        valid = _paired_valid(e, self.root, noisy, clean, self.min_support)
        if not valid.any():
            raise ValueError(f"Pair {index}: no valid finite supported samples")
        pair = np.stack((np.where(np.isfinite(noisy), noisy, 0),
                         np.where(np.isfinite(clean), clean, 0), valid))
        # Torch seeds are reset by callers and independently seeded per worker.
        rng = np.random.default_rng((int(torch.randint(0, 2**62, ()).item())+self.seed+index) % (2**63))
        if self.augment:
            pair = augment_mosaic(pair, *rng.integers(0, 2, 3).astype(bool))
        if self.crop_size:
            try:
                pair = _valid_crop(pair, self.crop_size, rng)
            except ValueError as exc:
                raise ValueError(f"Pair {index} ({e['noisy']}): {exc}") from exc
        inp, target, mask = pack(pair)
        iso = int(e.get("iso", 100))
        out = {"noisy": _tensor(inp), "clean": _tensor(target), "iso": iso,
               "valid_mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=bool)),
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
                 impulse_sigma_range=(3., 30.), impulse_positive=.6,
                 lag1_h_range=None, lag1_v_range=None, quantize_bits_choices=None):
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
        if lag1_h_range is not None:
            self.parameters["lag1_h_range"] = tuple(float(v) for v in lag1_h_range)
        if lag1_v_range is not None:
            self.parameters["lag1_v_range"] = tuple(float(v) for v in lag1_v_range)
        if quantize_bits_choices is not None:
            self.parameters["quantize_bits_choices"] = tuple(int(v) for v in quantize_bits_choices)

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
