"""Camera-calibrated noise synthesis from dark-frame profiles (calib/darks_*.yaml).

Per sample: pick a camera by weight, an ISO log-uniformly within its calibrated range (or, with
`processed_probability`, one of its in-camera-processed top ISOs), interpolate the profile in
log-ISO, jitter K (interim, flat-field-free estimate) and read noise, then synthesise:
shot (Poisson), read (Tukey-lambda, optional horizontal lag-1 correlation), row and column
offsets, black offset, hot pixels and the camera's quantisation (code step or Sony tone-curve
steps), clipped to the sensor's DN range. All values are normalised: (DN - black)/(white - black).
"""
import inspect

import numpy as np
import torch
import yaml

from .bayer import augment_mosaic, pack, random_crop, read_mosaic, unpack
from .clean_targets import BinnedBootstrapDataset, apply_channel_gains, apply_clip_highlight, bin_mosaic
from .data import BootstrapDataset, _tensor
from .defects import inject_defects
from .flat_augment import synthetic_smooth_mosaic
from .noise import noise_std, tukey_quantile
from .physics_noise import banding, correlate, moment_read, quantize_dn

# Sony ARW tone-curve quantisation steps (DN, black included), measured on A5100 and RX100 V files.
SONY_STEPS = ((3200, 4), (5708, 8), (8116, 16), (np.inf, 32))
INTERP_LOG = ("K", "read_std", "row_std", "col_std", "row_fixed_std", "col_fixed_std")
INTERP_LIN = ("tukey_lambda", "lag1_h", "hot_density", "black_offset")


_DIFF_MAD = {}


def _diff_mad_scale(lam):
    """Robust std of (T1-T2)/sqrt(2) for unit-MAD Tukey-lambda T, to match pair-difference calibration."""
    key = round(float(lam), 3)
    if key not in _DIFF_MAD:
        u = np.random.default_rng(12345).uniform(size=(2, 400000))
        t = tukey_quantile(u, key)/(1.4826*tukey_quantile(.75, key))
        d = (t[0]-t[1])/np.sqrt(2)
        _DIFF_MAD[key] = float(1.4826*np.median(np.abs(d-np.median(d))))
    return _DIFF_MAD[key]


def _corr_kernel(c):
    """Weight a of the [a, 1, a] filter whose output has lag-1 correlation c."""
    c = float(np.clip(c, 0, .45))
    return 0. if c < 1e-3 else (1-np.sqrt(1-2*c*c))/(2*c)


class CameraNoise:
    def __init__(self, calibration, cameras, K_jitter_log=.3, read_jitter_log=.1,
                 processed_probability=.1, iso_range=None, physics_options=None,
                 shutter_modes=None):
        with open(calibration) as f:
            cal = yaml.safe_load(f)
        if cal.get("units") != "normalized":
            raise ValueError("Calibration must declare units: normalized")
        self.cameras = {}
        self.physics_options = dict(physics_options or {})
        for name, weight in cameras.items():
            cam = cal["cameras"][name]
            shutter = (shutter_modes or {}).get(name)
            if shutter is not None:
                cam = {**cam, **cam["shutters"][shutter]}
            profiles = {int(k): v for k, v in cam["iso"].items()}
            processed = set(int(i) for i in cam.get("processed_isos", [])) & profiles.keys()
            normal = sorted(i for i in profiles if i not in processed)
            if iso_range:
                normal = [i for i in normal if iso_range[0] <= i <= iso_range[1]] or normal
            self.cameras[name] = {"weight": float(weight), "profiles": profiles, "isos": normal,
                                  "processed": sorted(processed), "quantization": cam.get("quantization"),
                                  "shutter": shutter or cam.get("shutter", name)}
        w = np.array([c["weight"] for c in self.cameras.values()])
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("Camera weights must be nonnegative with a positive sum")
        self.names, self.weights = list(self.cameras), w/w.sum()
        self.K_jitter_log, self.read_jitter_log = float(K_jitter_log), float(read_jitter_log)
        self.processed_probability = float(processed_probability)

    def sample(self, rng):
        name = self.names[int(rng.choice(len(self.names), p=self.weights))]
        cam = self.cameras[name]
        prof = cam["profiles"]
        if cam["processed"] and rng.random() < self.processed_probability:
            iso = int(rng.choice(cam["processed"]))
            p = {k: prof[iso][k] for k in INTERP_LOG+INTERP_LIN}
            p["processed"] = True
        else:
            isos = cam["isos"]
            iso = float(np.exp(rng.uniform(np.log(isos[0]), np.log(isos[-1]))))
            hi = min(max(1, int(np.searchsorted(isos, iso))), len(isos)-1)
            lo = hi-1
            t = (np.log(iso)-np.log(isos[lo]))/max(np.log(isos[hi])-np.log(isos[lo]), 1e-9)
            a, b = prof[isos[lo]], prof[isos[hi]]
            p = {k: float(np.exp((1-t)*np.log(max(a[k], 1e-12))+t*np.log(max(b[k], 1e-12)))) for k in INTERP_LOG}
            p.update({k: float((1-t)*a[k]+t*b[k]) for k in INTERP_LIN})
            p["processed"] = False
        ref = self._dn_profile(cam, iso, p["processed"])
        p["K"] *= float(np.exp(rng.normal(0, self.K_jitter_log)))
        read_multiplier = float(np.exp(rng.normal(0, self.read_jitter_log)))
        p["read_std"] *= read_multiplier
        p.update(camera=name, iso=iso, quantization=cam["quantization"], code_step=ref["code_step"],
                 black_dn=ref["black_dn"], white_dn=ref["white_dn"])
        self._physics_profile(p, cam, iso)
        if self.physics_options and "read_variance" in p["physics"]:
            p["physics"]["read_variance"] *= read_multiplier**2
        return p

    @staticmethod
    def _dn_profile(cam, iso, processed):
        """Discrete DN metadata: exact processed mode, otherwise nearest log-ISO.

        Normal ISOs clamp to the calibrated range; midpoint ties choose the lower
        ISO. Processed modes never enter normal interpolation.
        """
        if processed:
            return cam["profiles"][int(iso)]
        keys = cam["isos"]
        midpoints = np.sqrt(np.asarray(keys[:-1], float)*keys[1:])
        selected = int(np.searchsorted(midpoints, iso, side="left"))
        return cam["profiles"][keys[selected]]

    def _physics_profile(self, p, cam, iso):
        """Interpolate optional new keys without changing the legacy profile or RNG."""
        if not self.physics_options:
            return
        keys = sorted(cam["profiles"])
        hi = min(int(np.searchsorted(keys, iso)), len(keys)-1)
        lo = max(0, hi-1)
        t = np.clip((np.log(iso)-np.log(keys[lo]))/max(np.log(keys[hi]/keys[lo]), 1e-9), 0, 1)
        a, b = (cam["profiles"][keys[i]].get("physics", {}) for i in (lo, hi))
        # Numeric new keys interpolate linearly; zero endpoints remain zero.
        p["physics"] = {k: float((1-t)*a[k]+t*b[k]) for k in a.keys() & b.keys()
                        if isinstance(a[k], (int, float))}
        p["shutter"] = cam["shutter"]
        if self.physics_options.get("quantization"):
            p["physics"].update(black_dn=p["black_dn"], white_dn=p["white_dn"],
                                code_step_dn=p["code_step"]*(p["white_dn"]-p["black_dn"]))

    def at(self, name, iso, processed=False):
        """Deterministic interpolated profile (no jitter) for evaluation at a fixed ISO."""
        cam = self.cameras[name]
        prof = cam["profiles"]
        if processed:
            if iso not in cam["processed"]:
                raise ValueError(f"ISO {iso} is not a calibrated processed mode")
            p = {k: prof[int(iso)][k] for k in INTERP_LOG+INTERP_LIN}
        else:
            isos = cam["isos"]
            iso = float(np.clip(iso, isos[0], isos[-1]))
            hi = min(max(1, int(np.searchsorted(isos, iso))), len(isos)-1)
            lo = hi-1
            t = (np.log(iso)-np.log(isos[lo]))/max(np.log(isos[hi])-np.log(isos[lo]), 1e-9)
            a, b = prof[isos[lo]], prof[isos[hi]]
            p = {k: float(np.exp((1-t)*np.log(max(a[k], 1e-12))+t*np.log(max(b[k], 1e-12)))) for k in INTERP_LOG}
            p.update({k: float((1-t)*a[k]+t*b[k]) for k in INTERP_LIN})
        ref = self._dn_profile(cam, iso, processed)
        p.update(processed=processed, camera=name, iso=iso, quantization=cam["quantization"],
                 code_step=ref["code_step"], black_dn=ref["black_dn"], white_dn=ref["white_dn"])
        self._physics_profile(p, cam, iso)
        return p

    def apply(self, clean, p, rng, clip_ceiling=None):
        clean = np.asarray(clean, np.float64)
        if clean.ndim != 2 or clean.shape[0] % 2 or clean.shape[1] % 2:
            raise ValueError("Expected an even-sized RGGB mosaic")
        positive = np.maximum(clean, 0)
        noisy = rng.poisson(positive/p["K"])*p["K"]+(clean-positive)
        h, w = clean.shape
        lam = p["tukey_lambda"]
        opts = self.physics_options
        phys = {**p, **p.get("physics", {})}
        # Profiles store the robust (MAD) std of frame-pair differences/sqrt(2); scale to reproduce it.
        if opts.get("moment_read"):
            variance = phys.get("read_variance", p["read_std"]**2)
            read = moment_read((4, h//2, w//2), variance, phys.get("read_excess_kurtosis", 0.), rng)
            read_scale = 1.
        else:
            read = tukey_quantile(rng.uniform(size=(4, h//2, w//2)), lam)/(1.4826*tukey_quantile(.75, lam)*_diff_mad_scale(lam))
            read_scale = p["read_std"]
        if opts.get("horizontal_correlation"):
            read = correlate(read, phys["lag1_h"])
        else:
            a = _corr_kernel(p["lag1_h"])
            if a:
                read = (read+a*np.roll(read, 1, -1)+a*np.roll(read, -1, -1))/np.sqrt(1+2*a*a)
        if opts.get("vertical_correlation"):
            read = correlate(read, phys.get("lag1_v", 0.), axis=-2)
        noisy += read_scale*unpack(read)
        row = np.hypot(p["row_std"], p["row_fixed_std"])
        col = np.hypot(p["col_std"], p["col_fixed_std"])
        if opts.get("banding"):
            noisy += banding(clean.shape, phys, rng, f"{p['camera']}:{p.get('shutter', '')}", opts.get("fixed_seed", 0))
        else:
            noisy += rng.normal(0, row, (h, 1))+rng.normal(0, col, (1, w))
        noisy += p["black_offset"]
        if not opts.get("moment_read") and not p["processed"] and p["hot_density"] > 0:
            hit = rng.random((h, w)) < p["hot_density"]*float(np.exp(rng.uniform(np.log(.5), np.log(2.))))
            n = int(hit.sum())
            if n:
                noisy[hit] += np.exp(rng.uniform(np.log(8.), np.log(60.), n))*p["read_std"]
        span = p["white_dn"]-p["black_dn"]
        if opts.get("quantization"):
            if p["quantization"] == "sony_curve":
                phys = {**phys, "quantization_steps_dn": SONY_STEPS}
            return quantize_dn(noisy, phys, clip_ceiling=clip_ceiling).astype(np.float32)
        dn = noisy*span+p["black_dn"]
        raw_upper_dn = p["white_dn"] if clip_ceiling is None else p["black_dn"] + float(clip_ceiling) * span
        if p["quantization"] == "sony_curve":
            step = np.select([dn < lim for lim, _ in SONY_STEPS], [s for _, s in SONY_STEPS])
            upper_step = float(np.select([raw_upper_dn < lim for lim, _ in SONY_STEPS], [s for _, s in SONY_STEPS]))
        else:
            step = max(p["code_step"]*span, 1e-9)
            upper_step = step
        dn = np.round(dn/step)*step
        # Snap saturation ceiling down on the ceiling's own curve segment so one saturation code is emitted.
        upper_dn = p["white_dn"] if clip_ceiling is None else float(np.floor(raw_upper_dn / upper_step) * upper_step)
        dn = np.clip(dn, 0, upper_dn)
        return ((dn-p["black_dn"])/span).astype(np.float32)


class CameraNoiseDataset(BinnedBootstrapDataset):
    """Binned/flat-augmented crops with camera-calibrated noise instead of a generic prior.

    Exposure follows the parent (log-uniform plus a bright fraction). For cameras whose raw gain
    stops at `gain_cap_iso` (X1D II), the raw signal is additionally darkened by cap/ISO.
    """
    @staticmethod
    def validate_options(options):
        """Reject inherited settings unused by this dataset before any file IO."""
        unused = {"resite", "iso_coupled", "noise_map_jitter", "shot_range", "read_range",
                  "row_max", "fixed_parameters", "column_relative_max", "impulse_range",
                  "impulse_sigma_range", "impulse_positive",
                  "lag1_h_range", "lag1_v_range", "quantize_bits_choices"}
        defaults = {}
        for cls in CameraNoiseDataset.__mro__[1:]:
            if "__init__" in cls.__dict__:
                defaults.update({k: p.default for k, p in inspect.signature(cls.__init__).parameters.items()
                                 if k in unused})
        for key in unused & options.keys():
            value, default = options[key], defaults[key]
            if isinstance(default, tuple) and isinstance(value, (list, tuple)):
                value = tuple(value)
            if value != default:
                raise ValueError(f"CameraNoiseDataset does not use {key}; leave it at {default!r}")

    def __init__(self, *args, calibration=None, cameras=None, camera_options=None, gain_cap_iso=None,
                 zero_identity_noise_map=False, **kwargs):
        positional = inspect.signature(BootstrapDataset.__init__).bind_partial(None, *args).arguments
        self.validate_options({**positional, **kwargs})
        super().__init__(*args, **kwargs)
        self.zero_identity_noise_map = bool(zero_identity_noise_map)
        if calibration is None or not cameras:
            raise ValueError("camera noise needs a calibration file and camera weights")
        self.camera_noise = CameraNoise(calibration, cameras, **(camera_options or {}))
        self.gain_cap_iso = dict(gain_cap_iso or {})

    def __getitem__(self, index):
        seed = self.seed+index if self.deterministic else int(torch.randint(0, 2**62, ()).item())
        rng = np.random.default_rng(seed)
        clean = read_mosaic(self.root/self.entries[index]["clean"])
        factor = 1
        if self.bin_probability > 0 and rng.random() < self.bin_probability:
            factor = self.bin_factors[int(rng.integers(len(self.bin_factors)))]
            clean = bin_mosaic(clean, factor)
        if self.augment:
            clean = augment_mosaic(clean, *rng.integers(0, 2, 3).astype(bool))
        clean = random_crop(clean, self.crop_size, rng)
        if self.augment and self.channel_gain_probability > 0 and rng.random() < self.channel_gain_probability:
            clean = apply_channel_gains(clean, rng, self.channel_gain_range)
        if self.augment and self.flat_probability > 0 and rng.random() < self.flat_probability:
            clean = synthetic_smooth_mosaic(*clean.shape, rng, **self.flat_options)
        p = self.camera_noise.sample(rng)
        clip_ceiling = None
        if self.augment and self.clip_highlight_probability > 0 and rng.random() < self.clip_highlight_probability:
            signal_for_noise, clean, exposure, clip_ceiling = apply_clip_highlight(
                clean, self.exposure_scale, rng, self.clip_exposure_range, self.clip_ceiling_range
            )
        else:
            exposure = self.exposure_scale
            if self.augment:
                lo, hi = self.exposure_range
                if self.bright_exposure_probability > 0 and rng.random() < self.bright_exposure_probability:
                    exposure *= float(rng.uniform(.6, 1.))
                else:
                    exposure *= float(np.exp(rng.uniform(np.log(lo), np.log(hi)))
                                      if self.exposure_sampling == "log" else rng.uniform(lo, hi))
            cap = self.gain_cap_iso.get(p["camera"])
            if cap and p["iso"] > cap:
                exposure *= cap/p["iso"]
            clean = (clean*exposure).astype(np.float32)
            signal_for_noise = clean
        identity = self.augment and rng.random() < self.identity_probability
        if identity:
            noisy = clean.copy()
        else:
            noisy = self.camera_noise.apply(signal_for_noise, p, rng, clip_ceiling=clip_ceiling)
        if self.augment and self.defect_probability > 0 and rng.random() < self.defect_probability:
            noisy, _ = inject_defects(noisy, rng, **self.defect_options)
        inp, target = pack(noisy), pack(clean)
        parameters = {"K": 0., "read_std": 0.} if identity and self.zero_identity_noise_map else p
        std = noise_std({k: parameters[k] for k in ("K", "read_std")}, np.maximum(inp, 0).mean(0, keepdims=True))
        return {"noisy": _tensor(inp), "clean": _tensor(target), "noise_map": _tensor(std),
                "exposure_scale": exposure, "bin_factor": factor,
                "clip_ceiling": float(clip_ceiling if clip_ceiling is not None else 1.0)}
