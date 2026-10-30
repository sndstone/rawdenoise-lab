"""Provisional low-ISO single-frame pretraining, with explicitly uncalibrated noise."""
import hashlib
import numpy as np


def capture_group(camera, capture_time, source_path):
    day = str(capture_time or "")[:10]
    if len(day) != 10 or day[4:5] != ":" or day[7:8] != ":":
        day = source_path.rsplit("\\", 1)[0]
    return f"{camera}|{day}"


def split_for_group(group):
    # Whole capture days remain together, including bursts and backup copies.
    return "val" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10 < 2 else "train"


def screen_crop(mosaic):
    """Conservative exposure screening, not a certification of noiselessness/focus."""
    green = np.asarray(mosaic[0::2, 1::2], dtype=np.float32)
    if not np.isfinite(mosaic).all():
        return None
    median = float(np.median(green))
    dark = float(np.mean(green < .01))
    bright = float(np.mean(mosaic >= .98))
    if not .03 <= median <= .7 or dark > .15 or bright > .001:
        return None
    gradient = float(np.median(np.abs(np.diff(green, axis=0))) +
                     np.median(np.abs(np.diff(green, axis=1))))
    return {"median_green": median, "dark_fraction": dark,
            "near_white_fraction": bright, "gradient_proxy": gradient}


def add_bootstrap_noise(clean, rng, shot_range=(1e-5, 1e-3), read_range=(1e-4, .004),
                        row_max=.0005, fixed_parameters=None, column_relative_max=0.,
                        impulse_range=None, impulse_sigma_range=(3., 30.), impulse_positive=.6):
    """A broad shot/read/row prior; parameters are not measured camera/ISO profiles.

    Optional structural components (off by default, preserving the original
    random stream): per-column offsets with std U(0, column_relative_max)*read,
    and sparse impulses with per-pixel probability log-U(impulse_range) and
    amplitude log-U(impulse_sigma_range)*read (positive with impulse_positive).
    Their envelopes are uncalibrated; motivated by heavy-tailed optical-black
    statistics, not fitted to any third-party output.
    """
    if fixed_parameters is None:
        K = float(np.exp(rng.uniform(*np.log(shot_range))))
        read = float(np.exp(rng.uniform(*np.log(read_range))))
        row = float(rng.uniform(0, row_max))
        col = float(rng.uniform(0, column_relative_max))*read if column_relative_max else 0.
        p_imp = float(np.exp(rng.uniform(*np.log(impulse_range)))) if impulse_range else 0.
    else:
        K, read, row = [float(fixed_parameters[key]) for key in ("K", "read_std", "row_std")]
        col = float(fixed_parameters.get("col_std", 0.))
        p_imp = float(fixed_parameters.get("impulse_p", 0.))
    positive = np.maximum(clean, 0)
    noisy = rng.poisson(positive/K)*K + clean-positive
    noisy += rng.normal(0, read, clean.shape)
    noisy += rng.normal(0, row, (clean.shape[0], 1))
    if col:
        noisy += rng.normal(0, col, (1, clean.shape[1]))
    if p_imp:
        hit = rng.random(clean.shape) < p_imp
        n = int(hit.sum())
        amp = np.exp(rng.uniform(*np.log(impulse_sigma_range), n))*read
        sign = np.where(rng.random(n) < impulse_positive, 1., -1.)
        noisy[hit] += amp*sign
    parameters = {"K": K, "read_std": read, "row_std": row}
    if column_relative_max or col:
        parameters["col_std"] = col
    if impulse_range or p_imp:
        parameters["impulse_p"] = p_imp
    return noisy.astype(np.float32), parameters
