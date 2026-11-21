"""Opt-in dark-frame components; stds are normalized, quantisation steps are DN.

Band offsets are independent per CFA plane. Fixed patterns use a separate seeded
RNG, keyed by camera/shutter and crop shape, and persist across frames and workers.
They model band power, not a measured full-sensor shading map or crop coordinates.
"""
import hashlib
import numpy as np

from .bayer import unpack


def banding(shape, p, rng, key="sensor", seed=0):
    h, w = shape
    if h % 2 or w % 2:
        raise ValueError("Banding requires an even-sized Bayer mosaic")
    token = f"{seed}:{key}:{h}:{w}".encode()
    fixed_rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
    row = fixed_rng.normal(size=(4, h//2, 1))*p.get("row_fixed_std", 0.)
    col = fixed_rng.normal(size=(4, 1, w//2))*p.get("col_fixed_std", 0.)
    row += rng.normal(size=row.shape)*p.get("row_std", 0.)
    col += rng.normal(size=col.shape)*p.get("col_std", 0.)
    return unpack(row+col)


def correlate(read, correlation, axis=-1):
    """Variance-preserving same-colour [a,1,a] filter, with reflected boundaries.

    Supports signed lag-1 correlations up to 0.45; it approximates compression
    covariance only, without claiming to reproduce a camera's codec.
    """
    c = float(np.clip(correlation, -.45, .45))
    if abs(c) < 1e-6:
        return read
    a = (1-np.sqrt(1-2*c*c))/(2*c)
    pads = [(0, 0)]*read.ndim
    pads[axis] = (1, 1)
    padded = np.pad(read, pads, mode="reflect")
    n = read.shape[axis]
    return (read+a*np.take(padded, np.arange(n), axis=axis)
            +a*np.take(padded, np.arange(n)+2, axis=axis))/np.sqrt(1+2*a*a)


def moment_read(shape, variance, excess, rng):
    """Zero-mean Gaussian scale mixture matching variance and nonnegative kurtosis.

    Finite fourth moments avoid unstable extrapolation of Tukey fits below -0.25.
    Mixture frequency is a modelling choice, not identifiable from four moments.
    """
    excess = max(0., float(excess))
    probability = min(.01, 1/(2+2*excess/3))
    delta = np.sqrt(excess/(3*probability*(1-probability)))
    low = max(0., 1-probability*delta)
    high = low+delta
    scales = np.where(rng.random(shape) < probability, high, low)
    return rng.normal(size=shape)*np.sqrt(max(0., variance)*scales)


def quantize_dn(noisy, p):
    """Round absolute DN codes before normalization; preserve signed black residuals."""
    span = p["white_dn"]-p["black_dn"]
    if span <= 0:
        raise ValueError("white_dn must exceed black_dn")
    step = p.get("code_step_dn")
    if step is None:
        bits = p.get("effective_bits")
        step = 2.**(p.get("sensor_bits", int(np.ceil(np.log2(p["white_dn"]+1))))-bits) if bits else p.get("code_step", 0.)*span
    if step <= 0:
        raise ValueError("Quantisation needs a positive code_step_dn or effective_bits")
    dn = noisy*span+p["black_dn"]
    if p.get("quantization_steps_dn"):
        curve = p["quantization_steps_dn"]
        step = np.select([dn < limit for limit, _ in curve], [value for _, value in curve])
    dn = np.clip(np.round(dn/step)*step, 0, np.floor(p["white_dn"]/step)*step)
    return (dn-p["black_dn"])/span


def dark_statistics(frames):
    """Comparable statistics from N>=2 packed darks (N,4,H,W).

    Variance/kurtosis and lag-1 are on row/column-detrended disjoint pair
    differences / sqrt(2). Band powers subtract finite-width pixel leakage;
    fixed band powers also subtract temporal band power / N. Spatial moments
    include hot pixels and shading and are reported separately as diagnostics.
    """
    st = np.asarray(frames, np.float64)
    if st.ndim != 4 or st.shape[1] != 4 or len(st) < 2 or min(st.shape[-2:]) < 4:
        raise ValueError("Need at least two packed dark frames with H,W >= 4")
    n, _, h, w = st.shape
    d = (st[:n//2*2:2]-st[1:n//2*2:2])/np.sqrt(2)
    row = d.mean(-1, keepdims=True)
    col = d.mean(-2, keepdims=True)
    pix = d-row-col+d.mean((-1, -2), keepdims=True)
    variance = float(np.mean(pix**2))*h*w/((h-1)*(w-1))
    kurt = float(np.mean(pix**4)/max(float(np.mean(pix**2))**2, 1e-30)-3)
    hh = float(np.mean(pix[..., :, 1:]*pix[..., :, :-1])/max(variance, 1e-30))
    vv = float(np.mean(pix[..., 1:, :]*pix[..., :-1, :])/max(variance, 1e-30))
    # The 3-tap model has a small lag-2 term; include it in mean-noise leakage.
    h2 = float(np.mean(pix[..., :, 2:]*pix[..., :, :-2])/max(variance, 1e-30))
    v2 = float(np.mean(pix[..., 2:, :]*pix[..., :-2, :])/max(variance, 1e-30))
    row_leak = variance*max(0., 1+2*hh+2*h2)/w
    col_leak = variance*max(0., 1+2*vv+2*v2)/h
    row_power = max(0., float(np.var(row, axis=-2).mean())-row_leak)
    col_power = max(0., float(np.var(col, axis=-1).mean())-col_leak)
    mean = st.mean(0)
    row_fixed = max(0., float(np.var(mean.mean(-1), axis=-1).mean())-(row_power+row_leak)/n)
    col_fixed = max(0., float(np.var(mean.mean(-2), axis=-1).mean())-(col_power+col_leak)/n)
    spatial = st-st.mean((-1, -2), keepdims=True)
    sv = float(np.mean(spatial**2))
    return {"variance": variance, "excess_kurtosis": kurt, "lag1_h": hh, "lag1_v": vv,
            "row_power": row_power, "col_power": col_power,
            "row_fixed_power": row_fixed, "col_fixed_power": col_fixed,
            "spatial_variance": sv, "spatial_excess_kurtosis": float(np.mean(spatial**4)/max(sv*sv, 1e-30)-3)}


def fit_physics(frames, step_dn, black, white):
    """Moment fit from dark pairs; K is deliberately not inferred from darks."""
    stats = dark_statistics(frames)
    qvar = (step_dn/(white-black))**2/12
    variance = max(0., stats["variance"]-qvar)
    # Pair-difference excess is half single-frame excess. Uniform rounding has
    # fourth cumulant -q^4/120; approximation is weak when quantisation dominates.
    excess = max(0., (2*stats["excess_kurtosis"]*stats["variance"]**2+1.2*qvar*qvar)/max(variance**2, 1e-30))
    return {"read_variance": variance, "read_excess_kurtosis": excess,
            "lag1_h": stats["lag1_h"], "lag1_v": stats["lag1_v"],
            "row_std": np.sqrt(stats["row_power"]).item(), "col_std": np.sqrt(stats["col_power"]).item(),
            "row_fixed_std": np.sqrt(stats["row_fixed_power"]).item(),
            "col_fixed_std": np.sqrt(stats["col_fixed_power"]).item(),
            "code_step_dn": float(step_dn), "black_dn": float(black), "white_dn": float(white),
            "statistics": stats}
