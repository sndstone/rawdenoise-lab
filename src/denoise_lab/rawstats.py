"""Single-image RAW statistics for audits and noise-map estimation (numpy only).

Everything here is a *diagnostic*. Single-frame spatial statistics confound
temporal noise with texture, PRNU, shading and fixed pattern; nothing in this
module is a photon-transfer calibration. Optical-black (OB) statistics mix
temporal read noise with dark fixed pattern because only one frame exists.
"""
import numpy as np

GREEN_PAIR = (1, 3)


def box_mean(a, k):
    """Same-size box mean over the last two axes with edge replication."""
    if k <= 1:
        return np.asarray(a, np.float64)
    p = k//2
    a = np.pad(np.asarray(a, np.float64), [(0, 0)]*(a.ndim-2)+[(p, k-1-p), (p, k-1-p)], mode="edge")
    c = a.cumsum(-2).cumsum(-1)
    c = np.pad(c, [(0, 0)]*(a.ndim-2)+[(1, 0), (1, 0)])
    h, w = a.shape[-2]-k+1, a.shape[-1]-k+1
    s = c[..., k:k+h, k:k+w]-c[..., :h, k:k+w]-c[..., k:k+h, :w]+c[..., :h, :w]
    return s/(k*k)


def median3(a):
    """3x3 median over the last two axes (edge replicated), row-chunked."""
    a = np.asarray(a, np.float32)
    flat = a.reshape(-1, *a.shape[-2:])
    out = np.empty_like(flat)
    for i, plane in enumerate(flat):
        p = np.pad(plane, 1, mode="edge")
        h, w = plane.shape
        for y0 in range(0, h, 512):
            y1 = min(h, y0+512)
            stack = np.stack([p[y0+dy:y1+dy, dx:dx+w] for dy in range(3) for dx in range(3)])
            out[i, y0:y1] = np.partition(stack, 4, axis=0)[4]
    return out.reshape(a.shape)


def robust_std(x, axis=None):
    x = np.asarray(x, np.float64)
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826*np.median(np.abs(x-med), axis=axis)


def _blocks(a, b):
    h, w = (a.shape[-2]//b)*b, (a.shape[-1]//b)*b
    return a[..., :h, :w].reshape(*a.shape[:-2], h//b, b, w//b, b).swapaxes(-3, -2).reshape(
        *a.shape[:-2], h//b, w//b, b*b)


def green_pair_blocks(packed, block=16):
    """Per-block signal and robust variance of (G1-G2)/sqrt(2).

    G1/G2 sit on different rows and columns, so the difference also carries
    row/column offset differences, green imbalance and Nyquist texture.
    """
    packed = np.asarray(packed, np.float32)
    d = (packed[GREEN_PAIR[0]]-packed[GREEN_PAIR[1]])/np.sqrt(2)
    mu = _blocks((packed[GREEN_PAIR[0]]+packed[GREEN_PAIR[1]])/2, block).mean(-1)
    var = robust_std(_blocks(d, block), axis=-1)**2
    return mu.ravel(), var.ravel()


def fit_nlf(mu, var, quantile=.25, bins=24, min_blocks=20, max_signal=.9):
    """Fit var = K*mu + c on per-bin variance quantiles (non-negative LS).

    Low quantiles suppress texture inflation but are biased low by sampling
    scatter; report several quantiles as a bracket, not as a calibration.
    """
    mu, var = np.asarray(mu, float), np.asarray(var, float)
    keep = np.isfinite(mu) & np.isfinite(var)
    mu, var = mu[keep], var[keep]
    keep = mu < max_signal
    mu, var = mu[keep], var[keep]
    if len(mu) < min_blocks:
        return {"K": 0., "c": float(np.quantile(var, quantile)) if len(var) else 0., "bins": 0, "points": []}
    # Equal-count bins: real images are dominated by a few signal levels.
    edges = np.unique(np.quantile(mu, np.linspace(.005, .995, bins+1)))
    xs, ys, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (mu >= a) & (mu < b)
        if sel.sum() >= min_blocks:
            xs.append(np.median(mu[sel]))
            ys.append(np.quantile(var[sel], quantile))
            ns.append(int(sel.sum()))
    if len(xs) < 2:
        c = float(np.quantile(var, quantile)) if len(var) else 0.
        return {"K": 0., "c": max(c, 0.), "bins": len(xs), "points": list(zip(xs, ys, ns))}
    x, y, n = map(np.asarray, (xs, ys, ns))
    wts = np.sqrt(n)/np.maximum(y, 1e-14)
    A = np.stack([np.maximum(x, 0), np.ones_like(x)], 1)*wts[:, None]
    K, c = np.linalg.lstsq(A, y*wts, rcond=None)[0]
    if K < 0:
        K, c = 0., float(np.average(y, weights=wts**2))
    if c < 0:
        c = 0.
        K = float(max(0., np.sum(wts**2*np.maximum(x, 0)*y)/max(np.sum(wts**2*np.maximum(x, 0)**2), 1e-30)))
    return {"K": float(K), "c": float(c), "bins": len(xs),
            "points": [[float(a), float(b), int(m)] for a, b, m in zip(x, y, n)]}


def estimate_noise_parameters(packed, block=None, quantile=.5):
    """Blind estimate of {K, read_std} from one noisy packed image.

    read_std absorbs row/column components visible in the green pair.
    This is the 'estimated map' path; it is not a calibration. Small crops
    use 8x8 blocks and fewer bins so every bin keeps >= 40 blocks.
    """
    if block is None:
        block = 8 if min(packed.shape[-2:]) < 512 else 16
    mu, var = green_pair_blocks(packed, block)
    bins = int(np.clip(len(mu)//40, 2, 24))
    fit = fit_nlf(mu, var, quantile=quantile, bins=bins, min_blocks=min(20, max(5, len(mu)//(2*bins))))
    return {"K": fit["K"], "read_std": float(np.sqrt(fit["c"])), "row_std": 0.}


def line_structure(packed, axis, highpass=31):
    """Excess line (row axis=-1 mean / column axis=-2 mean) structure.

    Each plane is high-passed by subtracting a box mean, then averaged along
    the line. Returns the ratio of observed line-profile std to the value
    expected if pixels were independent (1 = no line structure).
    """
    packed = np.asarray(packed, np.float64)
    hp = packed-box_mean(packed, highpass)
    prof = hp.mean(axis=axis)
    n = packed.shape[axis]
    px = robust_std(hp.reshape(packed.shape[0], -1), axis=-1)
    obs = robust_std(prof, axis=-1)
    expected = px/np.sqrt(n)
    return {"ratio": (obs/np.maximum(expected, 1e-30)).tolist(), "profile_std": obs.tolist(),
            "pixel_std": px.tolist()}


def impulses(packed, sigma, threshold=8.):
    """Pixels deviating from their same-channel 3x3 median by > threshold*sigma.

    sigma may be scalar or broadcastable to packed (e.g. a noise map).
    """
    packed = np.asarray(packed, np.float32)
    dev = packed-median3(packed)
    s = np.maximum(np.asarray(sigma, np.float32), 1e-12)
    return dev > threshold*s, dev < -threshold*s


def block_mean_shift(inp, out, block=32, noise_var=None, clip=.98):
    """Per-channel block-mean difference out-in, its z-score and a validity mask.

    The z-score uses noise_var(mean) (e.g. an NLF bracket) when given, else a
    neighbour-difference scatter; blocks containing clipped input are invalid.
    """
    inp, out = np.asarray(inp, np.float64), np.asarray(out, np.float64)
    bi, bo = _blocks(inp, block), _blocks(out, block)
    diff = (bo-bi).mean(-1)
    mu = bi.mean(-1)
    if noise_var is not None:
        var = noise_var(mu)
    else:
        var = (robust_std(np.diff(bi, axis=-1), axis=-1)/np.sqrt(2))**2
    se = np.sqrt(np.maximum(var, 1e-14)/(block*block))
    valid = (bi.max(-1) < clip).all(0)
    return mu, diff, diff/se, valid


def lag1_autocorr(r):
    """Lag-1 horizontal/vertical autocorrelation per plane of a residual."""
    r = np.asarray(r, np.float64)
    r = r-r.mean(axis=(-2, -1), keepdims=True)
    v = (r*r).mean(axis=(-2, -1))
    h = (r[..., :, 1:]*r[..., :, :-1]).mean(axis=(-2, -1))/np.maximum(v, 1e-30)
    vv = (r[..., 1:, :]*r[..., :-1, :]).mean(axis=(-2, -1))/np.maximum(v, 1e-30)
    return {"horizontal": np.atleast_1d(h).tolist(), "vertical": np.atleast_1d(vv).tolist()}


def dark_region_stats(codes, black, scale):
    """Statistics of an optical-black (masked) region of ONE frame.

    codes: 2D same-channel or mixed array in DN; black: DN; scale: white-black.
    Temporal read noise and dark fixed pattern are inseparable here.
    """
    x = (np.asarray(codes, np.float64)-black)/scale
    med = float(np.median(x))
    rs = float(robust_std(x))
    z = (x-med)/max(rs, 1e-12)
    rows, cols = x.mean(1), x.mean(0)
    return {"median_offset": med, "mean_offset": float(x.mean()), "std": float(x.std()),
            "robust_std": rs,
            "excess_kurtosis": float(((x-x.mean())**4).mean()/max(x.var(), 1e-30)**2-3),
            "frac_gt_5sigma_pos": float((z > 5).mean()), "frac_gt_5sigma_neg": float((z < -5).mean()),
            "frac_gt_10sigma_pos": float((z > 10).mean()),
            "row_mean_std": float(robust_std(rows)), "row_white_expect": rs/np.sqrt(x.shape[1]),
            "col_mean_std": float(robust_std(cols)), "col_white_expect": rs/np.sqrt(x.shape[0]),
            "shape": list(x.shape)}
