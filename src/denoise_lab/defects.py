"""Deterministic photosite defects and conservative same-colour correction.

The detector is a single pass, not an iterative despiker. Heavy-tailed sensor
noise can still trigger it; report its masks on held-out real frames before use.
"""
import math
import numpy as np


def inject_defects(mosaic, rng, density_range=(1e-5, 3e-4),
                   kind_probs=None, pair_probability=.05, hot_range=(.15, 1.),
                   warm_sigma_range=(8., 80.), sigma=None):
    """Return a copy and an injected-site mask; never mutate the input.

    Density counts primary events; pairs add one orthogonally adjacent site.
    Repeated locations are allowed by the Poisson process. A pair shares its
    event's kind, with independently drawn amplitudes. Sigma may be scalar or
    broadcastable to the mosaic. The fallback is first-difference MAD / sqrt(2).
    """
    a = np.asarray(mosaic)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.floating) or not a.size:
        raise ValueError('Expected a nonempty HxW float mosaic')
    lo, hi = density_range
    probs = kind_probs if kind_probs is not None else dict(hot=.45, warm=.3, dead=.25)
    if not 0 < lo <= hi or not 0 <= pair_probability <= 1:
        raise ValueError('Invalid density or pair probability')
    if set(probs) - {'hot', 'warm', 'dead'} or not probs:
        raise ValueError('Unknown or empty defect kinds')
    p = np.array(list(probs.values()), dtype=float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Kind probabilities must sum to one')
    for bounds in (hot_range, warm_sigma_range):
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError('Invalid amplitude range')
    if sigma is None:
        # Differences two sensor pixels apart preserve Bayer colour.
        d = np.concatenate([np.diff(a[sy::2, sx::2], axis=1).ravel()
                            for sy in range(2) for sx in range(2)])
        sigma = np.median(np.abs(d-np.median(d)))/(.6744897501960817*math.sqrt(2)) if d.size else 0.
    scales = np.broadcast_to(np.asarray(sigma), a.shape)
    if np.any(scales < 0) or not np.isfinite(scales).all():
        raise ValueError('Sigma must be finite and nonnegative')
    out, mask = a.copy(), np.zeros(a.shape, dtype=bool)
    count = rng.poisson(a.size*np.exp(rng.uniform(np.log(lo), np.log(hi))))
    kinds = list(probs)
    for _ in range(count):
        y, x = int(rng.integers(a.shape[0])), int(rng.integers(a.shape[1]))
        kind = kinds[int(rng.choice(len(kinds), p=p))]
        sites = [(y, x)]
        if rng.random() < pair_probability:
            adjacent = [(v, u) for v, u in ((y-1,x),(y+1,x),(y,x-1),(y,x+1))
                        if 0 <= v < a.shape[0] and 0 <= u < a.shape[1]]
            if adjacent:
                sites.append(adjacent[int(rng.integers(len(adjacent)))])
        for y, x in sites:
            if kind == 'hot': out[y, x] = rng.uniform(*hot_range)
            elif kind == 'dead': out[y, x] = rng.uniform(-.002, .002)
            else: out[y, x] += rng.uniform(*warm_sigma_range)*scales[y, x]
            mask[y, x] = True
    return out, mask


# Bit flags may overlap: an unstable site can also have an elevated mean.
HOT, WARM, STUCK, DEAD, HIGH_VARIANCE = 1, 2, 4, 8, 16
DEFECT_KINDS = {HOT: 'hot', WARM: 'warm', STUCK: 'stuck_high',
                DEAD: 'low_dead_candidate', HIGH_VARIANCE: 'high_variance'}


def dark_statistics(frames):
    """Streaming moments of signed, normalized visible mosaics (same body/settings).

    Remove each frame's per-CFA median and row/column offsets before accumulating;
    broad banding is not a list of defective photosites. Memory is O(sensor size).
    Two frames are the minimum; variance evidence is weak with so few captures.
    """
    mean = m2 = positive = negative = None
    n = 0
    for frame in frames:
        a = np.array(frame, dtype=np.float32, copy=True)
        if a.ndim != 2 or min(a.shape) < 4 or not np.isfinite(a).all():
            raise ValueError('Expected finite normalized visible dark mosaics')
        if mean is None:
            mean, m2 = np.zeros_like(a), np.zeros_like(a)
            positive = np.zeros(a.shape, np.uint32)
            negative = np.zeros(a.shape, np.uint32)
        if a.shape != mean.shape:
            raise ValueError('Dark frame dimensions differ')
        for y in range(2):
            for x in range(2):
                plane = a[y::2, x::2]
                # Deterministic sparse samples keep full-resolution calibration cheap.
                plane -= np.median(plane[::8, ::8])
                plane -= np.median(plane[:, ::8], axis=1)[:, None]
                plane -= np.median(plane[::8, :], axis=0)[None, :]
                sample = plane[::8, ::8]
                sigma = max(float(np.median(np.abs(sample-np.median(sample)))/.67448975), 1e-7)
                positive[y::2, x::2] += plane > 3*sigma
                negative[y::2, x::2] += plane < -3*sigma
        n += 1
        delta = a-mean
        mean += delta/n
        m2 += delta*(a-mean)
    if n < 2:
        raise ValueError('Need at least two dark frames per ISO/exposure/shutter group')
    return dict(mean=mean, std=np.sqrt(np.maximum(m2/(n-1), 0)),
                positive=positive.astype(np.float32)/n, negative=negative.astype(np.float32)/n, frames=n)


def detect_dark_defects(stats, *, k=6., warm_floor=.0005, hot_floor=.01,
                        variance_floor=.001, persistence=.75):
    """Candidate masks from independent dark moments; confidence is empirical evidence.

    Hot/warm means exceed k times typical temporal noise / sqrt(N), with at least
    `persistence` of frames exceeding 3 sigma. RTS is only a high-variance candidate,
    not a confirmed telegraph process. Darks cannot establish dead photoresponse;
    DEAD denotes a persistent negative offset. Flat fields are needed to confirm it.
    """
    if k <= 0 or not 0 < persistence <= 1 or min(warm_floor, hot_floor, variance_floor) < 0:
        raise ValueError('Invalid dark detector thresholds')
    mean, std, n = stats['mean'], stats['std'], stats['frames']
    kinds = np.zeros(mean.shape, np.uint8)
    confidence = np.zeros(mean.shape, np.float32)
    for y in range(2):
        for x in range(2):
            mu, sd = mean[y::2, x::2], std[y::2, x::2]
            noise = max(float(np.median(sd[::8, ::8])), 1e-7)
            threshold = max(warm_floor, k*noise/math.sqrt(n))
            pos, neg = stats['positive'][y::2, x::2], stats['negative'][y::2, x::2]
            high = (mu > threshold) & (pos >= persistence)
            low = (mu < -threshold) & (neg >= persistence)
            hot = high & (mu >= hot_floor)
            stuck = hot & (mu >= .98) & (sd < max(3*noise, .0001))
            variable = sd > max(variance_floor, k*noise)
            kinds[y::2, x::2] = (hot*HOT | (high & ~hot)*WARM | stuck*STUCK |
                                  low*DEAD | variable*HIGH_VARIANCE).astype(np.uint8)
            confidence[y::2, x::2] = np.maximum(
                np.where(high, pos, np.where(low, neg, 0))*n/(n+1),
                variable*((n-1)/(n+1)))
    return kinds, confidence


def make_defect_map(groups, metadata, **thresholds):
    """Build sparse visible-coordinate records from (condition, frame iterator) pairs.

    Conditions hold iso_min/max, exposure_min/max (seconds), and shutter. Separate
    records retain non-contiguous observed settings; never bridge an unmeasured ISO.
    Metadata must include shape and phase, and should include model, serial, origin.
    """
    import json
    result = {key: [] for key in ('coordinates', 'kind', 'confidence', 'mean', 'std', 'group')}
    conditions = []
    for condition, frames in groups:
        stats = dark_statistics(frames)
        if tuple(stats['mean'].shape) != tuple(metadata['shape']):
            raise ValueError('Dark geometry differs from map metadata')
        kinds, confidence = detect_dark_defects(stats, **thresholds)
        loc = np.argwhere(kinds != 0)
        yy, xx = loc.T
        result['coordinates'].append(loc.astype(np.int32))
        for key, values in [('kind', kinds), ('confidence', confidence),
                            ('mean', stats['mean']), ('std', stats['std'])]:
            result[key].append(values[yy, xx])
        result['group'].append(np.full(len(loc), len(conditions), np.int32))
        conditions.append({**condition, 'frames': stats['frames']})
    if not conditions:
        raise ValueError('No dark frame groups')
    result = {key: np.concatenate(values) for key, values in result.items()}
    result['metadata'] = np.array(json.dumps({**metadata, 'version': 1, 'conditions': conditions,
                                            'thresholds': {**dict(k=6., warm_floor=.0005, hot_floor=.01,
                                                                 variance_floor=.001, persistence=.75), **thresholds},
                                            'confidence': 'Empirical support discounted by frame count; not a probability'}))
    validate_defect_map(result)
    return result


def validate_defect_map(defect_map):
    """Validate the versioned, pickle-free sparse format before indexing sensors."""
    import json
    from .bayer import PHASES
    meta = json.loads(str(defect_map['metadata']))
    shape = meta['shape']
    if meta['version'] != 1 or meta['phase'] not in PHASES or len(shape) != 2 or min(shape) < 4:
        raise ValueError('Invalid defect map version or Bayer geometry')
    loc = defect_map['coordinates']
    if loc.ndim != 2 or loc.shape[1] != 2 or not np.issubdtype(loc.dtype, np.integer):
        raise ValueError('Invalid defect coordinates')
    if np.any(loc < 0) or np.any(loc >= np.asarray(shape)):
        raise ValueError('Defect coordinates outside sensor')
    for key in ('kind', 'confidence', 'mean', 'std', 'group'):
        values = defect_map[key]
        if values.shape != (len(loc),) or not np.isfinite(values).all():
            raise ValueError(f'Invalid defect map {key}')
    if (not np.issubdtype(defect_map['kind'].dtype, np.integer) or
        np.any((defect_map['confidence'] < 0) | (defect_map['confidence'] > 1)) or
        np.any((defect_map['kind'] < 1) | (defect_map['kind'] > 31)) or
        np.any(defect_map['std'] < 0)):
        raise ValueError('Invalid defect attributes')
    g = defect_map['group']
    if not np.issubdtype(g.dtype, np.integer) or np.any(g < 0) or np.any(g >= len(meta['conditions'])):
        raise ValueError('Invalid defect condition index')
    for condition in meta['conditions']:
        for prefix in ('iso', 'exposure'):
            lo, hi = condition[prefix+'_min'], condition[prefix+'_max']
            if not np.isfinite([lo, hi]).all() or not 0 < lo <= hi:
                raise ValueError('Invalid ISO/exposure range')
    return meta


def save_defect_map(path, defect_map):
    from pathlib import Path
    validate_defect_map(defect_map)
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Defect maps must use .npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **defect_map)


def load_defect_map(path):
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in saved.files}
    validate_defect_map(result)
    return result


def defect_map_mask(defect_map, metadata, *, iso=None, exposure=None, shutter=None):
    """Validate body/visible geometry and select only observed ISO/exposure ranges.

    Unknown exposure/ISO is an error, never permission to apply every condition.
    Explicit selection permits a missing serial; known mismatches always fail.
    """
    from .bayer import pack
    meta = validate_defect_map(defect_map)
    for key in ('shape', 'phase', 'origin', 'raw_shape'):
        if key in meta and meta[key] != metadata.get(key):
            raise ValueError(f'Defect map {key} does not match input')
    for key in ('model', 'serial'):
        if meta.get(key) and metadata.get(key) and meta[key] != metadata[key]:
            raise ValueError(f'Defect map {key} does not match input')
    if iso is None or exposure is None or not np.isfinite([iso, exposure]).all() or min(iso, exposure) <= 0:
        raise ValueError('Defect map requires input ISO and exposure metadata')
    shutters = {c.get('shutter', '') for c in meta['conditions']}
    if shutter is None and len(shutters) > 1:
        raise ValueError('Map has multiple shutter modes; build/select a single-shutter map')
    selected = []
    for c in meta['conditions']:
        selected.append(c['iso_min'] <= iso <= c['iso_max'] and
                        c['exposure_min']*(1-1e-6) <= exposure <= c['exposure_max']*(1+1e-6) and
                        (shutter is None or not c.get('shutter') or c['shutter'] == shutter))
    if not any(selected):
        raise ValueError('Input settings outside observed defect map ranges')
    use = np.asarray(selected)[defect_map['group']]
    mask = np.zeros(meta['shape'], bool)
    yy, xx = defect_map['coordinates'][use].T
    mask[yy, xx] = True
    return pack(mask, meta['phase'])


def find_defect_map(directory, metadata):
    """Auto-select only an exact model AND known serial; ambiguous matches fail."""
    from pathlib import Path
    if not metadata.get('model') or not metadata.get('serial'):
        return None
    matches = []
    for path in sorted(Path(directory).glob('*.npz')):
        candidate = load_defect_map(path)
        meta = validate_defect_map(candidate)
        if all(meta.get(k) == metadata[k] for k in ('model', 'serial')):
            matches.append(path)
    if len(matches) > 1:
        raise ValueError('Multiple defect maps match this body; select --defect-map explicitly')
    return matches[0] if matches else None


def raw_defect_metadata(raw, source):
    """Standard TIFF/EXIF identity only; no maker-note parsing or guessed serials."""
    import tifffile
    desc = raw.color_desc.decode('ascii')
    phase = ''.join(desc[int(c)] for c in raw.raw_colors_visible[:2, :2].ravel())
    meta = dict(shape=list(raw.raw_image_visible.shape), phase=phase,
                origin=[int(raw.sizes.top_margin), int(raw.sizes.left_margin)],
                raw_shape=[int(raw.sizes.raw_height), int(raw.sizes.raw_width)], model='', serial='')
    iso = exposure = None
    try:
        with tifffile.TiffFile(source) as tif:
            tags = tif.pages[0].tags
            meta['model'] = str(tags[272].value).strip() if 272 in tags else ''
            exif = tags[34665].value if 34665 in tags else {}
            serial = exif.get('BodySerialNumber', exif.get('CameraSerialNumber', ''))
            meta['serial'] = str(tags[50735].value if 50735 in tags else serial).strip()
            iso = exif.get('PhotographicSensitivity', exif.get('ISOSpeedRatings'))
            value = exif.get('ExposureTime')
            if value is not None:
                exposure = value[0]/value[1] if isinstance(value, tuple) else float(value)
            if iso is not None:
                iso = float(iso[0] if isinstance(iso, tuple) else iso)
    except (tifffile.TiffFileError, OSError):
        pass
    return meta, iso, exposure


def read_dark_frame(path):
    """Read normalized visible coordinates, without Bayer unification/cropping."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        meta, iso, exposure = raw_defect_metadata(raw, path)
        colors = raw.raw_colors_visible
        black = np.asarray(raw.black_level_per_channel, np.float32)
        whites = raw.camera_white_level_per_channel
        white = np.asarray(whites if whites is not None else [raw.white_level]*len(black), np.float32)
        white = np.where(white > black, white, raw.white_level)
        if np.any(white <= black):
            raise ValueError('White levels must exceed black levels')
        image = (raw.raw_image_visible.astype(np.float32)-black[colors])/(white[colors]-black[colors])
    return image, meta, iso, exposure


def __getattr__(name):
    # Preserve existing torch helper imports without loading torch for map I/O.
    if name in {"correct_defects", "correct_mapped_defects", "_neighbours",
                "_chunk_neighbours", "_sigma_model", "_shift", "_ADJACENT"}:
        from . import _torch_defects
        return getattr(_torch_defects, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
