"""RAW burst screening using the paired evaluator mask, not target certification.

All image arrays are linear normalized R/G1/B/G2 planes. Noise expectations are
robust empirical temporal estimates, not independent camera calibration.
"""
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import html
import json

import numpy as np

from .burst_eval import block_view, upsample, dilate, shifted, static_mask, temporal_mask


def read_metadata(path):
    """Read standard TIFF/EXIF fields; optional exifread supports other RAW containers.

    A PATH.json sidecar can supply camera, iso, shutter, aperture, timestamp when
    EXIF is unavailable. Missing fields fail explicitly; file mtimes are not capture times.
    """
    import tifffile
    values = {}
    try:
        with tifffile.TiffFile(path) as t:
            tags = t.pages[0].tags
            exif = tags['ExifTag'].value if 'ExifTag' in tags else {}
            values = {'camera': tags['Model'].value, 'iso': exif.get('ISOSpeedRatings'),
                      'shutter': exif.get('ExposureTime'), 'aperture': exif.get('FNumber'),
                      'timestamp': exif.get('DateTimeOriginal')}
    except (ValueError, KeyError, OSError):
        pass
    if any(values.get(k) is None for k in ('camera', 'iso', 'shutter', 'aperture', 'timestamp')):
        try:
            import exifread
            with Path(path).open('rb') as f:
                tags = exifread.process_file(f, details=False, extract_thumbnail=False)
            for key, tag in [('camera', 'Image Model'), ('iso', 'EXIF ISOSpeedRatings'),
                             ('shutter', 'EXIF ExposureTime'), ('aperture', 'EXIF FNumber'),
                             ('timestamp', 'EXIF DateTimeOriginal')]:
                if tag in tags:
                    values[key] = str(tags[tag])
        except (ImportError, OSError, ValueError):
            pass
    sidecar = Path(str(path)+'.json')
    if sidecar.exists():
        values.update(json.loads(sidecar.read_text()))
    def number(v):
        return float(Fraction(*v)) if isinstance(v, tuple) else float(Fraction(str(v)))
    try:
        meta = {'path': str(path), 'camera': str(values['camera']).strip(),
                **{k: number(values[k]) for k in ('iso', 'shutter', 'aperture')},
                'timestamp': datetime.strptime(values['timestamp'][:19], '%Y:%m:%d %H:%M:%S').timestamp()}
        if not meta['camera'] or any(not np.isfinite(meta[k]) or meta[k] <= 0
                                     for k in ('iso', 'shutter', 'aperture')):
            raise ValueError('nonpositive or missing settings')
        return meta
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f'{path}: incomplete capture metadata ({e}); supply PATH.json') from e


def group_frames(records, max_gap=30.):
    """Split chronological per-camera runs on settings changes or timestamp gaps.

    Long exposures get their shutter duration added to max_gap. No camera whitelist.
    """
    groups = []
    for camera in sorted({r['camera'] for r in records}):
        current = []
        for r in sorted((r for r in records if r['camera'] == camera),
                        key=lambda r: (r['timestamp'], r['path'])):
            if current:
                prev = current[-1]
                if (any(r[k] != prev[k] for k in ('iso', 'shutter', 'aperture')) or
                        r['timestamp']-prev['timestamp'] > max_gap+prev['shutter']):
                    groups.append(current)
                    current = []
            current.append(r)
        if current:
            groups.append(current)
    return groups


def load_preview(path, max_side=0):
    """Area-bin same-colour sites, retaining native clipping fraction before binning."""
    from .bayer import normalize_raw, pack
    import rawpy
    with rawpy.imread(str(path)) as raw:
        a = pack(normalize_raw(raw))
    clipped = float((a >= .98).mean())
    scale = max(1, int(np.ceil(max(a.shape[-2:])/max_side))) if max_side else 1
    a = block_view(a, scale).mean((-3, -1)).astype(np.float32)
    return a, scale*2, clipped  # sensor pixels per preview pixel


def phase_shift(reference, image):
    """Return (dy, dx) sampling image[y+dy,x+dx], in preview pixels.

    Hann-windowed phase correlation with parabolic subpixel peak interpolation.
    Low-texture frames return an uncertain registration rather than a silent pass.
    """
    a, b = reference.mean(0), image.mean(0)
    if min(a.std(), b.std()) < 1e-6:
        return np.zeros(2), False
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.rfft2((a-a.mean())*window)
    fb = np.fft.rfft2((b-b.mean())*window)
    cross = fb*fa.conj()
    corr = np.fft.irfft2(cross/np.maximum(np.abs(cross), 1e-12), s=a.shape).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    delta = []
    for axis, index in enumerate(peak):
        left, right = list(peak), list(peak)
        left[axis], right[axis] = (index-1)%a.shape[axis], (index+1)%a.shape[axis]
        l, c, r = corr[tuple(left)], corr[peak], corr[tuple(right)]
        offset = .5*(l-r)/(l-2*c+r) if abs(l-2*c+r) > 1e-12 else 0.
        delta.append((index if index <= a.shape[axis]//2 else index-a.shape[axis])+offset)
    return np.array(delta), bool(corr[peak] > 8*np.std(corr))


def _blocks(a, b):
    return block_view(a, b).mean((-3, -1))


DEFAULT_ANALOG_ISO = {'x1d2': 6400.}


def camera_key(camera):
    key = ''.join(c for c in camera.lower() if c.isalnum())
    if key.startswith(('x1dii', 'x1d2')):
        return 'x1d2'
    return {'ilce7m3': 'a7m3', 'ilce5100': 'a5100'}.get(key, key)


def analog_iso_table(calibration=None, overrides=()):
    """Read explicit highest_analog_iso camera fields; never infer gain from dark noise.

    The X1D II default is the 2026-11-01 dark-calibration ceiling (6400).
    Overrides are CAMERA=ISO strings; zero disables a ceiling for that camera.
    """
    table = DEFAULT_ANALOG_ISO.copy()
    if calibration is not None and Path(calibration).is_file():
        import yaml
        data = yaml.safe_load(Path(calibration).read_text()) or {}
        for camera, profile in data.get('cameras', {}).items():
            if 'highest_analog_iso' in profile:
                table[camera_key(camera)] = float(profile['highest_analog_iso'])
    for setting in overrides:
        camera, value = setting.rsplit('=', 1)
        table[camera_key(camera)] = float(value)
    if any(not np.isfinite(v) or v < 0 for v in table.values()):
        raise ValueError('Analog ISO ceilings must be finite and nonnegative')
    return table


def exposure_factor(record, table):
    ceiling = table.get(camera_key(record['camera']), 0)
    iso = min(record['iso'], ceiling) if ceiling else record['iso']
    return iso*record['shutter']/record['aperture']**2


def _registration_preview(a):
    scale = max(1, int(np.ceil(max(a.shape[-2:])/768)))
    return _blocks(a, scale), scale


def analyze_group(frames, records, sensor_scale=2., clipping=None, block=16, analog_isos=None):
    """Stream native packed frames into mean/variance; register thumbnails for diagnostics.

    Welford accumulation avoids retaining a full burst. Temporal masking uses the
    evaluator's 40 brightness bins. Per-frame diagnostic shifts use <=768 previews;
    pair registration and scoring use the input resolution (native by default).
    """
    mean = variance = first = None
    shifts, confident, signals, native_clipping = [], [], [], []
    frame_valid = None
    count = 0
    for a in frames:
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 3 or a.shape[0] != 4 or min(a.shape[-2:]) < block:
            raise ValueError('Expected N x 4 x H x W frames, at least one block')
        if not np.isfinite(a).all() or (mean is not None and a.shape != mean.shape):
            raise ValueError('Nonfinite pixels or inconsistent frame dimensions')
        count += 1
        if mean is None:
            mean, variance = a.copy(), np.zeros_like(a)
            frame_valid = np.ones(a.shape[-2:], bool)
        else:
            delta = a-mean
            mean += delta/count
            variance += delta*(a-mean)
        frame_valid &= (a < .95).all(0)
        preview, scale = _registration_preview(a)
        if first is None:
            first = preview.copy()
        shift, certain = phase_shift(first, preview)
        shifts.append(shift*scale*sensor_scale)
        confident.append(certain)
        signals.append(float(a.mean()))
        native_clipping.append(float((a >= .98).mean()))
    if not count or count != len(records):
        raise ValueError('Empty burst or frame metadata count mismatch')
    variance /= max(count-1, 1)
    temporal, expected = temporal_mask(mean, variance, block)
    table = DEFAULT_ANALOG_ISO if analog_isos is None else analog_isos
    factors = np.array([exposure_factor(r, table) for r in records])
    signals = np.array(signals)
    normal = signals/factors
    consistency = normal/max(float(np.median(normal)), 1e-12)
    times = np.array([r['timestamp'] for r in records])
    elapsed = times-times[0]
    slope = float(np.polyfit(elapsed, consistency, 1)[0]) if np.ptp(elapsed) > 0 else 0.
    tv = _blocks(variance, block)
    summary = {'frames': records, 'sensor_pixels_per_preview_pixel': sensor_scale,
               'block_packed_pixels': block,
               'clipping_fraction': list(map(float, clipping if clipping is not None else native_clipping)),
               'mean_signal': signals.tolist(), 'expected_signal_ratio': (factors/factors[0]).tolist(),
               'exposure_consistency_ratio': consistency.tolist(),
               'light_or_gain_drift_percent_per_minute': 6000*slope,
               'light_or_gain_end_to_start_percent': float(100*(normal[-1]/max(normal[0], 1e-12)-1)),
               'exposure_range_ev': float(np.log2(max(normal.max(), 1e-12)/max(normal.min(), 1e-12))),
               'shift_sensor_pixels': np.array(shifts).tolist(),
               'registration_confident': list(map(bool, confident)),
               'temporal_variance': float(variance.mean()) if count > 1 else None,
               'expected_variance': float(expected.mean()) if count > 1 else None,
               'noise_excess_ratio': float(np.median(tv/np.maximum(expected, 1e-12))) if count > 1 else None,
               'temporal_static_fraction': float(temporal.mean()),
               'duration_seconds': records[-1]['timestamp']-records[0]['timestamp']}
    return {'summary': summary, 'mean': mean, 'variance': variance, 'expected': expected,
            'static': temporal, 'frame_valid': frame_valid, 'block': block}


def compare_groups(clean, noisy, min_static=.10, marginal_static=.03, min_blocks=20, analog_isos=None,
                   alignment=None):
    """Score evaluator-compatible static pixels after measured affine/local exposure fit.

    USABLE requires >=10% static and >=20 complete static blocks by default;
    3--10% is MARGINAL. Nominal exposure discrepancies are information only.
    """
    if not 0 <= marginal_static <= min_static <= 1 or min_blocks < 1:
        raise ValueError('Require 0 <= marginal-static <= min-static <= 1 and min-blocks >= 1')
    cs, ns = clean['summary'], noisy['summary']
    cr, nr = cs['frames'][0], ns['frames'][0]
    b = clean['block']
    if clean['mean'].shape != noisy['mean'].shape or clean['block'] != noisy['block']:
        raise ValueError('Packed dimensions/block differ')
    if cr['camera'] != nr['camera'] or cs['sensor_pixels_per_preview_pixel'] != ns['sensor_pixels_per_preview_pixel']:
        raise ValueError('Camera or sampling scale differ')
    table = DEFAULT_ANALOG_ISO if analog_isos is None else analog_isos
    nominal = (nr['iso']*nr['shutter']/nr['aperture']**2)/(cr['iso']*cr['shutter']/cr['aperture']**2)
    ratio = exposure_factor(nr, table)/exposure_factor(cr, table)
    shift, confident = phase_shift(clean['mean'], noisy['mean'])
    if alignment is not None:
        if len(alignment) != 2 or not np.isfinite(alignment).all() or not np.equal(alignment, np.rint(alignment)).all():
            raise ValueError('Alignment must contain two integer packed-pixel shifts')
        shift = np.array(alignment, dtype=float)
    dy, dx = np.rint(shift).astype(int)
    aligned = shifted(noisy['mean'], dy, dx)
    valid = (clean['mean'] < .95).all(0) & np.isfinite(aligned).all(0)
    valid &= shifted(noisy['frame_valid'][None].astype(np.float32), dy, dx)[0] == 1
    variance = shifted(noisy['variance'], dy, dx)
    pixels, target, coeffs, before = static_mask(clean['mean'], clean['variance'], aligned,
                                                variance, len(ns['frames']), valid, b)
    fraction = float(pixels.mean())
    static = block_view(pixels, b).all((-3, -1))
    blocks = int(static.sum())
    measured = float(np.median([c[0] for c in coeffs])) if coeffs else None
    ev = float(np.log2(measured/ratio)) if measured is not None and measured > 0 else None
    elapsed = abs(float(np.mean([r['timestamp'] for r in ns['frames']]))-
                  float(np.mean([r['timestamp'] for r in cs['frames']])))
    movement = float(np.hypot(dy, dx)*cs['sensor_pixels_per_preview_pixel'])
    within = max(np.linalg.norm(s, axis=1).max() for s in
                 (np.array(cs['shift_sensor_pixels']), np.array(ns['shift_sensor_pixels'])))
    reasons, info, severity = [], [], 0
    def flag(message, level):
        nonlocal severity
        reasons.append(message)
        severity = max(severity, level)
    if fraction < min_static:
        flag(f'scene moved or changed: {fraction:.2%} static after temporal/group-mean masks '
             f'({before:.2%} before group-mean test)', 2 if fraction < marginal_static else 1)
    if blocks < min_blocks:
        flag(f'insufficient static blocks: {blocks} < {min_blocks}', 1 if blocks else 2)
    if measured is None or measured <= 0:
        flag('measured exposure fit unavailable or nonpositive', 2)
    elif ev is not None:
        info.append(f'INFO: measured/analog-capped expectation {ev:+.2f} EV; '
                    f'measured/nominal {np.log2(measured/nominal):+.2f} EV over {elapsed/60:.1f} min')
    if movement > 1.:
        flag(f'camera moved {movement:.1f} sensor px between groups', 2 if movement > 2 else 1)
    if within > 1.:
        flag(f'preview within-burst shift {within:.1f} sensor px; verify native alignment', 1)
    if cr['aperture'] != nr['aperture']:
        flag('aperture changed; depth of field/vignetting can differ', 2)
    if max(cs['clipping_fraction']+ns['clipping_fraction']) > .01:
        flag('more than 1% clipped photosites', 1)
    if min(len(cs['frames']), len(ns['frames'])) < 3:
        flag('fewer than 3 frames; temporal evidence insufficient', 1)
    if not confident or not all(cs['registration_confident']+ns['registration_confident']):
        flag('registration uncertain (low texture or scene change)', 1)
    for summary in (cs, ns):
        if summary['exposure_range_ev'] > .1:
            flag(f'within-burst exposure/light step {summary["exposure_range_ev"]:.2f} EV', 2)
    result = {'verdict': ('USABLE', 'MARGINAL', 'RESHOOT')[severity],
              'reasons': reasons or ['enough static area and blocks for paired evaluation'], 'info': info,
              'static_fraction': fraction, 'static_blocks': blocks,
              'static_fraction_before_group_test': before,
              'thresholds': {'usable_static': min_static, 'marginal_static': marginal_static, 'min_blocks': min_blocks},
              'exposure_mismatch_ev': ev, 'metadata_exposure_ratio': nominal,
              'analog_capped_exposure_ratio': ratio, 'highest_analog_iso': table.get(camera_key(cr['camera'])),
              'measured_exposure_ratio': measured, 'gain_offset_local_coefficients': coeffs,
              'nominal_exposure_mismatch_ev': float(np.log2(measured/nominal)) if measured and measured > 0 else None,
              'elapsed_seconds': elapsed,
              'shift_sensor_pixels': (shift*cs['sensor_pixels_per_preview_pixel']).tolist(),
              'applied_shift_packed_pixels': [int(dy), int(dx)],
              'alignment_source': 'explicit' if alignment is not None else 'phase correlation',
              'registration_confident': confident, 'clean': cr['path'], 'noisy': nr['path']}
    return result, static


def write_report(output, groups, pairings, errors=()):
    """Write portable HTML, PNG previews and masks; source files are never changed."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError('HTML thumbnails require the optional Pillow audit/dev dependency') from e
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = ['<!doctype html><meta charset="utf-8"><title>Burst capture check</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto}img{max-width:45%}pre{white-space:pre-wrap}</style>',
             '<h1>Burst capture check</h1><p>Evaluator static-area screening. Green masks mark static blocks. '
             'Noise is estimated from each burst, not independent calibration. Light drift and gain errors '
             'cannot be distinguished. Subpixel alignment needs full-resolution verification.</p>']
    for i, g in enumerate(groups):
        a, _ = _registration_preview(g['mean'])
        rgb = np.stack([a[0], (a[1]+a[3])/2, a[2]], -1)
        rgb = np.clip(rgb/max(float(np.percentile(rgb, 99)), 1e-6), 0, 1)**(1/2.2)
        Image.fromarray((rgb*255).astype('uint8')).save(output/f'group_{i}.png')
        Image.fromarray((~g['static']*255).astype('uint8')).resize(
            (a.shape[-1], a.shape[-2]), Image.Resampling.NEAREST).save(output/f'group_{i}_motion.png')
        lines += [f'<h2>Group {i}</h2><img src="group_{i}.png"><img src="group_{i}_motion.png">',
                  '<details><summary>Group measurements (white = temporal motion)</summary><pre>'+html.escape(json.dumps(g['summary'], indent=2))+'</pre></details>']
    rows = []
    for i, (result, mask) in enumerate(pairings):
        name = f'pair_{i}_static.png'
        Image.fromarray(np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], -1).astype('uint8')*255).resize(
            (mask.shape[1]*8, mask.shape[0]*8), Image.Resampling.NEAREST).save(output/name)
        rows.append(result)
        lines += ['<h2>'+html.escape(f'{result["verdict"]}: {Path(result["noisy"]).name} vs {Path(result["clean"]).name}')+'</h2>',
                  '<p>'+html.escape('; '.join(result['reasons']+result['info']))+'</p>', f'<img src="{name}">',
                  '<details><summary>Pair measurements</summary><pre>'+html.escape(json.dumps(result, indent=2))+'</pre></details>']
    lines += ['<h2>Input / pairing errors</h2><pre>'+html.escape('\n'.join(errors))+'</pre>']
    (output/'index.html').write_text('\n'.join(lines))
    (output/'results.json').write_text(json.dumps({'groups': [g['summary'] for g in groups],
                                                 'pairings': rows, 'errors': list(errors)}, indent=2, allow_nan=False)+'\n')
