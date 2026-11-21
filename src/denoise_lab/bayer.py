"""Bayer geometry. Packed axes are (..., 4, H/2, W/2): R, G1, B, G2."""
from pathlib import Path
import numpy as np

PHASES = {"RGGB": (0, 0), "BGGR": (1, 1), "GRBG": (0, 1), "GBRG": (1, 0)}
SITES = ((0, 0), (0, 1), (1, 1), (1, 0))


def unify(mosaic, phase="RGGB"):
    """Crop to a red origin and even extent; never roll/wrap the sensor."""
    if phase not in PHASES:
        raise ValueError(f"Unsupported CFA: {phase}")
    y, x = PHASES[phase]
    h, w = mosaic.shape[-2:]
    h, w = ((h-y)//2)*2, ((w-x)//2)*2
    if min(h, w) < 2:
        raise ValueError("Mosaic too small after Bayer unification")
    return mosaic[..., y:y+h, x:x+w]


def pack(mosaic, phase="RGGB"):
    mosaic = unify(np.asarray(mosaic), phase)
    return np.stack([mosaic[..., y::2, x::2] for y, x in SITES], axis=-3)


def unpack(packed):
    packed = np.asarray(packed)
    if packed.shape[-3] != 4:
        raise ValueError("Expected four Bayer planes")
    h, w = packed.shape[-2:]
    out = np.empty((*packed.shape[:-3], h*2, w*2), dtype=packed.dtype)
    for c, (y, x) in enumerate(SITES):
        out[..., y::2, x::2] = packed[..., c, :, :]
    return out


def normalize_raw(raw):
    """Normalize visible rawpy pixels using the metadata's channel-index map.

    Returns an RGGB normalized mosaic. Negative values are intentionally retained.
    """
    codes = raw.raw_image_visible.astype(np.float32)
    colors = raw.raw_colors_visible
    black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    white_meta = raw.camera_white_level_per_channel
    white = np.asarray(white_meta if white_meta is not None else [raw.white_level]*4,
                       dtype=np.float32)
    white = np.where(white > black, white, raw.white_level)
    if np.any(white <= black):
        raise ValueError("White levels must exceed black levels")
    normalized = (codes - black[colors]) / (white[colors] - black[colors])
    desc = raw.color_desc.decode("ascii")
    phase = "".join(desc[int(c)] for c in colors[:2, :2].ravel())
    return unify(normalized, phase).copy()


def read_mosaic(path):
    """NPY/TIFF targets must already be normalized RGGB mosaics, not RGB images."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        a = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(path)
    else:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            return normalize_raw(raw)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D normalized mosaic: {path}")
    return unify(a.astype(np.float32))


def augment_mosaic(mosaic, flip_y=False, flip_x=False, transpose=False):
    """Transform pixels and CFA together, then crop back to a red origin.

    Use identical flags for paired images. Flips may discard one sensor row/col.
    """
    a = unify(mosaic)
    cfa = np.tile(np.array(list("RGGB")).reshape(2, 2), (a.shape[-2]//2, a.shape[-1]//2))
    if flip_y:
        a, cfa = a[..., ::-1, :], cfa[::-1, :]
    if flip_x:
        a, cfa = a[..., :, ::-1], cfa[:, ::-1]
    if transpose:
        a, cfa = a.swapaxes(-2, -1), cfa.T
    phase = "".join(cfa[:2, :2].ravel())
    return unify(a, phase).copy()


def random_crop(mosaic, packed_size=512, rng=None):
    """A packed 512 crop spans 1024 sensor pixels, with an even sensor origin."""
    rng = rng or np.random.default_rng()
    size = 2*packed_size
    h, w = mosaic.shape[-2:]
    if min(h, w) < size:
        raise ValueError(f"Need at least {size}x{size} mosaic, got {h}x{w}")
    y = 2*int(rng.integers((h-size)//2+1))
    x = 2*int(rng.integers((w-size)//2+1))
    return mosaic[..., y:y+size, x:x+size]
