"""Uncompressed visible-area CFA DNGs with guarded standard metadata copying.

Pixels retain their sensor orientation. LibRaw flip codes map to TIFF orientation.
ColorMatrix1 uses rawpy's camera RGB/XYZ coefficients (LibRaw cam_xyz),
or the source DNG ColorMatrix1 when available.
"""
from dataclasses import dataclass, field
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import uuid

import numpy as np
import tifffile

from .bayer import PHASES
from .tiff_metadata import append_ifds, extratag, read_metadata


@dataclass
class RawMetadata:
    pattern: np.ndarray = field(default_factory=lambda: np.array([[0, 1], [1, 2]], dtype=np.uint8))
    black: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    white: np.ndarray = field(default_factory=lambda: np.full((2, 2), 65535.))
    color_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    neutral: np.ndarray = field(default_factory=lambda: np.ones(3))
    make: str = "Clean-room"
    model: str = "Bayer RAW"
    orientation: int = 1
    illuminant: int = 21

    source_tags: dict = field(default_factory=lambda: {'root': {}, 'exif': {}, 'gps': {}})
    source_name: str = ''
    geometry_unchanged: bool = False
    visible_shape: tuple | None = None

    @property
    def phase(self):
        pattern = np.asarray(self.pattern)
        if pattern.shape != (2, 2) or np.any((pattern < 0) | (pattern > 2)):
            raise ValueError("Only 2x2 RGB Bayer CFA inputs are supported (no X-Trans/Foveon)")
        phase = "".join("RGB"[int(c)] for c in pattern.ravel())
        if phase not in PHASES:
            raise ValueError(f"Unsupported Bayer layout: {phase}")
        return phase

    @classmethod
    def from_raw(cls, raw, source=None):
        if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2) or raw.raw_image_visible.ndim != 2:
            raise ValueError("Only 2x2 Bayer RAW is supported (no X-Trans/Foveon)")
        colors = raw.raw_colors_visible[:2, :2]
        desc = raw.color_desc.decode("ascii")
        try:
            pattern = np.array([["RGB".index(desc[int(c)]) for c in row] for row in colors], dtype=np.uint8)
        except ValueError as exc:
            raise ValueError("Only RGB Bayer CFA inputs are supported") from exc
        black = np.asarray(raw.black_level_per_channel, dtype=float)
        white_meta = raw.camera_white_level_per_channel
        white = np.asarray(white_meta if white_meta is not None else [raw.white_level]*len(black), dtype=float)
        white = np.where(white > black, white, raw.white_level)
        if np.any(white <= black):
            raise ValueError("White levels must exceed black levels")
        matrix = np.asarray(raw.rgb_xyz_matrix, dtype=float)[:3, :3]
        color_matrix = None
        illuminant = 21
        source_tags = read_metadata(source) if source is not None else {'root': {}, 'exif': {}, 'gps': {}}
        tags = source_tags['root']
        if 50721 in tags:
            try:
                values = np.asarray(tags[50721][2], dtype=float).reshape(-1, 2)
                candidate = (values[:, 0]/values[:, 1]).reshape(3, 3)
                if np.isfinite(candidate).all() and abs(np.linalg.det(candidate)) > 1e-12:
                    color_matrix = candidate
            except (ValueError, ZeroDivisionError):
                pass
        if 50778 in tags:
            illuminant = int(tags[50778][2][0])
        if color_matrix is None:
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                raise ValueError("Source has no usable camera RGB-to-XYZ color matrix")
            color_matrix = matrix
        wb = np.asarray(raw.camera_whitebalance, dtype=float)[:3]
        if not np.isfinite(wb).all() or np.any(wb <= 0):
            wb = np.ones(3)
        meta = cls(pattern, black[colors], white[colors], color_matrix, wb[1]/wb,
                   orientation={0: 1, 3: 3, 5: 8, 6: 6}.get(raw.sizes.flip, 1), illuminant=illuminant)
        # rawpy does not expose camera names; standard TIFF-based RAWs do.
        for code, attr in [(271, 'make'), (272, 'model')]:
            if code in tags and tags[code][0] == 2:
                setattr(meta, attr, tags[code][2].rstrip(b'\0').decode('ascii', errors='backslashreplace'))
        meta.source_tags = source_tags
        meta.source_name = Path(source).name if source is not None else ''
        meta.visible_shape = tuple(raw.raw_image_visible.shape)
        sizes = raw.sizes
        meta.geometry_unchanged = (sizes.top_margin == sizes.left_margin == 0
                                   and sizes.raw_height == sizes.height
                                   and sizes.raw_width == sizes.width)
        meta.phase
        return meta


def sensor_codes(normalized, metadata):
    """Convert a visible mosaic to uint16 with source per-CFA-site levels."""
    normalized = np.asarray(normalized)
    if normalized.ndim != 2 or not np.isfinite(normalized).all():
        raise ValueError("Expected a finite 2D visible mosaic")
    out = np.empty(normalized.shape, dtype=np.uint16)
    black, white = np.asarray(metadata.black), np.asarray(metadata.white)
    if black.shape != (2, 2) or white.shape != (2, 2) or np.any(white <= black):
        raise ValueError("Expected 2x2 black/white levels with white > black")
    for y in range(2):
        for x in range(2):
            values = normalized[y::2, x::2]*(white[y, x]-black[y, x])+black[y, x]
            out[y::2, x::2] = np.clip(np.rint(values), 0, 65535).astype(np.uint16)
    return out


def _rationals(values):
    scale = 1000000
    values = np.asarray(values).ravel()
    if not np.isfinite(values).all():
        raise ValueError("DNG metadata must be finite")
    return tuple(v for value in values for v in (int(round(float(value)*scale)), scale))


def write_dng(path, mosaic, metadata, *, normalized=True, metadata_mode="full", preview=True):
    """Write a visible-area DNG (normalized floats, or uint16 sensor codes).

    Full mode preserves standard TIFF/DNG/EXIF/GPS metadata and an optional
    browsing preview. Minimal mode reproduces the legacy writer exactly.
    """
    if metadata_mode not in ('full', 'minimal'):
        raise ValueError("metadata_mode must be full or minimal")
    metadata.phase
    codes = sensor_codes(mosaic, metadata) if normalized else np.asarray(mosaic)
    if codes.ndim != 2 or codes.dtype != np.uint16:
        raise ValueError("Expected a uint16 2D sensor mosaic")
    h, w = codes.shape
    def tag(code, dtype, count, value):
        return (code, dtype, count, value, False)
    tags = [tag(254, 'I', 1, 0), tag(271, 's', 0, metadata.make),
            tag(272, 's', 0, metadata.model), tag(274, 'H', 1, metadata.orientation),
            tag(33421, 'H', 2, (2, 2)), tag(33422, 'B', 4, tuple(np.asarray(metadata.pattern).ravel())),
            tag(50706, 'B', 4, (1, 4, 0, 0)), tag(50707, 'B', 4, (1, 1, 0, 0)),
            tag(50708, 's', 0, f'{metadata.make} {metadata.model}'),
            tag(50710, 'B', 3, (0, 1, 2)), tag(50711, 'H', 1, 1),
            tag(50713, 'H', 2, (2, 2)), tag(50714, '2I', 4, _rationals(metadata.black)),
            tag(50717, 'I', 1, int(np.max(metadata.white))),
            tag(50719, 'I', 2, (0, 0)), tag(50720, 'I', 2, (w, h)),
            tag(50721, '2i', 9, _rationals(metadata.color_matrix)),
            tag(50728, '2I', 3, _rationals(metadata.neutral)), tag(50778, 'H', 1, metadata.illuminant)]
    # DNG WhiteLevel has one value per sample (one for a CFA image). Source
    # channel-specific whites still drive conversion to the original sensor codes.
    if metadata_mode == 'minimal':
        # Keep the legacy writer byte-for-byte, including tifffile's Software tag.
        tifffile.imwrite(Path(path), codes, photometric=32803, metadata=None, extratags=tags)
        return
    copied = dict(metadata.source_tags['root'])
    # Exported pixels are the visible area. Reset geometry; never copy OpcodeList1
    # (pre-demosaic operations can repeat corrections already applied by decoding).
    # OpcodeLists2/3 can also reference original coordinates: retain only when no
    # masked margins were removed. MaskedAreas, original data and hashes are omitted.
    for code in (50719, 50720, 50729, 50829, 51125):
        copied.pop(code, None)
    if not (metadata.geometry_unchanged and metadata.visible_shape == codes.shape):
        for code in (51009, 51022):
            copied.pop(code, None)
    # Decoded sensor levels/CFA/orientation always take priority; colour and WB
    # rationals can be copied exactly from the source DNG rather than quantized.
    replacements = {code: extratag(code, value) for code, value in copied.items()}
    tags = [replacements.pop(t[0], t) for t in tags]
    tags.extend(replacements.values())
    tags.extend([tag(50829, 'I', 4, (0, 0, h, w)),
                 tag(50781, 'B', 16, tuple(uuid.uuid4().bytes))])
    if metadata.source_name:
        tags.append(tag(50827, 's', 0, metadata.source_name.encode('utf-8')))
    for name, code in [('exif', 34664), ('gps', 34852)]:
        if metadata.source_tags[name]:
            tags.append(tag(code, 'I', 1, 0))
    from denoise_lab import __version__
    software = f'rawdenoise {__version__}'
    preview = preview and min(codes.shape) >= 2
    with tifffile.TiffWriter(Path(path), byteorder='<') as writer:
        writer.write(codes, photometric=32803, metadata=None, extratags=tags,
                     software=software, description='Processed by rawdenoise; visible sensor mosaic denoised.',
                     subifds=1 if preview else None)
        if preview:
            writer.write(_thumbnail(codes, metadata), photometric='rgb', metadata=None,
                         subfiletype=1, extratags=[tag(274, 'H', 1, metadata.orientation)])
    append_ifds(path, metadata.source_tags)


def _thumbnail(codes, metadata):
    """Cheap <=256px sRGB preview from Bayer cells, without a full demosaic.

    Uses camera WB and the DNG XYZ-to-camera matrix; intended for browsing only.
    """
    h, w = codes.shape
    step = max(1, (max(h, w)//2+255)//256)
    rgb = np.zeros(((h//2+step-1)//step, (w//2+step-1)//step, 3), dtype=np.float32)
    counts = np.zeros(3)
    for y in range(2):
        for x in range(2):
            channel = int(metadata.pattern[y, x])
            values = codes[y:h//2*2:2*step, x:w//2*2:2*step].astype(np.float32)
            rgb[..., channel] += (values-metadata.black[y, x])/(metadata.white[y, x]-metadata.black[y, x])
            counts[channel] += 1
    rgb /= counts
    rgb /= np.asarray(metadata.neutral)
    xyz_to_rgb = np.array([[3.2406, -1.5372, -.4986], [-.9689, 1.8758, .0415], [.0557, -.2040, 1.0570]])
    try:
        rgb = rgb @ (xyz_to_rgb @ np.linalg.inv(metadata.color_matrix)).T
    except np.linalg.LinAlgError:
        pass
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)
    return np.rint(rgb*255).astype(np.uint8)
