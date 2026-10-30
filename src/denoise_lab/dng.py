"""Minimal uncompressed visible-area CFA DNGs, without proprietary metadata.

Pixels retain their sensor orientation. LibRaw flip codes map to TIFF orientation.
ColorMatrix1 uses rawpy's camera RGB/XYZ coefficients (LibRaw cam_xyz),
or the source DNG ColorMatrix1 when available.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from .bayer import PHASES


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
        if source is not None:
            try:
                with tifffile.TiffFile(source) as tif:
                    tags = tif.pages[0].tags
                    if 50721 in tags:
                        values = np.asarray(tags[50721].value, dtype=float).reshape(-1, 2)
                        color_matrix = (values[:, 0]/values[:, 1]).reshape(3, 3)
                    if 50778 in tags:
                        illuminant = int(tags[50778].value)
            except (tifffile.TiffFileError, OSError):
                pass
        if color_matrix is None:
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                raise ValueError("Source has no usable camera RGB-to-XYZ color matrix")
            color_matrix = matrix
        wb = np.asarray(raw.camera_whitebalance, dtype=float)[:3]
        if not np.isfinite(wb).all() or np.any(wb <= 0):
            wb = np.ones(3)
        meta = cls(pattern, black[colors], white[colors], color_matrix, wb[1]/wb,
                   orientation={0: 1, 3: 3, 5: 8, 6: 6}.get(raw.sizes.flip, 1), illuminant=illuminant)
        # rawpy does not expose camera names; TIFF-based RAWs commonly carry them.
        if source is not None:
            try:
                with tifffile.TiffFile(source) as tif:
                    tags = tif.pages[0].tags
                    meta.make = str(tags[271].value) if 271 in tags else meta.make
                    meta.model = str(tags[272].value) if 272 in tags else meta.model
            except (tifffile.TiffFileError, OSError):
                pass
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


def write_dng(path, mosaic, metadata, *, normalized=True):
    """Write normalized visible sensor pixels, or uint16 codes with normalized=False."""
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
    tifffile.imwrite(Path(path), codes, photometric=32803, metadata=None, extratags=tags)
