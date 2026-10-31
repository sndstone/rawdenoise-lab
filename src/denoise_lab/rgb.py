"""Strict uint16 RGB TIFF I/O and versioned, color-preserving model contract.

No demosaic, quantization to eight bits, transfer curve or ICC transform occurs.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import tifffile

RGB_ARCH = "rgb_nafnet"
COLOR_SPACES = ("srgb", "linear-srgb", "icc")


def rgb_contract(color_space, icc_profile=None):
    """Describe the actual encoded values, including exact ICC identity."""
    contract = dict(version=1, channels="RGB", dtype="uint16",
                    normalization="uint16/65535", color_space=color_space)
    if icc_profile:
        contract["icc_sha256"] = hashlib.sha256(icc_profile).hexdigest()
    return validate_rgb_contract(contract)


def validate_rgb_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("RGB checkpoint requires an explicit rgb_contract")
    expected = dict(version=1, channels="RGB", dtype="uint16", normalization="uint16/65535")
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported RGB contract: expected v1 RGB uint16/65535")
    space = contract.get("color_space")
    if space not in COLOR_SPACES:
        raise ValueError(f"Unsupported RGB color space {space!r}; no color transforms are implemented")
    digest = contract.get("icc_sha256")
    if space == "icc":
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("ICC RGB contract requires an exact icc_sha256")
    elif digest is not None:
        raise ValueError("Profiled RGB must use color_space=icc; profile transforms are unsupported")
    if set(contract) != set(expected) | {"color_space"} | ({"icc_sha256"} if space == "icc" else set()):
        raise ValueError("Unsupported fields in RGB contract")
    return dict(contract)


def normalize_rgb(codes):
    codes = np.asarray(codes)
    if codes.dtype != np.uint16 or codes.ndim != 3 or codes.shape[2] != 3 or min(codes.shape) < 1:
        raise ValueError("Expected nonempty uint16 RGB [H,W,3]")
    return np.ascontiguousarray(codes.transpose(2, 0, 1), dtype=np.float32) / np.float32(65535)


def rgb_codes(image):
    image = np.asarray(image)
    if image.dtype != np.float32 or image.ndim != 3 or image.shape[0] != 3 or min(image.shape) < 1:
        raise ValueError("Expected nonempty float32 RGB [3,H,W]")
    if not np.isfinite(image).all():
        raise ValueError("RGB output must be finite")
    return np.rint(np.clip(image, 0, 1) * np.float32(65535)).astype(np.uint16).transpose(1, 2, 0)


@dataclass
class RGBMetadata:
    contract: dict
    icc_profile: bytes | None = None
    orientation: int = 1
    resolution: tuple | None = None
    resolution_unit: int = 1


def is_rgb_tiff(path):
    """Inspect TIFF photometric type, so Bayer TIFFs retain their old path."""
    if Path(path).suffix.lower() not in {".tif", ".tiff"}:
        return False
    with tifffile.TiffFile(path) as tif:
        return tif.pages[0].photometric == tifffile.PHOTOMETRIC.RGB


def read_rgb_tiff(path, *, color_space=None):
    """Read one uint16 RGB image; require metadata or an explicit declaration.

    ICC profiles are opaque: only an exact profile-matched model is allowed.
    Unprofiled images may declare srgb/linear-srgb via --rgb-color-space or our
    ImageDescription JSON. Conflicting declarations are always errors.
    """
    if Path(path).suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("RGB input must be a uint16 RGB .tif/.tiff file")
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise ValueError("RGB TIFF must contain exactly one image/page")
        page = tif.pages[0]
        if page.photometric != tifffile.PHOTOMETRIC.RGB or page.samplesperpixel != 3 or page.extrasamples:
            raise ValueError("Expected RGB TIFF with exactly three channels, no alpha/CFA")
        if page.dtype != np.dtype("uint16") or page.bitspersample != 16:
            raise ValueError("RGB TIFF input must be uint16 (16-bit); no 8-bit or float conversion")
        tags = page.tags
        if any(code in tags for code in (318, 319, 301)):
            raise ValueError("TIFF colorimetry/transfer-function tags are unsupported; no color transforms implemented")
        icc = bytes(tags[34675].value) if 34675 in tags else None
        if icc is not None and not icc:
            raise ValueError("Empty ICC profile is unsupported")
        # TIFF EXIF can explicitly identify encoded sRGB. Uncalibrated is not
        # a declaration. Other encodings need a supported exact ICC contract.
        exif = tags[34665].value if 34665 in tags else {}
        exif_space = exif.get("ColorSpace") if isinstance(exif, dict) else None
        if 40961 in tags:
            exif_space = tags[40961].value
        if exif_space not in (None, 1, 65535) and not icc:
            raise ValueError("Unsupported EXIF RGB color space without an ICC profile; no color transforms implemented")
        description = {}
        try:
            description = json.loads(page.description or "{}")
        except (ValueError, TypeError):
            pass
        declared = description.get("denoise_lab_rgb") if isinstance(description, dict) else None
        if declared is not None:
            declared = validate_rgb_contract(declared)
            if color_space is not None and color_space != declared["color_space"]:
                raise ValueError("Explicit RGB color space conflicts with TIFF metadata")
            color_space = declared["color_space"]
        if exif_space == 1 and not icc:
            if color_space is not None and color_space != "srgb":
                raise ValueError("Explicit RGB color space conflicts with EXIF sRGB metadata")
            color_space = "srgb"
        if color_space is None:
            color_space = "icc" if icc else None
        if color_space is None:
            raise ValueError("RGB TIFF has no explicit color space; supply --rgb-color-space srgb or linear-srgb")
        contract = rgb_contract(color_space, icc)
        if declared is not None and declared != contract:
            raise ValueError("RGB TIFF metadata and embedded ICC profile disagree")
        orientation = int(tags[274].value) if 274 in tags else 1
        if orientation not in range(1, 9):
            raise ValueError("Unsupported TIFF orientation")
        resolution = None
        if 282 in tags and 283 in tags:
            resolution = (tags[282].value, tags[283].value)
        metadata = RGBMetadata(contract, icc, orientation, resolution,
                               int(tags[296].value) if 296 in tags else 1)
        codes = page.asarray()
        if page.planarconfig == tifffile.PLANARCONFIG.SEPARATE:
            codes = codes.transpose(1, 2, 0)
        return normalize_rgb(codes), metadata


def write_rgb_tiff(path, image, metadata):
    """Round/clamp to uint16 and preserve supported color/geometry metadata.

    Exclusive creation protects both existing outputs and source files.
    """
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("RGB output must be uint16 TIFF; Bayer/DNG/NPY output is unsupported")
    contract = validate_rgb_contract(metadata.contract)
    if rgb_contract(contract["color_space"], metadata.icc_profile) != contract:
        raise ValueError("RGB output ICC profile does not match its color contract")
    if metadata.orientation not in range(1, 9):
        raise ValueError("Unsupported TIFF orientation")
    codes = rgb_codes(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("xb")
    except FileExistsError as exc:
        raise ValueError(f"Refusing to overwrite existing file: {path}") from exc
    try:
        with handle:
            tifffile.imwrite(handle, codes, photometric="rgb", metadata=None,
                description=json.dumps({"denoise_lab_rgb": contract}),
                iccprofile=metadata.icc_profile, resolution=metadata.resolution,
                resolutionunit=metadata.resolution_unit,
                extratags=[(274, "H", 1, metadata.orientation, False)])
    except BaseException:
        path.unlink(missing_ok=True)
        raise
