"""Local cache precision checks shared by the RAW preparation scripts.

No RAW/SMB dependencies: existing NPY headers are authoritative, including for
legacy journals that predate dtype metadata. Never repair precision by upcasting.
"""
import argparse
import json
from pathlib import Path

import numpy as np


class CachePrecisionError(ValueError):
    """An output directory cannot safely be used for the requested cache."""


def cache_dtype(value):
    aliases = {"fp16": "float16", "fp32": "float32"}
    name = aliases.get(str(value).lower(), str(value).lower())
    if name not in {"float16", "float32"}:
        raise ValueError(f"Unsupported cache dtype: {value}")
    return np.dtype(name)


def add_cache_dtype_argument(parser):
    def parse(value):
        try:
            return cache_dtype(value).name
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    parser.add_argument(
        "--cache-dtype", type=parse, default="float32", metavar="{float32,float16,fp16}",
        help="default: float32; float16/fp16 is a lossy storage approximation",
    )


def _refuse(detail):
    raise CachePrecisionError(
        f"Cache precision/safety preflight failed: {detail}. Use a fresh --output "
        "directory and original RAWs for float32; use explicit --cache-dtype float16 "
        "only to continue a matching approximate float16 cache. Casting old arrays "
        "does not restore lost levels."
    )


def preflight_cache(output, dtype="float32"):
    """Inspect all local arrays, journals and manifests before any network access.

    Return parsed documents with verified cache_dtype fields added in memory.
    No existing file is modified. References must resolve to an array inside this
    output directory; missing or unverifiable legacy crops cannot be marked done.
    """
    output, dtype = Path(output).resolve(), cache_dtype(dtype)
    arrays = {}
    for path in output.rglob("*.npy"):
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            actual = array.dtype
            del array
        except Exception as error:
            _refuse(f"cannot inspect {path}: {error}; an interrupted write may have left "
                    "a truncated array; preserve it for diagnosis and regenerate in a new directory")
        if actual != dtype:
            _refuse(f"{path} has actual dtype {actual.name}, requested {dtype.name}")
        arrays[path.resolve()] = actual.name

    def inspect(value, origin):
        if isinstance(value, list):
            return [inspect(item, origin) for item in value]
        if not isinstance(value, dict):
            return value
        value = {key: inspect(item, origin) for key, item in value.items()}
        declared = value.get("cache_dtype")
        if declared is not None:
            try:
                matches = cache_dtype(declared) == dtype
            except ValueError:
                matches = False
            if not matches:
                _refuse(f"{origin} declares cache_dtype={declared}, requested {dtype.name}")
        for key in ("clean", "crop"):
            reference = value.get(key)
            if not isinstance(reference, str) or not reference.lower().endswith(".npy"):
                continue
            # The three scripts historically use data-root-relative references
            # (one or two parents up), while standalone fixtures may be local.
            ref = Path(reference.replace("\\", "/"))
            candidates = [(root / ref).resolve() for root in (output, *output.parents)]
            actual = next((arrays[path] for path in candidates if path in arrays), None)
            if actual is None:
                _refuse(f"{origin} references missing/unverifiable local array {reference}")
            value["cache_dtype"] = actual
        if value.get("status") == "cached":
            crops = value.get("crops")
            # Screening legitimately produced empty legacy rows. With no arrays
            # to reuse there are no values whose precision needs verification.
            if (not isinstance(crops, list)
                    or any(not isinstance(c, dict) or c.get("cache_dtype") != dtype.name
                           or not isinstance(c.get("clean"), str)
                           or not c["clean"].lower().endswith(".npy") for c in crops)):
                _refuse(f"{origin} has unverifiable cached journal crops")
            value["cache_dtype"] = dtype.name
        return value

    documents = {}
    for path in sorted((*output.rglob("*.json"), *output.rglob("*.jsonl"))):
        try:
            text = path.read_text(encoding="utf-8")
            value = ([json.loads(line) for line in text.splitlines() if line.strip()]
                     if path.suffix == ".jsonl" else json.loads(text))
        except (ValueError, OSError) as error:
            _refuse(f"cannot inspect metadata {path}: {error}")
        documents[path] = inspect(value, path)
    return documents


def save_cache_array(path, values, dtype="float32"):
    """Create exclusively, or reuse an exactly matching array without rewriting it."""
    path, dtype = Path(path), cache_dtype(dtype)
    if np.asarray(values).dtype == np.float16 and dtype == np.dtype("float32"):
        _refuse("float16 source values cannot be promoted into a precision-preserving float32 cache")
    array = np.asarray(values, dtype=dtype)
    try:
        target = path.open("xb")
    except FileExistsError:
        try:
            existing = np.load(path, allow_pickle=False)
        except Exception as error:
            _refuse(f"cannot verify existing {path}: {error}")
        if (existing.dtype != dtype or existing.shape != array.shape
                or not np.array_equal(existing, array)):
            _refuse(f"existing {path} differs in dtype, shape or values; refusing overwrite")
        return existing.dtype.name
    try:
        with target:
            np.save(target, array, allow_pickle=False)
    except BaseException:
        # Only this call's exclusively created file can be removed on failure.
        path.unlink()
        raise
    return array.dtype.name


def manifest_rows(rows, dtype="float32"):
    """Guard against mixed/unverified precision when publishing a local manifest."""
    expected = cache_dtype(dtype).name
    if any(row.get("cache_dtype") != expected for row in rows):
        _refuse(f"manifest contains rows without verified cache_dtype={expected}")
    return rows
