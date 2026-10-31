"""Generated local fixtures only: cache precision, resume and non-destructive I/O."""
import argparse
import builtins
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from denoise_lab.bayer import normalize_raw, read_mosaic
from denoise_lab.cache_precision import (
    CachePrecisionError, add_cache_dtype_argument, manifest_rows, preflight_cache, save_cache_array,
)
from denoise_lab.clean_targets import bin_mosaic


SCRIPTS = ("prepare_bootstrap_cache", "prepare_binned_cache", "cache_real_examples")
ROOT = Path(__file__).resolve().parents[1]


def sensor_fixture(codes):
    return SimpleNamespace(
        raw_image_visible=codes,
        raw_colors_visible=np.tile(np.array([[0, 1], [3, 2]], np.uint8),
                                   (codes.shape[0]//2, codes.shape[1]//2)),
        black_level_per_channel=[4096, 1024, 2048, 8192],
        camera_white_level_per_channel=[65535, 64000, 63000, 60000],
        white_level=65535, color_desc=b"RGBG",
    )


def test_all_65536_codes_normalize_default_cache_read_denormalize(tmp_path):
    codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    raw = sensor_fixture(codes)
    normalized = normalize_raw(raw)
    path = tmp_path/"all_codes.npy"
    assert save_cache_array(path, normalized) == "float32"
    loaded = read_mosaic(path)
    np.testing.assert_array_equal(loaded, normalized)
    black = np.asarray(raw.black_level_per_channel, np.float32)[raw.raw_colors_visible]
    white = np.asarray(raw.camera_white_level_per_channel, np.float32)[raw.raw_colors_visible]
    restored = np.rint(loaded*(white-black) + black).astype(np.uint16)
    np.testing.assert_array_equal(restored, codes)
    assert np.unique(restored).size == 65536
    assert loaded.min() < 0 and loaded.max() > 1


@pytest.mark.parametrize("factor", [2, 4])
def test_signed_binned_fractions_survive_default_cache(tmp_path, factor):
    codes = (np.arange(1024, dtype=np.uint16).reshape(32, 32)*50)
    binned = bin_mosaic(normalize_raw(sensor_fixture(codes)), factor)
    path = tmp_path/"binned.npy"
    save_cache_array(path, binned)
    np.testing.assert_array_equal(read_mosaic(path), binned)
    assert binned.min() < 0
    assert np.any(binned != binned.astype(np.float16).astype(np.float32))


def test_explicit_fp16_is_lossy_and_metadata_uses_actual_dtype(tmp_path):
    parser = argparse.ArgumentParser()
    add_cache_dtype_argument(parser)
    assert parser.parse_args([]).cache_dtype == "float32"
    assert parser.parse_args(["--cache-dtype", "fp16"]).cache_dtype == "float16"
    codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    normalized = normalize_raw(sensor_fixture(codes))
    path = tmp_path/"approximate.npy"
    assert save_cache_array(path, normalized, "fp16") == "float16"
    loaded = read_mosaic(path)
    assert np.count_nonzero(loaded != normalized) > 60000
    assert loaded.min() < 0
    with pytest.raises(CachePrecisionError, match="actual dtype float16"):
        preflight_cache(tmp_path)
    preflight_cache(tmp_path, "float16")
    with pytest.raises(CachePrecisionError, match="float16 source values cannot be promoted"):
        save_cache_array(tmp_path/"upcast.npy", np.load(path))
    assert not (tmp_path/"upcast.npy").exists()


def test_existing_arrays_are_never_overwritten(tmp_path):
    path = tmp_path/"crop.npy"
    values = np.array([[-.01, .080001], [.50001, 1.01]], np.float32)
    save_cache_array(path, values)
    original, timestamp = path.read_bytes(), path.stat().st_mtime_ns
    assert save_cache_array(path, values) == "float32"
    assert path.stat().st_mtime_ns == timestamp
    for changed, dtype in ((values+1e-5, "float32"), (values, "float16"), (values.ravel(), "float32")):
        with pytest.raises(CachePrecisionError, match="refusing overwrite"):
            save_cache_array(path, changed, dtype)
        assert path.read_bytes() == original
    path.write_bytes(b"unreadable historical file")
    with pytest.raises(CachePrecisionError, match="cannot verify existing"):
        save_cache_array(path, values)
    assert path.read_bytes() == b"unreadable historical file"


def test_preflight_checks_orphans_manifests_and_unverifiable_journals(tmp_path):
    np.save(tmp_path/"orphan.npy", np.ones((2, 2), np.float16))
    with pytest.raises(CachePrecisionError, match="actual dtype float16"):
        preflight_cache(tmp_path)
    (tmp_path/"orphan.npy").unlink()
    (tmp_path/"manifest.json").write_text(json.dumps([{"crop": "missing.npy"}]))
    with pytest.raises(CachePrecisionError, match="missing/unverifiable"):
        preflight_cache(tmp_path)
    (tmp_path/"manifest.json").unlink()
    journal = tmp_path/"sources.jsonl"
    journal.write_text(json.dumps({"path": "source.raw", "status": "cached"})+"\n")
    with pytest.raises(CachePrecisionError, match="unverifiable cached journal"):
        preflight_cache(tmp_path)
    journal.unlink()
    np.save(tmp_path/"crop.npy", np.ones((2, 2), np.float32))
    (tmp_path/"manifest.json").write_text(json.dumps([{"crop": "crop.npy", "cache_dtype": "float16"}]))
    with pytest.raises(CachePrecisionError, match="declares cache_dtype=float16"):
        preflight_cache(tmp_path)
    with pytest.raises(CachePrecisionError, match="manifest contains rows"):
        manifest_rows([{"clean": "crop.npy"}])


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbid_smb_imports(monkeypatch):
    original = builtins.__import__
    attempts = []

    def guarded(name, *args, **kwargs):
        if name.startswith("impacket") or name == "sample_smb_metadata":
            attempts.append(name)
            raise AssertionError("SMB must not be imported/accessed for this fixture")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    return attempts


def arguments(name, output, catalog, password, extra=()):
    return [name, "--server", "unused", "--share", "unused", "--username", "unused",
            "--password-file", str(password), "--catalog", str(catalog), "--output", str(output), *extra]


def catalog_item(name):
    return {"path": "fixture.raw", "suffix": ".raw", "camera": "X1D II 50C", "bytes": 4,
            "iso": "6400" if name == "cache_real_examples" else "100", "exposure": "1/100",
            "capture_time": "2025:03:29 12:00:00", "prefix_64k_sha256": "fixture"}


def legacy_cache(name, output, dtype):
    output.mkdir(parents=True)
    path = output/"crop.npy"
    np.save(path, np.array([[-.01, .080001], [.50001, 1.01]], dtype=dtype))
    item = catalog_item(name)
    from denoise_lab.bootstrap import capture_group, split_for_group
    group = capture_group(item["camera"], item["capture_time"], item["path"])
    row = {"group": group, "split": split_for_group(group), "camera": item["camera"],
           "source_sha256": "fixture", "bin_factor": 2}
    if name == "cache_real_examples":
        row.update(item, crop=str(path.relative_to(output.parents[1])))
        metadata = output/"manifest.json"
        metadata.write_text(json.dumps([row]))
    else:
        row["clean"] = str(path.relative_to(output.parent))
        record = {**item, "status": "cached", "crops": [row]}
        metadata = output/"sources.jsonl"
        metadata.write_text(json.dumps(record)+"\n")
    return item, path, metadata


@pytest.mark.parametrize("name", SCRIPTS)
def test_legacy_fp16_preflight_fails_before_network_or_password_read(tmp_path, monkeypatch, capsys, name):
    attempts = forbid_smb_imports(monkeypatch)
    module = load_script(name)
    output = tmp_path/"data"/name
    _, path, metadata = legacy_cache(name, output, "float16")
    originals = {p: p.read_bytes() for p in (path, metadata)}
    monkeypatch.setattr(sys, "argv", arguments(name, output, tmp_path/"missing_catalog", tmp_path/"missing_secret"))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "actual dtype float16" in message
    assert "fresh --output" in message and "--cache-dtype float16" in message
    assert attempts == []
    assert all(p.read_bytes() == content for p, content in originals.items())


@pytest.mark.parametrize("name", SCRIPTS)
def test_unavailable_legacy_crop_is_not_assumed_float32(tmp_path, monkeypatch, capsys, name):
    attempts = forbid_smb_imports(monkeypatch)
    module = load_script(name)
    output = tmp_path/"data"/name
    _, path, metadata = legacy_cache(name, output, "float32")
    path.unlink()
    before = metadata.read_bytes()
    monkeypatch.setattr(sys, "argv", arguments(name, output, tmp_path/"missing_catalog", tmp_path/"missing_secret"))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "missing/unverifiable local array" in capsys.readouterr().err
    assert attempts == [] and metadata.read_bytes() == before


@pytest.mark.parametrize("name", SCRIPTS)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_matching_legacy_resume_verifies_dtype_without_network(tmp_path, monkeypatch, name, dtype):
    attempts = forbid_smb_imports(monkeypatch)
    module = load_script(name)
    output = tmp_path/"data"/name
    item, path, metadata = legacy_cache(name, output, dtype)
    before, timestamp = path.read_bytes(), path.stat().st_mtime_ns
    catalog, password = tmp_path/"catalog.jsonl", tmp_path/"password"
    catalog.write_text(json.dumps(item)+"\n")
    password.write_text("fixture-only")
    extra = ["--cache-dtype", "fp16"] if dtype == "float16" else []
    monkeypatch.setattr(sys, "argv", arguments(name, output, catalog, password, extra))
    module.main()
    assert attempts == []
    assert path.read_bytes() == before and path.stat().st_mtime_ns == timestamp
    manifests = [p for p in output.glob("*.json") if p.name != "summary.json"]
    rows = [row for p in manifests for row in json.loads(p.read_text())]
    assert len(rows) == 1 and rows[0]["cache_dtype"] == dtype
    # Resume again from the now-annotated manifest; journal still predates dtype.
    module.main()
    assert path.read_bytes() == before


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("status", ["cached", "error"])
def test_binned_legacy_empty_or_error_resume_does_not_retry_network(tmp_path, monkeypatch, dtype, status):
    attempts = forbid_smb_imports(monkeypatch)
    module = load_script("prepare_binned_cache")
    output = tmp_path/"data"/"binned"
    output.mkdir(parents=True)
    item = catalog_item("prepare_binned_cache")
    record = {**item, "status": status}
    if status == "cached":
        record["crops"] = []  # Historical screening can accept no windows.
    else:
        record["error"] = "historical fixture error"
    journal = output/"sources.jsonl"
    journal.write_text(json.dumps(record)+"\n")
    before = journal.read_bytes()
    catalog, password = tmp_path/"catalog.jsonl", tmp_path/"password"
    catalog.write_text(json.dumps(item)+"\n")
    password.write_text("fixture-only")
    monkeypatch.setattr(sys, "argv", arguments(
        "prepare_binned_cache", output, catalog, password, ["--cache-dtype", dtype]))
    for _ in range(2):
        module.main()
        assert attempts == [] and journal.read_bytes() == before
        assert not list(output.rglob("*.npy"))
        assert all(json.loads(path.read_text()) == [] for path in output.glob("*.json"))


def test_truncated_cache_preflight_preserves_file_and_explains_recovery(tmp_path):
    path = tmp_path/"interrupted.npy"
    path.write_bytes(b"\x93NUMPY")
    with pytest.raises(CachePrecisionError, match="interrupted write.*truncated array"):
        preflight_cache(tmp_path)
    assert path.read_bytes() == b"\x93NUMPY"


def fake_local_smb(monkeypatch):
    """An in-memory four-byte source. No network libraries or sockets are used."""
    calls = []
    package = ModuleType("impacket")
    smb = ModuleType("impacket.smbconnection")

    class Connection:
        def __init__(self, *args, **kwargs):
            calls.append("connect")

        def login(self, *args):
            pass

        def connectTree(self, *args):
            return 1

        def disconnectTree(self, *args):
            pass

        def logoff(self):
            pass

        def close(self):
            pass

    class Source(io.BytesIO):
        def __init__(self, *args, **kwargs):
            super().__init__(b"data")
            self.cache = {}

    smb.SMBConnection = Connection
    source = ModuleType("sample_smb_metadata")
    source.ReadOnlySMBFile = Source
    monkeypatch.setitem(sys.modules, "impacket", package)
    monkeypatch.setitem(sys.modules, "impacket.smbconnection", smb)
    monkeypatch.setitem(sys.modules, "sample_smb_metadata", source)
    return calls


@pytest.mark.parametrize("name", SCRIPTS)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_writers_generate_actual_dtype_and_metadata_locally(tmp_path, monkeypatch, name, dtype):
    module = load_script(name)
    calls = fake_local_smb(monkeypatch)
    raw = sensor_fixture(np.arange(16, dtype=np.uint16).reshape(4, 4))
    monkeypatch.setattr(module.rawpy, "imread", lambda path: nullcontext(raw))
    # Keep production crop sizes/policy/RNG; only the decoded source is generated.
    mosaic = np.full((1696, 1696), .080001, np.float32)
    mosaic[32, 32] = -.000023
    monkeypatch.setattr(module, "normalize_raw", lambda raw: mosaic.copy())
    item = catalog_item(name)
    output = tmp_path/"data"/name
    catalog, password = tmp_path/"catalog.jsonl", tmp_path/"password"
    catalog.write_text(json.dumps(item)+"\n")
    password.write_text("fixture-only")
    extra = ["--cache-dtype", "fp16"] if dtype == "float16" else []
    if name == "prepare_bootstrap_cache":
        extra += ["--images-per-camera", "1", "--patches-per-image", "1", "--reserve-gib", "0"]
    elif name == "prepare_binned_cache":
        extra += ["--train-images", "1", "--val-images", "1", "--crops", "2:1"]
    monkeypatch.setattr(sys, "argv", arguments(name, output, catalog, password, extra))
    module.main()
    assert calls == ["connect"]
    paths = list(output.rglob("*.npy"))
    assert len(paths) == 1
    cached = np.load(paths[0], allow_pickle=False)
    assert cached.dtype == np.dtype(dtype)
    documents = preflight_cache(output, dtype)
    if name == "cache_real_examples":
        rows = documents[(output/"manifest.json").resolve()]
    else:
        record, = documents[(output/"sources.jsonl").resolve()]
        assert record["cache_dtype"] == dtype
        rows = record["crops"]
    assert rows[0]["cache_dtype"] == dtype
    expected_mosaic = bin_mosaic(mosaic, 2) if name == "prepare_binned_cache" else mosaic
    crop = rows[0].get("sensor_crop", {"x": rows[0].get("crop_x"), "y": rows[0].get("crop_y")})
    x, y = crop["x"], crop["y"]
    np.testing.assert_array_equal(cached, expected_mosaic[y:y+cached.shape[0], x:x+cached.shape[1]].astype(dtype))
    assert rows[0]["group"] and rows[0]["split"] in {"train", "val"}
    assert hashlib.sha256(b"data").hexdigest() in paths[0].name


def test_real_example_collision_does_not_replace_array_or_manifest(tmp_path, monkeypatch):
    module = load_script("cache_real_examples")
    fake_local_smb(monkeypatch)
    raw = sensor_fixture(np.zeros((4, 4), np.uint16))
    monkeypatch.setattr(module.rawpy, "imread", lambda path: nullcontext(raw))
    monkeypatch.setattr(module, "normalize_raw", lambda raw: np.full((1100, 1100), .08, np.float32))
    output = tmp_path/"data"/"real_examples"
    output.mkdir(parents=True)
    destination = output/(hashlib.sha256(b"data").hexdigest()+".npy")
    np.save(destination, np.full((1024, 1024), .09, np.float32))
    manifest = output/"manifest.json"
    manifest.write_text("[]\n")
    before = {p: p.read_bytes() for p in (destination, manifest)}
    catalog, password = tmp_path/"catalog.jsonl", tmp_path/"password"
    catalog.write_text(json.dumps(catalog_item("cache_real_examples"))+"\n")
    password.write_text("fixture-only")
    monkeypatch.setattr(sys, "argv", arguments("cache_real_examples", output, catalog, password))
    with pytest.raises(CachePrecisionError, match="refusing overwrite"):
        module.main()
    assert all(p.read_bytes() == content for p, content in before.items())
