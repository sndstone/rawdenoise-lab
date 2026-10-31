"""Real sensor-code precision and explicit checkpoint-selection contracts."""
import numpy as np
import pytest
import tifffile

from denoise_lab.bayer import PHASES, normalize_codes, read_mosaic, unify
from denoise_lab.checkpoint import checkpoint_weights
from denoise_lab.arch_registry import build_model
from denoise_lab.dng import RawMetadata, write_dng
import rawpy
import subprocess
import sys
import torch
from pathlib import Path


@pytest.mark.parametrize("phase", PHASES)
def test_all_16bit_sensor_codes_survive_normalization(phase):
    codes = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    normalized = normalize_codes(codes, 4096, 65535, phase)
    restored = np.rint(normalized*61439 + 4096).astype(np.uint16)
    np.testing.assert_array_equal(restored, unify(codes, phase))
    assert normalized.dtype == np.float32
    assert normalized.min() < 0


@pytest.mark.parametrize("suffix", [".npy", ".tiff"])
def test_integer_targets_require_calibration(tmp_path, suffix):
    codes = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path = tmp_path / ("sensor" + suffix)
    if suffix == ".npy":
        np.save(path, codes)
    else:
        tifffile.imwrite(path, codes)
    with pytest.raises(ValueError, match="explicit black_level and white_level"):
        read_mosaic(path)
    np.testing.assert_allclose(read_mosaic(path, black_level=16, white_level=65535),
                               (codes.astype(np.float32)-16)/65519)


def test_per_site_codes_and_invalid_metadata():
    black = np.array([[512, 520], [528, 536]], np.float32)
    white = np.array([[65535, 65000], [64000, 63000]], np.float32)
    codes = np.tile(black, (4, 4)).astype(np.uint16)
    np.testing.assert_array_equal(normalize_codes(codes, black, white), np.zeros((8, 8)))
    with pytest.raises(ValueError, match="Finite white levels"):
        normalize_codes(codes, 10, 10)
    with pytest.raises(ValueError, match="Finite white levels"):
        normalize_codes(codes, np.nan, 65535)


def test_training_and_legacy_ema_weights():
    model, current, legacy = {"w": 1}, {"w": 2}, {"w": 3}
    saved = {"model": model, "ema_model": current, "ema": legacy}
    assert checkpoint_weights(saved) is model
    assert checkpoint_weights(saved, "ema") is current
    assert checkpoint_weights({"ema": legacy}, "ema") is legacy
    with pytest.raises(ValueError, match="no ema weights"):
        checkpoint_weights({"model": model}, "ema")


def test_normalized_targets_keep_signed_values_and_reject_nonfinite(tmp_path):
    path = tmp_path / "target.npy"
    values = np.linspace(-.01, 1.1, 64, dtype=np.float32).reshape(8, 8)
    np.save(path, values)
    np.testing.assert_array_equal(read_mosaic(path), values)
    values[0, 0] = np.nan
    np.save(path, values)
    with pytest.raises(ValueError, match="must be finite"):
        read_mosaic(path)


@pytest.mark.parametrize("phase", ["RGGB", "BGGR"])
def test_full_16bit_bayer_cli_identity_roundtrip(tmp_path, phase):
    codes = np.arange(65536,dtype=np.uint16).reshape(256,256)
    metadata = RawMetadata(pattern=np.array(["RGB".index(c) for c in phase],np.uint8).reshape(2,2),
                           black=np.array([[4096.,4100.],[4104.,4108.]]),white=np.full((2,2),65535.))
    source = tmp_path/"all_codes.dng"
    write_dng(source,codes,metadata,normalized=False)
    original = source.read_bytes()
    config=dict(width=2,enc_blk_nums=[1],middle_blk_num=1,dec_blk_nums=[1])
    model=build_model(config)
    with torch.no_grad():
        model.body.ending.weight.zero_();model.body.ending.bias.zero_()
    checkpoint=tmp_path/"identity.pt"
    torch.save(dict(model_config=config,model=model.state_dict()),checkpoint)
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/"scripts/denoise.py"),str(source),
             "--model",str(checkpoint),"--device","cpu","--no-prepass"]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    with rawpy.imread(str(tmp_path/"all_codes_denoised.dng")) as decoded:
        np.testing.assert_array_equal(decoded.raw_image_visible,codes)
        assert decoded.raw_image_visible.dtype == np.uint16
    assert source.read_bytes()==original
