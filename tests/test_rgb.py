"""RGB precision, channel identity, color contract and checkpoint boundaries."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile
import torch

from denoise_lab.arch_registry import build_model
from denoise_lab.inference import Denoiser
from denoise_lab.rgb import (RGBMetadata, normalize_rgb, read_rgb_tiff, rgb_codes,
                             rgb_contract, write_rgb_tiff)
from denoise_lab.rgb_data import PairedRGBDataset
from denoise_lab.tiling import tiled_inference

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(width=2, enc_blk_nums=[0], middle_blk_num=0, dec_blk_nums=[0])


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def save_model(path, *, rgb=True, contract=None):
    config = dict(TINY, **({"arch": "rgb_nafnet"} if rgb else {}))
    model = build_model(config)
    with torch.no_grad():
        model.body.ending.weight.zero_()
        model.body.ending.bias.zero_()
    saved = dict(model_config=config, model=model.state_dict())
    if rgb:
        saved["rgb_contract"] = contract or rgb_contract("srgb")
    torch.save(saved, path)
    return saved


@pytest.fixture
def rgb_checkpoint(tmp_path):
    path = tmp_path / "rgb.pt"
    save_model(path)
    return path


def test_all_uint16_values_survive_tiled_model_and_tiff(tmp_path, rgb_checkpoint):
    ramp = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    # All values in every channel, in distinct orders to expose channel swaps.
    original = np.stack((ramp, ramp[::-1], np.roll(ramp, 137, axis=1)), axis=-1)
    image = normalize_rgb(original)
    assert image.dtype == np.float32
    denoiser = Denoiser.load(rgb_checkpoint, device="cpu")
    predicted = denoiser.denoise_rgb(image, color_contract=rgb_contract("srgb"),
                                    tile_size=64, overlap=17)
    np.testing.assert_array_equal(rgb_codes(predicted), original)
    path = tmp_path / "all_codes.tiff"
    write_rgb_tiff(path, predicted, RGBMetadata(rgb_contract("srgb")))
    np.testing.assert_array_equal(tifffile.imread(path), original)
    reread, meta = read_rgb_tiff(path)
    np.testing.assert_array_equal(rgb_codes(reread), original)
    assert meta.contract == denoiser.rgb_contract


@pytest.mark.parametrize("shape", [(1, 1), (19, 27), (32, 32)])
def test_experimental_resize_model_does_not_invent_flat_spatial_structure(shape):
    # Random nonidentity weights exercise every resolution and pixel phase.
    # Resize + replicated boundaries preserve spatially constant features.
    torch.manual_seed(7)
    config = dict(arch="rgb_nafnet", width=4, enc_blk_nums=[1,1], middle_blk_num=1,
                  dec_blk_nums=[1,1], boundary_mode="replicate", upsample_mode="nearest")
    model = build_model(config).eval()
    image = torch.tensor([.1,.3,.7]).reshape(1,3,1,1).expand(1,3,*shape)
    with torch.inference_mode():
        prediction = model(image)
    torch.testing.assert_close(prediction, prediction[..., :1, :1].expand_as(prediction), rtol=1e-5, atol=1e-6)
    assert prediction.shape == image.shape


@pytest.mark.parametrize("shape", [(3, 1, 1), (3, 1, 7), (3, 19, 35)])
def test_rgb_tiling_geometry_and_strength(shape, rgb_checkpoint):
    image = np.random.default_rng(2).random(shape, dtype=np.float32)
    model = Denoiser.load(rgb_checkpoint, device="cpu")
    result = model.denoise_rgb(image, color_contract=rgb_contract("srgb"), tile_size=32, overlap=9)
    assert result.shape == shape and result.dtype == np.float32
    np.testing.assert_allclose(result, image, atol=2e-7)
    zero = model.denoise_rgb(image, color_contract=rgb_contract("srgb"), strength=0)
    np.testing.assert_array_equal(zero, image)
    assert not np.shares_memory(zero, image)


def test_no_bayer_reinterpretation_even_strength_zero(tmp_path, rgb_checkpoint):
    bayer = tmp_path / "bayer.pt"
    bayer_saved = save_model(bayer, rgb=False)
    rgb = Denoiser.load(rgb_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="Bayer input requires"):
        rgb.denoise_packed(np.zeros((4, 7, 9), np.float32), strength=0)
    with pytest.raises(ValueError, match="Bayer weights cannot process RGB"):
        Denoiser.load(bayer, device="cpu").denoise_rgb(np.zeros((3, 7, 9), np.float32),
            color_contract=rgb_contract("srgb"), strength=0)
    bayer_saved["model_config"]["arch"] = "rgb_nafnet"
    bayer_saved["rgb_contract"] = rgb_contract("srgb")
    torch.save(bayer_saved, bayer)
    with pytest.raises(ValueError, match="Checkpoint/model mismatch"):
        Denoiser.load(bayer, device="cpu")
    saved = save_model(bayer)
    saved.pop("rgb_contract")
    torch.save(saved, bayer)
    with pytest.raises(ValueError, match="requires an explicit rgb_contract"):
        Denoiser.load(bayer, device="cpu")
    saved = save_model(bayer, rgb=False)
    saved["rgb_contract"] = rgb_contract("srgb")
    torch.save(saved, bayer)
    with pytest.raises(ValueError, match="requires arch=rgb_nafnet"):
        Denoiser.load(bayer, device="cpu")
    torch.save({"model": saved["model"]}, bayer)
    with pytest.raises(ValueError, match="requires an explicit model_config"):
        Denoiser.load(bayer, device="cpu")


def test_rgb_architecture_fixed_three_channels():
    model = build_model(dict(TINY, arch="rgb_nafnet"))
    assert model.body.intro.in_channels == model.body.ending.out_channels == 3
    with pytest.raises(ValueError, match="exactly 3"):
        build_model(dict(TINY, arch="rgb_nafnet", in_channels=4))


def test_ema_training_key_selected_for_both_architectures(tmp_path):
    for rgb in (False, True):
        path = tmp_path / f"ema_{rgb}.pt"
        saved = save_model(path, rgb=rgb)
        ema = {k: v.clone() for k, v in saved["model"].items()}
        ema["body.ending.bias"].fill_(.1)
        saved["ema_model"] = ema
        saved["ema"] = saved["model"]  # Current training key wins over legacy.
        torch.save(saved, path)
        loaded = Denoiser.load(path, device="cpu", weights="ema")
        torch.testing.assert_close(loaded._call.body.ending.bias, ema["body.ending.bias"])


def test_tiff_metadata_icc_orientation_resolution_and_overwrite(tmp_path):
    path = tmp_path / "profiled.tif"
    profile = b"opaque ICC bytes retained exactly; no transform requested"
    contract = rgb_contract("icc", profile)
    image = np.random.default_rng(4).random((3, 17, 29), dtype=np.float32)
    metadata = RGBMetadata(contract, profile, orientation=6, resolution=((300, 1), (240, 1)), resolution_unit=2)
    write_rgb_tiff(path, image, metadata)
    reread, meta = read_rgb_tiff(path)
    assert meta.contract == contract and meta.icc_profile == profile and meta.orientation == 6
    assert meta.resolution == ((300, 1), (240, 1)) and meta.resolution_unit == 2
    np.testing.assert_array_equal(rgb_codes(reread), rgb_codes(image))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_rgb_tiff(path, image, metadata)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="must be uint16 TIFF"):
        write_rgb_tiff(tmp_path / "bad.dng", image, metadata)
    with pytest.raises(ValueError, match="conflicts with TIFF metadata"):
        read_rgb_tiff(path, color_space="srgb")


def test_metadata_declarations_and_color_mismatch(tmp_path, rgb_checkpoint):
    path = tmp_path / "bare.tif"
    codes = np.ones((5, 7, 3), np.uint16)
    tifffile.imwrite(path, codes, photometric="rgb", metadata=None)
    with pytest.raises(ValueError, match="no explicit color space"):
        read_rgb_tiff(path)
    image, _ = read_rgb_tiff(path, color_space="linear-srgb")
    with pytest.raises(ValueError, match="does not match checkpoint"):
        Denoiser.load(rgb_checkpoint, device="cpu").denoise_rgb(image,
            color_contract=rgb_contract("linear-srgb"), strength=0)
    path = tmp_path / "tagged.tif"
    write_rgb_tiff(path, image, RGBMetadata(rgb_contract("linear-srgb")))
    with pytest.raises(ValueError, match="conflicts with TIFF metadata"):
        read_rgb_tiff(path, color_space="srgb")
    colored = tmp_path / "colorimetry.tif"
    tifffile.imwrite(colored, codes, photometric="rgb", extratags=[(318, "2I", 2, (1, 3, 1, 3), False)])
    with pytest.raises(ValueError, match="colorimetry/transfer-function tags are unsupported"):
        read_rgb_tiff(colored, color_space="srgb")


def test_explicit_exif_color_space_is_respected(tmp_path):
    path = tmp_path / "exif_srgb.tif"
    # ColorSpace may also appear directly in an IFD; the nested EXIF form is
    # handled by the same reader after tifffile resolves the EXIF pointer.
    tifffile.imwrite(path, np.ones((5, 7, 3), np.uint16), photometric="rgb",
                     extratags=[(40961, "H", 1, 1, False)])
    _, metadata = read_rgb_tiff(path)
    assert metadata.contract == rgb_contract("srgb")
    with pytest.raises(ValueError, match="conflicts with EXIF sRGB"):
        read_rgb_tiff(path, color_space="linear-srgb")


def test_icc_checkpoint_requires_exact_profile_even_strength_zero(tmp_path):
    first, second = b"first exact ICC profile", b"second exact ICC profile"
    checkpoint = tmp_path / "icc.pt"
    save_model(checkpoint, contract=rgb_contract("icc", first))
    denoiser = Denoiser.load(checkpoint, device="cpu")
    image = np.full((3, 5, 7), .25, np.float32)
    result = denoiser.denoise_rgb(image, color_contract=rgb_contract("icc", first), tile_size=16, overlap=4)
    np.testing.assert_allclose(result, image, atol=1e-7)
    with pytest.raises(ValueError, match="does not match checkpoint"):
        denoiser.denoise_rgb(image, color_contract=rgb_contract("icc", second), strength=0)
    missing = tmp_path / "missing_profile.tif"
    tifffile.imwrite(missing, rgb_codes(image), photometric="rgb", metadata=None,
        description=json.dumps({"denoise_lab_rgb": rgb_contract("icc", first)}))
    with pytest.raises(ValueError, match="requires an exact icc_sha256"):
        read_rgb_tiff(missing)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_reject_non_uint16_tiff(tmp_path, dtype):
    path = tmp_path / "wrong.tif"
    tifffile.imwrite(path, np.zeros((4, 5, 3), dtype), photometric="rgb")
    with pytest.raises(ValueError, match="must be uint16"):
        read_rgb_tiff(path, color_space="srgb")


def test_reject_alpha_pages_and_accept_planar_rgb(tmp_path):
    alpha = tmp_path / "alpha.tif"
    tifffile.imwrite(alpha, np.zeros((4, 5, 4), np.uint16), photometric="rgb")
    with pytest.raises(ValueError, match="exactly three channels"):
        read_rgb_tiff(alpha, color_space="srgb")
    pages = tmp_path / "pages.tif"
    with tifffile.TiffWriter(pages) as writer:
        for _ in range(2):
            writer.write(np.zeros((4, 5, 3), np.uint16), photometric="rgb")
    with pytest.raises(ValueError, match="exactly one"):
        read_rgb_tiff(pages, color_space="srgb")
    codes = np.arange(60, dtype=np.uint16).reshape(3, 4, 5)
    planar = tmp_path / "planar.tif"
    tifffile.imwrite(planar, codes, photometric="rgb", planarconfig="separate")
    image, _ = read_rgb_tiff(planar, color_space="srgb")
    np.testing.assert_array_equal(rgb_codes(image), codes.transpose(1, 2, 0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rgb_rejected(bad, rgb_checkpoint):
    image = np.zeros((3, 4, 5), np.float32)
    image[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        Denoiser.load(rgb_checkpoint, device="cpu").denoise_rgb(image,
            color_contract=rgb_contract("srgb"), strength=0)
    with pytest.raises(ValueError, match="finite"):
        rgb_codes(image)
    model = Denoiser.load(rgb_checkpoint, device="cpu")
    model._call = lambda tile: torch.full_like(tile, bad)
    with pytest.raises(ValueError, match="finite"):
        model.denoise_rgb(np.zeros_like(image), color_contract=rgb_contract("srgb"), tile_size=16, overlap=4)


def test_quantization_clips_rounds_and_rejects_shape_broadcast():
    image = np.array([[[-.1, .49/65535, .51/65535, 1.1]]] * 3, np.float32)
    np.testing.assert_array_equal(rgb_codes(image)[0, :, 0], [0, 0, 1, 65535])
    with pytest.raises(ValueError, match="does not match input"):
        tiled_inference(lambda x: x[:, :1], torch.zeros(1, 3, 4, 5), tile_size=16, overlap=4)


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/denoise.py"), *map(str, args)],
                          capture_output=True, text=True)


def test_rgb_cli_geometry_default_tiff_metadata_and_overwrite(tmp_path, rgb_checkpoint):
    original = np.arange(17*29*3, dtype=np.uint16).reshape(17, 29, 3) + 20000
    source = tmp_path / "source.tiff"
    write_rgb_tiff(source, normalize_rgb(original), RGBMetadata(rgb_contract("srgb"), orientation=8))
    before = source.read_bytes()
    command = (source, "--model", rgb_checkpoint, "--device", "cpu", "--output-dir", tmp_path / "out")
    result = cli(*command)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "out/source_denoised.tiff"
    np.testing.assert_array_equal(tifffile.imread(output), original)
    _, meta = read_rgb_tiff(output)
    assert meta.orientation == 8 and meta.contract == rgb_contract("srgb")
    result = cli(*command)
    assert result.returncode == 1 and "Refusing to overwrite" in result.stdout
    assert source.read_bytes() == before


@pytest.mark.parametrize("output_format", ["dng", "npy"])
def test_rgb_cli_rejects_bayer_output(tmp_path, rgb_checkpoint, output_format):
    source = tmp_path / "source.tif"
    write_rgb_tiff(source, np.zeros((3, 5, 7), np.float32), RGBMetadata(rgb_contract("srgb")))
    result = cli(source, "--model", rgb_checkpoint, "--device", "cpu", "--format", output_format)
    assert result.returncode == 1 and "RGB output must be uint16 TIFF" in result.stdout
    assert not (tmp_path / f"source_denoised.{output_format}").exists()


def test_cli_rgb_bayer_model_mismatch(tmp_path):
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.zeros((5, 7, 3), np.uint16), photometric="rgb")
    checkpoint = tmp_path / "bayer.pt"
    save_model(checkpoint, rgb=False)
    result = cli(source, "--model", checkpoint, "--device", "cpu", "--strength", "0")
    assert result.returncode == 1 and "Input/model mismatch" in result.stdout
    assert not (tmp_path / "source_denoised.tiff").exists()


def manifest_fixture(tmp_path):
    pairs = []
    for index, split in enumerate(("train", "validation", "test")):
        for role in ("noisy", "clean"):
            codes = np.arange(16*20*3, dtype=np.uint16).reshape(16, 20, 3) + 10000 + index
            if role == "noisy":
                codes = codes + 3
            write_rgb_tiff(tmp_path / f"{split}_{role}.tif", normalize_rgb(codes), RGBMetadata(rgb_contract("srgb")))
        pairs.append(dict(noisy=f"{split}_noisy.tif", clean=f"{split}_clean.tif", split=split,
                          scene=f"scene-{index}", source="generated CPU fixture",
                          usage_terms="test fixture only", target_provenance="generated independent array",
                          camera="fixture", iso=100, processing="srgb fixture, no external model"))
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(dict(version=1, rgb_contract=rgb_contract("srgb"), pairs=pairs)))
    return path


def test_rgb_manifest_pair_crops_channels_and_provenance(tmp_path):
    path = manifest_fixture(tmp_path)
    ds = PairedRGBDataset(path, crop_size=8)
    torch.manual_seed(5)
    sample = ds[0]
    assert sample["noisy"].shape == (3, 8, 8) and sample["noisy"].dtype == torch.float32
    torch.testing.assert_close(sample["noisy"] - sample["clean"], torch.full((3, 8, 8), 3/65535), atol=2e-8, rtol=0)
    torch.testing.assert_close(sample["noisy"][1] - sample["noisy"][0], torch.full((8, 8), 1/65535), atol=2e-8, rtol=0)
    manifest = json.loads(path.read_text())
    manifest["pairs"][1]["scene"] = manifest["pairs"][0]["scene"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scene leakage"):
        PairedRGBDataset(path)
    manifest["pairs"][1]["scene"] = "other"
    manifest["pairs"][1]["noisy"] = manifest["pairs"][0]["noisy"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="file leakage"):
        PairedRGBDataset(path)
    manifest["pairs"][1]["noisy"] = "validation_noisy.tif"
    del manifest["pairs"][0]["target_provenance"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target_provenance"):
        PairedRGBDataset(path)


def test_rgb_manifest_rejects_unaligned_or_color_mismatched_pairs(tmp_path):
    manifest = manifest_fixture(tmp_path)
    clean = tmp_path / "train_clean.tif"
    clean.unlink()
    write_rgb_tiff(clean, np.zeros((3, 16, 20), np.float32),
                   RGBMetadata(rgb_contract("srgb"), orientation=6))
    with pytest.raises(ValueError, match="matching orientation"):
        PairedRGBDataset(manifest, crop_size=8)[0]
    clean.unlink()
    write_rgb_tiff(clean, np.zeros((3, 17, 20), np.float32), RGBMetadata(rgb_contract("srgb")))
    with pytest.raises(ValueError, match="matching registered geometry"):
        PairedRGBDataset(manifest, crop_size=8)[0]
    clean.unlink()
    write_rgb_tiff(clean, np.zeros((3, 16, 20), np.float32), RGBMetadata(rgb_contract("linear-srgb")))
    with pytest.raises(ValueError, match="conflicts with TIFF metadata"):
        PairedRGBDataset(manifest, crop_size=8)[0]


def test_rgb_training_script_one_step_and_checkpoint_inference(tmp_path):
    manifest = manifest_fixture(tmp_path)
    output = tmp_path / "trained.pt"
    command = [sys.executable, str(ROOT / "scripts/train_rgb.py"), "--manifest", str(manifest),
               "--output", str(output), "--steps", "1", "--crop-size", "8", "--width", "2",
               "--enc-blocks", "1", "--middle-blocks", "1", "--dec-blocks", "1", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = torch.load(output, weights_only=True)
    assert saved["experimental_rgb"] and saved["initialization"] == "independent_random"
    assert saved["step"] == 1 and saved["model_config"]["arch"] == "rgb_nafnet"
    assert len(saved["sources"]) == 6 and saved["metrics"]["train_last_l1"] > 0
    torch.manual_seed(saved["seed"])
    fresh = build_model(saved["model_config"])
    # The checkpoint contains a real optimizer update, not a dummy identity.
    assert not torch.equal(saved["model"]["body.ending.weight"], fresh.state_dict()["body.ending.weight"])
    denoiser = Denoiser.load(output, device="cpu")
    image, metadata = read_rgb_tiff(tmp_path / "validation_noisy.tif")
    prediction = denoiser.denoise_rgb(image, color_contract=metadata.contract, tile_size=16, overlap=4)
    assert prediction.shape == image.shape and np.isfinite(prediction).all()
    repeat = subprocess.run(command, capture_output=True, text=True)
    assert repeat.returncode != 0 and "Refusing to overwrite" in repeat.stderr


def test_rgb_stability_recipe_trains_and_reloads_architecture(tmp_path):
    manifest = manifest_fixture(tmp_path)
    output = tmp_path/"stable.pt"
    command = [sys.executable,str(ROOT/"scripts/train_rgb.py"),"--manifest",str(manifest),"--output",str(output),
        "--steps","1","--crop-size","8","--width","2","--enc-blocks","1","--dec-blocks","1",
        "--middle-blocks","1","--identity-init","--identity-weight",".2","--flat-identity-weight",".2",
        "--color-bias-weight",".5","--boundary-mode","replicate","--upsample-mode","nearest"]
    result = subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 0,result.stdout+result.stderr
    saved = torch.load(output,weights_only=True)
    assert saved['model_config']['upsample_mode'] == 'nearest'
    assert saved['model_config']['boundary_mode'] == 'replicate'
    assert saved['model']['body.ending.weight'].abs().max() > 0
    model = Denoiser.load(output,device='cpu')
    image = np.full((3,13,17),.2,np.float32)
    prediction = model.denoise_rgb(image,color_contract=rgb_contract('srgb'),tile_size=16,overlap=4)
    assert prediction.shape == image.shape and np.isfinite(prediction).all()
