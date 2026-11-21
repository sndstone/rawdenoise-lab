"""Synthetic CPU contracts for checkpoint interpolation and loader compatibility."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from denoise_lab.arch_registry import build_model
from denoise_lab.inference import Denoiser

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/"scripts/interpolate_checkpoints.py"
spec = importlib.util.spec_from_file_location("interpolate_checkpoints", SCRIPT)
interpolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interpolation)


@pytest.fixture
def parents(tmp_path):
    config = dict(width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])
    state = build_model(config).state_dict()
    paths = [tmp_path/"a.pt", tmp_path/"b.pt"]
    for i, path in enumerate(paths):
        torch.save({"model_config": config,
                    "model": {k: torch.full_like(v, i*.02) for k, v in state.items()},
                    "ema": {k: torch.full_like(v, .04+i*.02) for k, v in state.items()},
                    "optimizer": {"discard": True}}, path)
    return paths


@pytest.mark.parametrize("weights", ["model", "ema"])
def test_endpoints_midpoint_and_provenance(tmp_path, parents, weights):
    a, b = parents
    # Non-floating buffers must survive unchanged as well.
    for path in parents:
        saved = torch.load(path, weights_only=True)
        saved[weights]["counter"] = torch.tensor(7)
        saved[weights]["flag"] = torch.tensor(True)
        torch.save(saved, path)
    originals = [torch.load(p, weights_only=True) for p in parents]
    outputs = interpolation.interpolate_checkpoints(str(a), str(b), [0, .5, 1], tmp_path/"mix", weights)
    assert [p.name for p in outputs] == ["mix_a0.pt", "mix_a05.pt", "mix_a1.pt"]
    for alpha, path in zip([0, .5, 1], outputs):
        saved = torch.load(path, weights_only=True)
        assert set(saved) == {"model_config", "model", "provenance"}
        assert saved["model_config"] == originals[0]["model_config"]
        for key, tensor in saved["model"].items():
            left, right = (p[weights][key] for p in originals)
            expected = left if alpha == 0 or not left.is_floating_point() else (
                right if alpha == 1 else (left+right)/2)
            assert torch.equal(tensor, expected)
            assert tensor.dtype == left.dtype
        assert saved["provenance"] == {
            "parents": {label: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for label, p in zip(["a", "b"], parents)},
            "weights": weights, "alpha": alpha}


@pytest.mark.parametrize("mismatch", ["model_config", "keys", "shapes", "dtypes", "non-floating",
                                      "missing weights", "non-tensor"])
def test_incompatible_parents_rejected(tmp_path, parents, mismatch):
    a, b = parents
    saved = torch.load(b, weights_only=True)
    key = next(iter(saved["model"]))
    match = mismatch
    if mismatch == "model_config":
        saved["model_config"]["width"] = 3
    elif mismatch == "keys":
        del saved["model"][key]
    elif mismatch == "shapes":
        saved["model"][key] = torch.zeros(1)
    elif mismatch == "dtypes":
        saved["model"][key] = saved["model"][key].double()
    elif mismatch == "non-floating":
        first = torch.load(a, weights_only=True)
        first["model"]["counter"] = torch.tensor(1)
        torch.save(first, a)
        saved["model"]["counter"] = torch.tensor(2)
    elif mismatch == "missing weights":
        saved["model"] = None
        match = "no model weights"
    else:
        saved["model"][key] = 1
        match = "must be a state mapping"
    torch.save(saved, b)
    with pytest.raises(ValueError, match=match):
        interpolation.interpolate_checkpoints(a, b, [.5], tmp_path/"mix")
    assert not (tmp_path/"mix").exists()


@pytest.mark.parametrize("alphas", [[-.1], [1.1], [float("nan")], [float("inf")], [], [.5, .5]])
def test_invalid_alphas_rejected(tmp_path, parents, alphas):
    with pytest.raises(ValueError, match="alpha"):
        interpolation.interpolate_checkpoints(*parents, alphas, tmp_path/"mix")
    assert not (tmp_path/"mix").exists()


def test_cli_and_finite_inference(tmp_path, parents):
    output = tmp_path/"mix"
    result = subprocess.run([sys.executable, str(SCRIPT), "--a", str(parents[0]), "--b", str(parents[1]),
                             "--alpha", ".25", "--alpha", ".5", "--alpha", ".75",
                             "--weights", "ema", "--output-dir", str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert sorted(p.name for p in output.iterdir()) == ["mix_a025.pt", "mix_a05.pt", "mix_a075.pt"]
    for path in output.iterdir():
        denoiser = Denoiser.load(path, device="cpu")
        packed = np.full((4, 16, 16), .2, dtype=np.float32)
        prediction = denoiser.denoise_packed(packed, prepass=False, highlights="off", tile_size=16, overlap=0)
        assert prediction.shape == packed.shape
        assert np.isfinite(prediction).all()


def test_refuse_overwrite_and_directory(tmp_path, parents):
    output = tmp_path/"mix"
    paths = interpolation.interpolate_checkpoints(*parents, [.5], output)
    original = paths[0].read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        interpolation.interpolate_checkpoints(*parents, [.25, .5], output)
    assert paths[0].read_bytes() == original
    assert not (output/"mix_a025.pt").exists()
    with pytest.raises(ValueError, match="checkpoint file"):
        interpolation.interpolate_checkpoints(tmp_path, parents[1], [.5], output/"other")
