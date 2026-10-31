"""Exporter construction contracts that do not require coremltools."""
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from denoise_lab.arch_registry import build_model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_coreml", ROOT / "scripts/export_coreml.py")
export_coreml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_coreml)


@pytest.mark.parametrize("weights", ["model", "ema"])
def test_released_config_construction(weights):
    config = yaml.safe_load((ROOT / "configs/round4/r4_w40s4_plain.yaml").read_text())["model"]
    assert config["arch"] == "nafnet" and config["width"] == 40
    assert len(config["enc_blk_nums"]) == len(config["dec_blk_nums"]) == 4
    original = build_model(config)
    state = original.state_dict()
    # Distinguish selected weights without running a full-size inference.
    key = next(iter(state))
    state[key] = torch.full_like(state[key], 0.125)
    model = export_coreml.model_from_checkpoint({"model_config": config, weights: state}, weights)
    assert not model.training
    torch.testing.assert_close(model.state_dict()[key], state[key])


def test_missing_ema_and_conditioned_checkpoint_rejected():
    config = dict(width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)
    with pytest.raises(ValueError, match="no ema weights"):
        export_coreml.model_from_checkpoint({"model_config": config}, "ema")
    with pytest.raises(ValueError, match="cond=none"):
        export_coreml.model_from_checkpoint({"model_config": {**config, "cond": "noise_map"}})


def test_training_ema_and_rgb_export_contract():
    config = dict(width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)
    state = build_model(config).state_dict()
    model = export_coreml.model_from_checkpoint({"model_config": config, "ema_model": state}, "ema")
    assert not model.training
    with pytest.raises(ValueError, match="Bayer checkpoints only"):
        export_coreml.model_from_checkpoint({"model_config": {"arch": "rgb_nafnet"}})
