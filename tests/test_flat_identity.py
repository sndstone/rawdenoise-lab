"""Bounded training contracts for optional independent clean-flat supervision."""
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from denoise_lab.arch_registry import build_model
from denoise_lab.flat_identity import make_flat_identity
from denoise_lab.losses import DenoiseLoss

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("flat_train", ROOT/"scripts/train_arch_study.py")
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)
MODEL = dict(arch="nafnet", cond="none", width=2, enc_blk_nums=[1],
             middle_blk_num=1, dec_blk_nums=[1])


def test_disabled_and_enabled_private_rng():
    model = build_model(MODEL)
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    for cfg in (None, {}, {"enabled": False, "weight": float("nan")}):
        assert make_flat_identity(object(), cfg) is None
    reg = make_flat_identity(model, {"enabled": True, "size": 8}, seed=9)
    a = reg.sample("cpu")
    loss = reg(model, DenoiseLoss(), "cpu")
    loss.backward()
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == random_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    other = make_flat_identity(model, {"enabled": True, "size": 8}, seed=9)
    assert torch.equal(a, other.sample("cpu"))
    assert not torch.equal(a, reg.sample("cpu"))
    state = reg.generator.get_state()
    b = reg.sample("cpu")
    reg.generator.set_state(state)
    assert torch.equal(b, reg.sample("cpu"))


def test_continuous_fields_layout_and_coverage():
    reg = make_flat_identity(build_model(MODEL),
                             dict(enabled=True, batch_size=256, size=13), seed=130)
    x = reg.sample("cpu")
    assert x.shape == (256, 4, 13, 13) and x.dtype == torch.float32
    assert x.is_contiguous() and not x.requires_grad and torch.isfinite(x).all()
    assert x.min() >= 0 and x.max() <= 1
    assert (x < 1).flatten(1).any(1).all()  # never the exact all-white gate fixture
    assert torch.equal(x[:, 1], x[:, 3])
    assert (x[:, 1].mean((1, 2)) < 1e-3).any()
    assert (x[:, 1].mean((1, 2)) > .9).any()
    assert ((x == 1).flatten(1).any(1) & (x < .95).flatten(1).any(1)).any()
    scalar = (x[:, 0] == x[:, 1]).flatten(1).all(1) & (x[:, 2] == x[:, 1]).flatten(1).all(1)
    assert scalar.any() and (~scalar).any()
    assert (x[:, 0].mean((1, 2)) > x[:, 1].mean((1, 2))).any()
    assert (x[:, 2].mean((1, 2)) < x[:, 1].mean((1, 2))).any()
    # Some are exactly flat, others planar before sensor clipping.
    spatial = x.std((2, 3)).max(1).values
    assert (spatial == 0).any() and (spatial > 1e-5).any()
    unclipped = (x < 1).flatten(1).all(1)
    assert torch.diff(x[unclipped], n=2, dim=-1).abs().max() < 5e-7
    assert torch.diff(x[unclipped], n=2, dim=-2).abs().max() < 5e-7
    assert torch.unique(x[:, 1, 6, 6]).numel() > 240


@pytest.mark.parametrize("cfg", [{**MODEL, "arch": "study"},
                                 {**MODEL, "cond": "noise_map"},
                                 {**MODEL, "dropout": .1}])
def test_unsupported_contract_rejected(cfg):
    with pytest.raises(ValueError, match="unconditioned four-channel Bayer NAFNet"):
        make_flat_identity(build_model(cfg), dict(enabled=True))


def test_three_channel_contract_rejected():
    model = build_model(MODEL)
    model.body.intro = torch.nn.Conv2d(3, 2, 3, padding=1)
    model.body.ending = torch.nn.Conv2d(2, 3, 3, padding=1)
    with pytest.raises(ValueError, match="four-channel Bayer"):
        make_flat_identity(model, dict(enabled=True))


@pytest.mark.parametrize("options", [dict(weight=0), dict(weight=float("inf")),
                                     dict(size=0), dict(batch_size=1.5)])
def test_invalid_settings(options):
    with pytest.raises(ValueError):
        make_flat_identity(build_model(MODEL), dict(enabled=True, **options))


def test_weighted_signed_fp32_loss_and_gradient():
    model = build_model(MODEL)
    reg = make_flat_identity(model, dict(enabled=True, size=8, batch_size=2), seed=8)
    criterion = DenoiseLoss(eps=1e-5, gamma_weight=.25)
    target = reg.sample("cpu")
    reg.sample = lambda device: target
    # Force negative outputs and verify the exact signed loss, without clipping.
    with torch.no_grad():
        model.body.ending.weight.zero_()
        model.body.ending.bias.fill_(-2)
    prediction = model(target)
    assert (prediction < 0).all()
    expected = criterion(prediction, target)*reg.weight
    actual = reg(model, criterion, "cpu")*reg.weight
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    grad = model.body.ending.bias.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    expected_grad = grad.clone()
    model.zero_grad()
    expected.backward()
    torch.testing.assert_close(model.body.ending.bias.grad, expected_grad, rtol=0, atol=0)


def test_nonfinite_prediction_and_loss():
    model = build_model(MODEL)
    reg = make_flat_identity(model, dict(enabled=True, size=8))
    with pytest.raises(FloatingPointError, match="loss"):
        reg(model, lambda pred, target: pred.mean()*float("nan"), "cpu")
    with torch.no_grad():
        model.body.ending.bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="prediction"):
        reg(model, DenoiseLoss(), "cpu")


def test_one_step_cpu_and_disabled_exact_behavior(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT/"configs/arch_study/smoke_nafnet_s1.yaml").read_text())
    cfg["model"] = MODEL
    cfg["data"].update(crop_size=8, length=4)
    cfg["device"] = "cpu"
    cfg["training"].update(steps=1, batch_size=1, workers=0, grad_accum=2,
                           identity_init=False, ema_decay=.99)
    saved, histories = {}, {}
    for name, setting in (("absent", None), ("disabled", dict(enabled=False)),
                          ("enabled", dict(enabled=True, weight=.1, batch_size=2, size=8))):
        cfg["training"].pop("flat_identity", None)
        if setting is not None:
            cfg["training"]["flat_identity"] = setting
        config = tmp_path/(name+".yaml")
        config.write_text(yaml.safe_dump(cfg))
        run = tmp_path/name
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--run-dir", str(run)])
        train.main()
        saved[name] = torch.load(run/"last.pt", weights_only=True)
        histories[name] = json.loads((run/"history.json").read_text())[0]
    for key in saved["absent"]["model"]:
        assert torch.equal(saved["absent"]["model"][key], saved["disabled"]["model"][key])
    for name in ("disabled", "enabled"):
        assert torch.equal(saved["absent"]["torch_rng_state"], saved[name]["torch_rng_state"])
    assert "flat_identity_rng_state" not in saved["disabled"]
    assert "identity_loss" not in histories["disabled"]
    row = histories["enabled"]
    assert row["identity_loss"] > 0 and row["flat_identity_loss"] == pytest.approx(.1*row["identity_loss"])
    assert row["loss"] == row["data_loss"] + row["flat_identity_loss"]
    assert row["data_loss"] == histories["absent"]["loss"]
    assert any(not torch.equal(v, saved["absent"]["model"][k]) for k, v in saved["enabled"]["model"].items())
    reg = make_flat_identity(build_model(MODEL), cfg["training"]["flat_identity"], seed=cfg["seed"])
    reg.sample("cpu")  # exactly one auxiliary batch per optimizer step, even with accumulation
    assert torch.equal(reg.generator.get_state(), saved["enabled"]["flat_identity_rng_state"])


def test_experiment_recipe_matches_control():
    control = yaml.safe_load((ROOT/"configs/quality_loop/control.yaml").read_text())
    arm = yaml.safe_load((ROOT/"configs/quality_loop/flat_identity.yaml").read_text())
    assert arm["training"].pop("flat_identity") == dict(enabled=True, weight=.1, batch_size=4, size=128)
    arm["training"]["run_dir"] = control["training"]["run_dir"]
    assert arm == control


def test_resume_private_rng_and_missing_state(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT/"configs/arch_study/smoke_nafnet_s1.yaml").read_text())
    cfg["model"] = MODEL
    cfg["data"].update(crop_size=8, length=4)
    cfg["device"] = "cpu"
    setting = dict(enabled=True, weight=.1, batch_size=2, size=8)
    cfg["training"].update(steps=1, batch_size=1, workers=0, grad_accum=1,
                           identity_init=False, flat_identity=setting)
    config = tmp_path/"resume.yaml"
    config.write_text(yaml.safe_dump(cfg))
    run = tmp_path/"first"
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--run-dir", str(run)])
    train.main()
    checkpoint = run/"last.pt"
    resumed = tmp_path/"resumed"
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--run-dir", str(resumed),
                                    "--resume", str(checkpoint), "--steps", "2"])
    train.main()
    saved = torch.load(resumed/"last.pt", weights_only=True)
    reg = make_flat_identity(build_model(MODEL), setting, seed=cfg["seed"])
    reg.sample("cpu")
    reg.sample("cpu")
    assert saved["step"] == 2
    assert torch.equal(saved["flat_identity_rng_state"], reg.generator.get_state())
    saved.pop("flat_identity_rng_state")
    torch.save(saved, tmp_path/"missing.pt")
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--run-dir", str(tmp_path/"bad"),
                                    "--resume", str(tmp_path/"missing.pt"), "--steps", "3"])
    with pytest.raises(ValueError, match="private RNG state"):
        train.main()
