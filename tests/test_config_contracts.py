"""Training schemas and matched recipe checks without source-data IO."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch
import yaml

from denoise_lab import data

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("contract_"+name, ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_script("train")
research = load_script("train_arch_study")
TRAINING_CONFIGS = [path for path in sorted((ROOT/"configs").rglob("*.yaml"))
                    if "training" in yaml.safe_load(path.read_text())]


@pytest.mark.parametrize("path", TRAINING_CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_all_training_configs_validate_and_construct_without_data(path, monkeypatch):
    cfg = yaml.safe_load(path.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        baseline.validate_config(cfg, arch_study=True)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(data, "load_manifest", lambda _: [{"clean": "unused.npy", "noisy": "unused.npy"}])
    ds, _ = research.dataset(cfg["data"], require_conditioning=cfg["model"].get("cond") == "noise_map")
    assert len(ds) > 0


@pytest.mark.parametrize("key,value", [("betas", [.9, .99]), ("warmup_steps", 0),
                                      ("grad_accum", 1), ("ema_decay", 0), ("typo", 1)])
def test_baseline_rejects_unsupported_training_settings(key, value):
    cfg = yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    cfg["training"][key] = value
    with pytest.raises(ValueError, match="training."+key):
        baseline.validate_config(cfg)


@pytest.mark.parametrize("amp,explicit,expected", [(True, None, torch.float16), (False, None, torch.float16),
    ("bf16", None, torch.bfloat16), ("bfloat16", None, torch.bfloat16),
    (True, "bf16", torch.bfloat16), (True, "bfloat16", torch.bfloat16), ("fp16", None, torch.float16)])
def test_baseline_amp_precision(amp, explicit, expected):
    cfg = yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    cfg["training"]["amp"] = amp
    if explicit is not None:
        cfg["training"]["amp_dtype"] = explicit
    baseline.validate_config(cfg)
    assert baseline.amp_dtype(cfg["training"]) == expected


@pytest.mark.parametrize("section,key,value", [("training", "amp", "typo"), ("training", "amp_dtype", "typo"),
                                              ("data", "shot_ragne", [.1, .2])])
def test_unknown_settings_fail(section, key, value):
    cfg = yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    cfg[section][key] = value
    with pytest.raises(ValueError, match=section+"."+key):
        baseline.validate_config(cfg)


def test_camera_validation_before_io_and_nested_typos():
    cfg = yaml.safe_load((ROOT/"configs/round9/r9_phys35.yaml").read_text())
    camera = cfg["data"]["sources"][1]
    camera["noise_map_jitter"] = 1
    with pytest.raises(ValueError, match="noise_map_jitter"):
        baseline.validate_config(cfg, arch_study=True)
    del camera["noise_map_jitter"]
    camera["camera_options"]["physics_options"]["quantizaton"] = True
    with pytest.raises(ValueError, match="quantizaton"):
        baseline.validate_config(cfg, arch_study=True)


def test_iso_coupled_precedence_is_reported():
    cfg = yaml.safe_load((ROOT/"configs/round5/r5_data.yaml").read_text())
    with pytest.warns(UserWarning, match="iso_coupled overrides"):
        baseline.validate_config(cfg, arch_study=True)


@pytest.mark.parametrize("name,weights", [("r9_phys35", [.65, .15, .1, .1]),
    ("r9_legacy35", [.65, .15, .1, .1]),
    ("r9_phys20", [.80, .085714285714, .057142857143, .057142857143]),
    ("r9_phys10", [.90, .042857142857, .028571428571, .028571428572])])
def test_round9_exact_recipe_diffs(name, weights):
    base = yaml.safe_load((ROOT/"configs/round8/r8_physnoise.yaml").read_text())
    expected = copy.deepcopy(base)
    expected["training"].update(checkpoint_every=24000, keep_checkpoints=0, run_dir="runs/round9/"+name)
    expected["data"]["probabilities"] = weights
    if name == "r9_legacy35":
        for source in expected["data"]["sources"][1:]:
            del source["camera_options"]["physics_options"]
    cfg = yaml.safe_load((ROOT/"configs/round9"/(name+".yaml")).read_text())
    assert cfg == expected
    assert sum(weights) == pytest.approx(1)
    assert .5*weights[0] == pytest.approx({"r9_phys10": .45, "r9_phys20": .4}.get(name, .325))


@pytest.mark.parametrize("name", ["r9_phys35", "r9_legacy35", "r9_phys20", "r9_phys10"])
def test_round9_sources_produce_finite_synthetic_batches(name, monkeypatch):
    from denoise_lab import camera_noise, clean_targets

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(data, "load_manifest", lambda _: [{"clean": "unused.npy"}])
    clean = np.full((260, 260), .2, np.float32)
    for module in (data, camera_noise, clean_targets):
        monkeypatch.setattr(module, "read_mosaic", lambda _: clean.copy())
    cfg = yaml.safe_load((ROOT/"configs/round9"/(name+".yaml")).read_text())
    for source in cfg["data"]["sources"]:
        source.update(deterministic=True, seed=9)
    ds, _ = research.dataset(cfg["data"])
    for source in ds.datasets:
        sample = source[0]
        for key in ("noisy", "clean", "noise_map"):
            assert torch.isfinite(sample[key]).all()
            assert sample[key].shape[-2:] == (128, 128)


@pytest.mark.parametrize("script", ["train", "train_arch_study"])
def test_validation_precedes_dataset_io(script, tmp_path):
    cfg = yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    cfg["training"]["misspelled_setting"] = 1
    cfg["data"]["manifest"] = "missing.json"
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "PYTHONPATH": str(ROOT/"src"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, str(ROOT/"scripts"/(script+".py")), "--config", str(path)],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "training.misspelled_setting" in result.stderr
    assert "FileNotFoundError" not in result.stderr


def test_zero_checkpoint_retention_writes_only_last(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT/"configs/smoke.yaml").read_text())
    cfg["model"].update(width=4, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)
    cfg["data"].update(length=2, crop_size=8)
    cfg["training"].update(steps=2, batch_size=1, workers=0, checkpoint_every=1,
                            keep_checkpoints=0, run_dir=str(tmp_path/"run"), progress=False)
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    monkeypatch.setattr(sys, "argv", ["train_arch_study.py", "--config", str(path), "--device", "cpu"])
    research.main()
    assert [p.name for p in (tmp_path/"run").glob("*.pt")] == ["last.pt"]
    assert torch.load(tmp_path/"run/last.pt", weights_only=True)["step"] == 2


def test_changed_python_parses_on_311():
    for name in ("scripts/train.py", "scripts/train_arch_study.py", "src/denoise_lab/camera_noise.py"):
        ast.parse((ROOT/name).read_text(), feature_version=(3, 11))
