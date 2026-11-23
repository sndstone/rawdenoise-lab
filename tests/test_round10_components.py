"""Unit tests for Round 10 losses, extended small-sensor bootstrap physics, and slim-stage-0 NAFNet."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import yaml

from denoise_lab.arch_registry import build_model
from denoise_lab.bootstrap import add_bootstrap_noise
from denoise_lab.losses import DenoiseLoss
from denoise_lab.model import DenoiseNet, NAFNet, gamma

ROOT = Path(__file__).resolve().parents[1]


def test_loss_backward_compatible_and_chroma_green_terms():
    torch.manual_seed(0)
    pred = torch.rand(2, 4, 32, 32) * 0.4 + 0.05
    tgt = pred + torch.randn_like(pred) * 0.01

    base = DenoiseLoss(kind="charbonnier", linear_weight=1.0, gamma_weight=0.25, eps=1e-5)
    zero_new = DenoiseLoss(kind="charbonnier", linear_weight=1.0, gamma_weight=0.25, eps=1e-5,
                           chroma_weight=0.0, chroma_lowpass_weight=0.0, green_balance_weight=0.0)
    assert torch.equal(base(pred, tgt), zero_new(pred, tgt))

    # Pure chroma shift increases chroma loss significantly
    chroma_only = DenoiseLoss(linear_weight=0.0, gamma_weight=0.0, chroma_weight=1.0,
                              chroma_lowpass_weight=1.0, green_balance_weight=0.0, eps=1e-5)
    shifted = tgt.clone()
    shifted[:, 0] += 0.02  # R bias -> Cr shift
    shifted[:, 2] -= 0.02  # B bias -> Cb shift
    assert chroma_only(shifted, tgt) > chroma_only(tgt, tgt) * 100

    # Pure G1-G2 imbalance triggers green_balance_weight cleanly
    gb_only = DenoiseLoss(linear_weight=0.0, gamma_weight=0.0, green_balance_weight=1.0, eps=1e-5)
    imbal = tgt.clone()
    imbal[:, 1] += 0.015
    imbal[:, 3] -= 0.015
    assert gb_only(imbal, tgt) > gb_only(tgt, tgt) * 100

    # Near-black target with negative noisy predictions stays finite with valid gradients
    dark_tgt = torch.zeros(2, 4, 16, 16)
    dark_pred = (torch.randn(2, 4, 16, 16) * 0.005).requires_grad_(True)
    full = DenoiseLoss(kind="charbonnier", linear_weight=1.0, gamma_weight=0.25,
                       chroma_weight=0.1, chroma_lowpass_weight=0.1, green_balance_weight=0.05, eps=1e-5)
    val = full(dark_pred, dark_tgt)
    assert torch.isfinite(val)
    val.backward()
    assert torch.isfinite(dark_pred.grad).all()


def test_bootstrap_noise_backward_compatible_and_small_sensor_physics():
    clean = np.full((64, 64), 0.15, dtype=np.float32)
    # Default call matches exact legacy signature and returns no new keys
    n1, p1 = add_bootstrap_noise(clean, np.random.default_rng(42))
    assert set(p1.keys()) == {"K", "read_std", "row_std"}
    assert np.isfinite(n1).all()

    # Extended small-sensor physics: 12-bit quantization + horizontal/vertical lag-1 correlation
    n2, p2 = add_bootstrap_noise(
        clean,
        np.random.default_rng(7),
        shot_range=(0.005, 0.01),
        read_range=(0.008, 0.012),
        lag1_h_range=(0.15, 0.20),
        lag1_v_range=(0.04, 0.06),
        quantize_bits_choices=(12,),
    )
    assert p2["quantize_bits"] == 12
    assert 0.15 <= p2["lag1_h"] <= 0.20
    assert 0.04 <= p2["lag1_v"] <= 0.06
    # Verify 12-bit step quantization
    step = 1.0 / (2**12 - 1)
    rem = np.abs(n2 / step - np.round(n2 / step))
    assert float(rem.max()) < 1e-3

    # Negative black-subtracted residuals survive quantization on dark signal
    dark = np.zeros((64, 64), dtype=np.float32)
    nd, _ = add_bootstrap_noise(dark, np.random.default_rng(9), read_range=(0.01, 0.01), quantize_bits_choices=(12,))
    assert float(nd.min()) < -0.01


def test_nafnet_stage_widths_and_outer_fp32_deep_fp16():
    # Default stage_widths matches explicit doubling bit-for-bit
    torch.manual_seed(123)
    m1 = NAFNet(width=16, enc_blk_nums=(1, 1, 2), middle_blk_num=2, dec_blk_nums=(1, 1, 1)).eval()
    m2 = NAFNet(width=16, stage_widths=(16, 32, 64, 128), enc_blk_nums=(1, 1, 2), middle_blk_num=2, dec_blk_nums=(1, 1, 1)).eval()
    m2.load_state_dict(m1.state_dict(), strict=True)
    x = torch.rand(1, 4, 32, 32) * 0.2
    with torch.inference_mode():
        assert torch.equal(m1(x), m2(x))

    # Slim-stage-0 zero-block topology runs cleanly and satisfies flat identity on zero ending
    slim = DenoiseNet(width=16, stage_widths=(16, 32, 64, 128, 256),
                      enc_blk_nums=(0, 1, 1, 2), middle_blk_num=2, dec_blk_nums=(1, 1, 1, 0)).eval()
    with torch.inference_mode():
        y = slim(torch.rand(1, 4, 64, 64) * 0.2)
    assert y.shape == (1, 4, 64, 64) and torch.isfinite(y).all()

    # Outer FP32 + deep FP16 mode runs cleanly and preserves outer FP32 weights while casting deep stages
    net = DenoiseNet(width=16, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1)).eval()
    inp = torch.full((1, 4, 32, 32), 0.18)
    with torch.inference_mode():
        ref = net(inp)
        net.body.set_outer_fp32_deep_dtype(torch.float16)
        assert net.body.intro.weight.dtype == torch.float32
        assert net.body.ending.weight.dtype == torch.float32
        assert net.body.encoders[0][0].conv1.weight.dtype == torch.float32
        assert net.body.decoders[-1][0].conv1.weight.dtype == torch.float32
        assert net.body.middle_blks[0].conv1.weight.dtype == torch.float16
        mixed = net(inp)
        net.body.set_outer_fp32_deep_dtype(None)
        restored = net(inp)
    assert mixed.dtype == torch.float32 and torch.isfinite(mixed).all()
    torch.testing.assert_close(mixed, ref, rtol=1e-2, atol=1e-3)
    assert restored.dtype == torch.float32 and net.body.middle_blks[0].conv1.weight.dtype == torch.float32


def test_round10_configs_cpu_smoke_step(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/round10/r10_slim0_fast.yaml").read_text())
    # Tiny smoke override preserving exact Round 10 loss & slim-stage-0 architecture contract
    cfg["model"].update(width=8, stage_widths=[8, 16, 32], enc_blk_nums=[0, 1], middle_blk_num=1, dec_blk_nums=[1, 0])
    model = build_model(cfg["model"])
    criterion = DenoiseLoss(**cfg["loss"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.rand(2, 4, 32, 32) * 0.2
    y = torch.rand(2, 4, 32, 32) * 0.2
    loss = criterion(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()


def test_distill_config_validation_and_training_smoke(tmp_path):
    import subprocess
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from train import validate_config

    cfg = yaml.safe_load((ROOT / "configs/round10/r10_5stage_1blk_kd.yaml").read_text())
    validate_config(cfg, arch_study=True)

    bad = dict(cfg)
    bad["distill"] = {"checkpoint": "x.pt", "unknown_key": 1}
    with pytest.raises(ValueError, match="Unsupported distill.unknown_key"):
        validate_config(bad, arch_study=True)

    # Save tiny teacher checkpoint and run 2 real KD training steps on CPU
    teacher_cfg = {"arch": "nafnet", "cond": "none", "width": 8, "enc_blk_nums": [1, 1],
                   "middle_blk_num": 1, "dec_blk_nums": [1, 1]}
    teacher = build_model(teacher_cfg).eval()
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model": teacher.state_dict(), "model_config": teacher_cfg, "config": {"model": teacher_cfg}}, teacher_path)

    clean_npy = tmp_path / "clean.npy"
    np.save(clean_npy, np.full((80, 80), 0.25, dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps([{"clean": "clean.npy"}]))

    run_dir = tmp_path / "kd_run"
    smoke_cfg = {
        "seed": 7,
        "threads": 1,
        "device": "cpu",
        "model": {"arch": "nafnet", "cond": "none", "width": 8, "stage_widths": [8, 16, 32],
                  "enc_blk_nums": [1, 1], "middle_blk_num": 1, "dec_blk_nums": [1, 1]},
        "data": {"kind": "bootstrap", "manifest": str(manifest_path), "root": str(tmp_path),
                 "crop_size": 32, "flat_probability": 0.5, "identity_probability": 0.2},
        "loss": {"kind": "charbonnier", "linear_weight": 1.0, "gamma_weight": 0.25,
                 "chroma_weight": 0.04, "chroma_lowpass_weight": 0.06, "green_balance_weight": 0.02},
        "distill": {"checkpoint": str(teacher_path), "weights": "model", "weight": 0.5, "min_weight": 0.2},
        "training": {"steps": 2, "batch_size": 1, "workers": 0, "lr": 1e-3, "min_lr": 1e-4,
                     "amp": False, "identity_init": True, "ema_decay": 0.99, "progress": False,
                     "log_every": 1, "run_dir": str(run_dir)},
    }
    cfg_path = tmp_path / "smoke_kd.yaml"
    cfg_path.write_text(yaml.safe_dump(smoke_cfg))
    env = {**__import__("os").environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/train_arch_study.py"), "--config", str(cfg_path)],
                          capture_output=True, text=True, env=env, check=True)
    assert "distill: teacher=" in proc.stdout
    history = __import__("json").loads((run_dir / "history.json").read_text())
    assert len(history) == 2
    assert all(row["kd_loss"] >= 0 and np.isfinite(row["kd_loss"]) for row in history)
    manifest = __import__("json").loads((run_dir / "run_manifest.json").read_text())
    assert manifest["distill"]["checkpoint"] == str(teacher_path)


def test_stage2_highlight_chroma_and_green_dc_losses():
    torch.manual_seed(17)
    pred = torch.rand(2, 4, 32, 32) * 0.4 + 0.05
    tgt = pred + torch.randn_like(pred) * 0.01

    # Zero Stage 2 weights are bit-identical to legacy loss
    base = DenoiseLoss(kind="charbonnier", linear_weight=1.0, gamma_weight=0.25, eps=1e-5)
    zero_s2 = DenoiseLoss(kind="charbonnier", linear_weight=1.0, gamma_weight=0.25, eps=1e-5,
                          highlight_chroma_weight=0.0, green_dc_weight=0.0)
    assert torch.equal(base(pred, tgt), zero_s2(pred, tgt))

    # Highlight chroma loss activates on bright/clipped (>0.75..1.07) pixels and stays 0 on dark crops
    hl_only = DenoiseLoss(linear_weight=0.0, gamma_weight=0.0,
                          highlight_chroma_weight=1.0, highlight_threshold=0.75, eps=1e-5)
    assert float(hl_only(pred, tgt)) == 0.0
    hl_tgt = tgt.clone()
    hl_tgt[:, 0, 8:20, 8:20] = 0.55
    hl_tgt[:, 1, 8:20, 8:20] = 1.069
    hl_tgt[:, 2, 8:20, 8:20] = 1.028
    hl_tgt[:, 3, 8:20, 8:20] = 1.069
    hl_pred = hl_tgt.clone()
    hl_pred[:, 1, 8:20, 8:20] -= 0.05  # magenta shift (G drops vs R/B)
    hl_pred[:, 3, 8:20, 8:20] -= 0.05
    hl_pred.requires_grad_(True)
    v_hl = hl_only(hl_pred, hl_tgt)
    assert float(v_hl.detach()) > 0.01 and torch.isfinite(v_hl)
    v_hl.backward()
    assert torch.isfinite(hl_pred.grad).all()
    assert float(hl_pred.grad[:, :, 8:20, 8:20].abs().mean()) > float(hl_pred.grad[:, :, :6, :6].abs().mean()) * 100

    # Sparse barely-bright single pixel (w << 1) must stay bounded (floored denominator, no spike)
    sparse_tgt = torch.full((2, 4, 32, 32), 0.2)
    sparse_tgt[0, 1, 10, 10] = 0.76  # w = 0.04 on 1 pixel
    sparse_pred = sparse_tgt.clone()
    sparse_pred[0, 1, 10, 10] -= 0.05
    sparse_pred.requires_grad_(True)
    v_sparse = hl_only(sparse_pred, sparse_tgt)
    assert torch.isfinite(v_sparse) and float(v_sparse.detach()) < 1e-3
    v_sparse.backward()
    assert float(sparse_pred.grad.abs().max()) < 5e-3

    # Pooled DC green balance loss penalizes regional G1-G2 offset cleanly
    gdc = DenoiseLoss(linear_weight=0.0, gamma_weight=0.0, green_dc_weight=1.0, green_dc_kernel=16, eps=1e-5)
    g_shift = tgt.clone()
    g_shift[:, 1] += 0.02
    g_shift[:, 3] -= 0.02
    assert gdc(g_shift, tgt) > gdc(tgt, tgt) * 100


def _write_synthetic_calib(path):
    calib = {
        "units": "normalized",
        "cameras": {
            "test_cam": {
                "quantization": "linear",
                "iso": {
                    100: {"K": 1e-4, "read_std": 5e-4, "row_std": 5e-5, "col_std": 5e-5,
                          "row_fixed_std": 1e-5, "col_fixed_std": 1e-5, "tukey_lambda": 0.14,
                          "lag1_h": 0.05, "hot_density": 0.0, "black_offset": 0.0,
                          "code_step": 1.0 / (15360 - 512), "black_dn": 512.0, "white_dn": 15360.0,
                          "physics": {"read_variance": 2.5e-7, "read_excess_kurtosis": 0.2,
                                      "lag1_h": 0.05, "lag1_v": 0.02, "row_std": 5e-5, "col_std": 5e-5,
                                      "row_fixed_std": 1e-5, "col_fixed_std": 1e-5}},
                    6400: {"K": 2e-3, "read_std": 4e-3, "row_std": 2e-4, "col_std": 2e-4,
                           "row_fixed_std": 5e-5, "col_fixed_std": 5e-5, "tukey_lambda": 0.14,
                           "lag1_h": 0.08, "hot_density": 0.0, "black_offset": 0.0,
                           "code_step": 1.0 / (15360 - 512), "black_dn": 512.0, "white_dn": 15360.0,
                           "physics": {"read_variance": 1.6e-5, "read_excess_kurtosis": 0.5,
                                       "lag1_h": 0.08, "lag1_v": 0.03, "row_std": 2e-4, "col_std": 2e-4,
                                       "row_fixed_std": 5e-5, "col_fixed_std": 5e-5}},
                },
            }
        },
    }
    path.write_text(yaml.safe_dump(calib))


def test_stage2_clip_highlight_augmentation_and_zero_legacy_drift(tmp_path):
    from denoise_lab.camera_noise import CameraNoise, CameraNoiseDataset
    from denoise_lab.clean_targets import BinnedBootstrapDataset

    # Create a textured 80x80 RGGB mosaic with unequal channels so partial clipping occurs naturally
    y, x = np.mgrid[:80, :80]
    base_clean = (0.12 + 0.18 * np.exp(-((y - 40) ** 2 + (x - 40) ** 2) / (2 * 14 ** 2))).astype(np.float32)
    base_clean[0::2, 0::2] *= 0.55  # R lower than G
    np.save(tmp_path / "clean.npy", base_clean)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(__import__("json").dumps([{"clean": "clean.npy"}]))

    # Zero clip_highlight_probability matches legacy stream bit-for-bit
    ds_legacy = BinnedBootstrapDataset(manifest, root=tmp_path, crop_size=32, deterministic=True, seed=19)
    ds_zero = BinnedBootstrapDataset(manifest, root=tmp_path, crop_size=32, deterministic=True, seed=19,
                                     clip_highlight_probability=0.0)
    s_leg, s_zero = ds_legacy[0], ds_zero[0]
    assert torch.equal(s_leg["noisy"], s_zero["noisy"])
    assert torch.equal(s_leg["clean"], s_zero["clean"])

    # Enabled clip_highlight_probability=1.0 with >1.0 ceiling produces clipped headroom >1.0 in BOTH clean and noisy
    ds_clip = BinnedBootstrapDataset(manifest, root=tmp_path, crop_size=32, deterministic=True, seed=19,
                                     clip_highlight_probability=1.0, clip_exposure_range=(2.0, 2.5),
                                     clip_ceiling_range=(1.05, 1.09), quantize_bits_choices=(14,))
    s_clip = ds_clip[0]
    assert 1.04 < float(s_clip["clean"].max()) <= 1.09 + 1e-5
    assert 1.04 < float(s_clip["noisy"].max()) <= 1.09 + 1e-5
    assert 1.05 <= s_clip["clip_ceiling"] <= 1.09

    # Pre-noise clipping physics check: a 2x overexposed flat field must stay ~100% pinned at c_clip
    # (not ~50% from clipping before noise!)
    np.save(tmp_path / "flat.npy", np.full((80, 80), 0.5, dtype=np.float32))
    flat_manifest = tmp_path / "flat_manifest.json"
    flat_manifest.write_text(__import__("json").dumps([{"clean": "flat.npy"}]))
    ds_flat_clip = BinnedBootstrapDataset(flat_manifest, root=tmp_path, crop_size=32, deterministic=True, seed=5,
                                          clip_highlight_probability=1.0, clip_exposure_range=(2.0, 2.0),
                                          clip_ceiling_range=(1.037, 1.037), identity_probability=0.0,
                                          read_range=(0.01, 0.01), shot_range=(0.005, 0.005))
    sf = ds_flat_clip[0]
    sat_frac = float((sf["noisy"] >= 1.037 - 1e-5).float().mean())
    assert sat_frac > 0.99, f"Expected >99% saturated pixels for 2x overexposed flat field, got {sat_frac:.3f}"

    # Quantizer code-grid alignment check: both upper clip_ceiling=1.037 AND lower -0.1 floor snap on-grid
    step12 = 1.0 / (2**12 - 1)
    q_hi, _ = add_bootstrap_noise(np.full((32, 32), 2.0, dtype=np.float32), np.random.default_rng(3),
                                  read_range=(0.005, 0.005), quantize_bits_choices=(12,), clip_ceiling=1.037)
    assert float(np.abs(q_hi / step12 - np.round(q_hi / step12)).max()) < 1e-4
    assert 1.036 < float(q_hi.max()) <= 1.037
    q_lo, _ = add_bootstrap_noise(np.full((32, 32), -0.25, dtype=np.float32), np.random.default_rng(4),
                                  read_range=(0.005, 0.005), quantize_bits_choices=(12,), clip_ceiling=1.037)
    assert float(np.abs(q_lo / step12 - np.round(q_lo / step12)).max()) < 1e-4
    assert float(q_lo.min()) >= -0.1 - 1e-7

    # Sony piecewise curve saturation: signals from different curve segments (6700 DN vs 9000 DN)
    # must saturate to ONE identical code on the ceiling's segment (6520 DN -> 6512 DN) in both paths.
    from denoise_lab.camera_noise import SONY_STEPS
    from denoise_lab.physics_noise import quantize_dn
    sony_calib = tmp_path / "sony_calib.yaml"
    _write_synthetic_calib(tmp_path / "calib.yaml")
    calib_data = yaml.safe_load((tmp_path / "calib.yaml").read_text())
    calib_data["cameras"]["test_cam"]["quantization"] = "sony_curve"
    sony_calib.write_text(yaml.safe_dump(calib_data))
    span_s = 15360.0 - 512.0
    ceil_6520 = (6520.0 - 512.0) / span_s
    sig_pair = np.empty((32, 32), dtype=np.float32)
    sig_pair[:, :16] = (6700.0 - 512.0) / span_s   # in 16-DN segment
    sig_pair[:, 16:] = (9000.0 - 512.0) / span_s   # in 32-DN segment
    # Physics quantizer directly
    p_phys = {"black_dn": 512.0, "white_dn": 15360.0, "code_step_dn": 1.0, "quantization_steps_dn": SONY_STEPS}
    q_phys = quantize_dn(sig_pair, p_phys, clip_ceiling=ceil_6520)
    dn_phys = np.round(q_phys * span_s + 512.0)
    assert float(dn_phys.min()) == 6512.0 and float(dn_phys.max()) == 6512.0
    # CameraNoise fallback + physics apply paths
    for use_phys in (False, True):
        c_opts = {"physics_options": {"quantization": True}} if use_phys else {}
        cn_s = CameraNoise(str(sony_calib), {"test_cam": 1.0}, **c_opts)
        ps = cn_s.at("test_cam", 100)
        ps["K"] = 1e-12
        ps["read_std"] = 0.0
        ps["row_std"] = ps["col_std"] = ps["row_fixed_std"] = ps["col_fixed_std"] = 0.0
        if "physics" in ps:
            ps["physics"]["read_variance"] = 0.0
        q_s = cn_s.apply(sig_pair, ps, np.random.default_rng(1), clip_ceiling=ceil_6520)
        dn_s = np.round(q_s * span_s + 512.0)
        assert float(dn_s.min()) == 6512.0 and float(dn_s.max()) == 6512.0, f"Sony curve mismatch use_phys={use_phys}"

    # Self-contained CameraNoiseDataset test (always runs on CI; zero hardcoded paths)
    calib_path = tmp_path / "calib.yaml"
    # Test both fallback and physics quantization code-grid alignment + >1.0 headroom + 2x saturation plateau
    for use_phys in (False, True):
        cam_opts = {"physics_options": {"banding": True, "quantization": True, "horizontal_correlation": True,
                                        "vertical_correlation": True, "moment_read": True, "fixed_seed": 7}} if use_phys else {}
        cn = CameraNoise(str(calib_path), {"test_cam": 1.0}, **cam_opts)
        p = cn.at("test_cam", 6400)
        span = p["white_dn"] - p["black_dn"]
        q_cam = cn.apply(np.full((32, 32), 2.0, dtype=np.float32), p, np.random.default_rng(11), clip_ceiling=1.037)
        dn = q_cam * span + p["black_dn"]
        assert float(np.abs(dn - np.round(dn)).max()) < 1e-2, f"Off-grid DN codes for use_phys={use_phys}"
        assert 1.036 < float(q_cam.max()) <= 1.037

        ds_cam_leg = CameraNoiseDataset(manifest, root=tmp_path, crop_size=32, deterministic=True, seed=23,
                                        calibration=str(calib_path), cameras={"test_cam": 1.0}, camera_options=cam_opts)
        ds_cam_zero = CameraNoiseDataset(manifest, root=tmp_path, crop_size=32, deterministic=True, seed=23,
                                         calibration=str(calib_path), cameras={"test_cam": 1.0}, camera_options=cam_opts,
                                         clip_highlight_probability=0.0)
        assert torch.equal(ds_cam_leg[0]["noisy"], ds_cam_zero[0]["noisy"])

        ds_cam_flat = CameraNoiseDataset(flat_manifest, root=tmp_path, crop_size=32, deterministic=True, seed=23,
                                         calibration=str(calib_path), cameras={"test_cam": 1.0}, camera_options=cam_opts,
                                         clip_highlight_probability=1.0, clip_exposure_range=(2.0, 2.0),
                                         clip_ceiling_range=(1.05, 1.09), identity_probability=0.0)
        sc = ds_cam_flat[0]
        assert 1.04 < float(sc["clean"].max()) <= 1.09 + 1e-5
        assert float((sc["noisy"] >= sc["clip_ceiling"] - 2e-3).float().mean()) > 0.99


def test_stage2_protect_teacher_highlights_low_ceiling():
    from denoise_lab.inference import protect_highlights

    # Synthetic batch: sample 0 clipped at c_clip=0.94 (<0.98!), sample 1 unclipped at 0.40
    clean = torch.full((2, 4, 16, 16), 0.30)
    clean[0, :, 5:11, 5:11] = 0.94
    teacher_pred = clean.clone()
    teacher_pred[0, 1, 5:11, 5:11] -= 0.08  # magenta teacher error in 0.94 core
    teacher_pred[0, 0, 4, 5:11] = 0.97      # overshoot above 0.94 ceiling in ring
    teacher_pred[1] += 0.02

    c_ceil = torch.tensor([0.94, 1.0]).view(2, 1, 1, 1)
    hl_thr = torch.minimum(torch.full_like(c_ceil, 0.98), c_ceil * 0.98)
    c_max = torch.minimum(torch.ones_like(c_ceil), c_ceil)
    protected = protect_highlights(teacher_pred, clean, threshold=hl_thr, ceiling_max=c_max)

    # Sample 0 clipped core (0.94 < 0.98) is now exactly restored to noiseless clean (no teacher magenta shift!)
    torch.testing.assert_close(protected[0, :, 6:10, 6:10], clean[0, :, 6:10, 6:10], atol=0.0, rtol=0.0)
    # Ring overshoot above synthetic 0.94 ceiling is capped strictly below 0.94
    assert float(protected[0, 0, 4, 5:11].max()) < 0.94
    # Sample 1 unclipped sample is completely untouched
    torch.testing.assert_close(protected[1], teacher_pred[1], atol=0.0, rtol=0.0)


def test_stage2_config_validation_and_training_smoke(tmp_path):
    import importlib.util
    import json
    import os
    import subprocess

    train_spec = importlib.util.spec_from_file_location("train", ROOT / "scripts/train.py")
    train_mod = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_mod)

    cfg = yaml.safe_load((ROOT / "configs/round10/r10_5stage_1blk_s2_kd.yaml").read_text())
    train_mod.validate_config(cfg, arch_study=True)

    teacher_cfg = {"arch": "nafnet", "cond": "none", "width": 8, "enc_blk_nums": [1, 1],
                   "middle_blk_num": 1, "dec_blk_nums": [1, 1]}
    teacher = build_model(teacher_cfg).eval()
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model": teacher.state_dict(), "model_config": teacher_cfg, "config": {"model": teacher_cfg}}, teacher_path)

    np.save(tmp_path / "clean.npy", np.full((80, 80), 0.30, dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([{"clean": "clean.npy"}]))

    run_dir = tmp_path / "s2_run"
    smoke_cfg = {
        "seed": 11,
        "threads": 1,
        "device": "cpu",
        "model": {"arch": "nafnet", "cond": "none", "width": 8, "stage_widths": [8, 16, 32],
                  "enc_blk_nums": [1, 1], "middle_blk_num": 1, "dec_blk_nums": [1, 1]},
        "data": {"kind": "bootstrap", "manifest": str(manifest_path), "root": str(tmp_path),
                 "crop_size": 32, "clip_highlight_probability": 0.5,
                 "clip_exposure_range": [1.5, 2.5], "clip_ceiling_range": [0.95, 1.08]},
        "loss": {"kind": "charbonnier", "linear_weight": 1.0, "gamma_weight": 0.25,
                 "chroma_weight": 0.04, "chroma_lowpass_weight": 0.06,
                 "highlight_chroma_weight": 0.15, "highlight_threshold": 0.75,
                 "green_balance_weight": 0.02, "green_dc_weight": 0.05, "green_dc_kernel": 16},
        "distill": {"checkpoint": str(teacher_path), "weights": "model", "weight": 0.45, "min_weight": 0.15,
                    "highlight_chroma_weight": 0.10, "protect_teacher_highlights": True},
        "training": {"steps": 2, "batch_size": 2, "workers": 0, "lr": 1e-3, "min_lr": 1e-4,
                     "amp": False, "identity_init": True, "ema_decay": 0.99, "progress": False,
                     "transpose_consistency_weight": 0.08, "transpose_consistency_fraction": 0.5,
                     "log_every": 1, "run_dir": str(run_dir)},
    }
    cfg_path = tmp_path / "smoke_s2.yaml"
    cfg_path.write_text(yaml.safe_dump(smoke_cfg))
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/train_arch_study.py"), "--config", str(cfg_path)],
                          capture_output=True, text=True, env=env, check=True)
    assert "protect_hl=True" in proc.stdout
    history = json.loads((run_dir / "history.json").read_text())
    assert len(history) == 2
    assert all(row["tc_loss"] >= 0 and np.isfinite(row["tc_loss"]) for row in history)
    assert all(row["kd_loss"] >= 0 and np.isfinite(row["kd_loss"]) for row in history)


def test_stage3_green_split_head_and_aa_shallow2():
    # 1) green_split head: approximate low-frequency trend suppression on dG_diff, exact identity on zero-init,
    # unequal-block boundary behavior, non-multiple-of-16 fallback, and pixel-level HF G1-G2 pass-through.
    torch.manual_seed(5)
    m_gs = DenoiseNet(width=16, stage_widths=(16, 32, 64), enc_blk_nums=(1, 1),
                      middle_blk_num=1, dec_blk_nums=(1, 1), output_head="green_split").eval()
    flat = torch.full((1, 4, 96, 96), 0.25)
    with torch.inference_mode():
        y_flat = m_gs(flat)
        b_flat = m_gs.body(flat)
    assert y_flat.shape == flat.shape and torch.isfinite(y_flat).all()
    # Uniform interior cancels constant dG_diff trend cleanly.
    assert abs(float((b_flat[:, 1, 32:64, 32:64] - b_flat[:, 3, 32:64, 32:64]).mean())) < 1e-6

    # Non-multiple-of-16 fallback subtracts global mean across the full map (including borders).
    odd_in = torch.rand(1, 4, 38, 50) * 0.3 + 0.05
    with torch.inference_mode():
        b_odd = m_gs.body(odd_in)
    assert abs(float((b_odd[:, 1] - b_odd[:, 3]).mean())) < 1e-6

    # Explicit check of bilinear low-frequency trend subtraction on two unequal 16x16 blocks + HF impulse:
    # Construct a controlled body where ending_gdiff directly passes channel 0 of features.
    m_ctrl = DenoiseNet(width=16, stage_widths=(16, 32), enc_blk_nums=(0,),
                        middle_blk_num=0, dec_blk_nums=(0,), output_head="green_split").eval()
    with torch.no_grad():
        for p in m_ctrl.parameters():
            p.zero_()
        # Pass input channel 1 (G1) straight through intro[0] -> ending_gdiff[0] with weight 1 at center tap.
        m_ctrl.body.intro.weight[0, 1, 1, 1] = 1.0
        m_ctrl.body.ending_gdiff.weight[0, 0, 1, 1] = 1.0
        # Two adjacent constant 16x16 blocks (0.0 left, 1.0 right) -> whole-map mean of dG_diff is 0,
        # while bilinear blending across the step leaves nonzero opposite-sign local block means.
        step_in = torch.zeros(1, 4, 16, 32)
        step_in[:, 1, :, 16:] = 1.0
        b_step = m_ctrl.body(step_in)
        gd_eff = 0.5 * (b_step[:, 1] - b_step[:, 3])
        assert abs(float(gd_eff.mean())) < 1e-6
        assert float(gd_eff[:, :, :16].mean()) < -0.05 and float(gd_eff[:, :, 16:].mean()) > 0.05
        # High-frequency alternating +-0.25 G1-G2 signal passes through unattenuated (zero block mean).
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        hf_in = torch.zeros(1, 4, 32, 32)
        hf_pat = (0.25 * (((yy + xx) % 2) * 2 - 1)).float()
        hf_in[0, 1] = hf_pat
        b_hf = m_ctrl.body(hf_in)
        gd_hf = 0.5 * (b_hf[0, 1] - b_hf[0, 3])
        torch.testing.assert_close(gd_hf, hf_pat, atol=1e-5, rtol=1e-5)

    # Zero-init ending + ending_gdiff gives exact identity
    with torch.no_grad():
        m_gs.body.ending.weight.zero_()
        m_gs.body.ending.bias.zero_()
        m_gs.body.ending_gdiff.weight.zero_()
        x = torch.rand(1, 4, 32, 32) * 0.4 + 0.05
        torch.testing.assert_close(m_gs(x), x, atol=1e-6, rtol=1e-6)

    # 2) aa_shallow2 resampling runs cleanly on both multiple-of-16 and non-divisible sizes
    m_aa = DenoiseNet(width=16, stage_widths=(16, 32, 64, 128), enc_blk_nums=(1, 1, 1),
                      middle_blk_num=1, dec_blk_nums=(1, 1, 1), resample="aa_shallow2",
                      output_head="green_split").eval()
    for shape in ((1, 4, 32, 32), (1, 4, 38, 50)):
        with torch.inference_mode():
            y = m_aa(torch.rand(*shape) * 0.3)
        assert y.shape == shape and torch.isfinite(y).all()
