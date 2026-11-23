"""Unit tests for isolated 2x RAW SR package (Track A + Track B)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from denoise_lab.sr import (
    DegradationParams,
    LocalAvgPool2d,
    PhotoSRv1,
    PhotoSRv2,
    PhotoSRv2Config,
    RefineConfig,
    SRv2Loss,
    align_planes,
    anisotropic_psf_kernel,
    apply_tlc,
    cfa_chroma_guard,
    compressive_range_guard,
    context_tiled_sr,
    forward_degrade_v2,
    highlight_blend_sr,
    infer_photosr_v2,
    inverse_asinh,
    refine_sr_prediction,
    sample_degradation_params,
    signed_asinh,
    simulate_lr_observation,
    site_upsample,
)


def test_site_upsample_reproduces_linear_sensor_ramp():
    # Exact analytic sensor ramp f(y, x) = 0.2 + 0.003*y + 0.005*x sampled on LR CFA
    h, w = 16, 16
    sites = ((0, 0), (0, 1), (1, 1), (1, 0))
    yy_lr = torch.arange(h, dtype=torch.float32)
    xx_lr = torch.arange(w, dtype=torch.float32)
    lr_planes = []
    for sy, sx in sites:
        Y, X = torch.meshgrid(2 * yy_lr + sy, 2 * xx_lr + sx, indexing="ij")
        lr_planes.append(0.2 + 0.003 * Y + 0.005 * X)
    lr = torch.stack(lr_planes, dim=0)[None]
    up = site_upsample(lr, scale=2)

    # Expected interior sensor coordinates mapped to LR coordinate space: (o + 0.5)/2 - 0.5
    yy_hr = torch.arange(2 * h, dtype=torch.float32)
    xx_hr = torch.arange(2 * w, dtype=torch.float32)
    hr_planes = []
    for sy, sx in sites:
        oy = (2 * yy_hr + sy + 0.5) / 2.0 - 0.5
        ox = (2 * xx_hr + sx + 0.5) / 2.0 - 0.5
        Y, X = torch.meshgrid(oy, ox, indexing="ij")
        hr_planes.append(0.2 + 0.003 * Y + 0.005 * X)
    expected = torch.stack(hr_planes, dim=0)[None]
    assert torch.allclose(up[..., 2:-2, 2:-2], expected[..., 2:-2, 2:-2], atol=1e-5)


def test_signed_asinh_roundtrip_and_zero_smoothness():
    x = torch.tensor([-0.05, -1e-4, 0.0, 1e-4, 0.02, 0.5, 1.0, 2.0], requires_grad=True)
    z = signed_asinh(x, scale=0.05)
    x_rec = inverse_asinh(z, scale=0.05)
    assert torch.allclose(x_rec, x, atol=1e-6)
    # Odd symmetry and normalized +1 -> +1
    assert torch.isclose(z[2], torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(z[6], torch.tensor(1.0), atol=1e-6)
    z.sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad > 0).all()


def test_highlight_blend_identity_on_unclipped_and_protects_clipped():
    torch.manual_seed(0)
    guide = torch.rand(1, 4, 16, 16) * 0.5
    orig = guide.clone()
    pred = site_upsample(guide) + 0.05 * torch.randn(1, 4, 32, 32)
    # Unclipped (<0.98) must return exact same tensor
    out_unclipped = highlight_blend_sr(pred, guide, orig)
    assert torch.equal(out_unclipped, pred)

    # Clipped highlight (>=0.98) with extreme overshoot (+2.0) and undershoot (-0.5)
    orig_clip = orig.clone()
    orig_clip[:, 0, 7:9, 7:9] = 1.0
    pred_bad = pred.clone()
    pred_bad[:, :, 14:18, 14:18] = 2.1
    pred_bad[:, :, 13:14, 14:18] = -0.5
    out_clip = highlight_blend_sr(pred_bad, guide, orig_clip)
    assert float(out_clip[:, :, 14:18, 14:18].max()) <= 1.0 + 1e-6
    assert float(out_clip[:, :, 13:14, 14:18].min()) > -0.1


def test_compressive_range_guard_blocks_night_negative_undershoot():
    # Sharp 0.004 -> 1.0 step where linear SR produces -0.50 undershoot and +2.0 overshoot
    guide = torch.full((1, 4, 16, 16), 0.004)
    guide[..., 8:] = 1.0
    pred = site_upsample(guide)
    pred[..., 15] = -0.50
    pred[..., 16] = 2.10
    guarded = compressive_range_guard(pred, guide)
    assert float(guarded.min()) > 0.0, "Positive shadow must not undershoot into negative black-clip halo"
    assert float(guarded.max()) <= 1.0 + 1e-6, "Unclipped/white ceiling must not blow up to +2.1"

    # Dim isolated point source (star) retains recovered peak while blocking negative ring
    star_guide = torch.full((1, 4, 20, 20), 0.02)
    star_guide[..., 10, 10] = 0.14
    star_pred = site_upsample(star_guide)
    star_pred[..., 20, 20] = 0.28
    star_pred[..., 19, 20] = -0.01
    star_guarded = compressive_range_guard(star_pred, star_guide, preserve_point_sources=True)
    assert float(star_guarded[..., 20, 20].max()) > 0.25, "Isolated star peak should be preserved"
    assert float(star_guarded.min()) > 0.0, "Negative ring around star must still be eliminated"


def test_cfa_chroma_guard_reduces_neutral_edge_color_ringing():
    # Construct a physically neutral step edge on the sensor mosaic (no CFA-phase color bias)
    y, x = torch.meshgrid(torch.arange(32, dtype=torch.float32), torch.arange(32, dtype=torch.float32), indexing="ij")
    field = 0.05 + 0.60 * (x >= 16).float()
    sites = ((0, 0), (0, 1), (1, 1), (1, 0))
    guide = torch.stack([field[sy::2, sx::2] for sy, sx in sites], dim=0)[None]
    base = site_upsample(guide)
    # Inject synthetic cyan/red ringing (opposite R vs B residual) at the edge
    pred = base.clone()
    pred[:, 0, :, 15:17] -= 0.08
    pred[:, 2, :, 15:17] += 0.08
    cleaned = cfa_chroma_guard(pred, guide)
    res_before = align_planes(pred - base)
    res_after = align_planes(cleaned - base)
    diff_before = (res_before[:, 2] - res_before[:, 0]).abs().max()
    diff_after = (res_after[:, 2] - res_after[:, 0]).abs().max()
    assert float(diff_after) < 0.5 * float(diff_before)


def test_v1_tlc_and_context_tiling_shapes():
    torch.manual_seed(1)
    model = PhotoSRv1(mode="fusion", width=16, depth=2).eval()
    tlc_model = apply_tlc(model, base_kernel=15)
    # Confirm no AdaptiveAvgPool2d remains
    for m in tlc_model.modules():
        assert not isinstance(m, torch.nn.AdaptiveAvgPool2d)
    assert any(isinstance(m, LocalAvgPool2d) for m in tlc_model.modules())
    y = torch.rand(1, 4, 48, 48)
    d = torch.rand(1, 4, 48, 48)
    out = context_tiled_sr(tlc_model, y, d, center=32, apron=8, overlap=16, tile_batch_size=2)
    assert out.shape == (1, 4, 96, 96) and torch.isfinite(out).all()


def test_photosr_v2_zero_init_identity_and_domains():
    for domain in ("asinh", "gamma"):
        cfg = PhotoSRv2Config(width=16, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1), sca_kernels=(7, 5, 3), domain=domain)
        model = PhotoSRv2(cfg).eval()
        y = torch.rand(1, 4, 24, 24) * 0.8 - 0.01
        d = torch.rand(1, 4, 24, 24) * 0.8 - 0.01
        with torch.inference_mode():
            out = model(y, d)
            expected = site_upsample(d)
        assert out.shape == (1, 4, 48, 48)
        assert torch.allclose(out, expected, atol=1e-5)

    # Verify soft_white_cap strictly prevents exponential >1.0 blowup even with huge positive weights
    cfg = PhotoSRv2Config(width=16, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1), sca_kernels=(7, 5, 3), soft_white_cap=True)
    m = PhotoSRv2(cfg).eval()
    with torch.no_grad():
        m.hr_tail[-1].bias.fill_(2.0)
    y_sat = torch.ones(1, 4, 16, 16)
    with torch.inference_mode():
        out_sat = m(y_sat, y_sat)
    # With zero-bias start=1.0, knee=0.003, linear output stays <= 1.015 (never exponential blowup)
    assert float(out_sat.max()) <= 1.015

    # Exact 2-fold Bayer transpose TTA reproduces linear ramp identically at zero-init
    m_zero = PhotoSRv2(cfg).eval()
    y_ramp = torch.rand(1, 4, 16, 16) * 0.5 + 0.05
    out_tta = infer_photosr_v2(m_zero, y_ramp, y_ramp, tta=True, physical_floor_guard=True)
    assert torch.allclose(out_tta, site_upsample(y_ramp), atol=1e-5)


def test_photosr_v2_shift_equivariance_zero_tile_breathing():
    """LocalSCA guarantees interior predictions do NOT change when distant highlights change."""
    torch.manual_seed(7)
    cfg = PhotoSRv2Config(
        width=16,
        enc_blk_nums=(1, 1),
        middle_blk_num=1,
        dec_blk_nums=(1, 1),
        sca_kernels=(5, 3, 3),
        domain="asinh",
    )
    model = PhotoSRv2(cfg).eval()
    # Non-zero ending/tail weights and block scales so SCA is actively engaged
    with torch.no_grad():
        torch.nn.init.normal_(model.body.ending.weight, std=0.05)
        if model.hr_tail is not None:
            torch.nn.init.normal_(model.hr_tail[-1].weight, std=0.05)
        for name, p in model.named_parameters():
            if name.endswith((".beta", ".gamma")):
                p.fill_(0.5)

    y1 = torch.rand(1, 4, 64, 64) * 0.2
    d1 = y1.clone()
    y2 = y1.clone()
    d2 = d1.clone()
    # Inject an intense 1.0 specular light in the far bottom-right corner
    y2[..., 54:, 54:] = 1.0
    d2[..., 54:, 54:] = 1.0

    with torch.inference_mode():
        o1 = model(y1, d1)
        o2 = model(y2, d2)

    # Top-left interior (far beyond finite receptive field) must be IDENTICAL!
    diff_far = (o1[..., :20, :20] - o2[..., :20, :20]).abs().max().item()
    assert diff_far < 1e-6, f"Expected zero tile-wide leakage from distant specular, got {diff_far}"


def test_degradation_v2_psf_saturation_and_physics_noise():
    rng = np.random.default_rng(123)
    k = anisotropic_psf_kernel(0.4, 1.2, 0.7, radius=4)
    assert k.shape == (9, 9) and torch.isclose(k.sum(), torch.tensor(1.0), atol=1e-6)
    # Anisotropic kernel must not be transpose-symmetric at general angle
    assert float((k - k.T).abs().max()) > 1e-3

    hr = np.full((4, 32, 32), 0.005, dtype=np.float32)
    hr[:, :, 16:] = 0.9
    params = DegradationParams(
        sigma_x=0.5, sigma_y=1.1, theta=0.4, aperture_fill=0.9,
        exposure=2.0, iso=3200.0, shot_k=3e-4, read_std=2e-3,
        read_excess_kurtosis=1.0, row_std=2e-4, clip_highlights=True, quantize_bits=14,
    )
    noisy_lr, hr_clipped, sat_mask = simulate_lr_observation(hr, params, rng)
    assert noisy_lr.shape == (4, 16, 16)
    assert hr_clipped.shape == (4, 32, 32)
    assert sat_mask.any()
    assert float(noisy_lr.max()) <= 1.0 + 1e-6
    assert float(hr_clipped.max()) <= 1.0 + 1e-6
    assert float(noisy_lr.min()) < 0.005  # Signed noise preserved near black

    # Random sampler produces valid params
    p_rand = sample_degradation_params(rng)
    assert p_rand.sigma_x > 0 and p_rand.sigma_y > 0


def test_sr_v2_loss_and_cpu_smoke_step():
    torch.manual_seed(42)
    cfg = PhotoSRv2Config(
        width=16, enc_blk_nums=(1, 1), middle_blk_num=1, dec_blk_nums=(1, 1), sca_kernels=(7, 5, 3)
    )
    model = PhotoSRv2(cfg)
    criterion = SRv2Loss(domain="asinh", domain_scale=0.05, border=4)

    target = torch.full((2, 4, 32, 32), 0.01)
    target[..., 16:] = 0.95
    noisy_lr = forward_degrade_v2(target, DegradationParams()).clamp_max(1.0)
    guide_lr = noisy_lr.clone()

    # Overshoot penalty responds strictly to negative halos & >1.0 blowups
    bad = target.clone()
    bad[..., 15] = -0.3
    bad[..., 16] = 1.8
    l_clean = criterion(target, target)
    l_bad = criterion(bad, target)
    assert float(l_bad) > float(l_clean) + 0.05

    # AdamW updates reduce loss cleanly with finite gradients
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(25):
        pred = model(noisy_lr, guide_lr)
        loss = criterion(pred, target)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], f"Expected loss decrease, got {losses[0]:.5f} -> {losses[-1]:.5f}"
