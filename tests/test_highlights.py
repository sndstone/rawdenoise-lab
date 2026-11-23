"""Synthetic CPU checks for backend-independent, normalized Bayer protection."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from denoise_lab.bayer import PHASES, normalize_raw, pack
from denoise_lab.defects import correct_defects
from denoise_lab.inference import Denoiser, protect_highlights
from denoise_lab.tiling import tiled_inference


def denoiser(call):
    model = Denoiser()
    model.device = 'cpu'
    model._call = call
    return model


def srgb_chroma(packed):
    # Bayer-cell RGB with fixed WB, deliberately below display clipping.
    rgb = np.stack([packed[0]*1.7, (packed[1]+packed[3])*.5, packed[2]*2.])*.6
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(rgb <= .0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)
    return rgb-rgb.mean(axis=0, keepdims=True)


def test_clipped_boundary_chroma():
    y, x = np.mgrid[:64, :64]
    light = .03+1.5*np.exp(-((y-32)**2+(x-32)**2)/(2*12**2))
    clean = np.minimum(np.array([.55, 1., .4, 1.])[:, None, None]*light, 1).astype(np.float32)
    noisy = np.minimum(clean+np.random.default_rng(7).normal(0, .001, clean.shape), 1).astype(np.float32)
    # Simulate per-channel smoothing/bias at a clipped light's boundary.
    gains = torch.tensor([.9, 1., 1.1, 1.])[None, :, None, None]
    model = denoiser(lambda a: F.avg_pool2d(a, 3, stride=1, padding=1)*gains)
    off = model.denoise_packed(noisy, prepass=False, tile_size=64, overlap=0, highlights='off')
    on = model.denoise_packed(noisy, prepass=False, tile_size=64, overlap=0)
    core = torch.from_numpy((noisy >= .98).any(axis=0))[None, None].float()
    ring = (F.max_pool2d(core, 5, stride=1, padding=2) > 0)[0, 0].numpy() & ~core[0, 0].bool().numpy()
    target = srgb_chroma(clean)
    before = np.linalg.norm(srgb_chroma(off)-target, axis=0)[ring].mean()
    after = np.linalg.norm(srgb_chroma(on)-target, axis=0)[ring].mean()
    assert after < .007
    assert after < .25*before


@pytest.mark.parametrize('prepass,strength', [(False, 1.), (True, .4)])
def test_off_matches_legacy_exactly(prepass, strength):
    packed = np.random.default_rng(4).uniform(-.03, 1.03, (4, 23, 29)).astype(np.float32)
    model = denoiser(lambda a: a*.97+.02)
    original = torch.from_numpy(packed.copy())[None]
    image = correct_defects(original, k=8, abs_floor=.01, adjacency_ratio=.1)[0] if prepass else original
    legacy = tiled_inference(model._call, image, tile_size=32, overlap=8, feather_power=2.2)
    legacy = strength*legacy+(1-strength)*original
    result = model.denoise_packed(packed, prepass=prepass, strength=strength,
                                  tile_size=32, overlap=8, highlights='off')
    np.testing.assert_array_equal(result, legacy[0].numpy())


def test_unclipped_and_zero_strength_unchanged():
    packed = np.random.default_rng(3).uniform(-.1, .97, (4, 19, 21)).astype(np.float32)
    model = denoiser(lambda a: a*2)  # Even overshoot is unchanged away from clipping.
    on = model.denoise_packed(packed, tile_size=32, overlap=8)
    off = model.denoise_packed(packed, tile_size=32, overlap=8, highlights='off')
    np.testing.assert_array_equal(on, off)
    packed[:, 4:8, 4:8] = 1.01
    np.testing.assert_array_equal(model.denoise_packed(packed, strength=0), packed)
    with pytest.raises(ValueError, match='highlights'):
        model.denoise_packed(packed, highlights='invalid')


@pytest.mark.parametrize('phase', list(PHASES))
def test_per_channel_white_and_cfa_phase(phase):
    # Unequal black/white levels, including distinct green channels.
    indices = {'R': 0, 'G': 1, 'B': 2}
    pattern = np.array([indices[c] for c in phase]).reshape(2, 2)
    gy, gx = np.argwhere(pattern == 1)[1]
    pattern[gy, gx] = 3
    colors = np.tile(pattern, (24, 24))
    black = np.array([100, 200, 300, 400], dtype=np.float32)
    white = np.array([1100, 2200, 4300, 8400], dtype=np.float32)
    normalized = np.full((48, 48), .2, np.float32)
    normalized[20:26, 20:26][colors[20:26, 20:26] == 1] = 1
    raw = SimpleNamespace(raw_image_visible=normalized*(white[colors]-black[colors])+black[colors],
                          raw_colors_visible=colors, black_level_per_channel=black,
                          camera_white_level_per_channel=white, white_level=8400, color_desc=b'RGBG')
    packed = pack(normalize_raw(raw))
    original = torch.from_numpy(packed)[None]
    result = protect_highlights(original*.7, original)[0].numpy()
    core = (packed >= .98).any(axis=0)
    assert core.any()
    np.testing.assert_array_equal(result[:, core], packed[:, core])


def test_feather_footprint_and_white_ceiling():
    original = torch.full((1, 4, 15, 15), .2)
    original[0, 1, 7, 7] = 1.01
    output = torch.full_like(original, 1.2)
    result = protect_highlights(output, original)
    # Sites already >= 1 keep their exact original headroom rather than clamping to 1.0.
    assert result[0, 1, 7, 7] == original[0, 1, 7, 7]
    np.testing.assert_array_equal(result[0, [0, 2, 3], 7, 7], original[0, [0, 2, 3], 7, 7])
    area = result[..., 3:12, 3:12]
    orig_area = original[..., 3:12, 3:12]
    assert (area <= torch.maximum(orig_area, torch.ones_like(orig_area))).all()
    assert (area[orig_area < 1] < 1).all()
    np.testing.assert_array_equal(result[..., :3, :], output[..., :3, :])
    # Shared weights of 1, .75, .5, .25, 0 at increasing distances.
    soft = protect_highlights(torch.zeros_like(original), original)
    np.testing.assert_allclose(soft[0, 0, 7, 8:13], [.2, .15, .1, .05, 0], atol=1e-7)


def test_protect_highlights_preserves_above_white_headroom_and_backend_parity():
    from denoise_lab.bayer import unpack
    from denoise_lab.dng import RawMetadata, sensor_codes
    from denoise_lab.np_pipeline import protect_highlights as np_protect_highlights

    # Simulate Sony A7M3/A5100 partial-clipping headroom (camera_white_level=15360, sensor codes up to 16596).
    original = np.full((1, 4, 16, 16), 0.25, dtype=np.float32)
    original[0, 0, 6:10, 6:10] = 0.50532
    original[0, 1, 6:10, 6:10] = 1.06890
    original[0, 2, 6:10, 6:10] = 1.02755
    original[0, 3, 6:10, 6:10] = 1.06890
    # Model output with overshoot and green dip inside core/feather.
    output = original.copy()
    output[0, 0] += 0.05
    output[0, 1] -= 0.04
    output[0, 2] += 0.08
    output[0, 3] -= 0.04

    res_torch = protect_highlights(torch.from_numpy(output), torch.from_numpy(original)).numpy()
    res_np = np_protect_highlights(output, original)
    np.testing.assert_array_equal(res_torch, res_np)

    # Protected core preserves exact >1.0 channel values -> zero raw (R+B)/2 - G distortion!
    np.testing.assert_array_equal(res_torch[:, :, 6:10, 6:10], original[:, :, 6:10, 6:10])
    # DNG sensor_codes roundtrip preserves >15360 Sony headroom codes cleanly.
    meta = RawMetadata(black=np.full((2, 2), 512.0), white=np.full((2, 2), 15360.0))
    codes = sensor_codes(unpack(res_torch[0]), meta)
    assert int(codes.max()) == round(1.06890 * (15360.0 - 512.0) + 512.0)


def test_tta_transpose_cancels_g1_g2_bias_torch_and_numpy():
    from denoise_lab.np_pipeline import denoise_packed as np_denoise_packed

    rng = np.random.default_rng(11)
    packed = rng.uniform(0.05, 0.35, (4, 32, 32)).astype(np.float32)

    # Artificial denoiser with deliberate +0.02 G1 / -0.02 G2 asymmetry (mimics r4_plain green split).
    bias = np.array([0.0, +0.02, 0.0, -0.02], dtype=np.float32)[None, :, None, None]
    torch_bias = torch.from_numpy(bias)
    model_t = denoiser(lambda a: a + torch_bias)

    off_t = model_t.denoise_packed(packed, prepass=False, tile_size=32, overlap=0, highlights='off', tta_transpose=False)
    on_t = model_t.denoise_packed(packed, prepass=False, tile_size=32, overlap=0, highlights='off', tta_transpose=True)
    # Without transpose TTA, G1-G2 shifts by +0.04; with exact Bayer transpose TTA, bias cancels to 0.
    raw_g12 = float(np.mean(packed[1] - packed[3]))
    assert abs(float(np.mean(off_t[1] - off_t[3])) - (raw_g12 + 0.04)) < 1e-6
    assert abs(float(np.mean(on_t[1] - on_t[3])) - raw_g12) < 1e-6

    # Exact parity with NumPy / ONNX / Core ML pipeline.
    on_np = np_denoise_packed(
        lambda a: a + bias, packed, mode='off', defect_mask=None, overlap=0, strength=1.0,
        tile_size=32, highlights='off', offsets=[0], grid_alignment=1,
        feather_mode='linear', chroma_strength=1.0, gains=(1., 1., 1.), tta_transpose=True,
    )
    np.testing.assert_allclose(on_t, on_np, atol=1e-6)
    # Verify Denoiser wrapper rejects non-bool tta_transpose cleanly.
    with pytest.raises(ValueError, match='tta_transpose'):
        model_t.denoise_packed(packed, tile_size=32, overlap=0, tta_transpose=2)


def test_tta_transpose_rectangular_multitile_and_directional_operator():
    from denoise_lab.np_pipeline import denoise_packed as np_denoise_packed
    from denoise_lab.tiling import shift_offsets

    rng = np.random.default_rng(29)
    # Non-square multi-tile shape (H != W, triggers multiple tiles & feather seams in both axes).
    packed = rng.uniform(0.05, 0.45, (4, 40, 68)).astype(np.float32)

    # 1) Identity round-trip under multi-tile + tta_shift=2 + tta_transpose=True.
    id_torch = denoiser(lambda a: a).denoise_packed(
        packed, prepass=False, tile_size=32, overlap=8, tta_shift=2, highlights='off', tta_transpose=True,
    )
    id_np = np_denoise_packed(
        lambda a: a, packed, mode='off', defect_mask=None, overlap=8, strength=1.0,
        tile_size=32, highlights='off', offsets=shift_offsets(32, 8, 1, 2), grid_alignment=1,
        feather_mode='linear', chroma_strength=1.0, gains=(1., 1., 1.), tta_transpose=True,
    )
    np.testing.assert_allclose(id_torch, packed, atol=1e-6)
    np.testing.assert_allclose(id_np, packed, atol=1e-6)
    np.testing.assert_allclose(id_torch, id_np, atol=1e-6)

    # 2) Direction-sensitive anisotropic operator (horizontal shift + vertical gradient + G1/G2 bias).
    # Catches any bug in spatial axis inversion or G1<->G2 channel permutation across rectangular tiles.
    ch_bias = np.array([0.01, +0.03, -0.02, -0.01], dtype=np.float32)[None, :, None, None]
    ch_bias_t = torch.from_numpy(ch_bias)

    def aniso_torch(a):
        horiz = torch.roll(a, shifts=1, dims=-1)
        vert_diff = a - torch.roll(a, shifts=-1, dims=-2)
        return 0.7 * a + 0.2 * horiz + 0.1 * vert_diff + ch_bias_t

    def aniso_np(a):
        horiz = np.roll(a, shift=1, axis=-1)
        vert_diff = a - np.roll(a, shift=-1, axis=-2)
        return (0.7 * a + 0.2 * horiz + 0.1 * vert_diff + ch_bias).astype(np.float32)

    # Single-tile (tile_size=128) interior exact check against analytical Bayer transpose reference,
    # plus full-image (including reflected borders) exact check against explicit TTA-off average.
    perm = [0, 3, 2, 1]
    x_t = torch.from_numpy(packed)[None]
    x_t_tr = x_t[:, perm].transpose(-2, -1).contiguous()
    expected_interior = (0.5 * (aniso_torch(x_t) + aniso_torch(x_t_tr).transpose(-2, -1)[:, perm]))[0].numpy()

    model_aniso = denoiser(aniso_torch)
    out_t_1 = model_aniso.denoise_packed(
        packed, prepass=False, tile_size=128, overlap=0, highlights='off', tta_transpose=True,
    )
    out_np_1 = np_denoise_packed(
        aniso_np, packed, mode='off', defect_mask=None, overlap=0, strength=1.0,
        tile_size=128, highlights='off', offsets=[0], grid_alignment=1,
        feather_mode='linear', chroma_strength=1.0, gains=(1., 1., 1.), tta_transpose=True,
    )
    np.testing.assert_allclose(out_t_1[:, 1:-1, 1:-1], expected_interior[:, 1:-1, 1:-1], atol=1e-6)
    np.testing.assert_allclose(out_np_1[:, 1:-1, 1:-1], expected_interior[:, 1:-1, 1:-1], atol=1e-6)
    np.testing.assert_allclose(out_t_1, out_np_1, atol=1e-6)

    # Full-frame (borders + seams) check against two explicit non-transposed runs on multi-tile + tta_shift=2.
    packed_tr = np.ascontiguousarray(packed[perm].transpose(0, 2, 1))
    ref_fwd = model_aniso.denoise_packed(
        packed, prepass=False, tile_size=32, overlap=8, tta_shift=2, highlights='off', tta_transpose=False,
    )
    ref_tr = model_aniso.denoise_packed(
        packed_tr, prepass=False, tile_size=32, overlap=8, tta_shift=2, highlights='off', tta_transpose=False,
    ).transpose(0, 2, 1)[perm]
    expected_full_m = 0.5 * (ref_fwd + ref_tr)

    out_t_m = model_aniso.denoise_packed(
        packed, prepass=False, tile_size=32, overlap=8, tta_shift=2, highlights='off', tta_transpose=True,
    )
    out_np_m = np_denoise_packed(
        aniso_np, packed, mode='off', defect_mask=None, overlap=8, strength=1.0,
        tile_size=32, highlights='off', offsets=shift_offsets(32, 8, 1, 2), grid_alignment=1,
        feather_mode='linear', chroma_strength=1.0, gains=(1., 1., 1.), tta_transpose=True,
    )
    np.testing.assert_allclose(out_t_m, expected_full_m, atol=1e-6)
    np.testing.assert_allclose(out_np_m, expected_full_m, atol=1e-6)
    np.testing.assert_allclose(out_t_m, out_np_m, atol=1e-6)


def test_gate_highlight_metrics_detects_headroom_clamp_in_lab():
    import importlib.util
    from pathlib import Path
    from denoise_lab.dng import RawMetadata

    gate_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_foliage_highlights_gate.py"
    spec = importlib.util.spec_from_file_location("evaluate_foliage_highlights_gate", gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    highlight_metrics = gate.highlight_metrics

    # Sony-like partial clip with >1.0 headroom in G and B, unclipped R.
    orig = np.full((4, 32, 32), 0.20, dtype=np.float32)
    orig[0, 10:22, 10:22] = 0.50532
    orig[1, 10:22, 10:22] = 1.06890
    orig[2, 10:22, 10:22] = 1.02755
    orig[3, 10:22, 10:22] = 1.06890
    meta = RawMetadata(
        black=np.full((2, 2), 512.0),
        white=np.full((2, 2), 15360.0),
        neutral=(0.45, 1.0, 0.65),
    )

    # A broken denoiser that only clamps >1.0 headroom to 1.0 must show a clear non-zero Lab shift.
    clamped = np.minimum(orig, 1.0).astype(np.float32)
    m_clamped = highlight_metrics(orig, clamped, meta, exposure=0.5, headroom_scale=4.0, prepass_ref=orig)
    assert m_clamped["has_clip"]
    assert abs(m_clamped["core"]["mean_da"]) > 0.5
    assert m_clamped["core"]["raw_rb_minus_g_shift"] > 0.02
    assert "max_abs_diff" in m_clamped["ring_1_4"]
    assert "core_max_abs_diff" not in m_clamped["ring_1_4"]

    # Fixed protect_highlights preserves >1.0 headroom -> zero Lab shift and zero prepass diff.
    fixed = protect_highlights(torch.from_numpy(clamped)[None], torch.from_numpy(orig)[None])[0].numpy()
    m_fixed = highlight_metrics(orig, fixed, meta, exposure=0.5, headroom_scale=4.0, prepass_ref=orig)
    assert abs(m_fixed["core"]["mean_da"]) < 1e-6
    assert abs(m_fixed["core"]["mean_db"]) < 1e-6
    assert m_fixed["core"]["prepass_max_abs_diff"] == 0.0


def test_protection_does_not_restore_prepass_defects():
    packed = np.full((4, 32, 32), .2, np.float32)
    packed[:, 16:24, 16:24] = 1  # A real highlight shared across adjacent sites.
    packed[0, 6, 6] = 1  # Isolated saturated defect, far from that highlight.
    packed[0, 16, 14] = 1  # Another isolated defect inside the feather footprint.
    model = denoiser(lambda a: a)
    on = model.denoise_packed(packed, tile_size=32, overlap=0)
    assert on[0, 6, 6] == packed[1, 6, 6]
    assert on[0, 16, 14] == packed[1, 16, 14]
    np.testing.assert_array_equal(on[:, 18:22, 18:22], packed[:, 18:22, 18:22])
    no_prepass = model.denoise_packed(packed, prepass=False, tile_size=32, overlap=0)
    assert no_prepass[0, 6, 6] == 1
