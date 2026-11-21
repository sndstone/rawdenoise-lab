"""Synthetic NumPy contracts; torch parity checks skip in runtime-only installs."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from denoise_lab import np_pipeline as pipeline
from denoise_lab.bayer import pack, unpack
from denoise_lab.inference import Denoiser

ROOT = Path(__file__).resolve().parents[1]


def sample(shape=(1, 4, 37, 43)):
    a = np.random.default_rng(12).normal(.1, .002, shape).astype(np.float32)
    a[..., 0, 0] = .9
    a[..., -1, -1] = -.2
    return a


@pytest.mark.parametrize('shape', [(1, 4, 1, 1), (2, 4, 2, 3), (1, 4, 17, 19)])
def test_numpy_reflection_matches_sensor(shape):
    a = sample(shape)
    mosaic = unpack(a)
    y = pipeline.reflect_indices(mosaic.shape[-2], 12, 16)
    x = pipeline.reflect_indices(mosaic.shape[-1], 10, 14)
    expected = pack(mosaic.take(y, axis=-2).take(x, axis=-1))
    actual = pipeline.reflected_bayer_pad(a, 5, 7, 6, 8)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('mode', ['linear', 'cosine'])
@pytest.mark.parametrize('overlap,alignment,offset', [(0, 1, 0), (8, 1, 3), (13, 16, 0)])
def test_numpy_tiling_partition(mode, overlap, alignment, offset):
    a = sample()
    out = pipeline.tiled_inference(lambda tile: tile*.8+.01, a, tile_size=32,
                                   overlap=overlap, grid_alignment=alignment,
                                   grid_offset=offset, feather_mode=mode)
    np.testing.assert_allclose(out, a*.8+.01, atol=2e-7, rtol=2e-6)
    assert out.dtype == np.float32


def test_numpy_defects_map_and_highlights():
    a = np.full((1, 4, 32, 40), .1, np.float32)
    a[0, 0, 5, 6:8] = [.8, .9]
    a[0, 2, 0, 0] = -.5
    mask = np.zeros_like(a, bool)
    mask[0, 0, 5, 6:8] = True
    mask[0, 2, 0, 0] = True
    fixed, flags = pipeline.correct_mapped_defects(a, mask)
    np.testing.assert_array_equal(flags, mask)
    assert (fixed[mask] == .1).all()
    np.testing.assert_array_equal(fixed[~mask], a[~mask])
    with pytest.raises(ValueError, match='no valid'):
        pipeline.correct_mapped_defects(a, np.ones_like(mask))
    a = fixed
    a[0, 0, 5, 5] = 1
    a[:, :, 20:25, 20:25] = 1  # Shared real detail survives the heuristic.
    corrected, flags = pipeline.correct_defects(a, chunk_rows=3)
    assert flags.sum() == 1 and flags[0, 0, 5, 5]
    protected = pipeline.protect_highlights(corrected*.7, corrected)
    assert protected[0, 0, 5, 5] < .2
    np.testing.assert_array_equal(protected[..., 21:24, 21:24], corrected[..., 21:24, 21:24])
    assert np.isfinite(protected).all()


@pytest.mark.parametrize('shape', [(2, 4, 1, 1), (1, 4, 2, 3), (1, 4, 37, 43), (1, 4, 258, 260)])
def test_torch_prepass_parity(shape):
    torch = pytest.importorskip('torch')
    from denoise_lab.defects import correct_defects, correct_mapped_defects
    from denoise_lab.inference import protect_highlights
    a = sample(shape)
    tensor = torch.from_numpy(a)
    expected, flags = correct_defects(tensor, adjacency_ratio=.1, chunk_rows=7)
    actual, found = pipeline.correct_defects(a, chunk_rows=5)
    np.testing.assert_array_equal(found, flags.numpy())
    np.testing.assert_array_equal(actual, expected.numpy())
    if min(shape[-2:]) > 2:
        mask = np.zeros_like(a, bool)
        mask[..., 0, 0] = True
        mask[..., 2, 2:4] = True
        expected = correct_mapped_defects(tensor, torch.from_numpy(mask))[0].numpy()
        np.testing.assert_array_equal(pipeline.correct_mapped_defects(a, mask)[0], expected)
    a[..., -1, -1] = 1.02
    expected = protect_highlights(torch.from_numpy(a*.8), tensor).numpy()
    np.testing.assert_array_equal(pipeline.protect_highlights(a*.8, a), expected)


@pytest.mark.parametrize('mode', ['linear', 'cosine'])
@pytest.mark.parametrize('prepass', ['off', 'heuristic', 'map', 'map+heuristic'])
def test_torch_complete_pipeline_parity(mode, prepass):
    pytest.importorskip('torch')
    a = sample()[0]
    a[1, 20:25, 20:25] = 1.02
    mask = np.zeros(a.shape, bool)
    mask[0, 0, 0] = True
    numpy_model, torch_model = Denoiser(), Denoiser()
    numpy_model.backend = 'onnx'
    for model in (numpy_model, torch_model):
        model.device = 'cpu'
        model._call = lambda tile: tile*.8+.03
    kwargs = dict(prepass=prepass, defect_mask=mask, tile_size=64, overlap=9,
                  grid_alignment=16, feather_mode=mode, tta_shift=2, strength=.7)
    expected = torch_model.denoise_packed(a, **kwargs)
    actual = numpy_model.denoise_packed(a, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(numpy_model.denoise_packed(a, strength=0), a)
    numpy_model._call = lambda tile: tile[..., :1]
    with pytest.raises(ValueError, match='shape'):
        numpy_model.denoise_packed(a, tile_size=32, overlap=0)
    numpy_model._call = lambda tile: tile*np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        numpy_model.denoise_packed(a, tile_size=32, overlap=0)


def test_torch_blocked_subprocess(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from denoise_lab.arch_registry import build_model
    model = build_model(dict(width=2, enc_blk_nums=[1], middle_blk_num=1, dec_blk_nums=[1])).eval()
    path = tmp_path/'tiny.onnx'
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 4, 32, 32), str(path), opset_version=17,
                          dynamo=False, input_names=['inp_1'], output_names=['output'],
                          dynamic_axes={'inp_1': {2: 'height', 3: 'width'},
                                        'output': {2: 'height', 3: 'width'}})
    hook = tmp_path/'sitecustomize.py'
    hook.write_text('''import sys
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ImportError('torch deliberately blocked')
sys.meta_path.insert(0, BlockTorch())
''')
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(tmp_path), str(ROOT/'src')]),
           'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
    commands = [[sys.executable, '-m', 'pytest', '-q', str(Path(__file__)), '-k', 'numpy'],
                [sys.executable, str(ROOT/'tests/torchfree_smoke.py'), str(path), str(tmp_path/'smoke')]]
    for command in commands:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout+result.stderr
