"""Standalone deployment smoke for a tiny dynamic or fixed-512 ONNX NAFNet.

Run in a runtime-only environment or under a torch-blocking sitecustomize hook.
This intentionally avoids importing pytest or training/test fixture modules.
"""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import rawpy
import tifffile

from denoise_lab import cli
from denoise_lab.bayer import normalize_raw, pack
from denoise_lab.dng import RawMetadata, write_dng
from denoise_lab.inference import Denoiser


def main():
    model_path, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir/'source.dng'
    meta = RawMetadata(pattern=np.array([[2, 1], [1, 0]], np.uint8),
                       black=np.full((2, 2), 512.), white=np.full((2, 2), 16383.))
    codes = np.random.default_rng(7).integers(500, 13000, (64, 80), dtype=np.uint16)
    write_dng(source, codes, meta, normalized=False)
    with rawpy.imread(str(source)) as raw:
        packed = pack(normalize_raw(raw))
    denoiser = Denoiser.load(model_path, device='cpu')
    actual = denoiser.denoise_packed(packed)
    assert actual.dtype == np.float32 and actual.shape == packed.shape
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(denoiser.denoise_packed(packed, strength=0), packed)
    mask = np.zeros_like(packed, bool)
    mask[0, 0, 0] = True
    assert np.isfinite(denoiser.denoise_packed(packed, prepass='map+heuristic', defect_mask=mask)).all()
    for fmt in ('dng', 'tiff', 'npy'):
        sys.argv = ['rawdenoise', str(source), '--model', str(model_path), '--device', 'cpu',
                    '--output-dir', str(output_dir/fmt), '--format', fmt]
        cli.main()
        output = output_dir/fmt/f'source_denoised.{fmt}'
        if fmt == 'dng':
            with rawpy.imread(str(output)) as raw:
                assert raw.raw_image_visible.shape == codes.shape
                assert RawMetadata.from_raw(raw, output).phase == 'BGGR'
        elif fmt == 'tiff':
            assert tifffile.imread(output).shape == (*codes.shape, 3)
        else:
            assert np.load(output).shape == codes.shape
    # Test the Core ML array boundary without a platform/runtime requirement.
    spec = SimpleNamespace(description=SimpleNamespace(
        input=[SimpleNamespace(name='input', type=SimpleNamespace(
            multiArrayType=SimpleNamespace(shape=[1, 4, 512, 512])))],
        output=[SimpleNamespace(name='output')]))
    class Model:
        def __init__(self, *args, **kwargs):
            pass
        def get_spec(self):
            return spec
        def predict(self, inputs):
            assert isinstance(inputs['input'], np.ndarray)
            return {'output': inputs['input']*.9}
    units = SimpleNamespace(name='CPU_ONLY')
    sys.modules['coremltools'] = SimpleNamespace(models=SimpleNamespace(MLModel=Model),
                                               ComputeUnit=SimpleNamespace(CPU_ONLY=units, ALL=units))
    platform = sys.platform
    try:
        sys.platform = 'darwin'
        coreml = Denoiser.load('fixture.mlpackage', device='cpu')
        assert np.isfinite(coreml.denoise_packed(packed)).all()
    finally:
        sys.platform = platform
        del sys.modules['coremltools']
    assert 'torch' not in sys.modules
    missing = subprocess.run([sys.executable, '-m', 'denoise_lab.cli', str(source),
                              '--model', 'missing.pt'], capture_output=True, text=True)
    assert missing.returncode == 1
    assert "install 'denoise_lab[torch]'" in missing.stderr, missing.stderr
    print('Torch-free ONNX, CLI outputs, mapped prepass, Core ML array boundary and .pt error passed')


if __name__ == '__main__':
    main()
