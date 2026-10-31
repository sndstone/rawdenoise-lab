"""Export precision must not quantize the linear RAW skip to half precision."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import sys

from denoise_lab.arch_registry import build_model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_onnx_precision", ROOT / "scripts/export_onnx.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def identity_model():
    model = build_model(dict(width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)).eval()
    with torch.no_grad():
        model.body.ending.weight.zero_()
        model.body.ending.bias.zero_()
    return model


@torch.inference_mode()
def test_mixed_precision_preserves_16bit_skip_and_signed_shadows():
    # Every sensor code, plus subnormals/signed values, must survive an identity body.
    image = torch.arange(65536, dtype=torch.float32).reshape(1, 4, 128, 128)/65535
    image[..., 0, :5] = torch.tensor([-1e-7, -1e-5, 0., 1e-7, 1e-5])
    wrapper = export.MixedPrecisionIO(identity_model()).eval()
    torch.testing.assert_close(wrapper(image), image, rtol=0, atol=0)
    legacy = export.FloatIO(identity_model()).eval()(image)
    assert float((legacy-image).abs().max()*65535) > 1


def test_mixed_precision_rejects_conditioned_contract():
    model = build_model(dict(cond="noise_map", width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1))
    with pytest.raises(ValueError, match="unconditioned Bayer NAFNet"):
        export.MixedPrecisionIO(model)


@torch.inference_mode()
def test_mixed_precision_onnx_preserves_code_steps(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    wrapper = export.MixedPrecisionIO(identity_model()).eval()
    image = torch.linspace(-1e-4, 1, 4*16*16).reshape(1, 4, 16, 16)
    path = tmp_path / "mixed.onnx"
    torch.onnx.export(wrapper, image, str(path), input_names=["inp_1"], output_names=["output"],
                      dynamo=False, opset_version=17)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"inp_1": image.numpy()})[0]
    np.testing.assert_allclose(actual, image.numpy(), rtol=0, atol=1e-7)


def test_failed_export_does_not_leave_artifact_or_overwrite(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    path, destination = tmp_path/"identity.pt", tmp_path/"result.onnx"
    config = dict(width=2, enc_blk_nums=[1], dec_blk_nums=[1], middle_blk_num=1)
    torch.save(dict(model_config=config, model=identity_model().state_dict()), path)
    class BadRuntime:
        def __init__(self, *args, **kwargs): pass
        def run(self, *args): return [np.full((1, 4, 512, 512), np.nan, np.float32)]
    monkeypatch.setattr(ort, "InferenceSession", BadRuntime)
    monkeypatch.setattr(sys, "argv", ["export_onnx.py", "--checkpoint", str(path), "--output", str(destination)])
    with pytest.raises(RuntimeError, match="ONNX parity failed"):
        export.main()
    assert not destination.exists()
    assert not list(tmp_path.glob(".onnx-export-*"))
    destination.write_bytes(b"preserve existing model")
    with pytest.raises(SystemExit):
        export.main()
    assert destination.read_bytes() == b"preserve existing model"
