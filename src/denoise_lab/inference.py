"""Portable, unconditioned packed Bayer inference; optional runtimes load lazily."""
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

from .arch_registry import build_model
from .defects import correct_defects
from .tiling import tiled_inference


class Denoiser:
    @classmethod
    def load(cls, path, backend="auto", device="auto", providers=None, weights="model"):
        """Load model or EMA weights. ONNX/Core ML artifacts select weights at export."""
        path = Path(path)
        inferred = {".pt": "torch", ".onnx": "onnx", ".mlpackage": "coreml"}
        backend = inferred.get(path.suffix.lower()) if backend == "auto" else backend
        if backend not in {"torch", "onnx", "coreml"}:
            raise ValueError("Expected a .pt, .onnx or .mlpackage model, or an explicit backend")
        if weights not in {"model", "ema"}:
            raise ValueError("weights must be model or ema")
        if backend != "torch" and weights != "model":
            raise ValueError("Select EMA weights when exporting ONNX/Core ML")
        self = cls()
        self.backend = backend
        self.device = "cpu"
        self.providers = []
        if backend == "torch":
            saved = torch.load(path, map_location="cpu", weights_only=True)
            config = saved["model_config"]
            if config.get("cond", "none") != "none":
                raise ValueError("Only unconditioned (cond=none) models are supported")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu")
            self.device = str(device)
            model = build_model(config)
            if weights not in saved or saved[weights] is None:
                raise ValueError(f"Checkpoint has no {weights} weights")
            model.load_state_dict(saved[weights])
            self._call = model.to(device).eval()
        elif backend == "onnx":
            import onnxruntime as ort
            available = ort.get_available_providers()
            order = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
            if device == "cpu" and providers is None:
                order = ["CPUExecutionProvider"]
            elif device != "auto" and providers is None:
                raise ValueError("ONNX device must be auto or cpu; select accelerators with providers")
            selected = [p for p in (providers if providers is not None else order) if p in available]
            if not selected:
                raise ValueError(f"No requested ONNX providers available; installed: {available}")
            options = ort.SessionOptions()
            options.intra_op_num_threads = min(torch.get_num_threads(), 8)
            while True:
                try:
                    session = ort.InferenceSession(str(path), sess_options=options, providers=selected)
                    break
                except Exception as exc:
                    if providers is not None or len(selected) == 1:
                        raise RuntimeError(f"ONNX initialization failed for {selected}: {exc}") from exc
                    failed = selected.pop(0)
                    warnings.warn(f"ONNX provider {failed} failed to initialize; retrying {selected}: {exc}",
                                  RuntimeWarning)
            inputs = session.get_inputs()
            if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 4:
                raise ValueError("Only unconditioned ONNX models with one Bx4xHxW input are supported")
            if session.get_modelmeta().custom_metadata_map.get("cond", "none") != "none":
                raise ValueError("Only unconditioned (cond=none) models are supported")
            self.providers = session.get_providers()
            dtype = np.float16 if inputs[0].type == "tensor(float16)" else np.float32
            def run(tile):
                output = session.run(None, {inputs[0].name: tile.numpy().astype(dtype, copy=False)})[0]
                return torch.from_numpy(output.astype(np.float32, copy=False))
            self._call = run
        else:
            if sys.platform != "darwin":
                raise RuntimeError("Core ML .mlpackage inference requires macOS; use .pt or .onnx on this platform")
            import coremltools as ct
            units = ct.ComputeUnit.CPU_ONLY if device == "cpu" else ct.ComputeUnit.ALL
            if device not in {"auto", "cpu"}:
                raise ValueError("Core ML device must be auto or cpu")
            model = ct.models.MLModel(str(path), compute_units=units)
            spec = model.get_spec()
            inputs = spec.description.input
            if len(inputs) != 1 or list(inputs[0].type.multiArrayType.shape) != [1, 4, 512, 512]:
                raise ValueError("Only unconditioned Core ML 1x4x512x512 models are supported")
            input_name, output_name = inputs[0].name, spec.description.output[0].name
            self.providers = [f"CoreML ({units.name})"]
            def run(tile):
                output = model.predict({input_name: tile.numpy()})[output_name]
                return torch.from_numpy(np.asarray(output, dtype=np.float32))
            self._call = run
        return self

    @property
    def description(self):
        return f"{self.backend}: {', '.join(self.providers) if self.providers else self.device}"

    @torch.inference_mode()
    def denoise_packed(self, packed, *, prepass=True, overlap=60, strength=1., tile_size=512):
        """Denoise float32 R/G1/B/G2; strength blends against the original input."""
        packed = np.asarray(packed)
        if packed.dtype != np.float32 or packed.ndim != 3 or packed.shape[0] != 4 or min(packed.shape) < 1:
            raise ValueError("Expected nonempty float32 packed input [4,H,W]")
        if not np.isfinite(packed).all():
            raise ValueError("Packed input must be finite")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if not 0 <= overlap < tile_size or tile_size % 16:
            raise ValueError("overlap must be below tile_size; tile_size must be a multiple of 16")
        original = torch.from_numpy(np.array(packed, copy=True))[None].to(self.device)
        if strength == 0:
            return packed.copy()
        image = correct_defects(original, k=8, abs_floor=.01, adjacency_ratio=.1)[0] if prepass else original
        output = tiled_inference(self._call, image, tile_size=tile_size, overlap=overlap, feather_power=2.2)
        output = strength*output + (1-strength)*original
        result = output[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Model produced nonfinite output")
        return result
