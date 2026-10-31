"""Portable Bayer inference and distinct experimental PyTorch RGB inference."""
from pathlib import Path
import copy
import gc
import hashlib
import io
import sys
import warnings

import numpy as np
import torch

from .arch_registry import build_model
from .checkpoint import checkpoint_weights
from .defects import correct_defects
from .rgb import RGB_ARCH, validate_rgb_contract
from .tiling import tiled_inference


class Denoiser:
    @classmethod
    def load(cls, path, backend="auto", device="auto", providers=None, weights="model",
             experimental_mps_body=False):
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
        if experimental_mps_body:
            from .accelerator import require_mps_environment
            if backend != "torch" or device != "mps":
                raise ValueError("Experimental MPS body requires a .pt model and explicit device=mps")
            require_mps_environment()
        self = cls()
        self.backend = backend
        self.device = "cpu"
        self.providers = []
        self.image_kind = "bayer"
        self.rgb_contract = None
        self._experimental_mps_body = bool(experimental_mps_body)
        self._accelerator_active = False
        self._provenance = None
        if backend == "torch":
            if experimental_mps_body:
                # Hash exactly the bytes deserialized, avoiding a second path read.
                payload = path.read_bytes()
                checkpoint_sha = hashlib.sha256(payload).hexdigest()
                saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
                del payload
            else:
                saved = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(saved, dict) or not isinstance(saved.get("model_config"), dict):
                raise ValueError("Checkpoint requires an explicit model_config object identifying its architecture")
            config = saved["model_config"]
            if config.get("cond", "none") != "none":
                raise ValueError("Only unconditioned (cond=none) models are supported")
            if config.get("arch") == RGB_ARCH:
                self.image_kind = "rgb"
                self.rgb_contract = validate_rgb_contract(saved.get("rgb_contract"))
            elif "rgb_contract" in saved:
                raise ValueError("RGB checkpoint/model mismatch: rgb_contract requires arch=rgb_nafnet")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu")
            self.device = str(device)
            model = build_model(config)
            try:
                model.load_state_dict(checkpoint_weights(saved, weights))
            except RuntimeError as exc:
                raise ValueError(f"Checkpoint/model mismatch ({config.get('arch', 'nafnet')}): {exc}") from exc
            if experimental_mps_body:
                from .model import DenoiseNet
                if self.image_kind != "bayer" or not isinstance(model, DenoiseNet):
                    raise ValueError("Experimental MPS body supports only unconditioned Bayer DenoiseNet")
                self._initialize_mps_body(model.eval(), config, checkpoint_sha, weights)
            else:
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
            if (session.get_modelmeta().custom_metadata_map.get("precision") == "fp16io"
                    or inputs[0].type == "tensor(float16)"):
                warnings.warn("This ONNX model quantizes the sensor input to fp16; use an fp32 export for 16-bit color fidelity.",
                              RuntimeWarning)
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
                output = model.predict({input_name: np.ascontiguousarray(tile.numpy())})[output_name]
                return torch.from_numpy(np.asarray(output, dtype=np.float32))
            self._call = run
        return self

    @property
    def description(self):
        if self._experimental_mps_body:
            return (f"torch: {self._provenance['producing_backend']} "
                    f"(experimental MPS body; not quality-accepted; "
                    f"weights_sha256={self._provenance['checkpoint_sha256']})")
        return f"{self.backend}: {', '.join(self.providers) if self.providers else self.device}"

    @property
    def provenance(self):
        """Last producing backend and frozen artifact identity for opt-in logging."""
        return copy.deepcopy(self._provenance)

    def _initialize_mps_body(self, model, config, checkpoint_sha, weights):
        from .accelerator import AcceleratorError, make_mps_body, numeric_canary
        model = model.to(device="cpu", dtype=torch.float32)
        frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        frozen_config = copy.deepcopy(config)
        def cpu_factory():
            cpu = build_model(frozen_config).to(device="cpu", dtype=torch.float32).eval()
            cpu.load_state_dict(frozen, strict=True)
            return cpu
        self._cpu_factory = cpu_factory
        self.device = "cpu"  # Host apron, accumulation and transforms remain CPU.
        self._provenance = dict(checkpoint_sha256=checkpoint_sha, weights=weights,
                                requested_backend="experimental-mps-body", producing_backend="cpu",
                                quality_accepted=False, numeric_canary=None, fallback_reason=None)
        warnings.warn("Experimental MPS body is not quality-accepted; numeric canaries do not certify render/perceptual equivalence",
                      RuntimeWarning)
        reason = None
        try:
            if not torch.backends.mps.is_available():
                raise AcceleratorError("MPS is unavailable")
            # Never move the CPU reference's body in place.
            self._call = make_mps_body(model)
            self._provenance['numeric_canary'] = numeric_canary(model, self._call)
            self._accelerator_active = True
            self._provenance['producing_backend'] = "mps-body + cpu-transforms"
        except AcceleratorError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if reason is not None:
            self._activate_cpu(reason)

    def _activate_cpu(self, reason):
        # Called outside the failing except block, so its traceback does not pin
        # a discarded partial merge or device output during CPU reconstruction.
        self._call = None
        self._accelerator_active = False
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        self._call = self._cpu_factory()
        self._provenance.update(producing_backend="cpu fallback", fallback_reason=reason)
        warnings.warn(f"Experimental MPS body failed; whole-image CPU fallback: {reason}", RuntimeWarning)

    def _experimental_packed(self, packed, *, prepass, overlap, strength, tile_size, tile_batch_size):
        from .accelerator import AcceleratorError
        self._provenance['producing_backend'] = ("mps-body + cpu-transforms" if self._accelerator_active
                                                else "cpu fallback")
        def attempt():
            original = torch.from_numpy(np.array(packed, copy=True))[None]
            image = correct_defects(original, k=8, abs_floor=.01, adjacency_ratio=.1)[0] if prepass else original
            with torch.autocast('cpu', enabled=False):
                output = tiled_inference(self._call, image, tile_size=tile_size, overlap=overlap,
                                         feather_power=2.2, tile_batch_size=tile_batch_size)
            if output.shape != original.shape:
                error = AcceleratorError if self._accelerator_active else ValueError
                raise error("Model produced wrong final output shape")
            result = (strength*output + (1-strength)*original)[0].numpy()
            if result.dtype != np.float32 or not np.isfinite(result).all():
                error = AcceleratorError if self._accelerator_active else ValueError
                raise error("Model produced nonfinite or non-float32 output")
            return result
        if not self._accelerator_active:
            return attempt()
        reason = None
        try:
            result = attempt()
        except AcceleratorError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if reason is None:
            return result
        self._activate_cpu(reason)
        return attempt()  # Restart from original input; never retain partial GPU tiles.

    @torch.inference_mode()
    def denoise_packed(self, packed, *, prepass=True, overlap=60, strength=1., tile_size=512,
                       tile_batch_size=1):
        """Denoise float32 R/G1/B/G2; optional tile batching requires CPU PyTorch."""
        if self.image_kind != "bayer":
            raise ValueError("Bayer input requires a four-plane Bayer model; loaded checkpoint is RGB")
        packed = np.asarray(packed)
        if packed.dtype != np.float32 or packed.ndim != 3 or packed.shape[0] != 4 or min(packed.shape) < 1:
            raise ValueError("Expected nonempty float32 packed input [4,H,W]")
        if not np.isfinite(packed).all():
            raise ValueError("Packed input must be finite")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if not 0 <= overlap < tile_size or tile_size % 16:
            raise ValueError("overlap must be below tile_size; tile_size must be a multiple of 16")
        if self.backend == "coreml" and tile_size != 512:
            raise ValueError("Core ML tile_size must be 512")
        if isinstance(tile_batch_size, bool) or not isinstance(tile_batch_size, int) or tile_batch_size < 1:
            raise ValueError("tile_batch_size must be a positive integer")
        if tile_batch_size > 1 and (self.backend != "torch" or torch.device(self.device).type != "cpu"):
            raise ValueError("tile_batch_size > 1 requires the CPU PyTorch Bayer backend")
        if self._experimental_mps_body:
            if tile_batch_size not in {1, 2}:
                raise ValueError("Experimental MPS body supports tile_batch_size 1 or 2 only")
            if strength == 0:
                self._provenance['producing_backend'] = "identity (strength=0)"
                return packed.copy()
            return self._experimental_packed(packed, prepass=prepass, overlap=overlap,
                                             strength=strength, tile_size=tile_size,
                                             tile_batch_size=tile_batch_size)
        original = torch.from_numpy(np.array(packed, copy=True))[None].to(self.device)
        if strength == 0:
            return packed.copy()
        image = correct_defects(original, k=8, abs_floor=.01, adjacency_ratio=.1)[0] if prepass else original
        output = tiled_inference(self._call, image, tile_size=tile_size, overlap=overlap,
                                 feather_power=2.2, tile_batch_size=tile_batch_size)
        output = strength*output + (1-strength)*original
        result = output[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Model produced nonfinite output")
        return result

    @torch.inference_mode()
    def denoise_rgb(self, image, *, color_contract, overlap=60, strength=1., tile_size=512):
        """Denoise encoded RGB float32 [3,H,W], preserving its color contract.

        No Bayer packing, sensor defect prepass, transfer curve or ICC transform.
        uint16 normalization and output rounding are provided by rgb.py.
        """
        if self.image_kind != "rgb":
            raise ValueError("RGB input requires a matching rgb_nafnet checkpoint; Bayer weights cannot process RGB")
        if validate_rgb_contract(color_contract) != self.rgb_contract:
            raise ValueError("RGB input color-space/ICC contract does not match checkpoint; color transforms are unsupported")
        image = np.asarray(image)
        if image.dtype != np.float32 or image.ndim != 3 or image.shape[0] != 3 or min(image.shape) < 1:
            raise ValueError("Expected nonempty float32 RGB input [3,H,W]")
        if not np.isfinite(image).all():
            raise ValueError("RGB input must be finite")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if not 0 <= overlap < tile_size or tile_size % 16:
            raise ValueError("overlap must be below tile_size; tile_size must be a multiple of 16")
        if strength == 0:
            return image.copy()
        original = torch.from_numpy(np.array(image, copy=True))[None].to(self.device)
        output = tiled_inference(self._call, original, tile_size=tile_size,
                                 overlap=overlap, feather_power=2.2)
        result = (strength*output + (1-strength)*original)[0].cpu().numpy()
        if result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("RGB model must produce finite float32 output")
        return result
