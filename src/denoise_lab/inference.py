"""Portable, unconditioned packed Bayer inference; optional runtimes load lazily."""
from pathlib import Path
import os
import sys
import warnings

import numpy as np
from .tile_geometry import shift_offsets
from .checkpoints import load_checkpoint, model_config, resolve_weights


COREML_COMPUTE_UNITS = ("all", "cpu_and_ne", "cpu_and_gpu", "cpu_only")
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 60
DEFAULT_GRID_ALIGNMENT = 1
DEFAULT_FEATHER_MODE = "linear"
DEFAULT_TTA_SHIFT = 1
DEFAULT_FEATHER_POWER = 2.2


def coreml_compute_units(device="auto", compute_units=None):
    """Resolve runtime selection without importing the optional Core ML runtime."""
    if device not in {"auto", "cpu"}:
        raise ValueError("Core ML device must be auto or cpu")
    if compute_units is not None and compute_units not in COREML_COMPUTE_UNITS:
        raise ValueError("compute_units must be one of: " + ", ".join(COREML_COMPUTE_UNITS))
    if device == "cpu" and compute_units not in {None, "cpu_only"}:
        raise ValueError("Core ML device cpu conflicts with compute_units other than cpu_only")
    return compute_units or ("cpu_only" if device == "cpu" else "all")


def build_model(config):
    """Load the torch model registry lazily (also retains the existing hook)."""
    from .arch_registry import build_model as build
    return build(config)


def protect_highlights(output, original):
    """Protect normalized Bx4xHxW Bayer data (each channel's white is 1).

    Any photosite >= .98 seeds a shared Bayer-cell mask, preserving colour
    ratios rather than blending each plane differently. Preserve the original
    within one packed pixel, then feather over three more (eight sensor pixels
    in total). Only that footprint is capped at white; originally unclipped
    sites stay below white. Negative values and the rest of the image survive.
    This deliberately retains input noise near clipping; it cannot recover
    clipped colour. Use the pre-pass-corrected input as original so protection
    does not restore known saturated defects. With no pre-pass, use raw input.
    """
    import torch
    from torch.nn import functional as F

    mask = (original >= .98).any(dim=1, keepdim=True).to(output.dtype)
    if not mask.any():
        return output
    weight = torch.zeros_like(mask)
    for _ in range(4):
        mask = F.max_pool2d(mask, 3, stride=1, padding=1)
        weight += .25*mask
    blended = weight*original + (1-weight)*output
    # Keep the original exactly in the core, including near-white photosites.
    blended = torch.where(weight == 1, original, blended)
    ceiling = torch.where(original < 1, 1.-2**-24, 1.)
    return torch.where(weight > 0, torch.minimum(blended, ceiling), output)


class Denoiser:
    backend = "torch"

    @classmethod
    def load(cls, path, backend="auto", device="auto", providers=None, weights="model", compute_units=None):
        """Load model or EMA weights. ONNX/Core ML artifacts select weights at export.

        compute_units is Core ML only; None selects all, or cpu_only for device=cpu.
        An explicit accelerator selection conflicts with device=cpu.
        """
        path = Path(path)
        inferred = {".pt": "torch", ".onnx": "onnx", ".mlpackage": "coreml"}
        backend = inferred.get(path.suffix.lower()) if backend == "auto" else backend
        if backend not in {"torch", "onnx", "coreml"}:
            raise ValueError("Expected a .pt, .onnx or .mlpackage model, or an explicit backend")
        if weights not in {"model", "ema"}:
            raise ValueError("weights must be model or ema")
        if backend != "torch" and weights != "model":
            raise ValueError("Select EMA weights when exporting ONNX/Core ML")
        if backend != "coreml" and compute_units is not None:
            raise ValueError("compute_units is only supported for the Core ML backend")
        selected_units = coreml_compute_units(device, compute_units) if backend == "coreml" else None
        self = cls()
        self.compute_units = selected_units
        self.backend = backend
        self.device = "cpu"
        self.providers = []
        if backend == "torch":
            try:
                import torch
            except ImportError as exc:
                raise ImportError("PyTorch is required for .pt models; install 'denoise_lab[torch]' "
                                  "or use an ONNX/Core ML model") from exc
            saved = load_checkpoint(path)
            config = model_config(saved)
            if config.get("cond", "none") != "none":
                raise ValueError("Only unconditioned (cond=none) models are supported")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu")
            self.device = str(device)
            state = resolve_weights(saved, weights)
            try:
                model = build_model(config)
                model.load_state_dict(state)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise ValueError("Invalid checkpoint: weights or model configuration are incompatible") from exc
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
            options.intra_op_num_threads = min(os.cpu_count() or 1, 8)
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
                output = session.run(None, {inputs[0].name: np.ascontiguousarray(tile, dtype=dtype)})[0]
                return output.astype(np.float32, copy=False)
            self._call = run
        else:
            if sys.platform != "darwin":
                raise RuntimeError("Core ML .mlpackage inference requires macOS; use .pt or .onnx on this platform")
            import coremltools as ct
            units = getattr(ct.ComputeUnit, selected_units.upper())
            model = ct.models.MLModel(str(path), compute_units=units)
            spec = model.get_spec()
            inputs = spec.description.input
            if len(inputs) != 1 or list(inputs[0].type.multiArrayType.shape) != [1, 4, 512, 512]:
                raise ValueError("Only unconditioned Core ML 1x4x512x512 models are supported")
            input_name, output_name = inputs[0].name, spec.description.output[0].name
            self.providers = [f"CoreML ({units.name})"]
            def run(tile):
                output = model.predict({input_name: np.ascontiguousarray(tile)})[output_name]
                return np.asarray(output, dtype=np.float32)
            self._call = run
        return self

    @property
    def description(self):
        return f"{self.backend}: {', '.join(self.providers) if self.providers else self.device}"

    def denoise_packed(self, packed, *, prepass=True, defect_mask=None, overlap=DEFAULT_OVERLAP,
                       strength=1., tile_size=DEFAULT_TILE_SIZE, highlights="protect",
                       grid_alignment=DEFAULT_GRID_ALIGNMENT, feather_mode=DEFAULT_FEATHER_MODE,
                       tta_shift=DEFAULT_TTA_SHIFT,
                       chroma_strength=None, white_balance=None):
        """Denoise float32 R/G1/B/G2 with per-channel white normalized to 1.

        Strength blends against the original input; highlight protection is
        applied after that blend and tiling, identically for every backend.
        highlights='off' preserves the legacy processing path exactly.
        tta_shift>1 averages N diagonal tile grids (about N model passes); it
        reduces phase variance without guaranteeing invariance.
        chroma_strength (default = strength) blends chroma separately per Bayer
        cell in WB-scaled opponent space Y=(R+2G+B)/4, Cb=B-Y, Cr=R-Y; G1-G2
        follows luma strength. white_balance is positive RGB gains (unity default).
        """
        packed = np.asarray(packed)
        if packed.dtype != np.float32 or packed.ndim != 3 or packed.shape[0] != 4 or min(packed.shape) < 1:
            raise ValueError("Expected nonempty float32 packed input [4,H,W]")
        if not np.isfinite(packed).all():
            raise ValueError("Packed input must be finite")
        if not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        chroma_strength = strength if chroma_strength is None else chroma_strength
        if not 0 <= chroma_strength <= 1:
            raise ValueError("chroma_strength must be between 0 and 1")
        gains = np.asarray([1., 1., 1.] if white_balance is None else white_balance, dtype=np.float32)
        if gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("white_balance must contain three finite positive RGB gains")
        if not 0 <= overlap < tile_size or tile_size % 16:
            raise ValueError("overlap must be below tile_size; tile_size must be a multiple of 16")
        mode = ('auto' if prepass else 'off') if isinstance(prepass, bool) else prepass
        if mode not in {'auto', 'map', 'heuristic', 'map+heuristic', 'off'}:
            raise ValueError('Unknown prepass mode')
        if mode == 'auto':
            mode = 'map' if defect_mask is not None else 'heuristic'
        if mode in {'map', 'map+heuristic'} and defect_mask is None:
            raise ValueError('Selected prepass requires a defect map')
        if defect_mask is not None:
            defect_mask = np.asarray(defect_mask)
            if defect_mask.dtype != np.bool_ or defect_mask.shape != packed.shape:
                raise ValueError('Defect mask must be bool with the packed input shape')
        if highlights not in {"protect", "off"}:
            raise ValueError("highlights must be protect or off")
        offsets = shift_offsets(tile_size, overlap, grid_alignment, tta_shift)
        if feather_mode not in ("linear", "cosine"):
            raise ValueError("feather_mode must be linear or cosine")
        if strength == 0 and chroma_strength == 0:
            return packed.copy()
        if self.backend != "torch":
            from .np_pipeline import denoise_packed
            return denoise_packed(self._call, packed, mode=mode, defect_mask=defect_mask,
                                  overlap=overlap, strength=strength, tile_size=tile_size,
                                  highlights=highlights, offsets=offsets,
                                  grid_alignment=grid_alignment, feather_mode=feather_mode,
                                  chroma_strength=chroma_strength, gains=gains)
        import torch
        with torch.inference_mode():
            return self._denoise_torch(packed, mode, defect_mask, overlap, strength, tile_size,
                                       highlights, offsets, grid_alignment, feather_mode,
                                       chroma_strength, gains)

    def _denoise_torch(self, packed, mode, defect_mask, overlap, strength, tile_size,
                       highlights, offsets, grid_alignment, feather_mode, chroma_strength, gains):
        import torch
        from .defects import correct_defects, correct_mapped_defects
        from .tiling import tiled_inference

        original = torch.from_numpy(np.array(packed, copy=True))[None].to(self.device)
        image = original
        if mode in {'map', 'map+heuristic'}:
            mask = torch.from_numpy(np.array(defect_mask, copy=True))[None].to(self.device)
            image = correct_mapped_defects(image, mask)[0]
        if mode in {'heuristic', 'map+heuristic'}:
            image = correct_defects(image, k=8, abs_floor=.01, adjacency_ratio=.1)[0]
        output = None
        for offset in offsets:
            prediction = tiled_inference(self._call, image, tile_size=tile_size, overlap=overlap,
                                         feather_power=DEFAULT_FEATHER_POWER, grid_offset=offset, grid_alignment=grid_alignment,
                                         feather_mode=feather_mode)
            output = prediction if output is None else output + prediction
        if len(offsets) > 1:
            output /= len(offsets)
        if chroma_strength == strength:
            output = strength*output + (1-strength)*original
        else:
            wb = torch.as_tensor(gains[[0, 1, 2, 1]], device=self.device)[None, :, None, None]
            delta = (output-original)*wb
            r, g1, b, g2 = delta.unbind(1)
            green = (g1+g2)*.5
            y = (r+2*green+b)*.25
            cb, cr = b-y, r-y
            yy, cb, cr = strength*y, chroma_strength*cb, chroma_strength*cr
            g = yy-(cb+cr)*.5
            difference = strength*(g1-g2)*.5
            correction = torch.stack((yy+cr, g+difference, yy+cb, g-difference), dim=1)
            output = original + correction/wb
        if not torch.isfinite(output).all():
            raise ValueError("Model produced nonfinite output")
        if highlights == "protect":
            output = protect_highlights(output, image)
        return output[0].cpu().numpy()
