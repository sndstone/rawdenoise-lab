#!/usr/bin/env python3
"""Compare NumPy and torch preprocessing/tiling using the SAME CPU ONNX model.

Reads only the requested RAWs (the three approved examples by default). Outputs
JSON under runs/torchfree; this is runtime parity, not a model-quality assessment.
Requires torch for the reference path, not for deployment.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rawpy
import torch

from denoise_lab.bayer import normalize_raw, pack
from denoise_lab.defects import correct_defects
from denoise_lab.inference import Denoiser
from denoise_lab.np_pipeline import correct_defects as numpy_correct_defects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/torchfree/parity.json'))
    parser.add_argument('inputs', type=Path, nargs='*')
    args = parser.parse_args()
    inputs = args.inputs or sorted(Path('examples/raw').glob('*.dng'))
    torch.set_num_threads(8)
    numpy_model = Denoiser.load(args.model, device='cpu')
    reference = Denoiser()
    reference.device = 'cpu'
    reference._call = lambda tile: torch.from_numpy(numpy_model._call(tile.numpy()))
    results = []
    for source in inputs:
        with rawpy.imread(str(source)) as raw:
            packed = pack(normalize_raw(raw))
        expected, flags = correct_defects(torch.from_numpy(packed)[None], adjacency_ratio=.1)
        actual, found = numpy_correct_defects(packed[None])
        if not np.array_equal(found, flags.numpy()) or not np.array_equal(actual, expected.numpy()):
            raise AssertionError(f'Prepass mismatch: {source}')
        # Warm the shared session before timing each complete path once.
        numpy_model._call(np.zeros((1, 4, 512, 512), np.float32))
        start = time.perf_counter()
        actual = numpy_model.denoise_packed(packed)
        numpy_seconds = time.perf_counter()-start
        start = time.perf_counter()
        expected = reference.denoise_packed(packed)
        torch_seconds = time.perf_counter()-start
        diff = float(np.max(np.abs(actual-expected)))
        result = dict(input=str(source), packed_shape=list(packed.shape), max_abs_diff=diff,
                      numpy_seconds=numpy_seconds, torch_seconds=torch_seconds,
                      prepass_equal=True, corrected_sites=int(found.sum()))
        print(json.dumps(result), flush=True)
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model=str(args.model), providers=numpy_model.providers,
                                          threads=8, tile_size=512, overlap=60, strength=1,
                                          highlights='protect', results=results), indent=2)+'\n')
    if not results or any(r['max_abs_diff'] > 1e-5 for r in results):
        raise AssertionError('NumPy/torch pipeline parity exceeds 1e-5')


if __name__ == '__main__':
    main()
