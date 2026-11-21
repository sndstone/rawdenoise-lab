#!/usr/bin/env python3
"""Offline r4 tile-phase diagnostic; all generated artifacts stay under output.

Real images are unpaired owner examples, not accuracy references. No training,
clipping or defect prepass is applied. Cached predictions permit interrupted runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rawpy
import torch

from denoise_lab.bayer import normalize_raw, pack
from denoise_lab.inference import Denoiser
from denoise_lab.metrics import seam_energy
from denoise_lab.tiling import shift_offsets, tile_starts, tiled_inference


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', default=sorted(Path('examples/raw').glob('*.dng')))
    parser.add_argument('--output', type=Path, default=Path('runs/tile_seams'))
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--runtime-repeats', type=int, default=3,
                        help='Rotated-order timing repeats on the noisy flat, 0 disables')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    denoiser = Denoiser.load(args.checkpoint, device=args.device)
    checkpoint_hash = digest(args.checkpoint)
    modes = dict(legacy=dict(bayer_reflection=False), corrected=dict(),
        aligned=dict(grid_alignment=16), cosine=dict(grid_alignment=16, feather_mode='cosine'),
        wide=dict(grid_alignment=16, overlap=128))
    records, timing, stride_records, ensemble_records, sources = [], [], [], [], []

    def sync():
        if args.device == 'mps':
            torch.mps.synchronize()

    # Warm up model kernels once; excluded from inference timing.
    with torch.inference_mode():
        denoiser._call(torch.zeros(1, 4, 512, 512, device=args.device))
    sync()
    for path in [*args.inputs, None]:
        if path is None:
            name = 'flat_noise'
            generator = np.random.default_rng(1401)
            packed = np.full((4, 512, 512), .18, dtype=np.float32)
            packed += generator.normal(0, .01, packed.shape).astype(np.float32)
            source = dict(name=name, seed=1401, level=.18, gaussian_std=.01)
        else:
            name = path.stem
            with rawpy.imread(str(path)) as raw:
                packed = pack(normalize_raw(raw))
            source = dict(name=name, path=str(path), sha256=digest(path))
        image = torch.from_numpy(packed)[None].to(args.device)
        source['packed_shape'] = list(image.shape)
        sources.append(source)
        cache = {}

        def predict(mode, phase, **overrides):
            kwargs = dict(tile_size=512, overlap=60, feather_power=2.2)
            kwargs.update(modes[mode])
            kwargs.update(overrides)
            kwargs['grid_offset'] = phase
            # Include every setting and source/checkpoint in the cache identity.
            identity = dict(kwargs=kwargs, source=source, checkpoint=checkpoint_hash,
                            device=args.device, threads=args.threads, version=1)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            if key in cache:
                return cache[key]
            npy = args.output/f'{name}_{key}.npy'
            meta = npy.with_suffix('.json')
            if npy.exists() and meta.exists():
                output = torch.from_numpy(np.load(npy, allow_pickle=False))
                elapsed = json.loads(meta.read_text())['seconds']
            else:
                sync()
                start = time.monotonic()
                output = tiled_inference(denoiser._call, image, **kwargs)
                sync()
                elapsed = time.monotonic()-start
                output = output.cpu()
                np.save(npy, output.numpy(), allow_pickle=False)
                meta.write_text(json.dumps(dict(identity, seconds=elapsed), indent=2)+'\n')
            count = np.prod([len(tile_starts(n, kwargs['tile_size'], kwargs['overlap'], phase,
                kwargs.get('grid_alignment', 1))) for n in image.shape[-2:]])
            timing.append(dict(image=name, mode=mode, phase=phase, seconds=elapsed, tiles=int(count),
                               settings=kwargs))
            print(f'{name} {mode} phase={phase}: {elapsed:.3f}s / {count} tiles', flush=True)
            cache[key] = (output, elapsed)
            return output, elapsed

        for mode, options in modes.items():
            baseline, _ = predict(mode, 0)
            phases = [64, 128, 192 if options.get('grid_alignment') == 16 else 200]
            for phase in phases:
                shifted, _ = predict(mode, phase)
                metrics = seam_energy(baseline-shifted, overlap=options.get('overlap', 60),
                    grid_offsets=(0, phase), grid_alignment=options.get('grid_alignment', 1))
                records.append(dict(image=name, mode=mode, phase=phase, **metrics))
        # Isolate one lattice: step 448, apron 64, keeping every origin's
        # residue fixed within each run; only the global origin residue varies.
        aligned, _ = predict('aligned', 0)
        for phase in (1, 8, 15, 16):
            shifted, _ = predict('aligned', phase, overlap=64, grid_alignment=1)
            stride_records.append(dict(image=name, phase=phase, **seam_energy(aligned-shifted,
                overlap=64, grid_offsets=(0, phase))))
        # The same evenly spaced two-grid ensemble used by --tta-shift 2.
        # Compare ensemble origin 0 to origin 64, retaining the 16-pixel lattice.
        ensembles, times = [], []
        all_offsets = []
        for origin in (0, 64):
            predictions = [predict('aligned', origin+phase)
                           for phase in shift_offsets(512, 60, 16, 2)]
            ensembles.append(sum(pred[0] for pred in predictions)/2)
            times.append(sum(pred[1] for pred in predictions))
            all_offsets.extend(origin+phase for phase in shift_offsets(512, 60, 16, 2))
        ensemble_records.append(dict(image=name, seconds=times, offsets=all_offsets,
            **seam_energy(ensembles[0]-ensembles[1], overlap=60,
                          grid_offsets=all_offsets, grid_alignment=16)))
        result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
            device=args.device, threads=args.threads, torch_version=torch.__version__,
            sources=sources, comparisons=records, timings=timing, stride=stride_records,
            tta_shift_2=ensemble_records)
        (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    # Repeat timing in rotated order after measurements. Cached predictions are
    # never used here. Use the public API, including shift averaging overhead.
    runtime = []
    arms = [*modes, 'tta2']
    for repeat in range(args.runtime_repeats):
        for mode in arms[repeat % len(arms):]+arms[:repeat % len(arms)]:
            sync()
            start = time.monotonic()
            if mode == 'legacy':
                tiled_inference(denoiser._call, image, overlap=60, feather_power=2.2,
                                bayer_reflection=False)
            else:
                options = dict(grid_alignment=16, tta_shift=2) if mode == 'tta2' else modes[mode]
                denoiser.denoise_packed(packed, prepass=False, **options)
            sync()
            seconds = time.monotonic()-start
            runtime.append(dict(mode=mode, repeat=repeat, seconds=seconds))
            print(f'runtime {mode} repeat={repeat}: {seconds:.3f}s', flush=True)
    result['runtime_repeats_flat'] = runtime
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved {args.output}/results.json', flush=True)


if __name__ == '__main__':
    main()
