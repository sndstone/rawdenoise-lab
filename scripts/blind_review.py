#!/usr/bin/env python3
"""Prepare offline blind RAW comparisons or score exported browser votes."""
import argparse
import json
from pathlib import Path

import torch

from denoise_lab.blind_review import prepare, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('inputs', nargs='*', type=Path)
    p.add_argument('--manifest', type=Path, help='JSON list of paths or objects with path, iso, camera, scene, split; paths relative to manifest')
    p.add_argument('--arm', action='append', nargs=3, required=True,
                   metavar=('LABEL', 'MODEL', 'STRENGTH'), help='Repeat for 2+ arms; strength 0 is identity')
    p.add_argument('--output', type=Path, required=True, help='New review directory')
    p.add_argument('--crops', type=int, default=5)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exposure-ev', type=float, default=0.)
    p.add_argument('--device', default='cpu')
    p.add_argument('--no-prepass', action='store_true')
    p.add_argument('--overlap', type=int, default=60)
    p = commands.add_parser('score')
    p.add_argument('--votes', type=Path, required=True)
    p.add_argument('--key', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('review_results.md'))
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if args.command == 'prepare':
        sources = args.inputs
        if args.manifest:
            for entry in json.loads(args.manifest.read_text()):
                if isinstance(entry, str):
                    sources.append(args.manifest.parent/entry)
                else:
                    sources.append(dict(entry, path=str(args.manifest.parent/entry['path'])))
        arms = [dict(label=label, model=model, strength=float(strength)) for label, model, strength in args.arm]
        prepare(sources, arms, args.output, count=args.crops, size=args.crop_size, seed=args.seed,
                exposure_ev=args.exposure_ev, device=args.device, prepass=not args.no_prepass,
                overlap=args.overlap)
        print(args.output/'index.html')
    else:
        score(json.loads(args.key.read_text()), json.loads(args.votes.read_text()), args.output,
              bootstrap=args.bootstrap, seed=args.seed)
        print(args.output)


if __name__ == '__main__':
    main()
