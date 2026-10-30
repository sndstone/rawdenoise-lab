#!/usr/bin/env python3
"""Run project Python inside the prepared Linux CUDA container, offline.

Example: python3 scripts/run_cuda.py scripts/train.py --config configs/smoke.yaml --device cuda:0
Only this project directory is mounted. The NAS is never mounted.
"""
import os
from pathlib import Path
import subprocess
import sys

IMAGE = "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel"


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        return
    root = Path(__file__).resolve().parents[1]
    if not (root / ".venv/pyvenv.cfg").is_file():
        raise SystemExit("The container project environment must be prepared first.")
    command = ["docker", "run", "--rm", "--pull", "never", "--network", "none",
               "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size", "1g",
               "-e", "HOME=/workspace", "-v", f"{root}:/workspace", "-w", "/workspace",
               "--entrypoint", "/workspace/.venv/bin/python", IMAGE, *sys.argv[1:]]
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
