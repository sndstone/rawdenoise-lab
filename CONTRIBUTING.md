# Contributing

Use Python 3.11 or newer and a virtual environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,onnx]' onnx
rawdenoise --help
pytest -q tests/
python scripts/train.py --config configs/smoke.yaml --device cpu
```

The tests run on CPU with generated fixtures; Core ML checks may skip when its optional runtime is unavailable. Smoke training verifies plumbing, not denoising quality. Inference changes should preserve Bayer layout, normalization, output metadata and overwrite protection. Keep experimental training claims scoped to the data and checks actually used.

Record each training file's source, applicable usage terms and split assignment. Verify permissions for the exact dataset version and intended use. Keep related captures together across splits and keep evaluation inputs separate from training. Do not commit private data, weights, generated outputs or credentials. The three approved demo DNGs in `examples/raw/` are inference examples, never training data.

Training starts from random initialisation. Do not initialise, fine-tune or distil from third-party weights. Do not use third-party denoised outputs as training targets, pseudo-labels or model-selection signals. Training targets and selection evidence require documented independent provenance. See [data](docs/03-data.md) and [evaluation](docs/05-evaluation.md) guidance.

Describe the problem, change and checks in contributions. No license has been chosen yet; consult the author before assuming reuse permissions.
