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

For the captured development/export environment, rebuild from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install -c constraints/macos-arm64-py312.txt -e '.[dev,export,audit]'
PYTHONPATH=$PWD/src .venv/bin/python -m pytest -q -rs tests/
```

The constraints were captured on macOS arm64 with Python 3.12.14 using
`uv pip freeze --python .venv/bin/python`; the editable project entry and any
local paths are omitted. They pin the installed packages, including export and
lint tooling, without changing the supported dependency ranges. Core ML export
is only tested with these versions; palettized compression modes additionally
need scikit-learn (part of the `export` extra).
This is a platform snapshot rather than a cross-platform lock: other platforms
may need additional dependencies, and Python 3.11 compatibility checks resolve
their own compatible versions. Do not apply these Python 3.12 pins to 3.11.
After an intentional environment update, recapture the constraints, remove
editable/local entries, and rerun export and parity checks before claiming
support. No clean reinstall is implied by the captured-version tests.

CI runs locally via `./scripts/run_ci.sh` (lint with `ruff`, CLI smoke, and the
full CPU/ONNX/Core ML `pytest` suite with `-rs`; pass `--fast` to skip slow
export/parity tests). `.github/workflows/tests.yml` is manual
`workflow_dispatch`-only so routine pushes do not trigger cloud runners. When
`coremltools` is installed locally, `scripts/run_ci.sh` sets
`REQUIRE_COREMLTOOLS=1` so missing or broken Core ML imports fail collection.

Record each training file's source, applicable usage terms and split assignment. Verify permissions for the exact dataset version and intended use. Keep related captures together across splits and keep evaluation inputs separate from training. Do not commit private data, weights, generated outputs or credentials. The three approved demo DNGs in `examples/raw/` are inference examples, never training data.

Training starts from random initialisation. Do not initialise, fine-tune or distil from third-party weights. Do not use third-party denoised outputs as training targets, pseudo-labels or model-selection signals. Training targets and selection evidence require documented independent provenance. See [data](docs/03-data.md) and [evaluation](docs/05-evaluation.md) guidance.

Describe the problem, change and checks in contributions. No license has been chosen yet; consult the author before assuming reuse permissions.
