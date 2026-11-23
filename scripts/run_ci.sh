#!/usr/bin/env bash
# Local CI runner: lint, optional private leak guard, CLI smoke, and full pytest suite.
# Usage: scripts/run_ci.sh [--fast] [--skip-leak-check]
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

FAST=0
SKIP_LEAK=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --skip-leak-check) SKIP_LEAK=1 ;;
    -h|--help)
      echo "Usage: scripts/run_ci.sh [--fast] [--skip-leak-check]"
      echo "  --fast             Skip slow Core ML export & ONNX parity tests"
      echo "  --skip-leak-check  Skip tools/leak_check.py (private repo only)"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Auto-detect project virtual environment (including from git worktrees).
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  COMMON_DIR="$(git rev-parse --git-common-dir 2>/dev/null || true)"
  COMMON_PARENT="$(cd "$(dirname "$COMMON_DIR")" 2>/dev/null && pwd || true)"
  if [ -n "$COMMON_PARENT" ] && [ -x "$COMMON_PARENT/.venv/bin/python" ]; then
    PY="$COMMON_PARENT/.venv/bin/python"
  else
    PY="python3"
  fi
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "==> Local CI using: $PY"

# 1. Private export leak check (only present in the private working repository).
if [ "$SKIP_LEAK" != 1 ] && [ -f "$ROOT/tools/leak_check.py" ]; then
  echo "==> [1/4] Export leak check (committed HEAD)"
  "$PY" "$ROOT/tools/leak_check.py"
else
  echo "==> [1/4] Export leak check skipped"
fi

# 2. Lint (ruff preferred; offline fallback if ruff is absent).
echo "==> [2/4] Lint"
if "$PY" -m ruff --version >/dev/null 2>&1; then
  "$PY" -m ruff check .
elif command -v ruff >/dev/null 2>&1; then
  ruff check .
elif [ -f "$ROOT/tools/lint_basic.py" ]; then
  "$PY" "$ROOT/tools/lint_basic.py"
else
  "$PY" -m compileall -q src scripts tests
fi

# 3. CLI smoke check.
echo "==> [3/4] CLI smoke"
"$PY" -m denoise_lab.cli --version
"$PY" -m denoise_lab.cli --help >/dev/null

# 4. Pytest suite.
echo "==> [4/4] Pytest"
if [ "$FAST" = 1 ]; then
  "$PY" -m pytest -q -rs tests/ \
    --deselect=tests/test_inference.py::test_torch_onnx_parity \
    --ignore=tests/test_export_coreml.py
else
  if "$PY" -c 'import coremltools' 2>/dev/null; then
    export REQUIRE_COREMLTOOLS=1
  fi
  "$PY" -m pytest -q -rs tests/
fi

echo "==> Local CI passed."
