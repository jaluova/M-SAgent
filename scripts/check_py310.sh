#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3.10"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} is required for this repository. Install Python 3.10 and rerun scripts/check_py310.sh." >&2
  exit 1
fi

cd "${ROOT_DIR}"

echo "Using $(${PYTHON_BIN} --version 2>&1)"

if command -v ruff >/dev/null 2>&1; then
  echo "Running ruff syntax guard (target py310)"
  ruff check --select E9 src tests
else
  echo "warning: ruff is not installed; skipping ruff check and continuing with Python 3.10 compile/test guards." >&2
fi

echo "Running Python 3.10 compile smoke check"
mapfile -t PYTHON_FILES < <(find src tests -type f -name "*.py" | sort)
"${PYTHON_BIN}" -m py_compile "${PYTHON_FILES[@]}"

echo "Running Python 3.10 unittest smoke"
"${PYTHON_BIN}" -m unittest discover -s tests -p "test_minimal_vertical_slice.py"
