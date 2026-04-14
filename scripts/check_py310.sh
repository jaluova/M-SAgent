#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3.10"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} is required for this repository. Install Python 3.10 and rerun scripts/check_py310.sh." >&2
  exit 1
fi

cd "${ROOT_DIR}"

ACTUAL_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(".".join(str(part) for part in sys.version_info[:3]))')"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)'; then
  echo "error: scripts/check_py310.sh requires a real Python 3.10 interpreter, but ${PYTHON_BIN} resolves to Python ${ACTUAL_VERSION}." >&2
  exit 1
fi

echo "Using Python ${ACTUAL_VERSION}"

if command -v ruff >/dev/null 2>&1; then
  echo "Running ruff syntax guard (target py310)"
  ruff check --select E9 src tests
else
  echo "warning: ruff is not installed; skipping ruff check and continuing with Python 3.10 compile/test guards." >&2
fi

echo "Running repository-wide Python 3.10 compatibility guardrails"
"${PYTHON_BIN}" scripts/py310_guard.py

echo "Running Python 3.10 unittest smoke"
"${PYTHON_BIN}" -m unittest discover -s tests -p "test_minimal_vertical_slice.py"
