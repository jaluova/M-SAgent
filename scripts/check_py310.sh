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

echo "Checking Python 3.10 API test dependencies"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

required = ("fastapi", "httpx")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    joined = ", ".join(missing)
    print(
        "error: missing API test dependencies for scripts/check_py310.sh: "
        f"{joined}. Install them with "
        "\"python3.10 -m pip install -e '.[api-test]'\" and rerun.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

echo "Running Python 3.10 API entry smoke"
"${PYTHON_BIN}" -m unittest \
  tests.test_api_service \
  tests.test_api_assembly \
  tests.test_api_server_entry \
  tests.test_api_fastapi_integration
