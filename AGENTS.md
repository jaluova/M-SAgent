# Repository Guardrails

This repository's authoritative Python baseline is Python 3.10.

- Run checks and tests through `scripts/check_py310.sh` so the repo uses `python3.10` instead of the caller's default `python`.
- Do not introduce Python 3.11+ standard-library features or syntax unless a Python 3.10 compatibility layer lands first.
- A newer local interpreter may exist for editing, but it does not change the repository baseline or CI contract.
