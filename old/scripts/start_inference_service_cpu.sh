#!/usr/bin/env bash
set -euo pipefail

export TRAIN_ADAPTER_SERVICE_DEVICE="cpu"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_train_adapter_service.sh"
