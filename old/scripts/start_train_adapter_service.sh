#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${M_SAGENT_DEPLOYMENT_PROFILE:-single_gpu_stable}"

if [[ -n "${TRAIN_ADAPTER_ROOT:-}" ]]; then
  TRAIN_ADAPTER_ROOT="$(cd "${TRAIN_ADAPTER_ROOT}" && pwd)"
elif [[ -d "${PROJECT_ROOT}/../TrainAdapter" ]]; then
  TRAIN_ADAPTER_ROOT="$(cd "${PROJECT_ROOT}/../TrainAdapter" && pwd)"
else
  TRAIN_ADAPTER_ROOT="/root/autodl-tmp/TrainAdapter"
fi

TRAIN_ADAPTER_SERVICE_DEVICE="${TRAIN_ADAPTER_SERVICE_DEVICE:-${TRAIN_ADAPTER_EXPECTED_DEVICE:-cpu}}"
TRAIN_ADAPTER_SERVICE_PYTHON="${TRAIN_ADAPTER_SERVICE_PYTHON:-/root/autodl-tmp/conda-envs/adapter/bin/python}"
TRAIN_ADAPTER_QWEN_MODEL_PATH="${TRAIN_ADAPTER_QWEN_MODEL_PATH:-/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct}"
TRAIN_ADAPTER_ADAPTER_PATH="${TRAIN_ADAPTER_ADAPTER_PATH:-/root/autodl-tmp/Data/train_outputs/textguided_recover_20260405_shortval/checkpoints/best_model.pth}"
TRAIN_ADAPTER_CONFIG_PATH="${TRAIN_ADAPTER_CONFIG_PATH:-/root/autodl-tmp/Data/train_outputs/textguided_recover_20260405_shortval/config.json}"
TRAIN_ADAPTER_HOST="${TRAIN_ADAPTER_HOST:-127.0.0.1}"
TRAIN_ADAPTER_PORT="${TRAIN_ADAPTER_PORT:-8765}"
TRAIN_ADAPTER_LOG_PATH="${TRAIN_ADAPTER_LOG_PATH:-/root/autodl-tmp/inference_service_22605.log}"
TRAIN_ADAPTER_FOREGROUND="${TRAIN_ADAPTER_FOREGROUND:-0}"

if [[ ! -x "${TRAIN_ADAPTER_SERVICE_PYTHON}" ]]; then
  echo "TrainAdapter python not found: ${TRAIN_ADAPTER_SERVICE_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${TRAIN_ADAPTER_ROOT}" ]]; then
  echo "TrainAdapter root not found: ${TRAIN_ADAPTER_ROOT}" >&2
  exit 1
fi

if [[ "${PROFILE}" == "single_gpu_stable" && "${TRAIN_ADAPTER_SERVICE_DEVICE}" == "cuda" ]]; then
  echo "Refusing to start TrainAdapter on GPU in single_gpu_stable profile" >&2
  exit 1
fi

CMD=(
  "${TRAIN_ADAPTER_SERVICE_PYTHON}"
  -u
  "${TRAIN_ADAPTER_ROOT}/src/inference_service.py"
  --adapter_path "${TRAIN_ADAPTER_ADAPTER_PATH}"
  --config "${TRAIN_ADAPTER_CONFIG_PATH}"
  --qwen_model_path "${TRAIN_ADAPTER_QWEN_MODEL_PATH}"
  --device "${TRAIN_ADAPTER_SERVICE_DEVICE}"
  --host "${TRAIN_ADAPTER_HOST}"
  --port "${TRAIN_ADAPTER_PORT}"
)

echo "Starting TrainAdapter service"
echo "  profile: ${PROFILE}"
echo "  device: ${TRAIN_ADAPTER_SERVICE_DEVICE}"
echo "  host: ${TRAIN_ADAPTER_HOST}:${TRAIN_ADAPTER_PORT}"
echo "  log_path: ${TRAIN_ADAPTER_LOG_PATH}"

if [[ "${TRAIN_ADAPTER_FOREGROUND}" == "1" ]]; then
  exec "${CMD[@]}"
fi

mkdir -p "$(dirname "${TRAIN_ADAPTER_LOG_PATH}")"
nohup "${CMD[@]}" > "${TRAIN_ADAPTER_LOG_PATH}" 2>&1 &
PID=$!
sleep 2

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "TrainAdapter service exited during startup" >&2
  tail -n 40 "${TRAIN_ADAPTER_LOG_PATH}" >&2 || true
  exit 1
fi

echo "TrainAdapter service started with PID ${PID}"
