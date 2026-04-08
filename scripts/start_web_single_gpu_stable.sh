#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="single_gpu_stable"

if [[ -f /etc/network_turbo && "${M_SAGENT_ENABLE_NETWORK_TURBO:-0}" == "1" ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

export M_SAGENT_DEPLOYMENT_PROFILE="${M_SAGENT_DEPLOYMENT_PROFILE:-${PROFILE}}"
export M_SAGENT_BASE_DIR="${M_SAGENT_BASE_DIR:-${PROJECT_ROOT}}"
export M_SAGENT_QWEN_MODEL_PATH="${M_SAGENT_QWEN_MODEL_PATH:-/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct}"
export M_SAGENT_SAM3_CHECKPOINT_PATH="${M_SAGENT_SAM3_CHECKPOINT_PATH:-/root/autodl-tmp/modelscope_cache_sam3/facebook/sam3/sam3.pt}"
export GRIDGROUND_BACKEND="${GRIDGROUND_BACKEND:-embedded}"
export GRIDGROUND_MODEL_ID="${GRIDGROUND_MODEL_ID:-alpharho/GridGround-TextGuided}"
export GRIDGROUND_MODEL_DIR="${GRIDGROUND_MODEL_DIR:-/root/autodl-tmp/modelscope_cache_gridground/${GRIDGROUND_MODEL_ID}}"
export M_SAGENT_SYSTEM_PROMPT="${M_SAGENT_SYSTEM_PROMPT:-${PROJECT_ROOT}/prompts/system_prompt_en.txt}"
export TRAIN_ADAPTER_ENABLED="${TRAIN_ADAPTER_ENABLED:-false}"
export TRAIN_ADAPTER_URL="${TRAIN_ADAPTER_URL:-http://127.0.0.1:8765}"
export TRAIN_ADAPTER_EXPECTED_DEVICE="${TRAIN_ADAPTER_EXPECTED_DEVICE:-cpu}"
export TRAIN_ADAPTER_TIMEOUT="${TRAIN_ADAPTER_TIMEOUT:-120}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export M_SAGENT_MLLM_MIN_PIXELS="${M_SAGENT_MLLM_MIN_PIXELS:-786432}"
export M_SAGENT_MLLM_MAX_PIXELS="${M_SAGENT_MLLM_MAX_PIXELS:-1048576}"
export M_SAGENT_MIN_FREE_GPU_MB="${M_SAGENT_MIN_FREE_GPU_MB:-20000}"
export M_SAGENT_SERVER_HOST="${M_SAGENT_SERVER_HOST:-0.0.0.0}"
export M_SAGENT_SERVER_PORT="${M_SAGENT_SERVER_PORT:-8000}"
export M_SAGENT_SERVER_LOG_PATH="${M_SAGENT_SERVER_LOG_PATH:-${PROJECT_ROOT}/uvicorn.log}"
export M_SAGENT_SERVER_FOREGROUND="${M_SAGENT_SERVER_FOREGROUND:-0}"

M_SAGENT_PYTHON="${M_SAGENT_PYTHON:-/root/autodl-tmp/conda-envs/m_sagent/bin/python}"
M_SAGENT_SAM3_MODEL_PATH="${M_SAGENT_SAM3_MODEL_PATH:-/root/autodl-tmp/sam3}"
M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING="${M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING:-${M_SAGENT_SAM3_MODEL_PATH}/sam3/agent/system_prompts/system_prompt_iterative_checking.txt}"
export M_SAGENT_SAM3_MODEL_PATH
export M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING

if [[ ! -x "${M_SAGENT_PYTHON}" ]]; then
  echo "M-SAgent python not found: ${M_SAGENT_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${M_SAGENT_SAM3_MODEL_PATH}/sam3" ]]; then
  echo "SAM3 code dir not found under M_SAGENT_SAM3_MODEL_PATH: ${M_SAGENT_SAM3_MODEL_PATH}" >&2
  echo "Provide an external SAM3 checkout and export M_SAGENT_SAM3_MODEL_PATH before running." >&2
  exit 1
fi

if [[ ! -f "${M_SAGENT_SAM3_CHECKPOINT_PATH}" ]]; then
  echo "SAM3 checkpoint not found: ${M_SAGENT_SAM3_CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING}" ]]; then
  echo "SAM3 iterative checking prompt not found: ${M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING}" >&2
  exit 1
fi

if [[ "${GRIDGROUND_BACKEND}" != "embedded" ]]; then
  echo "Refusing to start ${PROFILE} web service with GRIDGROUND_BACKEND=${GRIDGROUND_BACKEND}" >&2
  echo "Use embedded GridGround so the server can reuse the already-loaded Qwen backbone." >&2
  exit 1
fi

if [[ ! -d "${GRIDGROUND_MODEL_DIR}" ]]; then
  echo "GridGround model dir not found: ${GRIDGROUND_MODEL_DIR}" >&2
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  HEALTH_JSON="$(curl -fsS --max-time 3 "${TRAIN_ADAPTER_URL}/health" 2>/dev/null || true)"
  if [[ -n "${HEALTH_JSON}" ]]; then
    SERVICE_DEVICE="$(
      printf '%s' "${HEALTH_JSON}" | python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("device", "")).lower())'
    )"
    if [[ "${SERVICE_DEVICE}" == "cuda" ]]; then
      echo "Refusing to start web service: TrainAdapter is already running on GPU at ${TRAIN_ADAPTER_URL}" >&2
      echo "That service loads a second Qwen model and will likely OOM together with the web backend." >&2
      exit 1
    fi
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_GPU_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ -n "${FREE_GPU_MB}" && "${FREE_GPU_MB}" -lt "${M_SAGENT_MIN_FREE_GPU_MB}" ]]; then
    echo "Refusing to start ${PROFILE}: free GPU memory ${FREE_GPU_MB} MB is below ${M_SAGENT_MIN_FREE_GPU_MB} MB" >&2
    exit 1
  fi
fi

CMD=(
  "${M_SAGENT_PYTHON}"
  -m
  uvicorn
  server.app:app
  --host "${M_SAGENT_SERVER_HOST}"
  --port "${M_SAGENT_SERVER_PORT}"
  --workers 1
)

echo "Starting M-SAgent web service (${PROFILE})"
echo "  host: ${M_SAGENT_SERVER_HOST}:${M_SAGENT_SERVER_PORT}"
echo "  gridground_backend: ${GRIDGROUND_BACKEND}"
echo "  train_adapter_enabled: ${TRAIN_ADAPTER_ENABLED}"
echo "  train_adapter_url: ${TRAIN_ADAPTER_URL}"
echo "  sam3_model_path: ${M_SAGENT_SAM3_MODEL_PATH}"
echo "  log_path: ${M_SAGENT_SERVER_LOG_PATH}"

if [[ "${M_SAGENT_SERVER_FOREGROUND}" == "1" ]]; then
  cd "${PROJECT_ROOT}"
  exec "${CMD[@]}"
fi

mkdir -p "$(dirname "${M_SAGENT_SERVER_LOG_PATH}")"
cd "${PROJECT_ROOT}"
nohup "${CMD[@]}" > "${M_SAGENT_SERVER_LOG_PATH}" 2>&1 &
PID=$!
sleep 3

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Web service exited during startup" >&2
  tail -n 80 "${M_SAGENT_SERVER_LOG_PATH}" >&2 || true
  exit 1
fi

echo "Web service started with PID ${PID}"
