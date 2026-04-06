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
export M_SAGENT_SAM3_MODEL_PATH="${M_SAGENT_SAM3_MODEL_PATH:-${PROJECT_ROOT}/sam3}"
export M_SAGENT_SAM3_CHECKPOINT_PATH="${M_SAGENT_SAM3_CHECKPOINT_PATH:-/root/autodl-tmp/modelscope_cache_sam3/facebook/sam3/sam3.pt}"
export GRIDGROUND_BACKEND="${GRIDGROUND_BACKEND:-embedded}"
export GRIDGROUND_MODEL_ID="${GRIDGROUND_MODEL_ID:-alpharho/GridGround-TextGuided}"
export GRIDGROUND_MODEL_DIR="${GRIDGROUND_MODEL_DIR:-/root/autodl-tmp/modelscope_cache_gridground/${GRIDGROUND_MODEL_ID}}"
export M_SAGENT_SYSTEM_PROMPT="${M_SAGENT_SYSTEM_PROMPT:-${PROJECT_ROOT}/1.txt}"
export M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING="${M_SAGENT_SYSTEM_PROMPT_ITERATIVE_CHECKING:-${PROJECT_ROOT}/sam3/sam3/agent/system_prompts/system_prompt_iterative_checking.txt}"
export TRAIN_ADAPTER_ENABLED="${TRAIN_ADAPTER_ENABLED:-true}"
export TRAIN_ADAPTER_URL="${TRAIN_ADAPTER_URL:-http://127.0.0.1:8765}"
export TRAIN_ADAPTER_EXPECTED_DEVICE="${TRAIN_ADAPTER_EXPECTED_DEVICE:-cpu}"
export TRAIN_ADAPTER_TIMEOUT="${TRAIN_ADAPTER_TIMEOUT:-120}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export M_SAGENT_MLLM_MIN_PIXELS="${M_SAGENT_MLLM_MIN_PIXELS:-262144}"
export M_SAGENT_MLLM_MAX_PIXELS="${M_SAGENT_MLLM_MAX_PIXELS:-262144}"
export M_SAGENT_MIN_FREE_GPU_MB="${M_SAGENT_MIN_FREE_GPU_MB:-20000}"

M_SAGENT_PYTHON="${M_SAGENT_PYTHON:-/root/autodl-tmp/conda-envs/m_sagent/bin/python}"

if [[ ! -x "${M_SAGENT_PYTHON}" ]]; then
  echo "M-SAgent python not found: ${M_SAGENT_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${M_SAGENT_SAM3_CHECKPOINT_PATH}" ]]; then
  echo "SAM3 checkpoint not found: ${M_SAGENT_SAM3_CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ "${GRIDGROUND_BACKEND}" == "embedded" && ! -d "${GRIDGROUND_MODEL_DIR}" ]]; then
  echo "GridGround model dir not found: ${GRIDGROUND_MODEL_DIR}" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_GPU_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ -n "${FREE_GPU_MB}" && "${FREE_GPU_MB}" -lt "${M_SAGENT_MIN_FREE_GPU_MB}" ]]; then
    echo "Refusing to start ${PROFILE}: free GPU memory ${FREE_GPU_MB} MB is below ${M_SAGENT_MIN_FREE_GPU_MB} MB" >&2
    exit 1
  fi
fi

SERVICE_DEVICE="embedded"
if [[ "${GRIDGROUND_BACKEND}" == "http" ]]; then
  HEALTH_JSON="$(curl -fsS --max-time 5 "${TRAIN_ADAPTER_URL}/health" 2>/dev/null || true)"
  if [[ -n "${HEALTH_JSON}" ]]; then
    SERVICE_DEVICE="$(printf '%s' "${HEALTH_JSON}" | python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("device", "")).lower())')"
    if [[ "${SERVICE_DEVICE}" == "cuda" ]]; then
      echo "Refusing to continue: TrainAdapter service is using GPU under ${PROFILE}" >&2
      exit 1
    fi
    if [[ -n "${TRAIN_ADAPTER_EXPECTED_DEVICE}" && "${SERVICE_DEVICE}" != "${TRAIN_ADAPTER_EXPECTED_DEVICE}" ]]; then
      echo "Refusing to continue: TrainAdapter device mismatch (expected ${TRAIN_ADAPTER_EXPECTED_DEVICE}, got ${SERVICE_DEVICE})" >&2
      exit 1
    fi
  else
    "${PROJECT_ROOT}/scripts/start_inference_service_cpu.sh"
  fi

  for _ in $(seq 1 60); do
    HEALTH_JSON="$(curl -fsS --max-time 5 "${TRAIN_ADAPTER_URL}/health" 2>/dev/null || true)"
    if [[ -n "${HEALTH_JSON}" ]]; then
      SERVICE_DEVICE="$(printf '%s' "${HEALTH_JSON}" | python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("device", "")).lower())')"
      if [[ -n "${TRAIN_ADAPTER_EXPECTED_DEVICE}" && "${SERVICE_DEVICE}" != "${TRAIN_ADAPTER_EXPECTED_DEVICE}" ]]; then
        echo "TrainAdapter health check returned wrong device: ${SERVICE_DEVICE}" >&2
        exit 1
      fi
      break
    fi
    sleep 2
  done

  if [[ -z "${HEALTH_JSON}" ]]; then
    echo "TrainAdapter service did not become healthy at ${TRAIN_ADAPTER_URL}" >&2
    exit 1
  fi
fi

echo "Running M-SAgent single_gpu_stable profile"
echo "  gridground_backend: ${GRIDGROUND_BACKEND}"
echo "  gridground_model_dir: ${GRIDGROUND_MODEL_DIR}"
echo "  train_adapter_url: ${TRAIN_ADAPTER_URL}"
echo "  localization_device: ${SERVICE_DEVICE}"
echo "  sam3_checkpoint: ${M_SAGENT_SAM3_CHECKPOINT_PATH}"
echo "  mllm_pixels: ${M_SAGENT_MLLM_MIN_PIXELS}-${M_SAGENT_MLLM_MAX_PIXELS}"

exec "${M_SAGENT_PYTHON}" "${PROJECT_ROOT}/main.py" "$@"
