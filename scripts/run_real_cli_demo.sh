#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-m_sagent}"

export M_SAGENT_QWEN_MODEL_PATH="${M_SAGENT_QWEN_MODEL_PATH:-/root/autodl-tmp/modelscope/Qwen2.5-VL-7B-Instruct}"
export GRIDGROUND_CONFIG_PATH="${GRIDGROUND_CONFIG_PATH:-/root/autodl-tmp/modelscope_cache_gridground/alpharho/GridGround-TextGuided/config.json}"
export GRIDGROUND_ADAPTER_PATH="${GRIDGROUND_ADAPTER_PATH:-/root/autodl-tmp/modelscope_cache_gridground/alpharho/GridGround-TextGuided/best_model.pth}"
export M_SAGENT_SAM3_MODEL_PATH="${M_SAGENT_SAM3_MODEL_PATH:-/root/autodl-tmp/M-SAgent/old/sam3}"
export M_SAGENT_SAM3_CHECKPOINT_PATH="${M_SAGENT_SAM3_CHECKPOINT_PATH:-/root/autodl-tmp/modelscope_cache_sam3/facebook/sam3/sam3.pt}"
export MSAGENT_REAL_CLI_IMAGE="${MSAGENT_REAL_CLI_IMAGE:-/root/autodl-tmp/M-SAgent/old/example/truck.jpg}"
export MSAGENT_REAL_CLI_QUERY="${MSAGENT_REAL_CLI_QUERY:-truck}"
export MSAGENT_REAL_CLI_OUTPUT_DIR="${MSAGENT_REAL_CLI_OUTPUT_DIR:-/root/autodl-tmp/msagent_cli_demo_output}"
export MSAGENT_REAL_CLI_ARTIFACT_ROOT="${MSAGENT_REAL_CLI_ARTIFACT_ROOT:-${MSAGENT_REAL_CLI_OUTPUT_DIR}/artifacts}"

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "error: ${label} does not exist: ${path}" >&2
    exit 1
  fi
}

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda is required to run scripts/run_real_cli_demo.sh." >&2
  exit 1
fi

require_path "Qwen model path" "${M_SAGENT_QWEN_MODEL_PATH}"
require_path "GridGround config path" "${GRIDGROUND_CONFIG_PATH}"
require_path "GridGround adapter path" "${GRIDGROUND_ADAPTER_PATH}"
require_path "SAM3 model path" "${M_SAGENT_SAM3_MODEL_PATH}"
require_path "SAM3 checkpoint path" "${M_SAGENT_SAM3_CHECKPOINT_PATH}"
require_path "CLI demo image" "${MSAGENT_REAL_CLI_IMAGE}"

mkdir -p "${MSAGENT_REAL_CLI_OUTPUT_DIR}" "${MSAGENT_REAL_CLI_ARTIFACT_ROOT}"

echo "Running real CLI demo with:"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  CONDA_ENV_NAME=${CONDA_ENV_NAME}"
echo "  IMAGE=${MSAGENT_REAL_CLI_IMAGE}"
echo "  QUERY=${MSAGENT_REAL_CLI_QUERY}"
echo "  OUTPUT_DIR=${MSAGENT_REAL_CLI_OUTPUT_DIR}"
echo "  ARTIFACT_ROOT=${MSAGENT_REAL_CLI_ARTIFACT_ROOT}"

cd "${ROOT_DIR}"

TMP_PY="$(mktemp "${TMPDIR:-/tmp}/msagent-real-cli-demo.XXXXXX.py")"
cleanup() {
  rm -f "${TMP_PY}"
}
trap cleanup EXIT

cat >"${TMP_PY}" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

from msagent.core.config.settings import MSAgentSettings
from msagent.core.contracts.types import EvaluationVerdict, SegmentationResult
from msagent.infra.mock_adapters import MockLLMAdapter
from msagent.infra.mock_artifacts import MockMask
from msagent.service import build_default_cli_service
from msagent.service.cli import CLIRequest


settings = MSAgentSettings()
settings.runtime.artifact_root = os.environ["MSAGENT_REAL_CLI_ARTIFACT_ROOT"]
settings.model_paths.qwen_model_path = os.environ["M_SAGENT_QWEN_MODEL_PATH"]
settings.model_paths.embedded_locator_config_path = os.environ["GRIDGROUND_CONFIG_PATH"]
settings.model_paths.embedded_locator_adapter_path = os.environ["GRIDGROUND_ADAPTER_PATH"]
settings.model_paths.sam_model_path = os.environ["M_SAGENT_SAM3_MODEL_PATH"]
settings.model_paths.sam_checkpoint_path = os.environ["M_SAGENT_SAM3_CHECKPOINT_PATH"]

assembly = build_default_cli_service(
    settings,
    llm_adapter=MockLLMAdapter(
        backend_name="mock-llm-real-cli-demo",
        evaluation_verdict_sequence=(EvaluationVerdict.ACCEPT,),
    ),
)

result = assembly.run(
    CLIRequest(
        image_path=os.environ["MSAGENT_REAL_CLI_IMAGE"],
        query_text=os.environ["MSAGENT_REAL_CLI_QUERY"],
        max_attempts=1,
        output_dir=os.environ["MSAGENT_REAL_CLI_OUTPUT_DIR"],
    )
)

attempt = result.task.attempt_history[0]
segmentation_ref = attempt.segmentation_ref
mask_ref = result.task.result.final_mask_ref
segmentation = (
    assembly.artifact_store.load_artifact(segmentation_ref, SegmentationResult)
    if segmentation_ref is not None
    else None
)
mask_payload = (
    assembly.artifact_store.load_artifact(mask_ref, MockMask)
    if mask_ref is not None
    else None
)
report_path = Path(os.environ["MSAGENT_REAL_CLI_OUTPUT_DIR"]) / "task_report.md"

summary = {
    "diagnostics": assembly.diagnostics,
    "task_id": result.task.identity.task_id,
    "task_status": result.task.runtime.status.value,
    "final_verdict": result.task.result.final_verdict.value if result.task.result.final_verdict else None,
    "locator_backend": assembly.locator_adapter.backend_name,
    "sam_backend": assembly.sam_adapter.backend_name,
    "segmentation_status": segmentation.status.value if segmentation is not None else None,
    "mask_backend_name": mask_payload.backend_name if mask_payload is not None else None,
    "mask_pixel_area": mask_payload.pixel_area if mask_payload is not None else None,
    "report_path": str(report_path),
    "artifact_root": os.environ["MSAGENT_REAL_CLI_ARTIFACT_ROOT"],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

PYTHONPATH="${ROOT_DIR}/src" conda run -n "${CONDA_ENV_NAME}" python "${TMP_PY}"

echo
echo "Demo report written to:"
echo "  ${MSAGENT_REAL_CLI_OUTPUT_DIR}/task_report.md"
