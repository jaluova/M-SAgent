from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_minimal_vertical_slice import build_cli_service
from msagent.core.contracts.types import (
    EvaluationResult,
    PromptPackage,
    ProposalResult,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.service.cli import CLIRequest
from msagent.core.contracts.types import EvaluationVerdict


def main() -> None:
    run_dir = ROOT / "experiments" / "out" / "mock_retry"
    artifact_dir = run_dir / "artifacts"
    image_path = run_dir / "input.png"

    if run_dir.exists():
        shutil.rmtree(run_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"mock-image")

    cli_service, store = build_cli_service(
        artifact_dir,
        evaluation_sequence=(
            EvaluationVerdict.REJECT,
            EvaluationVerdict.REJECT,
        ),
    )
    result = cli_service.run(
        CLIRequest(
            image_path=str(image_path),
            query_text="the red cup",
            max_attempts=2,
        )
    )

    task = result.task
    print("=== Mock Retry Run ===")
    print("task_id:", task.identity.task_id)
    print("status:", task.runtime.status.value)
    print("stage:", task.runtime.stage.value)
    print("stop_reason:", task.result.stop_reason.value if task.result.stop_reason else None)
    print("final_verdict:", task.result.final_verdict.value if task.result.final_verdict else None)
    print("failure_summary:", task.result.failure_summary)
    print("attempt_count:", len(task.attempt_history))
    print("artifact_dir:", artifact_dir)
    print()

    for attempt in task.attempt_history:
        print(f"=== Attempt {attempt.attempt_index} ===")
        print("route:", attempt.route.value)
        print("verdict:", attempt.verdict.value if attempt.verdict else None)
        print("failure_type:", attempt.failure_type.value if attempt.failure_type else None)
        print("notes:", attempt.notes)
        print()

        if attempt.query_understanding_ref:
            print("[QueryUnderstandingResult]")
            print(store.load_artifact(attempt.query_understanding_ref, QueryUnderstandingResult))
            print()

        if attempt.proposal_ref:
            print("[ProposalResult]")
            print(store.load_artifact(attempt.proposal_ref, ProposalResult))
            print()

        if attempt.prompt_package_ref:
            print("[PromptPackage]")
            print(store.load_artifact(attempt.prompt_package_ref, PromptPackage))
            print()

        if attempt.segmentation_ref:
            print("[SegmentationResult]")
            print(store.load_artifact(attempt.segmentation_ref, SegmentationResult))
            print()

        if attempt.evaluation_ref:
            print("[EvaluationResult]")
            print(store.load_artifact(attempt.evaluation_ref, EvaluationResult))
            print()

    print("artifact_ids:", [ref.artifact_id for ref in task.artifacts.artifact_refs])


if __name__ == "__main__":
    main()
