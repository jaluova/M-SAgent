from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.contracts.common import ArtifactKind, ArtifactRef, ImageRef
from msagent.core.contracts.types import EvaluationVerdict, ProposalResult, ProposalRoute
from msagent.core.contracts.types import ProposalStatus
from msagent.core.policies.retry_policy import RetryPolicy
from msagent.core.task.enums import StopReason, TaskSource, TaskStage, TaskStatus
from msagent.infra.adapters import LocatorAdapter
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.mock_adapters import MockLLMAdapter, MockLocatorAdapter, MockSAMAdapter
from msagent.modules.evaluator import LLMEvaluatorModule
from msagent.modules.prompt_bridge import RuleBasedPromptBridgeModule
from msagent.modules.proposal_engine import DefaultProposalEngineModule, LocateProposalRouteHandler
from msagent.modules.query_understanding import LLMQueryUnderstandingModule
from msagent.modules.segmenter import SAMSegmenterModule
from msagent.orchestrator.orchestrator import Orchestrator, OrchestratorDependencies
from msagent.core.task.models import RunTask, RunTaskIdentity, RunTaskRequest, RunTaskResult
from msagent.core.task.models import RunTaskArtifacts, RunTaskRuntime
from msagent.orchestrator.orchestrator import OrchestrationResult
from msagent.service.api import APIRequest, APIResponse, APIService


def make_task() -> RunTask:
    now = datetime(2026, 4, 14, 10, 0, 0)
    query_ref = ArtifactRef(
        artifact_id="artifact-query",
        artifact_type=ArtifactKind.QUERY_UNDERSTANDING_RESULT,
        attempt_index=1,
    )
    prompt_ref = ArtifactRef(
        artifact_id="artifact-prompt",
        artifact_type=ArtifactKind.PROMPT_PACKAGE,
        attempt_index=1,
    )
    eval_ref = ArtifactRef(
        artifact_id="artifact-eval",
        artifact_type=ArtifactKind.EVALUATION_RESULT,
        attempt_index=1,
    )
    mask_ref = ArtifactRef(
        artifact_id="artifact-mask",
        artifact_type=ArtifactKind.MASK,
        attempt_index=1,
    )
    return RunTask(
        identity=RunTaskIdentity(
            task_id="api-task-fixed",
            source=TaskSource.API,
            created_at=now,
            session_id="session-1",
            request_id="request-1",
        ),
        request=RunTaskRequest(
            image_ref=ImageRef(
                uri="https://example.invalid/image.png",
                image_id="image-1",
                sha256="sha256-1",
            ),
            raw_query="the red cup",
            user_context_text="from api",
            client_metadata={"page": "gallery"},
        ),
        runtime=RunTaskRuntime(
            stage=TaskStage.FINISHED,
            status=TaskStatus.SUCCEEDED,
            attempt_index=1,
            max_attempts=3,
            updated_at=now,
        ),
        artifacts=RunTaskArtifacts(
            artifact_refs=[query_ref, prompt_ref, eval_ref],
            latest_query_understanding_ref=query_ref,
            latest_prompt_package_ref=prompt_ref,
            latest_evaluation_ref=eval_ref,
        ),
        result=RunTaskResult(
            final_mask_ref=mask_ref,
            final_prompt_package_ref=prompt_ref,
            stop_reason=StopReason.ACCEPTED,
            final_summary="accepted on first attempt",
        ),
    )


class EmptyLocatorAdapter(MockLocatorAdapter):
    def locate(self, request):
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-empty",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.EMPTY,
            proposal_summary="no candidate produced",
            candidates=[],
            primary_candidate_id=None,
        )


class FailingLocatorAdapter(LocatorAdapter):
    def locate(self, request):
        raise RuntimeError("provider=embedded-locator session=session-123 crashed")


def build_api_service(
    artifact_root: Path,
    evaluation_sequence: tuple[EvaluationVerdict, ...] = (EvaluationVerdict.ACCEPT,),
    locator_adapter: LocatorAdapter | None = None,
) -> APIService:
    store = LocalFileArtifactStore(str(artifact_root))
    llm_adapter = MockLLMAdapter(
        backend_name="mock-llm",
        evaluation_verdict_sequence=evaluation_sequence,
    )
    orchestrator = Orchestrator(
        OrchestratorDependencies(
            query_understanding_module=LLMQueryUnderstandingModule(
                llm_adapter=llm_adapter,
                artifact_store=store,
            ),
            proposal_engine_module=DefaultProposalEngineModule(
                route_handlers={
                    ProposalRoute.LOCATE: LocateProposalRouteHandler(
                        locator_adapter=locator_adapter
                        or MockLocatorAdapter(backend_name="mock-locator"),
                    )
                },
                artifact_store=store,
            ),
            prompt_bridge_module=RuleBasedPromptBridgeModule(artifact_store=store),
            segmenter_module=SAMSegmenterModule(
                sam_adapter=MockSAMAdapter(
                    backend_name="mock-sam",
                    artifact_store=store,
                ),
                artifact_store=store,
            ),
            evaluator_module=LLMEvaluatorModule(
                llm_adapter=llm_adapter,
                artifact_store=store,
            ),
            retry_policy=RetryPolicy(),
        )
    )
    return APIService(orchestrator=orchestrator)


class APIServiceTests(unittest.TestCase):
    def test_build_task_maps_api_request_into_run_task(self) -> None:
        service = APIService(orchestrator=Mock())
        request = APIRequest(
            image_uri="s3://bucket/input.png",
            query_text="the red cup",
            max_attempts=0,
            request_metadata={
                "request_id": "request-1",
                "session_id": "session-1",
                "user_context_text": "from api",
                "image_id": "image-1",
                "sha256": "sha256-1",
                "page": "gallery",
                "viewport": "mobile",
            },
        )

        task = service.build_task(request)

        self.assertTrue(task.identity.task_id.startswith("api-task-"))
        self.assertIs(task.identity.source, TaskSource.API)
        self.assertEqual(task.identity.request_id, "request-1")
        self.assertEqual(task.identity.session_id, "session-1")
        self.assertEqual(task.request.image_ref.uri, "s3://bucket/input.png")
        self.assertEqual(task.request.image_ref.image_id, "image-1")
        self.assertEqual(task.request.image_ref.sha256, "sha256-1")
        self.assertEqual(task.request.raw_query, "the red cup")
        self.assertEqual(task.request.user_context_text, "from api")
        self.assertEqual(
            task.request.client_metadata,
            {"page": "gallery", "viewport": "mobile"},
        )
        self.assertIs(task.runtime.stage, TaskStage.CREATED)
        self.assertIs(task.runtime.status, TaskStatus.PENDING)
        self.assertEqual(task.runtime.attempt_index, 0)
        self.assertEqual(task.runtime.max_attempts, 1)

    def test_run_is_a_thin_orchestrator_wrapper(self) -> None:
        orchestrator = Mock()
        service = APIService(orchestrator=orchestrator)
        request = APIRequest(
            image_uri="https://example.invalid/image.png",
            query_text="the red cup",
        )
        task = make_task()
        expected_result = OrchestrationResult(task=task, last_attempt_result=None)

        with patch.object(service, "build_task", return_value=task) as build_task:
            orchestrator.run.return_value = expected_result

            result = service.run(request)

        build_task.assert_called_once_with(request)
        orchestrator.run.assert_called_once_with(task)
        self.assertIs(result, expected_result)

    def test_to_response_maps_task_snapshot_without_using_runtime_objects(self) -> None:
        service = APIService(orchestrator=Mock())
        task = make_task()
        result = OrchestrationResult(task=task, last_attempt_result=object())

        response = service.to_response(result)

        self.assertEqual(
            response,
            APIResponse(
                task_id="api-task-fixed",
                status="succeeded",
                summary="Task completed successfully.",
                result_refs=[
                    "artifact-mask",
                    "artifact-prompt",
                ],
            ),
        )

    def test_to_response_uses_safe_failure_summary_without_leaking_internal_text(self) -> None:
        service = APIService(orchestrator=Mock())
        task = make_task()
        task.runtime.status = TaskStatus.FAILED
        task.result.stop_reason = StopReason.UNRECOVERABLE_ERROR
        task.result.final_summary = "provider=embedded-locator session=session-123 crashed"
        task.result.failure_summary = "RuntimeError: session=session-123 crashed"

        response = service.to_response(OrchestrationResult(task=task))

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.summary, "Task failed due to an internal error.")
        self.assertNotIn("session-123", response.summary)
        self.assertNotIn("embedded-locator", response.summary)

    def test_to_response_only_exposes_final_result_refs(self) -> None:
        service = APIService(orchestrator=Mock())
        task = make_task()

        response = service.to_response(OrchestrationResult(task=task))

        self.assertEqual(response.result_refs, ["artifact-mask", "artifact-prompt"])

    def test_to_response_keeps_only_available_final_result_refs(self) -> None:
        service = APIService(orchestrator=Mock())
        task = make_task()
        task.result.final_prompt_package_ref = None

        response = service.to_response(OrchestrationResult(task=task))

        self.assertEqual(response.status, "succeeded")
        self.assertEqual(response.summary, "Task completed successfully.")
        self.assertEqual(response.result_refs, ["artifact-mask"])

    def test_api_end_to_end_accepted_response(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = build_api_service(Path(tmp_dir) / "artifacts")

            result = service.run(
                APIRequest(
                    image_uri="file:///tmp/input.png",
                    query_text="the red cup",
                    max_attempts=2,
                )
            )
            response = service.to_response(result)

        self.assertEqual(response.status, "succeeded")
        self.assertEqual(response.summary, "Task completed successfully.")
        self.assertIsNotNone(result.task.result.final_mask_ref)
        self.assertIsNotNone(result.task.result.final_prompt_package_ref)
        assert result.task.result.final_mask_ref is not None
        assert result.task.result.final_prompt_package_ref is not None
        self.assertEqual(
            response.result_refs,
            [
                result.task.result.final_mask_ref.artifact_id,
                result.task.result.final_prompt_package_ref.artifact_id,
            ],
        )
        self.assertEqual(result.task.result.stop_reason, StopReason.ACCEPTED)

    def test_api_end_to_end_empty_proposal_response(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = build_api_service(
                Path(tmp_dir) / "artifacts",
                locator_adapter=EmptyLocatorAdapter(backend_name="empty-locator"),
            )

            result = service.run(
                APIRequest(
                    image_uri="file:///tmp/input.png",
                    query_text="the red cup",
                    max_attempts=2,
                )
            )
            response = service.to_response(result)

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.summary, "Task could not produce a usable result.")
        self.assertEqual(response.result_refs, [])
        self.assertEqual(result.task.result.stop_reason, StopReason.EMPTY_PROPOSAL)

    def test_api_end_to_end_max_attempts_response(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = build_api_service(
                Path(tmp_dir) / "artifacts",
                evaluation_sequence=(
                    EvaluationVerdict.REJECT,
                    EvaluationVerdict.REJECT,
                ),
            )

            result = service.run(
                APIRequest(
                    image_uri="file:///tmp/input.png",
                    query_text="the red cup",
                    max_attempts=2,
                )
            )
            response = service.to_response(result)

        self.assertEqual(response.status, "failed")
        self.assertEqual(
            response.summary,
            "Task did not complete within the allowed attempts.",
        )
        self.assertEqual(response.result_refs, [])
        self.assertEqual(result.task.result.stop_reason, StopReason.MAX_ATTEMPTS_REACHED)

    def test_api_end_to_end_unrecoverable_error_response(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = build_api_service(
                Path(tmp_dir) / "artifacts",
                locator_adapter=FailingLocatorAdapter(backend_name="failing-locator"),
            )

            result = service.run(
                APIRequest(
                    image_uri="file:///tmp/input.png",
                    query_text="the red cup",
                    max_attempts=2,
                )
            )
            response = service.to_response(result)

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.summary, "Task failed due to an internal error.")
        self.assertEqual(response.result_refs, [])
        self.assertEqual(result.task.result.stop_reason, StopReason.UNRECOVERABLE_ERROR)
        self.assertNotIn("session=session-123", response.summary)
        self.assertNotIn("embedded-locator", response.summary)


if __name__ == "__main__":
    unittest.main()
