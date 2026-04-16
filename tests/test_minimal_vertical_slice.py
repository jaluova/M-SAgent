from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.contracts.common import ArtifactKind, ArtifactRef
from msagent.core.contracts.types import (
    EvaluationResult,
    EvaluationVerdict,
    FailureType,
    ProposalRoute,
    QueryUnderstandingResult,
    SegmentationCandidate,
    SegmentationResult,
    SegmentationStatus,
)
from msagent.core.contracts.types import (
    ImplicitnessLevel,
    NormalizedBox,
    PointHint,
    ProposalCandidate,
    ProposalResult,
    ProposalStatus,
    ReferentNumber,
    TargetType,
)
from msagent.core.contracts.adapter_requests import EvaluationAdapterRequest
from msagent.core.policies.retry_policy import RetryPolicy
from msagent.core.task.enums import StopReason, TaskStage, TaskStatus
from msagent.infra.adapters import LocatorAdapter
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.mask_artifact import MaskArtifact
from msagent.modules.evaluator import EvaluatorModuleInput, LLMEvaluatorModule
from msagent.modules.prompt_bridge import PromptBridgeModuleInput, RuleBasedPromptBridgeModule
from msagent.modules.proposal_engine import DefaultProposalEngineModule, LocateProposalRouteHandler
from msagent.modules.query_understanding import LLMQueryUnderstandingModule
from msagent.modules.segmenter import SegmenterModuleInput
from msagent.modules.segmenter import SAMSegmenterModule
from msagent.orchestrator.orchestrator import Orchestrator, OrchestratorDependencies
from msagent.service.cli import CLIRequest, CLIService
from tests.support.deterministic_adapters import (
    DeterministicLLMAdapter,
    DeterministicLocatorAdapter,
    DeterministicSAMAdapter,
)


def build_cli_service(
    artifact_root: Path,
    evaluation_sequence: tuple[EvaluationVerdict, ...] = (EvaluationVerdict.ACCEPT,),
    locator_adapter: LocatorAdapter | None = None,
    llm_adapter: DeterministicLLMAdapter | None = None,
) -> tuple[CLIService, LocalFileArtifactStore]:
    store = LocalFileArtifactStore(str(artifact_root))
    llm_adapter = llm_adapter or DeterministicLLMAdapter(
        backend_name="deterministic-llm",
        evaluation_verdict_sequence=evaluation_sequence,
    )
    query_module = LLMQueryUnderstandingModule(
        llm_adapter=llm_adapter,
        artifact_store=store,
    )
    proposal_module = DefaultProposalEngineModule(
        route_handlers={
            ProposalRoute.LOCATE: LocateProposalRouteHandler(
                locator_adapter=locator_adapter
                or DeterministicLocatorAdapter(backend_name="deterministic-locator"),
            )
        },
        artifact_store=store,
    )
    prompt_bridge_module = RuleBasedPromptBridgeModule(artifact_store=store)
    segmenter_module = SAMSegmenterModule(
        sam_adapter=DeterministicSAMAdapter(
            backend_name="deterministic-sam",
            artifact_store=store,
        ),
        artifact_store=store,
    )
    evaluator_module = LLMEvaluatorModule(
        llm_adapter=llm_adapter,
        artifact_store=store,
    )
    orchestrator = Orchestrator(
        OrchestratorDependencies(
            query_understanding_module=query_module,
            proposal_engine_module=proposal_module,
            prompt_bridge_module=prompt_bridge_module,
            segmenter_module=segmenter_module,
            evaluator_module=evaluator_module,
            retry_policy=RetryPolicy(),
        )
    )
    return CLIService(orchestrator=orchestrator), store


class RecordingLLMAdapter(DeterministicLLMAdapter):
    def __init__(self) -> None:
        super().__init__(backend_name="recording-llm")
        self.last_evaluation_request: EvaluationAdapterRequest | None = None

    def run_evaluation(self, request: EvaluationAdapterRequest):
        self.last_evaluation_request = request
        return super().run_evaluation(request)


class EmptyLocatorAdapter(DeterministicLocatorAdapter):
    def locate(self, request):
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-empty",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.EMPTY,
            proposal_summary="no candidate produced",
            candidates=[],
            primary_candidate_id=None,
        )


class AlwaysAcceptLLMAdapter(DeterministicLLMAdapter):
    def __init__(self, *, backend_name: str) -> None:
        super().__init__(backend_name=backend_name)
        self.last_evaluation_request: EvaluationAdapterRequest | None = None

    def run_evaluation(self, request: EvaluationAdapterRequest) -> EvaluationResult:
        self.last_evaluation_request = request
        primary_candidate = request.segmentation.candidates[0]
        return EvaluationResult(
            evaluation_id=f"{request.task_id}-evaluation-accept",
            verdict=EvaluationVerdict.ACCEPT,
            summary="LLM considered the candidate acceptable.",
            accepted_candidate_id=primary_candidate.candidate_id,
            accepted_mask_ref=primary_candidate.mask_ref,
            confidence=0.9,
        )


def make_understanding() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        understanding_id="u-1",
        normalized_query="the red cup",
        target_summary="a red cup",
        target_type=TargetType.OBJECT,
        implicitness=ImplicitnessLevel.EXPLICIT,
        canonical_referent_text="red cup",
        referent_number=ReferentNumber.SINGLE,
        focus_terms=["red", "cup"],
        attribute_clues=["red"],
    )


def make_proposal() -> ProposalResult:
    return ProposalResult(
        proposal_id="p-1",
        route=ProposalRoute.LOCATE,
        status=ProposalStatus.READY,
        proposal_summary="one stable candidate",
        candidates=[
            ProposalCandidate(
                candidate_id="candidate-1",
                rank=1,
                confidence=0.9,
                region_box=NormalizedBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9),
                positive_point_hints=[
                    PointHint(x=0.3, y=0.4, confidence=0.95, reason="center"),
                    PointHint(x=0.5, y=0.6, confidence=0.85, reason="support"),
                ],
                negative_point_hints=[
                    PointHint(x=0.05, y=0.1, confidence=0.7, reason="background"),
                ],
            )
        ],
        primary_candidate_id="candidate-1",
    )


class MinimalVerticalSliceTests(unittest.TestCase):
    def test_happy_path_accept(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"test-image")

            cli_service, store = build_cli_service(tmp_path / "artifacts")
            result = cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the red cup",
                    max_attempts=3,
                )
            )

            task = result.task
            self.assertIs(task.runtime.status, TaskStatus.SUCCEEDED)
            self.assertIs(task.result.final_verdict, EvaluationVerdict.ACCEPT)
            self.assertIsNotNone(task.result.final_mask_ref)
            self.assertIs(task.result.stop_reason, StopReason.ACCEPTED)
            self.assertEqual(len(task.attempt_history), 1)

            attempt = task.attempt_history[0]
            self.assertIs(attempt.route, ProposalRoute.LOCATE)
            self.assertIs(attempt.verdict, EvaluationVerdict.ACCEPT)
            self.assertIsNotNone(attempt.query_understanding_ref)
            self.assertIsNotNone(attempt.proposal_ref)
            self.assertIsNotNone(attempt.prompt_package_ref)
            self.assertIsNotNone(attempt.segmentation_ref)
            self.assertIsNotNone(attempt.evaluation_ref)

            loaded_mask = store.load_artifact(task.result.final_mask_ref, MaskArtifact)
            self.assertEqual(loaded_mask.label, "deterministic_mask")
            self.assertAlmostEqual(loaded_mask.active_box.x1, 0.2)
            self.assertEqual(len(loaded_mask.mask_bitmap), loaded_mask.height)
            self.assertEqual(len(loaded_mask.mask_bitmap[0]), loaded_mask.width)

    def test_artifact_store_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            payload = make_understanding()

            artifact_ref = store.save_artifact(
                ArtifactKind.QUERY_UNDERSTANDING_RESULT,
                payload,
            )
            loaded = store.load_artifact(artifact_ref, QueryUnderstandingResult)

            self.assertEqual(loaded, payload)
            with self.assertRaises(TypeError):
                store.load_artifact(artifact_ref, MaskArtifact)
            with self.assertRaises(TypeError):
                store.save_artifact(
                    ArtifactKind.QUERY_UNDERSTANDING_RESULT,
                    MaskArtifact(
                        mask_id="m-1",
                        width=1,
                        height=1,
                        active_box=NormalizedBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0),
                    ),
                )

            class DerivedUnderstanding(QueryUnderstandingResult):
                extra_field: str = "extra"

            with self.assertRaises(TypeError):
                store.save_artifact(
                    ArtifactKind.QUERY_UNDERSTANDING_RESULT,
                    DerivedUnderstanding(
                        understanding_id="u-2",
                        normalized_query="subclass query",
                        target_summary="subclass",
                        target_type=TargetType.OBJECT,
                        implicitness=ImplicitnessLevel.EXPLICIT,
                    ),
                )

    def test_prompt_bridge_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            module = RuleBasedPromptBridgeModule(artifact_store=store)

            output = module.run(
                PromptBridgeModuleInput(
                    task_id="task-1",
                    attempt_index=1,
                    raw_query="the red cup",
                    understanding=make_understanding(),
                    proposal=make_proposal(),
                    upstream_refs=[
                        ArtifactRef(
                            artifact_id="u-1",
                            artifact_type=ArtifactKind.QUERY_UNDERSTANDING_RESULT,
                            attempt_index=1,
                        ),
                        ArtifactRef(
                            artifact_id="p-1",
                            artifact_type=ArtifactKind.PROPOSAL_RESULT,
                            attempt_index=1,
                        ),
                    ],
                )
            )

            package = output.primary_payload
            self.assertIsNotNone(package)
            assert package is not None
            self.assertEqual(package.text_prompts.normalized_text, "the red cup")
            self.assertAlmostEqual(package.spatial_prompts.boxes[0].x1, 0.1)
            self.assertEqual(len(package.spatial_prompts.positive_points), 2)
            self.assertEqual(len(package.spatial_prompts.negative_points), 1)
            self.assertEqual(len(package.metadata.source_refs), 2)
            self.assertIsNotNone(package.execution_hints)
            self.assertTrue(package.execution_hints.crop_to_box)

    def test_deterministic_segmenter_accepts_points_only_prompt(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            segmenter = SAMSegmenterModule(
                sam_adapter=DeterministicSAMAdapter(
                    backend_name="deterministic-sam",
                    artifact_store=store,
                ),
                artifact_store=store,
            )
            package = RuleBasedPromptBridgeModule(artifact_store=store).run(
                PromptBridgeModuleInput(
                    task_id="task-points",
                    attempt_index=1,
                    raw_query="the red cup",
                    understanding=make_understanding(),
                    proposal=ProposalResult(
                        proposal_id="points-only",
                        route=ProposalRoute.LOCATE,
                        status=ProposalStatus.READY,
                        proposal_summary="points only",
                        candidates=[
                            ProposalCandidate(
                                candidate_id="candidate-points",
                                rank=1,
                                confidence=0.9,
                                positive_point_hints=[
                                    PointHint(x=0.4, y=0.5, confidence=0.9, reason="center"),
                                    PointHint(x=0.6, y=0.55, confidence=0.8, reason="support"),
                                ],
                            )
                        ],
                        primary_candidate_id="candidate-points",
                    ),
                )
            ).primary_payload
            assert package is not None
            package.spatial_prompts.boxes = []

            output = segmenter.run(
                SegmenterModuleInput(
                    task_id="task-points",
                    attempt_index=1,
                    image_uri=str(tmp_path / "input.png"),
                    prompt_package=package,
                )
            )

            self.assertIsNotNone(output.primary_payload)
            result = output.primary_payload
            assert result is not None
            self.assertEqual(len(result.candidates), 1)
            loaded_mask = store.load_artifact(result.candidates[0].mask_ref, MaskArtifact)
            self.assertAlmostEqual(loaded_mask.active_box.x1, 0.32)
            self.assertAlmostEqual(loaded_mask.active_box.x2, 0.68)

    def test_retry_boundary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"test-image")

            cli_service, _ = build_cli_service(
                tmp_path / "artifacts",
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
            self.assertIs(task.runtime.status, TaskStatus.FAILED)
            self.assertIs(task.result.final_verdict, EvaluationVerdict.REJECT)
            self.assertIs(task.result.stop_reason, StopReason.MAX_ATTEMPTS_REACHED)
            self.assertEqual(len(task.attempt_history), 2)
            self.assertTrue(
                all(
                    attempt.verdict is EvaluationVerdict.REJECT
                    for attempt in task.attempt_history
                )
            )

    def test_empty_proposal_closes_task_state(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"test-image")

            cli_service, _ = build_cli_service(
                tmp_path / "artifacts",
                locator_adapter=EmptyLocatorAdapter(backend_name="empty-locator"),
            )
            result = cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the red cup",
                    max_attempts=2,
                )
            )

            task = result.task
            self.assertIs(task.runtime.status, TaskStatus.FAILED)
            self.assertIs(task.result.stop_reason, StopReason.EMPTY_PROPOSAL)
            self.assertIs(task.runtime.stage, TaskStage.FINISHED)
            self.assertEqual(len(task.attempt_history), 1)
            self.assertIsNotNone(task.attempt_history[0].finished_at)
            self.assertIsNotNone(task.attempt_history[0].proposal_ref)
            self.assertIsNone(task.attempt_history[0].prompt_package_ref)

    def test_evaluator_receives_prompt_package(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"test-image")

            llm_adapter = RecordingLLMAdapter()
            cli_service, _ = build_cli_service(
                tmp_path / "artifacts",
                llm_adapter=llm_adapter,
            )
            cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the red cup",
                    max_attempts=1,
                )
            )

            self.assertIsNotNone(llm_adapter.last_evaluation_request)
            assert llm_adapter.last_evaluation_request is not None
            self.assertEqual(
                llm_adapter.last_evaluation_request.prompt_package.text_prompts.normalized_text,
                "the red cup",
            )
            self.assertIsNotNone(llm_adapter.last_evaluation_request.proposal)

    def test_evaluator_rejects_single_pixel_mask_before_accepting(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            llm_adapter = AlwaysAcceptLLMAdapter(backend_name="always-accept")
            module = LLMEvaluatorModule(
                llm_adapter=llm_adapter,
                artifact_store=store,
            )
            mask_ref = store.save_artifact(
                ArtifactKind.MASK,
                MaskArtifact(
                    mask_id="mask-single-pixel",
                    width=1000,
                    height=1000,
                    active_box=NormalizedBox(x1=0.0, y1=0.0031, x2=0.0020, y2=0.0061),
                    mask_bitmap=[],
                    active_points=[(0, 1)],
                    pixel_area=1,
                ),
            )
            segmentation = SegmentationResult(
                segmentation_id="seg-single-pixel",
                status=SegmentationStatus.READY,
                result_summary="one candidate",
                candidates=[
                    SegmentationCandidate(
                        candidate_id="candidate-1",
                        mask_ref=mask_ref,
                        score=0.82,
                    )
                ],
                primary_candidate_id="candidate-1",
            )
            prompt = RuleBasedPromptBridgeModule(artifact_store=store).run(
                PromptBridgeModuleInput(
                    task_id="task-single-pixel",
                    attempt_index=1,
                    raw_query="the banana",
                    understanding=make_understanding(),
                    proposal=make_proposal(),
                )
            ).primary_payload
            assert prompt is not None

            output = module.run(
                EvaluatorModuleInput(
                    task_id="task-single-pixel",
                    attempt_index=1,
                    raw_query="the banana",
                    prompt_package=prompt,
                    segmentation=segmentation,
                )
            )

            self.assertIsNotNone(output.primary_payload)
            result = output.primary_payload
            assert result is not None
            self.assertIs(result.verdict, EvaluationVerdict.REJECT)
            self.assertIs(result.failure_type, FailureType.PARTIAL_MASK)
            self.assertIsNone(result.accepted_mask_ref)
            self.assertIn("single active pixel", result.summary)
            self.assertEqual(llm_adapter.evaluation_counter, 0)

    def test_evaluator_downgrades_accept_when_mask_is_near_empty(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            llm_adapter = AlwaysAcceptLLMAdapter(backend_name="always-accept")
            module = LLMEvaluatorModule(
                llm_adapter=llm_adapter,
                artifact_store=store,
            )
            mask_ref = store.save_artifact(
                ArtifactKind.MASK,
                MaskArtifact(
                    mask_id="mask-near-empty",
                    width=1000,
                    height=1000,
                    active_box=NormalizedBox(x1=0.10, y1=0.10, x2=0.12, y2=0.12),
                    mask_bitmap=[],
                    active_points=[(110, 110)],
                    pixel_area=16,
                ),
            )
            segmentation = SegmentationResult(
                segmentation_id="seg-near-empty",
                status=SegmentationStatus.READY,
                result_summary="one candidate",
                candidates=[
                    SegmentationCandidate(
                        candidate_id="candidate-1",
                        mask_ref=mask_ref,
                        score=0.79,
                    )
                ],
                primary_candidate_id="candidate-1",
            )
            prompt = RuleBasedPromptBridgeModule(artifact_store=store).run(
                PromptBridgeModuleInput(
                    task_id="task-near-empty",
                    attempt_index=1,
                    raw_query="the banana",
                    understanding=make_understanding(),
                    proposal=make_proposal(),
                )
            ).primary_payload
            assert prompt is not None

            output = module.run(
                EvaluatorModuleInput(
                    task_id="task-near-empty",
                    attempt_index=1,
                    raw_query="the banana",
                    prompt_package=prompt,
                    segmentation=segmentation,
                )
            )

            self.assertIsNotNone(output.primary_payload)
            result = output.primary_payload
            assert result is not None
            self.assertIs(result.verdict, EvaluationVerdict.REVIEW)
            self.assertIs(result.failure_type, FailureType.PARTIAL_MASK)
            self.assertIsNone(result.accepted_mask_ref)
            self.assertIn("Geometry guard requested review", result.summary)
            self.assertIsNotNone(llm_adapter.last_evaluation_request)
            assert llm_adapter.last_evaluation_request is not None
            self.assertIn(
                "pixel_area=16",
                llm_adapter.last_evaluation_request.primary_mask_summary or "",
            )
            self.assertTrue(llm_adapter.last_evaluation_request.mask_quality_warnings)


if __name__ == "__main__":
    unittest.main()
