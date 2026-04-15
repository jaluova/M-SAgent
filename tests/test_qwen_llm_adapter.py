from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.contracts.adapter_requests import (
    EvaluationAdapterRequest,
    QueryUnderstandingAdapterRequest,
)
from msagent.core.contracts.common import ArtifactRef, ArtifactKind
from msagent.core.contracts.types import (
    EvaluationVerdict,
    FailureType,
    PromptMetadata,
    PromptPackage,
    PromptTextBundle,
    ProposalRoute,
    SegmentationCandidate,
    SegmentationResult,
    SegmentationStatus,
    SpatialPromptBundle,
)
from msagent.infra.qwen_llm_adapter import RealQwenLLMAdapter


class FakeProvider:
    def get_loaded_components(self):
        raise AssertionError("generation_override should bypass provider loading")


class RealQwenLLMAdapterTests(unittest.TestCase):
    def test_query_understanding_uses_generated_json_payload(self) -> None:
        adapter = RealQwenLLMAdapter(
            backend_name="real-qwen-test",
            model_path="/models/qwen",
            backbone_provider=FakeProvider(),
            generation_override=lambda _prompt: """
            {
              "normalized_query": "left sandwich",
              "target_summary": "the left sandwich on the plate",
              "target_type": "object",
              "implicitness": "explicit",
              "canonical_referent_text": "left sandwich",
              "referent_number": "single",
              "focus_terms": ["left", "sandwich"],
              "attribute_clues": ["left"]
            }
            """,
        )

        result = adapter.run_query_understanding(
            QueryUnderstandingAdapterRequest(
                task_id="task-1",
                raw_query="the left sandwich",
            )
        )

        self.assertEqual(result.normalized_query, "left sandwich")
        self.assertEqual(result.target_summary, "the left sandwich on the plate")
        self.assertEqual(result.focus_terms, ["left", "sandwich"])

    def test_query_understanding_falls_back_on_invalid_json(self) -> None:
        adapter = RealQwenLLMAdapter(
            backend_name="real-qwen-test",
            model_path="/models/qwen",
            backbone_provider=FakeProvider(),
            generation_override=lambda _prompt: "not-json",
        )

        result = adapter.run_query_understanding(
            QueryUnderstandingAdapterRequest(
                task_id="task-2",
                raw_query="the red cup",
            )
        )

        self.assertEqual(result.normalized_query, "the red cup")
        self.assertIn("fallback", result.notes[0])

    def test_evaluation_uses_generated_json_payload(self) -> None:
        adapter = RealQwenLLMAdapter(
            backend_name="real-qwen-test",
            model_path="/models/qwen",
            backbone_provider=FakeProvider(),
            generation_override=lambda _prompt: """
            {
              "verdict": "accept",
              "summary": "candidate looks correct",
              "failure_type": null,
              "confidence": 0.88,
              "retry_hints": []
            }
            """,
        )
        mask_ref = ArtifactRef(
            artifact_id="mask-1",
            artifact_type=ArtifactKind.MASK,
            attempt_index=1,
        )
        segmentation = SegmentationResult(
            segmentation_id="seg-1",
            status=SegmentationStatus.READY,
            result_summary="one candidate",
            candidates=[
                SegmentationCandidate(
                    candidate_id="candidate-1",
                    mask_ref=mask_ref,
                    score=0.91,
                )
            ],
            primary_candidate_id="candidate-1",
        )
        prompt = PromptPackage(
            package_id="pkg-1",
            package_version="v1",
            text_prompts=PromptTextBundle(normalized_text="the red cup"),
            spatial_prompts=SpatialPromptBundle(),
            metadata=PromptMetadata(produced_from_route=ProposalRoute.LOCATE),
        )

        result = adapter.run_evaluation(
            EvaluationAdapterRequest(
                task_id="task-3",
                raw_query="the red cup",
                segmentation=segmentation,
                prompt_package=prompt,
            )
        )

        self.assertIs(result.verdict, EvaluationVerdict.ACCEPT)
        self.assertEqual(result.summary, "candidate looks correct")
        self.assertEqual(result.accepted_candidate_id, "candidate-1")
        self.assertEqual(result.accepted_mask_ref, mask_ref)
        self.assertIsNone(result.failure_type)

    def test_evaluation_falls_back_when_generation_is_invalid(self) -> None:
        adapter = RealQwenLLMAdapter(
            backend_name="real-qwen-test",
            model_path="/models/qwen",
            backbone_provider=FakeProvider(),
            generation_override=lambda _prompt: "oops",
        )
        prompt = PromptPackage(
            package_id="pkg-2",
            package_version="v1",
            text_prompts=PromptTextBundle(normalized_text="the red cup"),
            spatial_prompts=SpatialPromptBundle(),
            metadata=PromptMetadata(produced_from_route=ProposalRoute.LOCATE),
        )
        segmentation = SegmentationResult(
            segmentation_id="seg-2",
            status=SegmentationStatus.EMPTY,
            result_summary="no candidate",
            candidates=[],
            primary_candidate_id=None,
        )

        result = adapter.run_evaluation(
            EvaluationAdapterRequest(
                task_id="task-4",
                raw_query="the red cup",
                segmentation=segmentation,
                prompt_package=prompt,
            )
        )

        self.assertIs(result.verdict, EvaluationVerdict.REJECT)
        self.assertIs(result.failure_type, FailureType.LOCALIZATION_ERROR)

    def test_evaluation_invalid_json_with_candidate_requires_review(self) -> None:
        adapter = RealQwenLLMAdapter(
            backend_name="real-qwen-test",
            model_path="/models/qwen",
            backbone_provider=FakeProvider(),
            generation_override=lambda _prompt: "```json\nnot actually json\n```",
        )
        mask_ref = ArtifactRef(
            artifact_id="mask-2",
            artifact_type=ArtifactKind.MASK,
            attempt_index=1,
        )
        segmentation = SegmentationResult(
            segmentation_id="seg-3",
            status=SegmentationStatus.READY,
            result_summary="one candidate",
            candidates=[
                SegmentationCandidate(
                    candidate_id="candidate-2",
                    mask_ref=mask_ref,
                    score=0.93,
                )
            ],
            primary_candidate_id="candidate-2",
        )
        prompt = PromptPackage(
            package_id="pkg-3",
            package_version="v1",
            text_prompts=PromptTextBundle(normalized_text="the red cup"),
            spatial_prompts=SpatialPromptBundle(),
            metadata=PromptMetadata(produced_from_route=ProposalRoute.LOCATE),
        )

        result = adapter.run_evaluation(
            EvaluationAdapterRequest(
                task_id="task-5",
                raw_query="the red cup",
                segmentation=segmentation,
                prompt_package=prompt,
            )
        )

        self.assertIs(result.verdict, EvaluationVerdict.REVIEW)
        self.assertIsNone(result.accepted_candidate_id)
        self.assertIsNone(result.accepted_mask_ref)
        self.assertEqual(result.retry_hints, ["retry_with_same_route"])

    def test_evaluation_unknown_verdict_with_candidate_requires_review(self) -> None:
        adapter = RealQwenLLMAdapter(
            backend_name="real-qwen-test",
            model_path="/models/qwen",
            backbone_provider=FakeProvider(),
            generation_override=lambda _prompt: """
            {
              "verdict": "looks_good",
              "summary": "candidate seems fine",
              "failure_type": null,
              "confidence": 0.74,
              "retry_hints": []
            }
            """,
        )
        mask_ref = ArtifactRef(
            artifact_id="mask-3",
            artifact_type=ArtifactKind.MASK,
            attempt_index=1,
        )
        segmentation = SegmentationResult(
            segmentation_id="seg-4",
            status=SegmentationStatus.READY,
            result_summary="one candidate",
            candidates=[
                SegmentationCandidate(
                    candidate_id="candidate-3",
                    mask_ref=mask_ref,
                    score=0.89,
                )
            ],
            primary_candidate_id="candidate-3",
        )
        prompt = PromptPackage(
            package_id="pkg-4",
            package_version="v1",
            text_prompts=PromptTextBundle(normalized_text="the red cup"),
            spatial_prompts=SpatialPromptBundle(),
            metadata=PromptMetadata(produced_from_route=ProposalRoute.LOCATE),
        )

        result = adapter.run_evaluation(
            EvaluationAdapterRequest(
                task_id="task-6",
                raw_query="the red cup",
                segmentation=segmentation,
                prompt_package=prompt,
            )
        )

        self.assertIs(result.verdict, EvaluationVerdict.REVIEW)
        self.assertIsNone(result.accepted_candidate_id)
        self.assertIsNone(result.accepted_mask_ref)
        self.assertEqual(result.retry_hints, ["retry_with_same_route"])

    def test_evaluation_propagates_generation_runtime_failures(self) -> None:
        adapter = RealQwenLLMAdapter(
            backend_name="real-qwen-test",
            model_path="/models/qwen",
            backbone_provider=FakeProvider(),
            generation_override=lambda _prompt: (_ for _ in ()).throw(RuntimeError("oom")),
        )
        prompt = PromptPackage(
            package_id="pkg-5",
            package_version="v1",
            text_prompts=PromptTextBundle(normalized_text="the red cup"),
            spatial_prompts=SpatialPromptBundle(),
            metadata=PromptMetadata(produced_from_route=ProposalRoute.LOCATE),
        )
        segmentation = SegmentationResult(
            segmentation_id="seg-5",
            status=SegmentationStatus.READY,
            result_summary="one candidate",
            candidates=[],
            primary_candidate_id=None,
        )

        with self.assertRaisesRegex(RuntimeError, "oom"):
            adapter.run_evaluation(
                EvaluationAdapterRequest(
                    task_id="task-7",
                    raw_query="the red cup",
                    segmentation=segmentation,
                    prompt_package=prompt,
                )
            )
