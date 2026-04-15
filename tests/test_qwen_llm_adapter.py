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
