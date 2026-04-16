from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.contracts.adapter_requests import LocateAdapterRequest
from msagent.core.contracts.types import (
    NormalizedBox,
    PointHint,
    ProposalCandidate,
    ProposalResult,
    ProposalRoute,
    ProposalStatus,
)
from msagent.evals.embedded_locator import (
    EmbeddedLocatorEvaluationHarness,
    EmbeddedLocatorFailureCategory,
    EmbeddedLocatorParameterGroup,
    classify_embedded_locator_failure,
    load_embedded_locator_manifest,
    write_embedded_locator_evaluation_report,
)
from msagent.infra.adapters import LocatorAdapter
from msagent.infra.embedded_locator import EmbeddedLocatorAdapter
from msagent.infra.runtime.train_adapter_runtime import (
    EmbeddedLocatePoint,
    EmbeddedLocatePrediction,
    TrainAdapterRuntime,
    TrainAdapterRuntimeRequest,
)


class RecordingEmbeddedRuntime(TrainAdapterRuntime):
    def __init__(self) -> None:
        super().__init__(runtime_name="recording-embedded-runtime")
        self.last_request: TrainAdapterRuntimeRequest | None = None

    def predict_embedded_locate(
        self,
        request: TrainAdapterRuntimeRequest,
    ) -> EmbeddedLocatePrediction:
        self.last_request = request
        if request.query_text == "empty target":
            return EmbeddedLocatePrediction(
                runtime_name=self.runtime_name,
                diagnostics=[
                    "selected_k=0",
                    "no_points_after_runtime_filter",
                ],
                limitations=["runtime produced no point above confidence threshold"],
            )
        if request.query_text == "sweep target":
            abs_threshold = float(request.options.get("abs_threshold", 0.5))
            if abs_threshold >= 0.8:
                return EmbeddedLocatePrediction(
                    runtime_name=self.runtime_name,
                    diagnostics=[
                        "selected_k=0",
                        "no_points_after_runtime_filter",
                    ],
                    limitations=["runtime produced no point above confidence threshold"],
                )
        return EmbeddedLocatePrediction(
            runtime_name=self.runtime_name,
            points=[
                EmbeddedLocatePoint(x=0.33, y=0.44, confidence=0.88, reason="peak_1"),
                EmbeddedLocatePoint(x=0.41, y=0.49, confidence=0.72, reason="peak_2"),
            ],
            coarse_box=NormalizedBox(x1=0.2, y1=0.3, x2=0.6, y2=0.7),
            diagnostics=["selected_k=2"],
            limitations=["coarse_box derived from selected point envelope"],
        )


class FailedLocatorAdapter(LocatorAdapter):
    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        return ProposalResult(
            proposal_id=f"{request.task_id}-failed",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.FAILED,
            proposal_summary="runtime failed",
            diagnostics=["embedded_locator.failed", "reason=runtime_not_ready"],
        )


class CoarseBoxOnlyLocatorAdapter(LocatorAdapter):
    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        return ProposalResult(
            proposal_id=f"{request.task_id}-coarse-only",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.READY,
            proposal_summary="coarse box only",
            candidates=[
                ProposalCandidate(
                    candidate_id="coarse-only",
                    rank=1,
                    confidence=None,
                    region_box=NormalizedBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9),
                )
            ],
            primary_candidate_id="coarse-only",
            diagnostics=["coarse_box_only"],
        )


class MixedPolarityLocatorAdapter(LocatorAdapter):
    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        abs_threshold = float(request.options.get("abs_threshold", 0.5))
        if request.understanding.normalized_query == "positive target":
            if abs_threshold <= 0.4:
                return ProposalResult(
                    proposal_id=f"{request.task_id}-positive-ready",
                    route=ProposalRoute.LOCATE,
                    status=ProposalStatus.READY,
                    proposal_summary="positive sample located",
                    candidates=[
                        ProposalCandidate(
                            candidate_id="positive-ready",
                            rank=1,
                            confidence=0.91,
                            region_box=NormalizedBox(x1=0.2, y1=0.2, x2=0.6, y2=0.6),
                            positive_point_hints=[
                                PointHint(x=0.4, y=0.4, confidence=0.91),
                            ],
                        )
                    ],
                    primary_candidate_id="positive-ready",
                    diagnostics=["selected_k=1"],
                )
            return ProposalResult(
                proposal_id=f"{request.task_id}-positive-failed",
                route=ProposalRoute.LOCATE,
                status=ProposalStatus.FAILED,
                proposal_summary="positive sample failed",
                diagnostics=["embedded_locator.failed", "reason=positive_runtime_failed"],
            )
        if abs_threshold <= 0.4:
            return ProposalResult(
                proposal_id=f"{request.task_id}-negative-wrong-ready",
                route=ProposalRoute.LOCATE,
                status=ProposalStatus.READY,
                proposal_summary="negative sample false positive",
                candidates=[
                    ProposalCandidate(
                        candidate_id="negative-wrong-ready",
                        rank=1,
                        confidence=0.8,
                        region_box=NormalizedBox(x1=0.1, y1=0.1, x2=0.8, y2=0.8),
                        positive_point_hints=[
                            PointHint(x=0.5, y=0.5, confidence=0.8),
                        ],
                    )
                ],
                primary_candidate_id="negative-wrong-ready",
                diagnostics=["selected_k=1"],
            )
        return ProposalResult(
            proposal_id=f"{request.task_id}-negative-empty",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.EMPTY,
            proposal_summary="negative sample stayed empty",
            diagnostics=["embedded_locator.empty", "reason=no_points_after_runtime_filter"],
        )


def build_manifest_payload(
    query_text: str,
    *,
    image_path: str | None = None,
    should_locate: bool = True,
) -> dict[str, object]:
    return {
        "manifest_id": "embedded-locator-test-manifest",
        "samples": [
            {
                "id": "sample-1",
                "image_path": image_path or str(ROOT / "old/example/truck.jpg"),
                "query_text": query_text,
                "should_locate": should_locate,
                "acceptable_point_standard": "At least one point on the target is acceptable.",
                "acceptable_box_standard": "A coarse box around the target is acceptable.",
                "remark": "Unit test sample.",
                "minimum_points": 1,
                "minimum_top_confidence": 0.3,
                "require_coarse_box": True,
            }
        ],
    }


class EmbeddedLocatorEvalTests(unittest.TestCase):
    def test_manifest_loads_and_validates_repo_baseline(self) -> None:
        manifest = load_embedded_locator_manifest(ROOT / "embedded_locator_eval_manifest.json")

        self.assertEqual(manifest.manifest_id, "embedded_locator_old_examples_v1")
        self.assertEqual(len(manifest.samples), 8)
        self.assertTrue(
            any(sample.image_path == "old/example/ tv.jpg" for sample in manifest.samples)
        )
        resolved = manifest.resolve_image_path(manifest.samples[-1])
        self.assertTrue(resolved.is_file())

    def test_manifest_validation_rejects_missing_image(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    build_manifest_payload(
                        "truck",
                        image_path="old/example/does-not-exist.jpg",
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing image"):
                load_embedded_locator_manifest(manifest_path)

    def test_single_sample_evaluation_writes_expected_structure(self) -> None:
        runtime = RecordingEmbeddedRuntime()
        adapter = EmbeddedLocatorAdapter(
            backend_name="embedded-locator-test",
            runtime=runtime,
        )

        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(build_manifest_payload("truck")),
                encoding="utf-8",
            )
            manifest = load_embedded_locator_manifest(manifest_path)
            harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=adapter,
                artifact_root=Path(tmp_dir) / "artifacts",
            )

            report = harness.evaluate(
                manifest=manifest,
                parameter_groups=[
                    EmbeddedLocatorParameterGroup(
                        label="default",
                        runtime_options={
                            "abs_threshold": 0.4,
                            "rel_ratio": 0.7,
                            "min_k": 1,
                            "max_k": 3,
                            "min_point_confidence": 0.2,
                        },
                    )
                ],
            )
            report_path = write_embedded_locator_evaluation_report(
                report,
                Path(tmp_dir) / "report.json",
            )
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["run_mode"], "single")
        self.assertEqual(payload["runs"][0]["samples"][0]["sample_id"], "sample-1")
        self.assertEqual(payload["runs"][0]["samples"][0]["proposal_status"], "ready")
        self.assertEqual(payload["runs"][0]["samples"][0]["point_count"], 2)
        self.assertEqual(payload["runs"][0]["samples"][0]["top_confidence"], 0.88)
        self.assertTrue(payload["runs"][0]["samples"][0]["has_coarse_box"])
        self.assertEqual(
            payload["runs"][0]["samples"][0]["diagnostics"],
            ["selected_k=2"],
        )
        self.assertEqual(
            payload["runs"][0]["samples"][0]["runtime_options_snapshot"],
            {
                "abs_threshold": 0.4,
                "rel_ratio": 0.7,
                "min_k": 1,
                "max_k": 3,
                "min_point_confidence": 0.2,
            },
        )
        self.assertEqual(
            payload["runs"][0]["samples"][0]["runtime_option_overrides"],
            {
                "abs_threshold": 0.4,
                "rel_ratio": 0.7,
                "min_k": 1,
                "max_k": 3,
                "min_point_confidence": 0.2,
            },
        )
        self.assertEqual(payload["runs"][0]["summary"]["ready_count"], 1)
        self.assertEqual(payload["review"]["best_parameter_group_label"], "default")
        self.assertEqual(
            payload["comparison"]["entries"][0]["runtime_option_overrides"],
            {
                "abs_threshold": 0.4,
                "rel_ratio": 0.7,
                "min_k": 1,
                "max_k": 3,
                "min_point_confidence": 0.2,
            },
        )

    def test_parameter_overrides_flow_into_runtime_request_options(self) -> None:
        runtime = RecordingEmbeddedRuntime()
        adapter = EmbeddedLocatorAdapter(
            backend_name="embedded-locator-test",
            runtime=runtime,
        )

        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(build_manifest_payload("truck")),
                encoding="utf-8",
            )
            manifest = load_embedded_locator_manifest(manifest_path)
            harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=adapter,
                artifact_root=Path(tmp_dir) / "artifacts",
            )
            harness.evaluate(
                manifest=manifest,
                parameter_groups=[
                    EmbeddedLocatorParameterGroup(
                        label="override-check",
                        runtime_options={
                            "abs_threshold": 0.31,
                            "rel_ratio": 0.61,
                            "min_k": 2,
                            "max_k": 4,
                            "min_point_confidence": 0.27,
                        },
                    )
                ],
            )

        self.assertIsNotNone(runtime.last_request)
        assert runtime.last_request is not None
        self.assertEqual(
            runtime.last_request.options,
            {
                "abs_threshold": 0.31,
                "rel_ratio": 0.61,
                "min_k": 2,
                "max_k": 4,
                "min_point_confidence": 0.27,
            },
        )

    def test_empty_override_still_persists_effective_runtime_defaults(self) -> None:
        runtime = RecordingEmbeddedRuntime()
        adapter = EmbeddedLocatorAdapter(
            backend_name="embedded-locator-defaults",
            runtime=runtime,
        )

        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(build_manifest_payload("truck")),
                encoding="utf-8",
            )
            manifest = load_embedded_locator_manifest(manifest_path)
            harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=adapter,
                artifact_root=Path(tmp_dir) / "artifacts",
            )

            report = harness.evaluate(
                manifest=manifest,
                parameter_groups=[EmbeddedLocatorParameterGroup(label="default")],
            )

        self.assertEqual(
            report.runs[0].samples[0].runtime_options_snapshot,
            {
                "abs_threshold": 0.5,
                "rel_ratio": 0.75,
                "min_k": 1,
                "max_k": 3,
                "min_point_confidence": 0.0,
            },
        )
        self.assertEqual(report.runs[0].samples[0].runtime_option_overrides, {})
        self.assertEqual(
            report.review.best_parameter_group_options,
            {
                "abs_threshold": 0.5,
                "rel_ratio": 0.75,
                "min_k": 1,
                "max_k": 3,
                "min_point_confidence": 0.0,
            },
        )
        self.assertEqual(report.review.best_parameter_group_overrides, {})

    def test_failure_classification_logic(self) -> None:
        self.assertEqual(
            classify_embedded_locator_failure(
                proposal_status="empty",
                diagnostics=["selected_k=0", "no_points_after_runtime_filter"],
                point_count=0,
                has_coarse_box=False,
                top_confidence=None,
            ),
            EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value,
        )
        self.assertEqual(
            classify_embedded_locator_failure(
                proposal_status="failed",
                diagnostics=["runtime_exception=RuntimeError"],
                point_count=0,
                has_coarse_box=False,
                top_confidence=None,
            ),
            EmbeddedLocatorFailureCategory.RUNTIME_FAILED.value,
        )
        self.assertEqual(
            classify_embedded_locator_failure(
                proposal_status="empty",
                diagnostics=["embedded_locator.empty"],
                point_count=0,
                has_coarse_box=False,
                top_confidence=None,
            ),
            EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value,
        )
        self.assertEqual(
            classify_embedded_locator_failure(
                proposal_status="ready",
                diagnostics=["selected_k=1"],
                point_count=1,
                has_coarse_box=True,
                top_confidence=0.22,
                minimum_top_confidence=0.3,
            ),
            EmbeddedLocatorFailureCategory.LOW_CONFIDENCE_OR_SPARSE_POINTS.value,
        )
        self.assertEqual(
            classify_embedded_locator_failure(
                proposal_status="ready",
                diagnostics=["coarse_box_only"],
                point_count=0,
                has_coarse_box=True,
                top_confidence=None,
            ),
            EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value,
        )

    def test_ready_empty_and_failed_integration_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manifest_ready_path = Path(tmp_dir) / "ready.json"
            manifest_ready_path.write_text(
                json.dumps(build_manifest_payload("truck")),
                encoding="utf-8",
            )
            manifest_empty_path = Path(tmp_dir) / "empty.json"
            manifest_empty_path.write_text(
                json.dumps(build_manifest_payload("empty target")),
                encoding="utf-8",
            )

            ready_manifest = load_embedded_locator_manifest(manifest_ready_path)
            empty_manifest = load_embedded_locator_manifest(manifest_empty_path)

            ready_harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=EmbeddedLocatorAdapter(
                    backend_name="embedded-ready",
                    runtime=RecordingEmbeddedRuntime(),
                ),
                artifact_root=Path(tmp_dir) / "ready-artifacts",
            )
            empty_harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=EmbeddedLocatorAdapter(
                    backend_name="embedded-empty",
                    runtime=RecordingEmbeddedRuntime(),
                ),
                artifact_root=Path(tmp_dir) / "empty-artifacts",
            )
            failed_harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=FailedLocatorAdapter(backend_name="embedded-failed"),
                artifact_root=Path(tmp_dir) / "failed-artifacts",
            )

            ready_report = ready_harness.evaluate(
                manifest=ready_manifest,
                parameter_groups=[EmbeddedLocatorParameterGroup(label="ready")],
            )
            empty_report = empty_harness.evaluate(
                manifest=empty_manifest,
                parameter_groups=[EmbeddedLocatorParameterGroup(label="empty")],
            )
            failed_report = failed_harness.evaluate(
                manifest=ready_manifest,
                parameter_groups=[EmbeddedLocatorParameterGroup(label="failed")],
            )

        self.assertEqual(ready_report.runs[0].samples[0].proposal_status, "ready")
        self.assertIsNone(ready_report.runs[0].samples[0].failure_category)
        self.assertEqual(empty_report.runs[0].samples[0].proposal_status, "empty")
        self.assertEqual(
            empty_report.runs[0].samples[0].failure_category,
            EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value,
        )
        self.assertEqual(failed_report.runs[0].samples[0].proposal_status, "failed")
        self.assertEqual(
            failed_report.runs[0].samples[0].failure_category,
            EmbeddedLocatorFailureCategory.RUNTIME_FAILED.value,
        )

    def test_parameter_sweep_comparison_prefers_higher_pass_count(self) -> None:
        runtime = RecordingEmbeddedRuntime()
        adapter = EmbeddedLocatorAdapter(
            backend_name="embedded-locator-sweep",
            runtime=runtime,
        )

        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(build_manifest_payload("sweep target")),
                encoding="utf-8",
            )
            manifest = load_embedded_locator_manifest(manifest_path)
            harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=adapter,
                artifact_root=Path(tmp_dir) / "artifacts",
            )

            report = harness.evaluate(
                manifest=manifest,
                parameter_groups=[
                    EmbeddedLocatorParameterGroup(
                        label="loose",
                        runtime_options={"abs_threshold": 0.4},
                    ),
                    EmbeddedLocatorParameterGroup(
                        label="strict",
                        runtime_options={"abs_threshold": 0.85},
                    ),
                ],
            )

        self.assertEqual(report.run_mode, "sweep")
        self.assertEqual(len(report.comparison.entries), 2)
        self.assertEqual(report.comparison.best_parameter_group_label, "loose")
        self.assertEqual(report.review.best_parameter_group_label, "loose")
        comparison_by_label = {
            entry.label: entry for entry in report.comparison.entries
        }
        self.assertEqual(comparison_by_label["loose"].passed_sample_count, 1)
        self.assertEqual(comparison_by_label["strict"].passed_sample_count, 0)
        self.assertEqual(
            comparison_by_label["strict"].failure_category_counts[
                EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value
            ],
            1,
        )

    def test_coarse_box_only_ready_result_is_classified(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(build_manifest_payload("truck")),
                encoding="utf-8",
            )
            manifest = load_embedded_locator_manifest(manifest_path)
            harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=CoarseBoxOnlyLocatorAdapter(
                    backend_name="embedded-coarse-only"
                ),
                artifact_root=Path(tmp_dir) / "artifacts",
            )

            report = harness.evaluate(
                manifest=manifest,
                parameter_groups=[EmbeddedLocatorParameterGroup(label="coarse-only")],
            )

        sample = report.runs[0].samples[0]
        self.assertEqual(
            sample.failure_category,
            EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value,
        )

    def test_best_run_tie_break_does_not_bias_toward_more_ready_with_negative_samples(self) -> None:
        mixed_manifest_payload = {
            "manifest_id": "embedded-locator-mixed-manifest",
            "samples": [
                {
                    "id": "positive-sample",
                    "image_path": str(ROOT / "old/example/truck.jpg"),
                    "query_text": "positive target",
                    "should_locate": True,
                    "acceptable_point_standard": "At least one point on the truck is acceptable.",
                    "acceptable_box_standard": "A coarse box around the truck is acceptable.",
                    "remark": "Positive sample.",
                    "minimum_points": 0,
                    "minimum_top_confidence": None,
                    "require_coarse_box": False,
                },
                {
                    "id": "negative-sample",
                    "image_path": str(ROOT / "old/example/truck.jpg"),
                    "query_text": "negative target",
                    "should_locate": False,
                    "acceptable_point_standard": "No target should be produced for this synthetic negative.",
                    "acceptable_box_standard": None,
                    "remark": "Negative sample.",
                    "minimum_points": 0,
                    "minimum_top_confidence": None,
                    "require_coarse_box": False,
                },
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "mixed-manifest.json"
            manifest_path.write_text(
                json.dumps(mixed_manifest_payload),
                encoding="utf-8",
            )
            manifest = load_embedded_locator_manifest(manifest_path)
            harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
                locator_adapter=MixedPolarityLocatorAdapter(backend_name="mixed-polarity"),
                artifact_root=Path(tmp_dir) / "artifacts",
            )

            report = harness.evaluate(
                manifest=manifest,
                parameter_groups=[
                    EmbeddedLocatorParameterGroup(
                        label="more-ready-but-worse",
                        runtime_options={"abs_threshold": 0.4},
                    ),
                    EmbeddedLocatorParameterGroup(
                        label="less-ready-but-better",
                        runtime_options={"abs_threshold": 0.8},
                    ),
                ],
            )

        comparison_by_label = {
            entry.label: entry for entry in report.comparison.entries
        }
        self.assertEqual(comparison_by_label["more-ready-but-worse"].passed_sample_count, 1)
        self.assertEqual(comparison_by_label["less-ready-but-better"].passed_sample_count, 1)
        self.assertEqual(report.comparison.best_parameter_group_label, "less-ready-but-better")
        self.assertEqual(report.review.best_parameter_group_label, "less-ready-but-better")
