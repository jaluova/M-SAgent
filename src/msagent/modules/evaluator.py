"""定义 Evaluator 模块骨架。

该模块负责判断当前分割结果是否可以接受，并返回结构化失败类型，
从而支撑 V1 的 failure-aware retry。
跨模块共享的评估类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from msagent.core.contracts.adapter_requests import EvaluationAdapterRequest
from msagent.core.contracts.common import ArtifactKind, BaseModuleInput, BaseModuleOutput
from msagent.core.contracts.common import ModuleStatus
from msagent.core.contracts.types import (
    EvaluationIssue,
    EvaluationResult,
    EvaluationVerdict,
    FailureType,
    PromptPackage,
    ProposalResult,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.infra.adapters import ArtifactStore, LLMAdapter
from msagent.infra.mask_artifact import MaskArtifact


@dataclass(slots=True, kw_only=True)
class EvaluatorModuleInput(BaseModuleInput):
    """Evaluator 模块输入 DTO。"""

    raw_query: str
    # 用户原始 query，便于对齐 prompt 与结果语义。

    understanding: QueryUnderstandingResult | None = None
    # query understanding 的权威实体对象，可辅助解释失败原因。

    proposal: ProposalResult | None = None
    # 本轮进入分割前的 proposal，可辅助分析 prompt mismatch。

    prompt_package: PromptPackage
    # 实际喂给 segmenter 的 PromptPackage，是 evaluator 判断 prompt mismatch 的关键输入。

    segmentation: SegmentationResult
    # 当前要评估的权威结构化分割结果。


class EvaluatorModule:
    """Evaluator 模块接口。"""

    module_name: str = "evaluator"
    # 模块稳定名称。

    def run(self, module_input: EvaluatorModuleInput) -> BaseModuleOutput[EvaluationResult]:
        """执行评估并返回结构化 verdict 与 failure type。"""
        raise NotImplementedError


@dataclass(slots=True)
class LLMEvaluatorModule(EvaluatorModule):
    """基于 LLM 后端的 Evaluator 默认实现骨架。"""

    llm_adapter: LLMAdapter
    # 负责执行 accept / reject 与 failure taxonomy 判断的后端。

    artifact_store: ArtifactStore
    # 负责保存 evaluation 结果及其引用。

    module_name: str = "evaluator"
    # 模块稳定名称。

    def run(self, module_input: EvaluatorModuleInput) -> BaseModuleOutput[EvaluationResult]:
        """调用评估后端并产出结构化 EvaluationResult。"""
        primary_mask = self._load_primary_mask(module_input.segmentation)
        mask_quality = self._assess_primary_mask(primary_mask)
        if mask_quality.hard_reject_summary is not None:
            payload = self._build_mask_quality_rejection(
                task_id=module_input.task_id,
                summary=mask_quality.hard_reject_summary,
                issues=mask_quality.issues,
            )
        else:
            payload = self.llm_adapter.run_evaluation(
                EvaluationAdapterRequest(
                    task_id=module_input.task_id,
                    raw_query=module_input.raw_query,
                    segmentation=module_input.segmentation,
                    prompt_package=module_input.prompt_package,
                    proposal=module_input.proposal,
                    understanding=module_input.understanding,
                    primary_mask_summary=mask_quality.summary,
                    mask_quality_warnings=list(mask_quality.warnings),
                    trace_context=module_input.trace_context,
                )
            )
            payload = self._apply_mask_quality_postcheck(payload, mask_quality)
        artifact_ref = self.artifact_store.save_artifact(
            ArtifactKind.EVALUATION_RESULT,
            payload,
        )
        artifact_ref.attempt_index = module_input.attempt_index
        artifact_ref.summary = payload.summary
        return BaseModuleOutput(
            module_name=self.module_name,
            status=ModuleStatus.SUCCESS,
            primary_payload=payload,
            artifact_ref=artifact_ref,
            consumed_refs=list(module_input.upstream_refs),
        )

    def _load_primary_mask(self, segmentation: SegmentationResult) -> MaskArtifact | None:
        primary_candidate = next(
            (
                candidate
                for candidate in segmentation.candidates
                if candidate.candidate_id == segmentation.primary_candidate_id
            ),
            segmentation.candidates[0] if segmentation.candidates else None,
        )
        if primary_candidate is None:
            return None
        return self.artifact_store.load_artifact(primary_candidate.mask_ref, MaskArtifact)

    def _assess_primary_mask(self, mask: MaskArtifact | None) -> "_MaskQualityAssessment":
        if mask is None:
            return _MaskQualityAssessment(
                summary="primary mask unavailable",
                warnings=["primary mask could not be loaded for evaluator geometry checks"],
            )

        pixel_area = mask.pixel_area
        if pixel_area is None:
            pixel_area = sum(1 for row in mask.mask_bitmap for value in row if value)
        total_pixels = mask.width * mask.height
        coverage_ratio = (
            float(pixel_area) / float(total_pixels) if total_pixels > 0 else None
        )
        box_width = max(0.0, mask.active_box.x2 - mask.active_box.x1)
        box_height = max(0.0, mask.active_box.y2 - mask.active_box.y1)
        box_area = box_width * box_height
        active_point_count = len(mask.active_points)

        summary = (
            f"primary mask stats: pixel_area={pixel_area}, total_pixels={total_pixels}, "
            f"coverage_ratio={_format_ratio(coverage_ratio)}, "
            f"box_area={box_area:.6f}, active_points={active_point_count}, "
            f"active_box=({mask.active_box.x1:.4f}, {mask.active_box.y1:.4f}, "
            f"{mask.active_box.x2:.4f}, {mask.active_box.y2:.4f})"
        )
        warnings: list[str] = []
        issues: list[EvaluationIssue] = []

        if pixel_area <= 0:
            issues.append(
                EvaluationIssue(
                    issue_type="empty_mask",
                    summary="Primary segmentation mask has no active pixels.",
                    severity="high",
                )
            )
            return _MaskQualityAssessment(
                summary=summary,
                warnings=warnings,
                issues=issues,
                hard_reject_summary=(
                    "Evaluator rejected the primary segmentation candidate because the "
                    "mask is empty after geometry validation."
                ),
            )

        if pixel_area == 1:
            issues.append(
                EvaluationIssue(
                    issue_type="single_pixel_mask",
                    summary="Primary segmentation mask collapsed to a single active pixel.",
                    severity="high",
                )
            )
            return _MaskQualityAssessment(
                summary=summary,
                warnings=warnings,
                issues=issues,
                hard_reject_summary=(
                    "Evaluator rejected the primary segmentation candidate because the "
                    "mask collapsed to a single active pixel."
                ),
            )

        if pixel_area <= 4 and box_area <= 0.0001 and active_point_count <= 1:
            issues.append(
                EvaluationIssue(
                    issue_type="degenerate_tiny_mask",
                    summary=(
                        "Primary segmentation mask is extremely small relative to its "
                        "normalized box and prompt support."
                    ),
                    severity="high",
                )
            )
            return _MaskQualityAssessment(
                summary=summary,
                warnings=warnings,
                issues=issues,
                hard_reject_summary=(
                    "Evaluator rejected the primary segmentation candidate because the "
                    "mask is a degenerate tiny fragment rather than a usable object region."
                ),
            )

        if (
            coverage_ratio is not None
            and coverage_ratio <= 0.00005
            and box_area <= 0.0005
            and active_point_count <= 2
        ):
            warnings.append(
                "primary mask is near-empty relative to the image and should not be "
                "accepted without strong visual evidence"
            )
            issues.append(
                EvaluationIssue(
                    issue_type="near_empty_mask",
                    summary=(
                        "Primary segmentation mask is unusually small and may not cover "
                        "the intended referent."
                    ),
                    severity="medium",
                )
            )

        return _MaskQualityAssessment(
            summary=summary,
            warnings=warnings,
            issues=issues,
        )

    def _build_mask_quality_rejection(
        self,
        *,
        task_id: str,
        summary: str,
        issues: list[EvaluationIssue],
    ) -> EvaluationResult:
        return EvaluationResult(
            evaluation_id=f"{task_id}-evaluation",
            verdict=EvaluationVerdict.REJECT,
            summary=summary,
            failure_type=FailureType.PARTIAL_MASK,
            confidence=0.0,
            issues=list(issues),
            retry_hints=["retry_with_same_route"],
        )

    def _apply_mask_quality_postcheck(
        self,
        payload: EvaluationResult,
        assessment: "_MaskQualityAssessment",
    ) -> EvaluationResult:
        if (
            payload.verdict is EvaluationVerdict.ACCEPT
            and assessment.warnings
        ):
            issues = [*payload.issues, *assessment.issues]
            warning_text = "; ".join(assessment.warnings)
            return replace(
                payload,
                verdict=EvaluationVerdict.REVIEW,
                summary=f"{payload.summary} Geometry guard requested review: {warning_text}.",
                failure_type=payload.failure_type or FailureType.PARTIAL_MASK,
                accepted_candidate_id=None,
                accepted_mask_ref=None,
                issues=issues,
                retry_hints=payload.retry_hints or ["retry_with_same_route"],
            )
        if assessment.issues:
            return replace(
                payload,
                issues=[*payload.issues, *assessment.issues],
            )
        return payload


@dataclass(slots=True)
class _MaskQualityAssessment:
    summary: str
    warnings: list[str]
    issues: list[EvaluationIssue]
    hard_reject_summary: str | None = None


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.8f}"
