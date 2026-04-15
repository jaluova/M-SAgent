"""把 RunTask 与 artifacts 转成可阅读的 demo 报告。"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.common import ArtifactKind, ArtifactRef
from msagent.core.contracts.types import (
    EvaluationResult,
    PromptPackage,
    ProposalResult,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.core.task.models import RunTask
from msagent.infra.adapters import ArtifactStore
from msagent.orchestrator.orchestrator import OrchestrationResult


@dataclass(slots=True)
class DemoAttemptSummary:
    """单轮尝试的人类可读摘要。"""

    attempt_index: int
    route: str
    verdict: str | None = None
    failure_type: str | None = None
    query_summary: str | None = None
    proposal_summary: str | None = None
    prompt_summary: str | None = None
    segmentation_summary: str | None = None
    evaluation_summary: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DemoTaskReport:
    """一次任务运行的 demo 报告快照。"""

    task_id: str
    source: str
    image_uri: str
    raw_query: str
    status: str
    stage: str
    attempt_count: int
    max_attempts: int
    active_route: str | None = None
    detected_language: str | None = None
    stop_reason: str | None = None
    final_verdict: str | None = None
    final_summary: str | None = None
    attempts: list[DemoAttemptSummary] = field(default_factory=list)
    load_warnings: list[str] = field(default_factory=list)


def build_demo_task_report(
    result: OrchestrationResult,
    *,
    artifact_store: ArtifactStore,
) -> DemoTaskReport:
    """从 orchestrator 结果和 artifact store 构造 demo 报告。"""
    task = result.task
    report = DemoTaskReport(
        task_id=task.identity.task_id,
        source=task.identity.source.value,
        image_uri=task.request.image_ref.uri,
        raw_query=task.request.raw_query,
        status=task.runtime.status.value,
        stage=task.runtime.stage.value,
        attempt_count=len(task.attempt_history),
        max_attempts=task.runtime.max_attempts,
        active_route=task.runtime.active_route.value if task.runtime.active_route else None,
        detected_language=(
            task.normalized_input.detected_language
            if task.normalized_input is not None
            else None
        ),
        stop_reason=task.result.stop_reason.value if task.result.stop_reason else None,
        final_verdict=task.result.final_verdict.value if task.result.final_verdict else None,
        final_summary=task.result.final_summary,
    )
    for attempt in task.attempt_history:
        report.attempts.append(
            DemoAttemptSummary(
                attempt_index=attempt.attempt_index,
                route=attempt.route.value,
                verdict=attempt.verdict.value if attempt.verdict else None,
                failure_type=attempt.failure_type.value if attempt.failure_type else None,
                query_summary=_build_query_summary(
                    _load_artifact(
                        artifact_store,
                        attempt.query_understanding_ref,
                        QueryUnderstandingResult,
                        report.load_warnings,
                    )
                ),
                proposal_summary=_build_proposal_summary(
                    _load_artifact(
                        artifact_store,
                        attempt.proposal_ref,
                        ProposalResult,
                        report.load_warnings,
                    )
                ),
                prompt_summary=_build_prompt_summary(
                    _load_artifact(
                        artifact_store,
                        attempt.prompt_package_ref,
                        PromptPackage,
                        report.load_warnings,
                    )
                ),
                segmentation_summary=_build_segmentation_summary(
                    _load_artifact(
                        artifact_store,
                        attempt.segmentation_ref,
                        SegmentationResult,
                        report.load_warnings,
                    )
                ),
                evaluation_summary=_build_evaluation_summary(
                    _load_artifact(
                        artifact_store,
                        attempt.evaluation_ref,
                        EvaluationResult,
                        report.load_warnings,
                    )
                ),
                notes=list(attempt.notes),
            )
        )
    return report


def render_demo_task_report_markdown(report: DemoTaskReport) -> str:
    """把 demo 报告快照渲染成 Markdown。"""
    lines = [
        "# M-SAgent Demo Report",
        "",
        "## Task",
        f"- Task ID: `{report.task_id}`",
        f"- Source: `{report.source}`",
        f"- Image: `{report.image_uri}`",
        f"- Query: `{report.raw_query}`",
        f"- Status: `{report.status}`",
        f"- Stage: `{report.stage}`",
        f"- Attempts: `{report.attempt_count}` / `{report.max_attempts}`",
    ]
    if report.active_route is not None:
        lines.append(f"- Active route: `{report.active_route}`")
    if report.detected_language is not None:
        lines.append(f"- Detected language: `{report.detected_language}`")
    if report.stop_reason is not None:
        lines.append(f"- Stop reason: `{report.stop_reason}`")
    if report.final_verdict is not None:
        lines.append(f"- Final verdict: `{report.final_verdict}`")
    if report.final_summary is not None:
        lines.append(f"- Final summary: {report.final_summary}")

    lines.extend(["", "## Attempts"])
    if not report.attempts:
        lines.append("- No attempts recorded.")
    for attempt in report.attempts:
        lines.extend(
            [
                "",
                f"### Attempt {attempt.attempt_index}",
                f"- Route: `{attempt.route}`",
            ]
        )
        if attempt.verdict is not None:
            lines.append(f"- Verdict: `{attempt.verdict}`")
        if attempt.failure_type is not None:
            lines.append(f"- Failure type: `{attempt.failure_type}`")
        if attempt.query_summary is not None:
            lines.append(f"- Query understanding: {attempt.query_summary}")
        if attempt.proposal_summary is not None:
            lines.append(f"- Proposal: {attempt.proposal_summary}")
        if attempt.prompt_summary is not None:
            lines.append(f"- Prompt package: {attempt.prompt_summary}")
        if attempt.segmentation_summary is not None:
            lines.append(f"- Segmentation: {attempt.segmentation_summary}")
        if attempt.evaluation_summary is not None:
            lines.append(f"- Evaluation: {attempt.evaluation_summary}")
        for note in attempt.notes:
            lines.append(f"- Note: {note}")

    if report.load_warnings:
        lines.extend(["", "## Load Warnings"])
        for warning in report.load_warnings:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def _load_artifact(
    artifact_store: ArtifactStore,
    artifact_ref: ArtifactRef | None,
    expected_type: type[object],
    warnings: list[str],
) -> object | None:
    if artifact_ref is None:
        return None
    try:
        return artifact_store.load_artifact(artifact_ref, expected_type)
    except Exception as exc:
        warnings.append(
            f"artifact `{artifact_ref.artifact_id}` could not be loaded as "
            f"`{expected_type.__name__}`: {type(exc).__name__}"
        )
        return None


def _build_query_summary(payload: object | None) -> str | None:
    if not isinstance(payload, QueryUnderstandingResult):
        return None
    return (
        f"`{payload.normalized_query}` -> {payload.target_type.value}, "
        f"implicitness={payload.implicitness.value}, "
        f"focus_terms={len(payload.focus_terms)}"
    )


def _build_proposal_summary(payload: object | None) -> str | None:
    if not isinstance(payload, ProposalResult):
        return None
    return (
        f"status={payload.status.value}, route={payload.route.value}, "
        f"candidates={len(payload.candidates)}, summary={payload.proposal_summary}"
    )


def _build_prompt_summary(payload: object | None) -> str | None:
    if not isinstance(payload, PromptPackage):
        return None
    return (
        f"version={payload.package_version}, boxes={len(payload.spatial_prompts.boxes)}, "
        f"positive_points={len(payload.spatial_prompts.positive_points)}, "
        f"negative_points={len(payload.spatial_prompts.negative_points)}, "
        f"strategy_tags={','.join(payload.metadata.strategy_tags) or 'none'}"
    )


def _build_segmentation_summary(payload: object | None) -> str | None:
    if not isinstance(payload, SegmentationResult):
        return None
    return (
        f"status={payload.status.value}, candidates={len(payload.candidates)}, "
        f"summary={payload.result_summary}"
    )


def _build_evaluation_summary(payload: object | None) -> str | None:
    if not isinstance(payload, EvaluationResult):
        return None
    failure_text = payload.failure_type.value if payload.failure_type else "none"
    return (
        f"verdict={payload.verdict.value}, failure_type={failure_text}, "
        f"summary={payload.summary}"
    )


__all__ = [
    "DemoAttemptSummary",
    "DemoTaskReport",
    "build_demo_task_report",
    "render_demo_task_report_markdown",
]
