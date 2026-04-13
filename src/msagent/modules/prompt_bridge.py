"""定义 Prompt Bridge 模块骨架。

Prompt Bridge 是 V1 新架构的关键层，负责把上游理解结果和 proposal
整理成 segmenter 可稳定消费的 PromptPackage。
跨模块共享的 prompt 类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.common import BaseModuleInput, BaseModuleOutput, DiagnosticMessage
from msagent.core.contracts.common import ModuleStatus
from msagent.core.contracts.types import (
    BoxPrompt,
    ExecutionHints,
    PointPrompt,
    PromptMetadata,
    PromptPackage,
    PromptTextBundle,
    ProposalResult,
    QueryUnderstandingResult,
    SpatialPromptBundle,
)
from msagent.infra.adapters import ArtifactKind, ArtifactStore


@dataclass(slots=True, kw_only=True)
class PromptBridgeModuleInput(BaseModuleInput):
    """Prompt Bridge 模块输入 DTO。"""

    understanding: QueryUnderstandingResult
    # query understanding 的权威实体对象，是本模块的正式业务输入。

    proposal: ProposalResult
    # proposal engine 的权威实体对象，是本模块的正式业务输入。

    raw_query: str
    # 用户原始 query，便于构造 raw_text prompt。


class PromptBridgeModule:
    """Prompt Bridge 模块接口。"""

    module_name: str = "prompt_bridge"
    # 模块稳定名称。

    def run(
        self, module_input: PromptBridgeModuleInput
    ) -> BaseModuleOutput[PromptPackage]:
        """将上游结构化结果转换为 PromptPackage。"""
        raise NotImplementedError


@dataclass(slots=True)
class RuleBasedPromptBridgeModule(PromptBridgeModule):
    """基于规则的 Prompt Bridge 默认实现骨架。"""

    artifact_store: ArtifactStore
    # 负责保存 prompt package 与桥接阶段中间产物。

    package_version: str = "v1"
    # 当前输出 PromptPackage 所采用的 schema 版本。

    module_name: str = "prompt_bridge"
    # 模块稳定名称。

    def run(
        self, module_input: PromptBridgeModuleInput
    ) -> BaseModuleOutput[PromptPackage]:
        """把 understanding 与 proposal 组织成 PromptPackage。"""
        primary_candidate = next(
            (
                candidate
                for candidate in module_input.proposal.candidates
                if candidate.candidate_id == module_input.proposal.primary_candidate_id
            ),
            module_input.proposal.candidates[0] if module_input.proposal.candidates else None,
        )
        if primary_candidate is None:
            return BaseModuleOutput(
                module_name=self.module_name,
                status=ModuleStatus.EMPTY,
                consumed_refs=list(module_input.upstream_refs),
                diagnostics=[
                    DiagnosticMessage(
                        level="warning",
                        message="Prompt bridge received a proposal with no candidate.",
                        code="empty_proposal_candidates",
                    )
                ],
            )

        boxes: list[BoxPrompt] = []
        if primary_candidate.region_box is not None:
            boxes.append(
                BoxPrompt(
                    x1=primary_candidate.region_box.x1,
                    y1=primary_candidate.region_box.y1,
                    x2=primary_candidate.region_box.x2,
                    y2=primary_candidate.region_box.y2,
                    confidence=primary_candidate.confidence,
                    source="proposal.primary_candidate",
                    note="primary proposal box",
                )
            )

        positive_points = [
            PointPrompt(
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                source="proposal.positive_point_hint",
                note=point.reason,
            )
            for point in primary_candidate.positive_point_hints
        ]
        negative_points = [
            PointPrompt(
                x=point.x,
                y=point.y,
                confidence=point.confidence,
                source="proposal.negative_point_hint",
                note=point.reason,
            )
            for point in primary_candidate.negative_point_hints
        ]

        strategy_tags = ["normalized_text"]
        if boxes:
            strategy_tags.append("box")
        if positive_points:
            strategy_tags.append("positive_points")
        if negative_points:
            strategy_tags.append("negative_points")

        payload = PromptPackage(
            package_id=f"{module_input.task_id}-prompt-{module_input.attempt_index}",
            package_version=self.package_version,
            text_prompts=PromptTextBundle(
                normalized_text=module_input.understanding.normalized_query,
                raw_text=module_input.raw_query,
                focus_terms=list(module_input.understanding.focus_terms),
            ),
            spatial_prompts=SpatialPromptBundle(
                positive_points=positive_points,
                negative_points=negative_points,
                boxes=boxes,
                coarse_region_ref=primary_candidate.coarse_region_ref,
            ),
            metadata=PromptMetadata(
                produced_from_route=module_input.proposal.route,
                source_refs=list(module_input.upstream_refs),
                confidence=primary_candidate.confidence,
                strategy_tags=strategy_tags,
                notes=[
                    f"primary_candidate_id={primary_candidate.candidate_id}",
                    *[hint.hint_type for hint in module_input.proposal.bridge_hints],
                ],
            ),
            execution_hints=ExecutionHints(
                multimask=False,
                crop_to_box=bool(boxes),
                preferred_prompt_order=[
                    "box",
                    "positive_points",
                    "negative_points",
                    "text",
                ],
                return_top_k=1,
                fallback_to_text_only=not boxes and not positive_points,
            ),
        )
        artifact_ref = self.artifact_store.save_artifact(
            ArtifactKind.PROMPT_PACKAGE,
            payload,
        )
        artifact_ref.attempt_index = module_input.attempt_index
        artifact_ref.summary = payload.text_prompts.normalized_text
        return BaseModuleOutput(
            module_name=self.module_name,
            status=ModuleStatus.SUCCESS,
            primary_payload=payload,
            artifact_ref=artifact_ref,
            consumed_refs=list(module_input.upstream_refs),
        )
