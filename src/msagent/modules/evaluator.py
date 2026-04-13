"""定义 Evaluator 模块骨架。

该模块负责判断当前分割结果是否可以接受，并返回结构化失败类型，
从而支撑 V1 的 failure-aware retry。
跨模块共享的评估类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.adapter_requests import EvaluationAdapterRequest
from msagent.core.contracts.common import ArtifactKind, BaseModuleInput, BaseModuleOutput
from msagent.core.contracts.common import ModuleStatus
from msagent.core.contracts.types import (
    EvaluationResult,
    PromptPackage,
    ProposalResult,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.infra.adapters import ArtifactStore, LLMAdapter


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
        payload = self.llm_adapter.run_evaluation(
            EvaluationAdapterRequest(
                task_id=module_input.task_id,
                raw_query=module_input.raw_query,
                segmentation=module_input.segmentation,
                prompt_package=module_input.prompt_package,
                proposal=module_input.proposal,
                understanding=module_input.understanding,
                trace_context=module_input.trace_context,
            )
        )
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
