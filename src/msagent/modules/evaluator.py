"""定义 Evaluator 模块骨架。

该模块负责判断当前分割结果是否可以接受，并返回结构化失败类型，
从而支撑 V1 的 failure-aware retry。
跨模块共享的评估类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.common import BaseModuleInput, BaseModuleOutput
from msagent.core.contracts.types import EvaluationResult, QueryUnderstandingResult, SegmentationResult
from msagent.infra.adapters import ArtifactStore, LLMAdapter


@dataclass(slots=True)
class EvaluatorModuleInput(BaseModuleInput):
    """Evaluator 模块输入 DTO。"""

    raw_query: str = ""
    # 用户原始 query，便于对齐 prompt 与结果语义。

    understanding: QueryUnderstandingResult | None = None
    # query understanding 的权威实体对象，可辅助解释失败原因。

    segmentation: SegmentationResult | None = None
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
        raise NotImplementedError
