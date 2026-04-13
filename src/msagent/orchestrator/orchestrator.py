"""定义 M-SAgent V1 的唯一主控制器骨架。

本文件严格落实架构文档中的核心原则：

- `Orchestrator` 是唯一允许推进阶段和决定下一步动作的模块；
- 其他模块只消费输入、产出结果，不直接操控全局流程；
- 有限重试和失败感知重试都由这里统一串接。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.common import BaseModuleOutput
from msagent.core.contracts.types import (
    EvaluationResult,
    PromptPackage,
    ProposalResult,
    ProposalRoute,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.core.policies.retry_policy import RetryDecision, RetryPolicy
from msagent.core.task.models import RunTask
from msagent.modules.evaluator import EvaluatorModule
from msagent.modules.prompt_bridge import PromptBridgeModule
from msagent.modules.proposal_engine import ProposalEngineModule
from msagent.modules.query_understanding import QueryUnderstandingModule
from msagent.modules.segmenter import SegmenterModule


@dataclass(slots=True)
class AttemptExecutionResult:
    """单轮 orchestrator 执行结果。

    orchestrator 是系统总调度台，因此这里按步骤显式保存每个模块的输出，
    而不是重新把所有结果抹平成 `BaseModuleOutput[object]`。
    """

    query_understanding_output: BaseModuleOutput[QueryUnderstandingResult] | None = None
    # Query Understanding 阶段的结构化输出。

    proposal_output: BaseModuleOutput[ProposalResult] | None = None
    # Proposal Engine 阶段的结构化输出。

    prompt_bridge_output: BaseModuleOutput[PromptPackage] | None = None
    # Prompt Bridge 阶段的结构化输出。

    segmentation_output: BaseModuleOutput[SegmentationResult] | None = None
    # Segmenter 阶段的结构化输出。

    evaluation_output: BaseModuleOutput[EvaluationResult] | None = None
    # Evaluator 阶段的结构化输出。


@dataclass(slots=True)
class OrchestratorDependencies:
    """orchestrator 运行所需的模块依赖集合。"""

    query_understanding_module: QueryUnderstandingModule
    # 负责轻量语义理解，产出 QueryUnderstandingResult。

    proposal_engine_module: ProposalEngineModule
    # 负责生成 locate / crop / rewrite 等 route proposal。

    prompt_bridge_module: PromptBridgeModule
    # 负责把理解结果与 proposal 转成 PromptPackage。

    segmenter_module: SegmenterModule
    # 负责执行分割并产出结构化分割结果。

    evaluator_module: EvaluatorModule
    # 负责给出 accept / reject 及失败原因。

    retry_policy: RetryPolicy
    # 负责封装有限重试与失败感知重试规则。


@dataclass(slots=True)
class OrchestrationResult:
    """一次 orchestrator 运行后的返回骨架。"""

    task: RunTask
    # 运行结束后的完整任务账本快照。

    last_attempt_result: AttemptExecutionResult | None = None
    # 最近一轮完整或部分执行的强类型结果快照。


class Orchestrator:
    """M-SAgent V1 的总调度员。"""

    def __init__(self, dependencies: OrchestratorDependencies) -> None:
        self.dependencies = dependencies
        # 所有模块与策略依赖都集中挂在这里，避免散落跨层调用。

    def run(self, task: RunTask) -> OrchestrationResult:
        """驱动单个任务完成完整的 V1 主流程。"""
        raise NotImplementedError

    def choose_initial_route(self, task: RunTask) -> ProposalRoute:
        """选择任务的首轮 route。"""
        raise NotImplementedError

    def run_single_attempt(self, task: RunTask) -> AttemptExecutionResult:
        """执行单轮理解、proposal、bridge、segment 和 evaluate 流程。"""
        raise NotImplementedError

    def apply_retry_decision(self, task: RunTask, decision: RetryDecision) -> None:
        """将重试决策回写到任务运行时状态中。"""
        raise NotImplementedError
