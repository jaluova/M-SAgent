"""定义失败感知重试策略的骨架。

文档明确要求有限重试由 orchestrator 规则主导，因此这里提供策略对象，
但不在当前阶段写入任何具体算法。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.types import EvaluationVerdict, ProposalRoute
from msagent.core.task.enums import StopReason
from msagent.core.task.models import RunTask


@dataclass(slots=True)
class RetryDecision:
    """一次评估后的重试决策结果。"""

    should_retry: bool
    # 是否允许继续进入下一轮。

    next_route: ProposalRoute | None = None
    # 若允许重试，下一轮建议走的 route。

    reason: str | None = None
    # 对本次决策的说明，便于 trace 和调试。

    policy_tags: list[str] = field(default_factory=list)
    # 命中的策略标签，例如 "failure-aware"、"max-attempt-guard"。


@dataclass(slots=True)
class RetryPolicy:
    """负责封装有限重试规则的策略对象。"""

    default_route: ProposalRoute = ProposalRoute.LOCATE
    # 第一轮默认采用的 route；V1 以 locate 为主。

    allow_failure_aware_retry: bool = True
    # 是否启用按失败类型驱动的重试规则。

    max_attempts_fallback_route: ProposalRoute | None = None
    # 当达到边界前需要最后一次保守尝试时预留的 route。

    def choose_initial_route(self, task: RunTask) -> ProposalRoute:
        """根据任务状态选择初始 route。"""
        return self.default_route

    def decide_retry(self, task: RunTask) -> RetryDecision:
        """根据任务当前状态和最近评估结果决定是否重试。"""
        if not task.attempt_history:
            return RetryDecision(
                should_retry=False,
                reason="No attempt has been executed yet.",
                policy_tags=["no-attempt-history"],
            )

        latest_attempt = task.attempt_history[-1]
        if latest_attempt.verdict is EvaluationVerdict.ACCEPT:
            return RetryDecision(
                should_retry=False,
                reason="Latest evaluation accepted the result.",
                policy_tags=["accepted"],
            )

        if len(task.attempt_history) >= task.runtime.max_attempts:
            return RetryDecision(
                should_retry=False,
                reason="Maximum attempts reached.",
                policy_tags=["max-attempt-guard", StopReason.MAX_ATTEMPTS_REACHED.value],
            )

        return RetryDecision(
            should_retry=True,
            next_route=latest_attempt.route,
            reason="Retry allowed because the latest evaluation rejected the result.",
            policy_tags=["simple-retry"],
        )
