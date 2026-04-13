"""定义 Evaluator 相关公共类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from msagent.core.contracts.common import ArtifactRef


class EvaluationVerdict(StrEnum):
    """评估结论。"""

    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"


class FailureType(StrEnum):
    """V1 第一版失败类型。"""

    LOCALIZATION_ERROR = "localization_error"
    PARTIAL_MASK = "partial_mask"
    WRONG_INSTANCE = "wrong_instance"
    PROMPT_MISMATCH = "prompt_mismatch"


@dataclass(slots=True)
class EvaluationIssue:
    """评估阶段识别出的具体问题项。"""

    issue_type: str
    summary: str
    severity: str | None = None


@dataclass(slots=True)
class EvaluationResult:
    """Evaluator 模块的主输出对象。"""

    evaluation_id: str
    verdict: EvaluationVerdict
    summary: str
    failure_type: FailureType | None = None
    accepted_candidate_id: str | None = None
    accepted_mask_ref: ArtifactRef | None = None
    confidence: float | None = None
    issues: list[EvaluationIssue] = field(default_factory=list)
    retry_hints: list[str] = field(default_factory=list)

