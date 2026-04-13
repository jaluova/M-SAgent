"""定义适配器层的结构化请求对象。

这些请求类型位于公共 contract 层，目的是让 infra adapter 的输入输出
也尽量保持显式，而不是重新退化成松散的 Any。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.types import (
    PromptPackage,
    ProposalResult,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.core.trace.models import TraceContext


@dataclass(slots=True)
class QueryUnderstandingAdapterRequest:
    """给理解后端的结构化请求。"""

    task_id: str
    raw_query: str
    user_context_text: str | None = None
    image_uri: str | None = None
    trace_context: TraceContext | None = None


@dataclass(slots=True)
class LocateAdapterRequest:
    """给定位后端的结构化请求。"""

    task_id: str
    understanding: QueryUnderstandingResult
    trace_context: TraceContext | None = None
    options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentAdapterRequest:
    """给分割后端的结构化请求。"""

    task_id: str
    image_uri: str
    prompt_package: PromptPackage
    trace_context: TraceContext | None = None


@dataclass(slots=True)
class EvaluationAdapterRequest:
    """给评估后端的结构化请求。"""

    task_id: str
    raw_query: str
    segmentation: SegmentationResult
    prompt_package: PromptPackage
    proposal: ProposalResult | None = None
    understanding: QueryUnderstandingResult | None = None
    trace_context: TraceContext | None = None
