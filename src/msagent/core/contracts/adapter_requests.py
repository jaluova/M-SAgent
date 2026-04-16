"""定义适配器层的结构化请求对象。

这些请求类型位于公共 contract 层，目的是让 infra adapter 的输入输出
也尽量保持显式，而不是重新退化成松散的 Any。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.common import ImageRef
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
    image_ref: ImageRef | None = None
    image_uri: str | None = None
    trace_context: TraceContext | None = None
    options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """维护 image_ref / image_uri 的单一真相源约束。"""
        if self.image_ref is not None and self.image_uri is not None:
            if self.image_ref.uri != self.image_uri:
                raise ValueError(
                    "LocateAdapterRequest received conflicting image_ref.uri and image_uri: "
                    f"{self.image_ref.uri!r} != {self.image_uri!r}"
                )
        if self.image_ref is not None and self.image_uri is None:
            self.image_uri = self.image_ref.uri

    def resolved_image_ref(self) -> ImageRef | None:
        """返回优先使用 image_ref 的统一图像引用。"""
        if self.image_ref is not None:
            return self.image_ref
        if self.image_uri is None:
            return None
        return ImageRef(uri=self.image_uri)

    def resolved_image_uri(self) -> str | None:
        """兼容仍只需要 URI 的定位后端。"""
        image_ref = self.resolved_image_ref()
        if image_ref is None:
            return None
        return image_ref.uri


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
    primary_mask_summary: str | None = None
    mask_quality_warnings: list[str] = field(default_factory=list)
    trace_context: TraceContext | None = None
