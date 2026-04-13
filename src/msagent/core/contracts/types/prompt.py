"""定义 Prompt Bridge 相关公共类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.common import ArtifactRef
from msagent.core.contracts.types.proposal import ProposalRoute


@dataclass(slots=True)
class PromptTextBundle:
    """给 segmenter 的文本提示集合。"""

    normalized_text: str
    raw_text: str | None = None
    rewritten_text: str | None = None
    focus_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PointPrompt:
    """最终交给 segmenter 的点提示。"""

    x: float
    y: float
    confidence: float | None = None
    source: str | None = None
    note: str | None = None


@dataclass(slots=True)
class BoxPrompt:
    """最终交给 segmenter 的框提示。"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None
    source: str | None = None
    note: str | None = None


@dataclass(slots=True)
class SpatialPromptBundle:
    """给 segmenter 的空间提示集合。"""

    positive_points: list[PointPrompt] = field(default_factory=list)
    negative_points: list[PointPrompt] = field(default_factory=list)
    boxes: list[BoxPrompt] = field(default_factory=list)
    coarse_region_ref: ArtifactRef | None = None
    coarse_mask_ref: ArtifactRef | None = None


@dataclass(slots=True)
class PromptMetadata:
    """PromptPackage 的来源和解释信息。"""

    produced_from_route: ProposalRoute
    source_refs: list[ArtifactRef] = field(default_factory=list)
    confidence: float | None = None
    strategy_tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionHints:
    """segmenter 执行偏好。"""

    multimask: bool | None = None
    crop_to_box: bool | None = None
    preferred_prompt_order: list[str] = field(default_factory=list)
    return_top_k: int | None = None
    fallback_to_text_only: bool | None = None


@dataclass(slots=True)
class PromptPackage:
    """Prompt Bridge 模块的主输出对象。"""

    package_id: str
    package_version: str
    text_prompts: PromptTextBundle
    spatial_prompts: SpatialPromptBundle
    metadata: PromptMetadata
    execution_hints: ExecutionHints | None = None

