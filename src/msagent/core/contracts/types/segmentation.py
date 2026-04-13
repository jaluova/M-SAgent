"""定义 Segmenter 相关公共类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from msagent.core.contracts.common import ArtifactRef


class SegmentationStatus(StrEnum):
    """分割结果状态。"""

    READY = "ready"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(slots=True)
class SegmentationCandidate:
    """单个 mask 候选结果。"""

    candidate_id: str
    mask_ref: ArtifactRef
    score: float | None = None
    preview_ref: ArtifactRef | None = None
    prompt_summary: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SegmentationResult:
    """Segmenter 模块的主输出对象。"""

    segmentation_id: str
    status: SegmentationStatus
    result_summary: str
    candidates: list[SegmentationCandidate] = field(default_factory=list)
    primary_candidate_id: str | None = None
    prompt_package_ref: ArtifactRef | None = None
    diagnostics: list[str] = field(default_factory=list)

