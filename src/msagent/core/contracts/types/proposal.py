"""定义 Proposal Engine 相关公共类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.common import ArtifactRef
from msagent.core.enum_compat import StrEnum


class ProposalRoute(StrEnum):
    """proposal 所属 route。"""

    LOCATE = "locate"
    CROP = "crop"
    REWRITE = "rewrite"


class ProposalStatus(StrEnum):
    """proposal 执行状态。"""

    READY = "ready"
    EMPTY = "empty"
    FAILED = "failed"


class ProposalQualityFlag(StrEnum):
    """proposal 质量风险标记。"""

    LOW_CONFIDENCE = "low_confidence"
    LOW_DIVERSITY = "low_diversity"
    DUPLICATE_RISK = "duplicate_risk"
    RELATION_UNRESOLVED = "relation_unresolved"
    SMALL_TARGET_RISK = "small_target_risk"


@dataclass(slots=True)
class NormalizedBox:
    """归一化图像坐标系下的矩形框。"""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(slots=True)
class PointHint:
    """proposal 阶段的建议点提示。"""

    x: float
    y: float
    confidence: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class ProposalBridgeHint:
    """从 proposal 传递给 prompt bridge 的弱提示。"""

    hint_type: str
    reason: str


@dataclass(slots=True)
class ProposalCandidate:
    """单个空间先验候选。"""

    candidate_id: str
    rank: int
    confidence: float | None = None
    matched_clues: list[str] = field(default_factory=list)
    region_box: NormalizedBox | None = None
    positive_point_hints: list[PointHint] = field(default_factory=list)
    negative_point_hints: list[PointHint] = field(default_factory=list)
    coarse_region_ref: ArtifactRef | None = None
    rationale: str | None = None
    limitations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProposalResult:
    """Proposal Engine 模块的主输出对象。"""

    proposal_id: str
    route: ProposalRoute
    status: ProposalStatus
    proposal_summary: str
    candidates: list[ProposalCandidate] = field(default_factory=list)
    primary_candidate_id: str | None = None
    matched_text_clues: list[str] = field(default_factory=list)
    bridge_hints: list[ProposalBridgeHint] = field(default_factory=list)
    quality_flags: list[ProposalQualityFlag] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    source_refs: list[ArtifactRef] = field(default_factory=list)
