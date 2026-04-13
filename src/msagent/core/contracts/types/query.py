"""定义 Query Understanding 相关公共类型。

这些对象是跨模块共享的语义结构化结果，因此放在公共类型层，
而不是继续放在 query_understanding 模块文件里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TargetType(StrEnum):
    """目标类别类型。"""

    PERSON = "person"
    OBJECT = "object"
    STUFF = "stuff"
    PART = "part"
    TEXT = "text"
    UNKNOWN = "unknown"


class ReferentNumber(StrEnum):
    """目标数量类型。"""

    SINGLE = "single"
    MULTIPLE = "multiple"
    UNKNOWN = "unknown"


class ImplicitnessLevel(StrEnum):
    """query 的显式程度。"""

    EXPLICIT = "explicit"
    MIXED = "mixed"
    IMPLICIT = "implicit"


class QueryRiskFlag(StrEnum):
    """下游执行风险标记。"""

    SMALL_TARGET_RISK = "small_target_risk"
    THIN_STRUCTURE_RISK = "thin_structure_risk"
    CROWDED_SCENE_RISK = "crowded_scene_risk"
    MULTI_INSTANCE_RISK = "multi_instance_risk"
    RELATION_SENSITIVE = "relation_sensitive"
    IMPLICIT_QUERY_RISK = "implicit_query_risk"


class AmbiguityFlag(StrEnum):
    """表达歧义标记。"""

    AMBIGUOUS_REFERENT = "ambiguous_referent"
    AMBIGUOUS_RELATION_ANCHOR = "ambiguous_relation_anchor"
    INSUFFICIENT_ATTRIBUTES = "insufficient_attributes"


@dataclass(slots=True)
class RelationCue:
    """关系线索。"""

    relation_type: str
    anchor_text: str
    cue_text: str


@dataclass(slots=True)
class SpatialCue:
    """纯空间线索。"""

    cue_type: str
    cue_text: str


@dataclass(slots=True)
class RouteHint:
    """给 orchestrator 的弱路线提示。"""

    hint_type: str
    reason: str
    confidence: float | None = None


@dataclass(slots=True)
class QueryBridgeHint:
    """给 prompt bridge 的弱提示。"""

    hint_type: str
    reason: str


@dataclass(slots=True)
class QueryUnderstandingResult:
    """Query Understanding 模块的主输出对象。"""

    understanding_id: str
    normalized_query: str
    target_summary: str
    target_type: TargetType
    implicitness: ImplicitnessLevel
    canonical_referent_text: str | None = None
    referent_number: ReferentNumber | None = None
    focus_terms: list[str] = field(default_factory=list)
    attribute_clues: list[str] = field(default_factory=list)
    relation_clues: list[RelationCue] = field(default_factory=list)
    spatial_clues: list[SpatialCue] = field(default_factory=list)
    ambiguity_flags: list[AmbiguityFlag] = field(default_factory=list)
    risk_flags: list[QueryRiskFlag] = field(default_factory=list)
    route_hints: list[RouteHint] = field(default_factory=list)
    bridge_hints: list[QueryBridgeHint] = field(default_factory=list)
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)

