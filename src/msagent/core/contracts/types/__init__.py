"""定义跨模块共享的公共类型层。

这个目录只放“公共协议对象”，不放模块本体逻辑。
这样各模块依赖的是统一类型，而不是彼此的实现文件，从而避免边界缠绕。
"""

from msagent.core.contracts.types.evaluation import (
    EvaluationIssue,
    EvaluationResult,
    EvaluationVerdict,
    FailureType,
)
from msagent.core.contracts.types.prompt import (
    BoxPrompt,
    ExecutionHints,
    PointPrompt,
    PromptMetadata,
    PromptPackage,
    PromptTextBundle,
    SpatialPromptBundle,
)
from msagent.core.contracts.types.proposal import (
    NormalizedBox,
    PointHint,
    ProposalBridgeHint,
    ProposalCandidate,
    ProposalQualityFlag,
    ProposalResult,
    ProposalRoute,
    ProposalStatus,
)
from msagent.core.contracts.types.query import (
    AmbiguityFlag,
    ImplicitnessLevel,
    QueryBridgeHint,
    QueryRiskFlag,
    QueryUnderstandingResult,
    ReferentNumber,
    RelationCue,
    RouteHint,
    SpatialCue,
    TargetType,
)
from msagent.core.contracts.types.segmentation import (
    SegmentationCandidate,
    SegmentationResult,
    SegmentationStatus,
)

__all__ = [
    "AmbiguityFlag",
    "BoxPrompt",
    "EvaluationIssue",
    "EvaluationResult",
    "EvaluationVerdict",
    "ExecutionHints",
    "FailureType",
    "ImplicitnessLevel",
    "NormalizedBox",
    "PointHint",
    "PointPrompt",
    "PromptMetadata",
    "PromptPackage",
    "PromptTextBundle",
    "ProposalBridgeHint",
    "ProposalCandidate",
    "ProposalQualityFlag",
    "ProposalResult",
    "ProposalRoute",
    "ProposalStatus",
    "QueryBridgeHint",
    "QueryRiskFlag",
    "QueryUnderstandingResult",
    "ReferentNumber",
    "RelationCue",
    "RouteHint",
    "SegmentationCandidate",
    "SegmentationResult",
    "SegmentationStatus",
    "SpatialCue",
    "SpatialPromptBundle",
    "TargetType",
]

