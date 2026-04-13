"""定义外部能力适配器骨架。

核心模块只依赖这些抽象接口，不直接依赖具体模型实现。
这样后续接回 Qwen、定位器、SAM3 或对象存储时，可以保持核心内核稳定。
同时 adapter 的输入输出也尽量采用显式结构，而不是退化成 Any。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from msagent.core.contracts.adapter_requests import (
    EvaluationAdapterRequest,
    LocateAdapterRequest,
    QueryUnderstandingAdapterRequest,
    SegmentAdapterRequest,
)
from msagent.core.contracts.common import ArtifactRef
from msagent.core.contracts.types import (
    EvaluationResult,
    MockMask,
    PromptPackage,
    ProposalResult,
    QueryUnderstandingResult,
    SegmentationResult,
)

LoadedArtifactT = TypeVar("LoadedArtifactT")


class ArtifactKind(StrEnum):
    """受控 artifact kind 注册表。"""

    QUERY_UNDERSTANDING_RESULT = "query_understanding_result"
    PROPOSAL_RESULT = "proposal_result"
    PROMPT_PACKAGE = "prompt_package"
    SEGMENTATION_RESULT = "segmentation_result"
    EVALUATION_RESULT = "evaluation_result"
    MOCK_MASK = "mock_mask"


ARTIFACT_PAYLOAD_TYPES: dict[ArtifactKind, type[object]] = {
    ArtifactKind.QUERY_UNDERSTANDING_RESULT: QueryUnderstandingResult,
    ArtifactKind.PROPOSAL_RESULT: ProposalResult,
    ArtifactKind.PROMPT_PACKAGE: PromptPackage,
    ArtifactKind.SEGMENTATION_RESULT: SegmentationResult,
    ArtifactKind.EVALUATION_RESULT: EvaluationResult,
    ArtifactKind.MOCK_MASK: MockMask,
}


def get_artifact_payload_type(artifact_kind: ArtifactKind) -> type[object]:
    """返回 artifact kind 对应的权威 payload 类型。"""

    return ARTIFACT_PAYLOAD_TYPES[artifact_kind]


def resolve_artifact_kind(artifact_type: str) -> ArtifactKind:
    """把 artifact_type 字符串解析成受控 kind。"""

    return ArtifactKind(artifact_type)


@dataclass(slots=True)
class LLMAdapter:
    """语言或多模态理解后端的统一适配接口。"""

    backend_name: str
    # 当前接入的后端名称，例如 "qwen2.5-vl"。

    model_path: str | None = None
    # 该后端对应的本地模型路径或远端模型标识。

    def run_query_understanding(
        self, request: QueryUnderstandingAdapterRequest
    ) -> QueryUnderstandingResult:
        """执行 query understanding 所需的结构化推理。"""
        raise NotImplementedError

    def run_evaluation(self, request: EvaluationAdapterRequest) -> EvaluationResult:
        """执行 evaluator 所需的结构化评估。"""
        raise NotImplementedError


@dataclass(slots=True)
class LocatorAdapter:
    """定位后端的统一适配接口。"""

    backend_name: str
    # 当前定位后端名称，例如 "gridground" 或 "train_adapter"。

    endpoint: str | None = None
    # 若定位后端是远程服务，可在这里记录服务地址。

    model_path: str | None = None
    # 若定位后端是本地运行时，可在这里记录模型路径。

    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        """执行 locate route 所需的空间先验生成。"""
        raise NotImplementedError


@dataclass(slots=True)
class SAMAdapter:
    """分割后端的统一适配接口。"""

    backend_name: str
    # 当前分割后端名称，例如 "sam3"。

    model_path: str | None = None
    # 分割后端代码或模型目录。

    checkpoint_path: str | None = None
    # 分割权重路径。

    def segment(self, request: SegmentAdapterRequest) -> SegmentationResult:
        """执行一次基于 PromptPackage 的分割。"""
        raise NotImplementedError


@dataclass(slots=True)
class ArtifactStore:
    """中间产物与结果对象的统一存储接口。"""

    root_uri: str
    # 默认产物根位置，可以是本地目录或远端存储前缀。

    def save_artifact(self, artifact_type: ArtifactKind, payload: object) -> ArtifactRef:
        """保存结构化产物并返回其引用。"""
        raise NotImplementedError

    def load_artifact(
        self,
        artifact_ref: ArtifactRef,
        expected_type: type[LoadedArtifactT],
    ) -> LoadedArtifactT:
        """根据 ArtifactRef 按声明类型加载已保存的产物。

        调用方必须显式给出期望类型，避免“按 ref 读回对象”时重新丢失类型边界。
        """
        raise NotImplementedError
