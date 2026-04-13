"""定义跨模块共享的基础输入输出 contract。

本文件对应 docs/architecture/module_contracts.md 中的公共骨架。
这里的对象不描述具体算法，只描述模块边界和可追踪的结构化通信格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from msagent.core.trace.models import TraceContext

ModulePayloadT = TypeVar("ModulePayloadT")


class ModuleStatus(StrEnum):
    """模块执行状态枚举。"""

    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    FAILED = "failed"


class ArtifactKind(StrEnum):
    """受控 artifact kind 注册表。"""

    QUERY_UNDERSTANDING_RESULT = "query_understanding_result"
    PROPOSAL_RESULT = "proposal_result"
    PROMPT_PACKAGE = "prompt_package"
    SEGMENTATION_RESULT = "segmentation_result"
    EVALUATION_RESULT = "evaluation_result"
    MASK = "mask"


@dataclass(slots=True)
class ImageRef:
    """输入图像的统一引用。"""

    uri: str
    # 图像定位方式，可以是本地路径、URI 或对象存储地址。

    image_id: str | None = None
    # 业务侧图像 ID，方便上层系统关联。

    sha256: str | None = None
    # 图像内容摘要，用于去重、缓存或校验。


@dataclass(slots=True)
class ArtifactRef:
    """任务账本中可被追踪的产物引用。"""

    artifact_id: str
    # 产物唯一标识，是任务内外统一引用某个中间结果的主键。

    artifact_type: ArtifactKind
    # 产物类型，例如 "proposal_result"、"segmentation_result"。

    attempt_index: int | None = None
    # 该产物产生于第几轮尝试；某些任务级产物可以为空。

    summary: str | None = None
    # 一行摘要，供 trace、日志或前端快速展示。


@dataclass(slots=True)
class DiagnosticMessage:
    """模块执行期间产生的诊断信息。"""

    level: str
    # 推荐取值为 "info"、"warning"、"error"。

    message: str
    # 面向开发与调试的一段可读说明。

    code: str | None = None
    # 可选结构化错误码，便于未来做前端映射与统计。


@dataclass(slots=True)
class BaseModuleInput:
    """所有模块输入 DTO 的公共骨架。"""

    task_id: str
    # 当前模块执行所属的任务 ID，对应 RunTask.identity.task_id。

    attempt_index: int
    # 当前属于第几轮尝试，由 orchestrator 统一维护。

    upstream_refs: list[ArtifactRef] = field(default_factory=list)
    # 本模块依赖的上游产物引用列表，仅用于追踪和记账。
    # 真正参与业务计算的权威输入，应由模块专属字段显式给出。

    module_options: dict[str, object] = field(default_factory=dict)
    # 模块局部选项，如阈值、开关、候选上限等。

    trace_context: TraceContext | None = None
    # 当前调用的 trace 上下文，便于未来接调试系统。


@dataclass(slots=True)
class BaseModuleOutput(Generic[ModulePayloadT]):
    """所有模块输出 DTO 的公共骨架。"""

    module_name: str
    # 产出该结果的模块名，例如 "query_understanding"。

    status: ModuleStatus
    # 当前模块本轮执行的状态，不直接承担调度职责。

    primary_payload: ModulePayloadT | None = None
    # 模块主产物，例如 QueryUnderstandingResult 或 PromptPackage。

    artifact_ref: ArtifactRef | None = None
    # 若本轮写入了可追踪产物，则在这里登记其引用。

    consumed_refs: list[ArtifactRef] = field(default_factory=list)
    # 当前模块实际消费的上游产物引用，便于回溯依赖链。

    diagnostics: list[DiagnosticMessage] = field(default_factory=list)
    # 调试信息、警告和不确定性说明，不用于主流程控制。
