"""定义 RunTask 及其关联状态对象。

本文件对应 docs/architecture/run_task.md。
这里的对象是任务总账本，只负责承载稳定状态，不混入具体业务实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from msagent.core.contracts.common import ArtifactRef, ImageRef
from msagent.core.contracts.types import EvaluationVerdict, FailureType, ProposalRoute
from msagent.core.task.enums import StopReason, TaskSource, TaskStage, TaskStatus


@dataclass(slots=True)
class ImageMeta:
    """图像基础元信息。"""

    width: int
    # 图像宽度。

    height: int
    # 图像高度。

    mode: str | None = None
    # 颜色模式，例如 "RGB"、"L"。


@dataclass(slots=True)
class RunTaskIdentity:
    """任务身份信息。"""

    task_id: str
    # 当前任务在系统中的唯一标识。

    source: TaskSource
    # 任务来源，用于区分 CLI、API、测试等入口。

    created_at: datetime
    # 任务创建时间，便于排序、追踪和审计。

    session_id: str | None = None
    # 上层会话 ID，便于把多个任务聚合到同一会话下。

    request_id: str | None = None
    # 外部请求 ID，便于 API 层透传与排障。


@dataclass(slots=True)
class RunTaskRequest:
    """用户原始请求快照。"""

    image_ref: ImageRef
    # 本次任务要处理的输入图像引用。

    raw_query: str
    # 用户原始指代表达，不做任何重写。

    user_context_text: str | None = None
    # 可选补充上下文，例如上层系统附带的说明。

    client_metadata: dict[str, object] = field(default_factory=dict)
    # 客户端透传元数据，例如展示尺寸、来源页面等。


@dataclass(slots=True)
class RunTaskNormalizedInput:
    """规范化后的输入信息。"""

    normalized_query: str | None = None
    # 预处理或理解模块产出的规范化文本。

    detected_language: str | None = None
    # 识别出的语言标记，例如 "zh"、"en"。

    image_meta: ImageMeta | None = None
    # 图像基础元信息，供后续模块使用。

    preprocessing_ref: ArtifactRef | None = None
    # 若存在预处理产物，则在这里挂接其引用。


@dataclass(slots=True)
class RunTaskRuntime:
    """任务运行时状态。"""

    stage: TaskStage
    # 当前进行到哪个处理阶段。

    status: TaskStatus
    # 任务整体运行状态。

    attempt_index: int
    # 当前轮次索引，需要与 attempt_history 保持一致。

    max_attempts: int
    # 最多允许尝试多少轮，由策略层控制。

    updated_at: datetime
    # 最近一次状态推进时间。

    active_route: ProposalRoute | None = None
    # 当前正在走哪条 route，由统一枚举维护。

    policy_snapshot_ref: ArtifactRef | None = None
    # 与当前调度策略配置对应的产物引用，便于追踪。


@dataclass(slots=True)
class RunTaskArtifacts:
    """任务重要中间产物的索引入口。"""

    artifact_refs: list[ArtifactRef] = field(default_factory=list)
    # 本任务涉及到的全部产物引用总表。

    latest_query_understanding_ref: ArtifactRef | None = None
    # 最近一次理解结果引用。

    latest_proposal_ref: ArtifactRef | None = None
    # 最近一次 proposal 结果引用。

    latest_prompt_package_ref: ArtifactRef | None = None
    # 最近一次 prompt package 引用。

    latest_segmentation_ref: ArtifactRef | None = None
    # 最近一次 segmentation 结果引用。

    latest_evaluation_ref: ArtifactRef | None = None
    # 最近一次 evaluation 结果引用。


@dataclass(slots=True)
class AttemptRecord:
    """单轮尝试的结构化记录。"""

    attempt_index: int
    # 当前是第几轮尝试。

    route: ProposalRoute
    # 本轮选择的 route，由统一枚举维护，避免字符串漂移。

    started_at: datetime
    # 本轮开始时间。

    finished_at: datetime | None = None
    # 本轮结束时间；未结束时为空。

    query_understanding_ref: ArtifactRef | None = None
    # 本轮关联的 query understanding 结果。

    proposal_ref: ArtifactRef | None = None
    # 本轮关联的 proposal 结果。

    prompt_package_ref: ArtifactRef | None = None
    # 本轮使用的 prompt package。

    segmentation_ref: ArtifactRef | None = None
    # 本轮分割结果引用。

    evaluation_ref: ArtifactRef | None = None
    # 本轮评估结果引用。

    verdict: EvaluationVerdict | None = None
    # 本轮结论，由评估结论枚举维护。

    failure_type: FailureType | None = None
    # 本轮主要失败类型，便于失败感知重试。

    notes: list[str] = field(default_factory=list)
    # 面向 trace 的补充说明。


@dataclass(slots=True)
class RunTaskResult:
    """任务最终结果摘要。"""

    final_mask_ref: ArtifactRef | None = None
    # 最终接受的 mask 产物引用。

    final_prompt_package_ref: ArtifactRef | None = None
    # 产出最终结果时使用的提示包引用。

    final_verdict: EvaluationVerdict | None = None
    # 最终结论，由统一枚举维护。

    stop_reason: StopReason | None = None
    # 当前任务为什么停止。

    failure_summary: str | None = None
    # 若任务失败或中止，可在这里写简短失败摘要。

    final_summary: str | None = None
    # 面向 trace 或服务层返回的一行总结。


@dataclass(slots=True)
class RunTask:
    """单次运行任务的总账本。

    该对象是系统状态中心，但它本身不承担 orchestrator 的调度逻辑。
    """

    identity: RunTaskIdentity
    # 任务身份与来源信息。

    request: RunTaskRequest
    # 用户原始输入请求。

    runtime: RunTaskRuntime
    # 当前运行状态快照。

    artifacts: RunTaskArtifacts = field(default_factory=RunTaskArtifacts)
    # 中间产物索引入口。

    attempt_history: list[AttemptRecord] = field(default_factory=list)
    # 所有轮次的结构化尝试记录。

    result: RunTaskResult = field(default_factory=RunTaskResult)
    # 最终结果与停止原因。

    normalized_input: RunTaskNormalizedInput | None = None
    # 规范化输入快照；在尚未完成规范化前可以为空。
