"""定义新架构 V1 的追踪上下文骨架。

本文件只放轻量 trace 对象，不承载具体日志实现。
这些对象的职责是给模块调用链预留稳定的调试与可观测性入口。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TraceContext:
    """模块级追踪上下文。

    该对象会跟随模块输入一起传递，帮助未来接入 trace UI、日志聚合
    或链路排障能力。
    """

    trace_id: str | None = None
    # 全链路追踪 ID，用于串起一次完整任务的所有模块调用。

    span_id: str | None = None
    # 当前模块调用的 span 标识，用于区分同一 trace 下的不同步骤。

    tags: list[str] = field(default_factory=list)
    # 面向调试的标签集合，例如 "locate-route"、"retry-1"。

