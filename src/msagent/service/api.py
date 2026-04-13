"""定义 API 薄入口骨架。

API 层后续可承接 FastAPI 或其他 Web 框架，但在 V1 中只保留清晰边界：

- 解析外部请求；
- 组装 RunTask；
- 调用 orchestrator；
- 返回结构化响应。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.task.models import RunTask
from msagent.orchestrator.orchestrator import Orchestrator, OrchestrationResult


@dataclass(slots=True)
class APIRequest:
    """API 层接收的请求对象。"""

    image_uri: str
    # 输入图像引用，可以是本地路径、上传对象或远端 URI。

    query_text: str
    # 用户原始指代表达。

    max_attempts: int = 3
    # 当前请求允许的最大尝试次数。

    request_metadata: dict[str, object] = field(default_factory=dict)
    # API 网关或前端透传的请求元数据。


@dataclass(slots=True)
class APIResponse:
    """API 层对外返回的响应骨架。"""

    task_id: str
    # 当前响应对应的任务 ID。

    status: str
    # 当前任务状态摘要。

    summary: str | None = None
    # 面向调用方的一行结果说明。

    result_refs: list[str] = field(default_factory=list)
    # 结果与关键产物引用列表，便于前端二次拉取。


class APIService:
    """API 服务层骨架。"""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        # API 层只依赖 orchestrator，避免推理逻辑渗入接口层。

    def build_task(self, request: APIRequest) -> RunTask:
        """把 API 请求转换为 RunTask。"""
        raise NotImplementedError

    def run(self, request: APIRequest) -> OrchestrationResult:
        """执行单次 API 推理任务。"""
        raise NotImplementedError

    def to_response(self, result: OrchestrationResult) -> APIResponse:
        """把 orchestrator 结果映射为 API 响应。"""
        raise NotImplementedError
