"""定义 CLI 薄入口骨架。

CLI 层的职责仅限于：

- 接收本地命令行参数；
- 组装 RunTask；
- 调用 orchestrator；
- 把最终结果格式化给用户。

它不承载核心推理逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from msagent.core.task.enums import TaskSource, TaskStage, TaskStatus
from msagent.core.task.models import RunTask
from msagent.core.task.models import ImageRef, RunTaskIdentity, RunTaskRequest, RunTaskRuntime
from msagent.orchestrator.orchestrator import Orchestrator, OrchestrationResult


@dataclass(slots=True)
class CLIRequest:
    """CLI 层接收的请求对象。"""

    image_path: str
    # 待处理图像路径。

    query_text: str
    # 用户输入的指代表达。

    max_attempts: int | None = None
    # 当前任务允许的最大尝试次数；未显式给出时由 service 默认值补齐。

    output_dir: str | None = None
    # 结果输出目录，供后续实现时使用。


class CLIService:
    """CLI 服务层骨架。"""

    def __init__(self, orchestrator: Orchestrator, *, default_max_attempts: int = 3) -> None:
        self.orchestrator = orchestrator
        # CLI 层只依赖 orchestrator，不直接接触模块细节。
        self.default_max_attempts = max(1, default_max_attempts)
        # request 未显式指定 max_attempts 时，回退到装配期默认值。

    def build_task(self, request: CLIRequest) -> RunTask:
        """把 CLI 请求转换为 RunTask。"""
        now = datetime.now()
        task_id = f"cli-task-{uuid4().hex[:8]}"
        return RunTask(
            identity=RunTaskIdentity(
                task_id=task_id,
                source=TaskSource.CLI,
                created_at=now,
            ),
            request=RunTaskRequest(
                image_ref=ImageRef(uri=str(Path(request.image_path).expanduser().resolve())),
                raw_query=request.query_text,
            ),
            runtime=RunTaskRuntime(
                stage=TaskStage.CREATED,
                status=TaskStatus.PENDING,
                attempt_index=0,
                max_attempts=max(
                    1,
                    request.max_attempts
                    if request.max_attempts is not None
                    else self.default_max_attempts,
                ),
                updated_at=now,
            ),
        )

    def run(self, request: CLIRequest) -> OrchestrationResult:
        """执行单次 CLI 推理任务。"""
        task = self.build_task(request)
        return self.orchestrator.run(task)
