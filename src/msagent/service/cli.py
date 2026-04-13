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

from msagent.core.task.models import RunTask
from msagent.orchestrator.orchestrator import Orchestrator, OrchestrationResult


@dataclass(slots=True)
class CLIRequest:
    """CLI 层接收的请求对象。"""

    image_path: str
    # 待处理图像路径。

    query_text: str
    # 用户输入的指代表达。

    max_attempts: int = 3
    # 当前任务允许的最大尝试次数。

    output_dir: str | None = None
    # 结果输出目录，供后续实现时使用。


class CLIService:
    """CLI 服务层骨架。"""

    def __init__(self, orchestrator: Orchestrator) -> None:
        self.orchestrator = orchestrator
        # CLI 层只依赖 orchestrator，不直接接触模块细节。

    def build_task(self, request: CLIRequest) -> RunTask:
        """把 CLI 请求转换为 RunTask。"""
        raise NotImplementedError

    def run(self, request: CLIRequest) -> OrchestrationResult:
        """执行单次 CLI 推理任务。"""
        raise NotImplementedError

