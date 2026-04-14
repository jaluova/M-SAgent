"""定义 RunTask 相关的任务级枚举。

本文件用于承接 docs/architecture/run_task.md 中的推荐枚举值，
帮助任务状态在整个系统内保持统一表达。
"""

from msagent.core.enum_compat import StrEnum


class TaskSource(StrEnum):
    """任务来源。"""

    CLI = "cli"
    API = "api"
    BATCH = "batch"
    TEST = "test"


class TaskStage(StrEnum):
    """任务当前所处阶段。"""

    CREATED = "created"
    QUERY_UNDERSTANDING = "query_understanding"
    PROPOSAL = "proposal"
    PROMPT_BRIDGE = "prompt_bridge"
    SEGMENTATION = "segmentation"
    EVALUATION = "evaluation"
    FINISHED = "finished"


class TaskStatus(StrEnum):
    """任务整体运行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    HALTED = "halted"


class StopReason(StrEnum):
    """任务停止原因。"""

    ACCEPTED = "accepted"
    MAX_ATTEMPTS_REACHED = "max_attempts_reached"
    EMPTY_PROPOSAL = "empty_proposal"
    UNRECOVERABLE_ERROR = "unrecoverable_error"
    MANUAL_STOP = "manual_stop"
