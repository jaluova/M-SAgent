"""定义 API 薄入口骨架。

API 层后续可承接 FastAPI 或其他 Web 框架，但在 V1 中只保留清晰边界：

- 解析外部请求；
- 组装 RunTask；
- 调用 orchestrator；
- 返回结构化响应。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from msagent.core.contracts.common import ArtifactRef, ImageRef
from msagent.core.task.enums import StopReason, TaskSource, TaskStage, TaskStatus
from msagent.core.task.models import RunTask
from msagent.core.task.models import RunTaskIdentity, RunTaskRequest, RunTaskRuntime
from msagent.infra.adapters import ArtifactStore
from msagent.orchestrator.orchestrator import Orchestrator, OrchestrationResult
from msagent.service.demo_report import DemoTaskReport, build_demo_task_report


@dataclass(slots=True)
class APIRequest:
    """API 层接收的请求对象。"""

    image_uri: str
    # 输入图像引用，可以是本地路径、上传对象或远端 URI。

    query_text: str
    # 用户原始指代表达。

    max_attempts: int | None = None
    # 当前请求允许的最大尝试次数；未显式给出时由 service 默认值补齐。

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

    report: DemoTaskReport | None = None
    # 面向调试与可视化前端的结构化迭代报告。


class APIService:
    """API 服务层骨架。"""

    def __init__(
        self,
        orchestrator: Orchestrator,
        artifact_store: ArtifactStore | None = None,
        *,
        include_debug_report: bool = False,
        default_max_attempts: int = 3,
    ) -> None:
        self.orchestrator = orchestrator
        # API 层只依赖 orchestrator，避免推理逻辑渗入接口层。
        self.artifact_store = artifact_store
        # 若存在 artifact store，则允许把结构化迭代摘要返回给前端。
        self.include_debug_report = include_debug_report
        # 调试报告只在显式 debug 模式下对外返回。
        self.default_max_attempts = max(1, default_max_attempts)
        # request 未显式指定 max_attempts 时，回退到装配期默认值。

    def build_task(self, request: APIRequest) -> RunTask:
        """把 API 请求转换为 RunTask。"""
        now = datetime.now()
        task_id = f"api-task-{uuid4().hex[:8]}"
        metadata = dict(request.request_metadata)

        request_id = self._pop_optional_str(metadata, "request_id")
        session_id = self._pop_optional_str(metadata, "session_id")
        user_context_text = self._pop_optional_str(metadata, "user_context_text")
        image_id = self._pop_optional_str(metadata, "image_id")
        sha256 = self._pop_optional_str(metadata, "sha256")

        return RunTask(
            identity=RunTaskIdentity(
                task_id=task_id,
                source=TaskSource.API,
                created_at=now,
                session_id=session_id,
                request_id=request_id,
            ),
            request=RunTaskRequest(
                image_ref=ImageRef(
                    uri=request.image_uri,
                    image_id=image_id,
                    sha256=sha256,
                ),
                raw_query=request.query_text,
                user_context_text=user_context_text,
                client_metadata=metadata,
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

    def run(self, request: APIRequest) -> OrchestrationResult:
        """执行单次 API 推理任务。"""
        task = self.build_task(request)
        return self.orchestrator.run(task)

    def to_response(self, result: OrchestrationResult) -> APIResponse:
        """把 orchestrator 结果映射为 API 响应。"""
        task = result.task
        return APIResponse(
            task_id=task.identity.task_id,
            status=task.runtime.status.value,
            summary=self._build_safe_summary(task),
            result_refs=self._collect_result_refs(task),
            report=(
                build_demo_task_report(result, artifact_store=self.artifact_store)
                if self.include_debug_report and self.artifact_store is not None
                else None
            ),
        )

    def _collect_result_refs(self, task: RunTask) -> list[str]:
        refs = [
            task.result.final_mask_ref,
            task.result.final_prompt_package_ref,
        ]
        return self._dedupe_artifact_ids(refs)

    def _build_safe_summary(self, task: RunTask) -> str | None:
        if task.runtime.status is TaskStatus.SUCCEEDED:
            return "Task completed successfully."
        if task.runtime.status is TaskStatus.RUNNING:
            return "Task is in progress."
        if task.runtime.status is TaskStatus.PENDING:
            return "Task is pending."
        if task.runtime.status is TaskStatus.HALTED:
            return "Task was halted before completion."

        stop_reason = task.result.stop_reason
        if stop_reason is None:
            return "Task failed."
        if stop_reason is StopReason.MAX_ATTEMPTS_REACHED:
            return "Task did not complete within the allowed attempts."
        if stop_reason is StopReason.EMPTY_PROPOSAL:
            return "Task could not produce a usable result."
        if stop_reason is StopReason.MANUAL_STOP:
            return "Task was stopped manually."
        if stop_reason is StopReason.UNRECOVERABLE_ERROR:
            return "Task failed due to an internal error."
        if stop_reason is StopReason.ACCEPTED:
            return "Task completed successfully."
        return "Task failed."

    @staticmethod
    def _dedupe_artifact_ids(refs: list[ArtifactRef | None]) -> list[str]:
        artifact_ids: list[str] = []
        seen: set[str] = set()
        for ref in refs:
            if ref is None or ref.artifact_id in seen:
                continue
            artifact_ids.append(ref.artifact_id)
            seen.add(ref.artifact_id)
        return artifact_ids

    @staticmethod
    def _pop_optional_str(metadata: dict[str, object], key: str) -> str | None:
        value = metadata.get(key)
        if isinstance(value, str):
            metadata.pop(key, None)
            return value
        return None
