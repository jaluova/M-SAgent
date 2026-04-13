"""定义 M-SAgent V1 的唯一主控制器骨架。

本文件严格落实架构文档中的核心原则：

- `Orchestrator` 是唯一允许推进阶段和决定下一步动作的模块；
- 其他模块只消费输入、产出结果，不直接操控全局流程；
- 有限重试和失败感知重试都由这里统一串接。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from msagent.core.contracts.common import ArtifactRef, BaseModuleOutput, ModuleStatus
from msagent.core.contracts.types import (
    EvaluationResult,
    EvaluationVerdict,
    PromptPackage,
    ProposalResult,
    ProposalRoute,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.core.policies.retry_policy import RetryDecision, RetryPolicy
from msagent.core.task.enums import StopReason, TaskStage, TaskStatus
from msagent.core.task.models import RunTask
from msagent.core.task.models import AttemptRecord, RunTaskNormalizedInput
from msagent.modules.evaluator import EvaluatorModule
from msagent.modules.evaluator import EvaluatorModuleInput
from msagent.modules.prompt_bridge import PromptBridgeModule
from msagent.modules.prompt_bridge import PromptBridgeModuleInput
from msagent.modules.proposal_engine import ProposalEngineModule
from msagent.modules.proposal_engine import ProposalEngineModuleInput
from msagent.modules.query_understanding import QueryUnderstandingModule
from msagent.modules.query_understanding import QueryUnderstandingModuleInput
from msagent.modules.segmenter import SegmenterModule
from msagent.modules.segmenter import SegmenterModuleInput

PayloadT = TypeVar("PayloadT")


@dataclass(slots=True)
class AttemptExecutionResult:
    """单轮 orchestrator 执行结果。

    orchestrator 是系统总调度台，因此这里按步骤显式保存每个模块的输出，
    而不是重新把所有结果抹平成 `BaseModuleOutput[object]`。
    """

    query_understanding_output: BaseModuleOutput[QueryUnderstandingResult] | None = None
    # Query Understanding 阶段的结构化输出。

    proposal_output: BaseModuleOutput[ProposalResult] | None = None
    # Proposal Engine 阶段的结构化输出。

    prompt_bridge_output: BaseModuleOutput[PromptPackage] | None = None
    # Prompt Bridge 阶段的结构化输出。

    segmentation_output: BaseModuleOutput[SegmentationResult] | None = None
    # Segmenter 阶段的结构化输出。

    evaluation_output: BaseModuleOutput[EvaluationResult] | None = None
    # Evaluator 阶段的结构化输出。


@dataclass(slots=True)
class OrchestratorDependencies:
    """orchestrator 运行所需的模块依赖集合。"""

    query_understanding_module: QueryUnderstandingModule
    # 负责轻量语义理解，产出 QueryUnderstandingResult。

    proposal_engine_module: ProposalEngineModule
    # 负责生成 locate / crop / rewrite 等 route proposal。

    prompt_bridge_module: PromptBridgeModule
    # 负责把理解结果与 proposal 转成 PromptPackage。

    segmenter_module: SegmenterModule
    # 负责执行分割并产出结构化分割结果。

    evaluator_module: EvaluatorModule
    # 负责给出 accept / reject 及失败原因。

    retry_policy: RetryPolicy
    # 负责封装有限重试与失败感知重试规则。


@dataclass(slots=True)
class OrchestrationResult:
    """一次 orchestrator 运行后的返回骨架。"""

    task: RunTask
    # 运行结束后的完整任务账本快照。

    last_attempt_result: AttemptExecutionResult | None = None
    # 最近一轮完整或部分执行的强类型结果快照。


class Orchestrator:
    """M-SAgent V1 的总调度员。"""

    def __init__(self, dependencies: OrchestratorDependencies) -> None:
        self.dependencies = dependencies
        # 所有模块与策略依赖都集中挂在这里，避免散落跨层调用。

    def run(self, task: RunTask) -> OrchestrationResult:
        """驱动单个任务完成完整的 V1 主流程。"""
        task.runtime.status = TaskStatus.RUNNING
        task.runtime.stage = TaskStage.CREATED
        task.runtime.updated_at = datetime.now()

        if task.runtime.active_route is None:
            task.runtime.active_route = self.choose_initial_route(task)

        last_attempt_result: AttemptExecutionResult | None = None
        while len(task.attempt_history) < task.runtime.max_attempts:
            task.runtime.attempt_index = len(task.attempt_history) + 1
            try:
                last_attempt_result = self.run_single_attempt(task)
            except Exception as exc:
                latest_attempt = task.attempt_history[-1] if task.attempt_history else None
                if latest_attempt is not None and latest_attempt.finished_at is None:
                    latest_attempt.finished_at = datetime.now()
                    latest_attempt.notes.append(f"unrecoverable_error={exc}")
                task.runtime.status = TaskStatus.FAILED
                task.runtime.stage = TaskStage.FINISHED
                task.runtime.updated_at = datetime.now()
                task.result.stop_reason = StopReason.UNRECOVERABLE_ERROR
                task.result.failure_summary = str(exc)
                task.result.final_summary = str(exc)
                break

            if task.runtime.stage is TaskStage.FINISHED:
                break

            evaluation_output = last_attempt_result.evaluation_output
            evaluation_result = (
                evaluation_output.primary_payload if evaluation_output is not None else None
            )
            if evaluation_result is None:
                task.runtime.status = TaskStatus.FAILED
                task.runtime.stage = TaskStage.FINISHED
                task.result.stop_reason = StopReason.UNRECOVERABLE_ERROR
                task.result.failure_summary = "Missing evaluation result."
                task.result.final_summary = "Task stopped because evaluation did not return."
                break

            decision = self.dependencies.retry_policy.decide_retry(task)
            self.apply_retry_decision(task, decision)
            if not decision.should_retry:
                break

        return OrchestrationResult(task=task, last_attempt_result=last_attempt_result)

    def choose_initial_route(self, task: RunTask) -> ProposalRoute:
        """选择任务的首轮 route。"""
        route = self.dependencies.retry_policy.choose_initial_route(task)
        task.runtime.active_route = route
        task.runtime.updated_at = datetime.now()
        return route

    def run_single_attempt(self, task: RunTask) -> AttemptExecutionResult:
        """执行单轮理解、proposal、bridge、segment 和 evaluate 流程。"""
        if task.runtime.active_route is None:
            raise ValueError("Active route must be selected before running an attempt.")

        attempt_record = AttemptRecord(
            attempt_index=task.runtime.attempt_index,
            route=task.runtime.active_route,
            started_at=datetime.now(),
        )
        task.attempt_history.append(attempt_record)

        task.runtime.stage = TaskStage.QUERY_UNDERSTANDING
        query_output = self.dependencies.query_understanding_module.run(
            QueryUnderstandingModuleInput(
                task_id=task.identity.task_id,
                attempt_index=task.runtime.attempt_index,
                raw_query=task.request.raw_query,
                user_context_text=task.request.user_context_text,
                image_uri=task.request.image_ref.uri,
            )
        )
        self._record_artifact(
            task=task,
            output=query_output,
            attempt_record=attempt_record,
            artifact_attr_name="latest_query_understanding_ref",
            attempt_attr_name="query_understanding_ref",
        )
        understanding = self._consume_step_output(
            task=task,
            attempt_record=attempt_record,
            output=query_output,
            step_name="query_understanding",
            stop_reason=StopReason.UNRECOVERABLE_ERROR,
            failure_summary="Query understanding did not produce a usable result.",
        )
        if understanding is None:
            return AttemptExecutionResult(query_understanding_output=query_output)
        task.normalized_input = RunTaskNormalizedInput(
            normalized_query=understanding.normalized_query,
            detected_language=self._detect_language(understanding.normalized_query),
            image_meta=task.normalized_input.image_meta if task.normalized_input else None,
            preprocessing_ref=task.normalized_input.preprocessing_ref
            if task.normalized_input
            else None,
        )

        task.runtime.stage = TaskStage.PROPOSAL
        proposal_output = self.dependencies.proposal_engine_module.run(
            ProposalEngineModuleInput(
                task_id=task.identity.task_id,
                attempt_index=task.runtime.attempt_index,
                upstream_refs=self._collect_attempt_refs(attempt_record),
                understanding=understanding,
                preferred_route=task.runtime.active_route,
            )
        )
        self._record_artifact(
            task=task,
            output=proposal_output,
            attempt_record=attempt_record,
            artifact_attr_name="latest_proposal_ref",
            attempt_attr_name="proposal_ref",
        )
        proposal = self._consume_step_output(
            task=task,
            attempt_record=attempt_record,
            output=proposal_output,
            step_name="proposal",
            stop_reason=StopReason.EMPTY_PROPOSAL,
            failure_summary="Proposal engine returned no usable candidate.",
        )
        if proposal is None:
            return AttemptExecutionResult(
                query_understanding_output=query_output,
                proposal_output=proposal_output,
            )

        task.runtime.stage = TaskStage.PROMPT_BRIDGE
        prompt_output = self.dependencies.prompt_bridge_module.run(
            PromptBridgeModuleInput(
                task_id=task.identity.task_id,
                attempt_index=task.runtime.attempt_index,
                upstream_refs=self._collect_attempt_refs(attempt_record),
                understanding=understanding,
                proposal=proposal,
                raw_query=task.request.raw_query,
            )
        )
        self._record_artifact(
            task=task,
            output=prompt_output,
            attempt_record=attempt_record,
            artifact_attr_name="latest_prompt_package_ref",
            attempt_attr_name="prompt_package_ref",
        )
        prompt_package = self._consume_step_output(
            task=task,
            attempt_record=attempt_record,
            output=prompt_output,
            step_name="prompt_bridge",
            stop_reason=StopReason.UNRECOVERABLE_ERROR,
            failure_summary="Prompt bridge did not produce a usable prompt package.",
        )
        if prompt_package is None:
            return AttemptExecutionResult(
                query_understanding_output=query_output,
                proposal_output=proposal_output,
                prompt_bridge_output=prompt_output,
            )

        task.runtime.stage = TaskStage.SEGMENTATION
        segmentation_output = self.dependencies.segmenter_module.run(
            SegmenterModuleInput(
                task_id=task.identity.task_id,
                attempt_index=task.runtime.attempt_index,
                upstream_refs=self._collect_attempt_refs(attempt_record),
                image_uri=task.request.image_ref.uri,
                prompt_package=prompt_package,
            )
        )
        self._record_artifact(
            task=task,
            output=segmentation_output,
            attempt_record=attempt_record,
            artifact_attr_name="latest_segmentation_ref",
            attempt_attr_name="segmentation_ref",
        )
        segmentation = self._consume_step_output(
            task=task,
            attempt_record=attempt_record,
            output=segmentation_output,
            step_name="segmentation",
            stop_reason=StopReason.UNRECOVERABLE_ERROR,
            failure_summary="Segmenter did not produce a usable segmentation result.",
        )
        if segmentation is None:
            return AttemptExecutionResult(
                query_understanding_output=query_output,
                proposal_output=proposal_output,
                prompt_bridge_output=prompt_output,
                segmentation_output=segmentation_output,
            )
        self._record_refs(task, [candidate.mask_ref for candidate in segmentation.candidates])

        task.runtime.stage = TaskStage.EVALUATION
        evaluation_output = self.dependencies.evaluator_module.run(
            EvaluatorModuleInput(
                task_id=task.identity.task_id,
                attempt_index=task.runtime.attempt_index,
                upstream_refs=self._collect_attempt_refs(attempt_record),
                raw_query=task.request.raw_query,
                understanding=understanding,
                proposal=proposal,
                prompt_package=prompt_package,
                segmentation=segmentation,
            )
        )
        self._record_artifact(
            task=task,
            output=evaluation_output,
            attempt_record=attempt_record,
            artifact_attr_name="latest_evaluation_ref",
            attempt_attr_name="evaluation_ref",
        )
        evaluation = self._consume_step_output(
            task=task,
            attempt_record=attempt_record,
            output=evaluation_output,
            step_name="evaluation",
            stop_reason=StopReason.UNRECOVERABLE_ERROR,
            failure_summary="Evaluator did not produce a usable verdict.",
        )
        if evaluation is None:
            return AttemptExecutionResult(
                query_understanding_output=query_output,
                proposal_output=proposal_output,
                prompt_bridge_output=prompt_output,
                segmentation_output=segmentation_output,
                evaluation_output=evaluation_output,
            )
        attempt_record.verdict = evaluation.verdict
        attempt_record.failure_type = evaluation.failure_type
        attempt_record.finished_at = datetime.now()

        if evaluation.verdict is EvaluationVerdict.ACCEPT:
            task.result.final_verdict = evaluation.verdict
            task.result.final_mask_ref = evaluation.accepted_mask_ref
            task.result.final_prompt_package_ref = attempt_record.prompt_package_ref
            task.result.stop_reason = StopReason.ACCEPTED
            task.result.final_summary = evaluation.summary

        task.runtime.updated_at = datetime.now()
        return AttemptExecutionResult(
            query_understanding_output=query_output,
            proposal_output=proposal_output,
            prompt_bridge_output=prompt_output,
            segmentation_output=segmentation_output,
            evaluation_output=evaluation_output,
        )

    def apply_retry_decision(self, task: RunTask, decision: RetryDecision) -> None:
        """将重试决策回写到任务运行时状态中。"""
        task.runtime.updated_at = datetime.now()
        if decision.should_retry:
            task.runtime.status = TaskStatus.RUNNING
            task.runtime.stage = TaskStage.CREATED
            task.runtime.active_route = decision.next_route or task.runtime.active_route
            return

        task.runtime.stage = TaskStage.FINISHED
        if task.result.final_verdict is EvaluationVerdict.ACCEPT:
            task.runtime.status = TaskStatus.SUCCEEDED
            task.result.stop_reason = StopReason.ACCEPTED
            return

        task.runtime.status = TaskStatus.FAILED
        task.result.final_verdict = task.attempt_history[-1].verdict
        if len(task.attempt_history) >= task.runtime.max_attempts:
            task.result.stop_reason = StopReason.MAX_ATTEMPTS_REACHED
        if task.result.failure_summary is None:
            task.result.failure_summary = decision.reason
        if task.result.final_summary is None:
            task.result.final_summary = decision.reason

    def _record_artifact(
        self,
        task: RunTask,
        output: BaseModuleOutput[PayloadT],
        attempt_record: AttemptRecord,
        artifact_attr_name: str,
        attempt_attr_name: str,
    ) -> None:
        if output.artifact_ref is None:
            return
        setattr(task.artifacts, artifact_attr_name, output.artifact_ref)
        setattr(attempt_record, attempt_attr_name, output.artifact_ref)
        self._record_refs(task, [output.artifact_ref])

    def _record_refs(self, task: RunTask, refs: list[ArtifactRef]) -> None:
        existing_ids = {ref.artifact_id for ref in task.artifacts.artifact_refs}
        for ref in refs:
            if ref.artifact_id in existing_ids:
                continue
            task.artifacts.artifact_refs.append(ref)
            existing_ids.add(ref.artifact_id)

    def _collect_attempt_refs(self, attempt_record: AttemptRecord) -> list[ArtifactRef]:
        refs = [
            attempt_record.query_understanding_ref,
            attempt_record.proposal_ref,
            attempt_record.prompt_package_ref,
            attempt_record.segmentation_ref,
            attempt_record.evaluation_ref,
        ]
        return [ref for ref in refs if ref is not None]

    def _consume_step_output(
        self,
        task: RunTask,
        attempt_record: AttemptRecord,
        output: BaseModuleOutput[PayloadT],
        step_name: str,
        stop_reason: StopReason,
        failure_summary: str,
    ) -> PayloadT | None:
        if output.status is ModuleStatus.SUCCESS and output.primary_payload is not None:
            return output.primary_payload

        attempt_record.finished_at = datetime.now()
        diagnostic_summary = "; ".join(message.message for message in output.diagnostics)
        attempt_record.notes.append(
            f"{step_name}.status={output.status.value}"
        )
        if diagnostic_summary:
            attempt_record.notes.append(diagnostic_summary)

        task.runtime.stage = TaskStage.FINISHED
        task.runtime.status = TaskStatus.FAILED
        task.runtime.updated_at = datetime.now()
        task.result.stop_reason = stop_reason
        task.result.failure_summary = diagnostic_summary or failure_summary
        task.result.final_summary = f"{step_name}: {diagnostic_summary or failure_summary}"
        return None

    def _detect_language(self, text: str) -> str:
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return "zh"
        return "en"
