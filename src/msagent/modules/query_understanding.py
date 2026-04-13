"""定义 Query Understanding 模块骨架。

该模块只负责把原始 query 整理成轻量语义结构，
不负责定位、不负责分割，也不拥有调度权。
跨模块共享的结果类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.common import BaseModuleInput, BaseModuleOutput
from msagent.core.contracts.types import QueryUnderstandingResult
from msagent.infra.adapters import ArtifactStore, LLMAdapter


@dataclass(slots=True)
class QueryUnderstandingModuleInput(BaseModuleInput):
    """Query Understanding 模块的输入 DTO。"""

    raw_query: str = ""
    # 用户原始 query，是当前模块最核心的文本输入。

    user_context_text: str | None = None
    # 上层请求附带的上下文，可辅助理解但不一定存在。

    image_uri: str | None = None
    # 可选图像引用；V1 允许理解模块只做轻量文本理解。


class QueryUnderstandingModule:
    """Query Understanding 模块接口。"""

    module_name: str = "query_understanding"
    # 模块稳定名称，供 orchestrator 和日志系统引用。

    def run(
        self, module_input: QueryUnderstandingModuleInput
    ) -> BaseModuleOutput[QueryUnderstandingResult]:
        """执行轻量 query understanding。"""
        raise NotImplementedError


@dataclass(slots=True)
class LLMQueryUnderstandingModule(QueryUnderstandingModule):
    """基于 LLM 后端的 Query Understanding 默认实现骨架。"""

    llm_adapter: LLMAdapter
    # 负责执行轻量语义结构化的理解后端。

    artifact_store: ArtifactStore
    # 负责保存 understanding 结构化结果和相关产物引用。

    module_name: str = "query_understanding"
    # 模块稳定名称，供 orchestrator 和 trace 使用。

    def run(
        self, module_input: QueryUnderstandingModuleInput
    ) -> BaseModuleOutput[QueryUnderstandingResult]:
        """调用理解后端并产出 QueryUnderstandingResult。"""
        raise NotImplementedError
