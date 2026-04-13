"""定义 Prompt Bridge 模块骨架。

Prompt Bridge 是 V1 新架构的关键层，负责把上游理解结果和 proposal
整理成 segmenter 可稳定消费的 PromptPackage。
跨模块共享的 prompt 类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.common import BaseModuleInput, BaseModuleOutput
from msagent.core.contracts.types import PromptPackage, ProposalResult, QueryUnderstandingResult
from msagent.infra.adapters import ArtifactStore


@dataclass(slots=True)
class PromptBridgeModuleInput(BaseModuleInput):
    """Prompt Bridge 模块输入 DTO。"""

    understanding: QueryUnderstandingResult | None = None
    # query understanding 的权威实体对象，是本模块的正式业务输入。

    proposal: ProposalResult | None = None
    # proposal engine 的权威实体对象，是本模块的正式业务输入。

    raw_query: str = ""
    # 用户原始 query，便于构造 raw_text prompt。


class PromptBridgeModule:
    """Prompt Bridge 模块接口。"""

    module_name: str = "prompt_bridge"
    # 模块稳定名称。

    def run(
        self, module_input: PromptBridgeModuleInput
    ) -> BaseModuleOutput[PromptPackage]:
        """将上游结构化结果转换为 PromptPackage。"""
        raise NotImplementedError


@dataclass(slots=True)
class RuleBasedPromptBridgeModule(PromptBridgeModule):
    """基于规则的 Prompt Bridge 默认实现骨架。"""

    artifact_store: ArtifactStore
    # 负责保存 prompt package 与桥接阶段中间产物。

    package_version: str = "v1"
    # 当前输出 PromptPackage 所采用的 schema 版本。

    module_name: str = "prompt_bridge"
    # 模块稳定名称。

    def run(
        self, module_input: PromptBridgeModuleInput
    ) -> BaseModuleOutput[PromptPackage]:
        """把 understanding 与 proposal 组织成 PromptPackage。"""
        raise NotImplementedError
