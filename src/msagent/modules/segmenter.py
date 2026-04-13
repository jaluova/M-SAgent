"""定义 Segmenter 模块骨架。

该模块只负责消费 PromptPackage 并执行分割，不参与高层理解和调度。
V1 默认以 SAM3 作为后端，但这里保持 segmenter-agnostic 的接口形态。
跨模块共享的分割类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.common import BaseModuleInput, BaseModuleOutput
from msagent.core.contracts.types import PromptPackage, SegmentationResult
from msagent.infra.adapters import ArtifactStore, SAMAdapter


@dataclass(slots=True)
class SegmenterModuleInput(BaseModuleInput):
    """Segmenter 模块输入 DTO。"""

    image_uri: str = ""
    # 当前待分割图像的引用。

    prompt_package: PromptPackage | None = None
    # Prompt Bridge 产出的权威实体对象，是本模块的正式业务输入。


class SegmenterModule:
    """Segmenter 模块接口。"""

    module_name: str = "segmenter"
    # 模块稳定名称。

    def run(self, module_input: SegmenterModuleInput) -> BaseModuleOutput[SegmentationResult]:
        """执行一次分割并返回结构化分割结果。"""
        raise NotImplementedError


@dataclass(slots=True)
class SAMSegmenterModule(SegmenterModule):
    """基于 SAM 后端的 Segmenter 默认实现骨架。"""

    sam_adapter: SAMAdapter
    # 分割后端适配器，当前默认目标是 SAM3。

    artifact_store: ArtifactStore
    # 负责保存 mask、预览图和结构化分割结果引用。

    module_name: str = "segmenter"
    # 模块稳定名称。

    def run(self, module_input: SegmenterModuleInput) -> BaseModuleOutput[SegmentationResult]:
        """调用 SAM 后端完成一次基于 PromptPackage 的分割。"""
        raise NotImplementedError
