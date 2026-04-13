"""定义 Proposal Engine 模块骨架。

该模块负责产出“怎么找目标”的候选空间先验。
V1 先把 locate route 做成必选通路，同时为 crop / rewrite 预留接口。
跨模块共享的 proposal 类型统一放在 contracts/types 下。
"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.common import BaseModuleInput, BaseModuleOutput
from msagent.core.contracts.types import ProposalResult, ProposalRoute, QueryUnderstandingResult
from msagent.infra.adapters import ArtifactStore, LocatorAdapter


@dataclass(slots=True)
class ProposalEngineModuleInput(BaseModuleInput):
    """Proposal Engine 模块输入 DTO。"""

    understanding: QueryUnderstandingResult | None = None
    # 本轮 query understanding 的权威实体对象。

    preferred_route: ProposalRoute = ProposalRoute.LOCATE
    # orchestrator 当前希望优先走的 route。


class ProposalRouteHandler:
    """单条 proposal route 的处理接口。"""

    route_name: ProposalRoute
    # 当前处理器负责的 route 类型。

    def build_proposal(self, module_input: ProposalEngineModuleInput) -> ProposalResult:
        """基于输入构建 proposal 结果。"""
        raise NotImplementedError


class ProposalEngineModule:
    """Proposal Engine 模块接口。"""

    module_name: str = "proposal_engine"
    # 模块稳定名称，供 orchestrator 与 trace 使用。

    route_handlers: dict[ProposalRoute, ProposalRouteHandler]
    # route 到处理器的注册表，用于扩展 locate / crop / rewrite。

    def run(
        self, module_input: ProposalEngineModuleInput
    ) -> BaseModuleOutput[ProposalResult]:
        """根据 route 调用对应处理器，产出 ProposalResult。"""
        raise NotImplementedError


@dataclass(slots=True)
class LocateProposalRouteHandler(ProposalRouteHandler):
    """V1 必须实现的 locate route 处理器骨架。"""

    locator_adapter: LocatorAdapter
    # 对接 GridGround、TrainAdapter 或未来其他定位后端。

    route_name: ProposalRoute = ProposalRoute.LOCATE
    # 当前处理器固定负责 locate route。

    def build_proposal(self, module_input: ProposalEngineModuleInput) -> ProposalResult:
        """基于定位后端构建 locate proposal。"""
        raise NotImplementedError


@dataclass(slots=True)
class CropProposalRouteHandler(ProposalRouteHandler):
    """未来 crop route 的预留处理器骨架。"""

    route_name: ProposalRoute = ProposalRoute.CROP
    # 当前处理器固定负责 crop route。

    def build_proposal(self, module_input: ProposalEngineModuleInput) -> ProposalResult:
        """为未来 crop route 预留 proposal 构建入口。"""
        raise NotImplementedError


@dataclass(slots=True)
class RewriteProposalRouteHandler(ProposalRouteHandler):
    """未来 rewrite route 的预留处理器骨架。"""

    route_name: ProposalRoute = ProposalRoute.REWRITE
    # 当前处理器固定负责 rewrite route。

    def build_proposal(self, module_input: ProposalEngineModuleInput) -> ProposalResult:
        """为未来 rewrite route 预留 proposal 构建入口。"""
        raise NotImplementedError


@dataclass(slots=True)
class DefaultProposalEngineModule(ProposalEngineModule):
    """Proposal Engine 默认实现骨架。"""

    route_handlers: dict[ProposalRoute, ProposalRouteHandler]
    # route 到处理器的映射表，是 proposal engine 的核心成员。

    artifact_store: ArtifactStore
    # 负责保存 proposal 结果与相关调试产物。

    module_name: str = "proposal_engine"
    # 模块稳定名称。

    def run(
        self, module_input: ProposalEngineModuleInput
    ) -> BaseModuleOutput[ProposalResult]:
        """根据 preferred_route 派发到对应 route handler。"""
        raise NotImplementedError
