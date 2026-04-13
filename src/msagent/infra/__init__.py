"""外部模型与存储适配层。

当前阶段对上层稳定暴露的 shared-backbone 入口只有 provider 抽象。
feature handle、session 句柄和具体 embedded locator 实现都视为 infra 内部细节。
"""

from msagent.infra.backbones import SharedQwenBackboneProvider

__all__ = ["SharedQwenBackboneProvider"]
