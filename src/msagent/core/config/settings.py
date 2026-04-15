"""定义新内核的配置对象骨架。

当前只给出结构化配置类，便于后续接入环境变量、文件配置或实验配置。
本阶段不承担具体解析逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.types import ProposalRoute


@dataclass(slots=True)
class ModelPathConfig:
    """模型与权重路径配置。"""

    qwen_model_path: str | None = None
    # LLM 或多模态理解后端的模型路径。

    locator_model_path: str | None = None
    # 定位后端相关模型或资源路径。

    embedded_locator_adapter_path: str | None = None
    # embedded locate runtime 的 adapter checkpoint 路径。

    embedded_locator_config_path: str | None = None
    # embedded locate runtime 的结构化配置路径。

    sam_model_path: str | None = None
    # 分割后端代码或模型目录。

    sam_checkpoint_path: str | None = None
    # 分割后端权重路径。

    def has_embedded_locator_runtime(self) -> bool:
        """判断真实 embedded locator 装配所需路径是否齐全。"""
        required_paths = (
            self.qwen_model_path,
            self.embedded_locator_adapter_path,
            self.embedded_locator_config_path,
        )
        return all(path is not None and path.strip() for path in required_paths)

    def has_partial_embedded_locator_runtime(self) -> bool:
        """判断 embedded locator 路径是否处于半配置状态。"""
        configured = [
            path
            for path in (
                self.qwen_model_path,
                self.embedded_locator_adapter_path,
                self.embedded_locator_config_path,
            )
            if path is not None and path.strip()
        ]
        return bool(configured) and len(configured) < 3


@dataclass(slots=True)
class ServiceConfig:
    """服务层配置。"""

    host: str = "127.0.0.1"
    # API 服务默认监听地址。

    port: int = 8000
    # API 服务默认端口。

    enable_api: bool = True
    # 是否启用 API 层骨架。

    enable_cli: bool = True
    # 是否启用 CLI 层骨架。

    enable_real_llm: bool = False
    # 是否启用基于真实 Qwen 的 LLM adapter 装配。


@dataclass(slots=True)
class RuntimeConfig:
    """运行时基础配置。"""

    max_attempts: int = 3
    # 有限重试的默认最大轮次。

    default_route: ProposalRoute = ProposalRoute.LOCATE
    # V1 默认优先采用 locate route。

    artifact_root: str = "artifacts"
    # 任务产物默认根目录。


@dataclass(slots=True)
class MSAgentSettings:
    """系统总配置入口。"""

    model_paths: ModelPathConfig = field(default_factory=ModelPathConfig)
    # 模型、权重与外部资源路径配置。

    service: ServiceConfig = field(default_factory=ServiceConfig)
    # CLI / API 等服务入口配置。

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # 运行时策略与通用参数配置。
