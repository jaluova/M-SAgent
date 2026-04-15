"""真实 embedded runtime 的装配工厂。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from msagent.infra.embedded_locator import EmbeddedLocatorAdapter
from msagent.infra.runtime.shared_qwen_backbone import QwenSharedBackboneProvider
from msagent.infra.runtime.train_adapter_runtime import EmbeddedGridGroundTrainAdapterRuntime


@dataclass(slots=True)
class EmbeddedLocatorRuntimeFactoryConfig:
    """构造 embedded locator 运行时所需的最小配置。"""

    qwen_model_path: str
    adapter_path: str
    config_path: str
    provider_name: str = "service-shared-qwen-provider"
    runtime_name: str = "service-embedded-gridground-runtime"
    locator_backend_name: str = "embedded-locator"
    device_map: str | None = "auto"
    torch_dtype: str | None = "auto"
    attn_implementation: str | None = None
    abs_threshold: float = 0.50
    rel_ratio: float = 0.75
    min_k: int = 1
    max_k: int = 3
    min_point_confidence: float = 0.0
    box_margin_ratio: float = 0.12


@dataclass(slots=True)
class EmbeddedLocatorRuntimeBundle:
    """把 provider / runtime / adapter 作为一个受控装配单元返回。"""

    locator_adapter: EmbeddedLocatorAdapter
    runtime: EmbeddedGridGroundTrainAdapterRuntime
    backbone_provider: QwenSharedBackboneProvider

    def close(self) -> None:
        self.runtime.close()
        self.backbone_provider.close()


def build_embedded_locator_runtime_bundle(
    config: EmbeddedLocatorRuntimeFactoryConfig,
) -> EmbeddedLocatorRuntimeBundle:
    """构造可直接接入 proposal engine 的 embedded locator bundle。"""
    backbone_provider = QwenSharedBackboneProvider(
        provider_name=config.provider_name,
        model_path=str(Path(config.qwen_model_path).expanduser()),
        device_map=config.device_map,
        torch_dtype=config.torch_dtype,
        attn_implementation=config.attn_implementation,
    )
    runtime = EmbeddedGridGroundTrainAdapterRuntime.from_files(
        runtime_name=config.runtime_name,
        backbone_provider=backbone_provider,
        adapter_path=str(Path(config.adapter_path).expanduser()),
        config_path=str(Path(config.config_path).expanduser()),
        abs_threshold=config.abs_threshold,
        rel_ratio=config.rel_ratio,
        min_k=config.min_k,
        max_k=config.max_k,
        min_point_confidence=config.min_point_confidence,
        box_margin_ratio=config.box_margin_ratio,
    )
    return EmbeddedLocatorRuntimeBundle(
        locator_adapter=EmbeddedLocatorAdapter(
            backend_name=config.locator_backend_name,
            model_path=str(Path(config.adapter_path).expanduser()),
            runtime=runtime,
        ),
        runtime=runtime,
        backbone_provider=backbone_provider,
    )
