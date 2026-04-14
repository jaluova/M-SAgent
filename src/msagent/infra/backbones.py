"""共享视觉语言骨干的 infra 抽象。

这里承接的是“共享 Qwen + embedded locate”所需的底层运行时能力，
只负责骨干访问、编码入口和特征会话，不上浮业务语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from msagent.core.contracts.common import ImageRef


@dataclass(slots=True, frozen=True)
class FeatureSessionHandle:
    """一次共享骨干特征会话的句柄。"""

    session_id: str
    backbone_name: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EncodedFeatureHandle:
    """编码后特征的受控句柄。"""

    feature_id: str
    feature_kind: str
    backbone_name: str
    session_id: str | None = None
    token_count: int | None = None
    hidden_dim: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ResolvedFeaturePayload:
    """infra 内部消费的特征载荷。"""

    tensor: object
    attention_mask: object | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SharedVisionLanguageBackbone:
    """共享视觉语言骨干的最小接口骨架。"""

    backbone_name: str
    tokenizer: object | None = None
    device: str = "cpu"
    dtype: str = "float32"

    def open_feature_session(
        self,
        *,
        task_id: str,
        metadata: dict[str, object] | None = None,
    ) -> FeatureSessionHandle:
        """创建一次受控特征会话。"""
        return FeatureSessionHandle(
            session_id=f"{task_id}-{uuid4().hex[:8]}",
            backbone_name=self.backbone_name,
            metadata=dict(metadata or {}),
        )

    def encode_image(
        self,
        image_ref: ImageRef,
        *,
        session: FeatureSessionHandle | None = None,
    ) -> EncodedFeatureHandle:
        """编码图像，返回受控特征句柄。"""
        raise NotImplementedError

    def encode_text(
        self,
        text: str,
        *,
        session: FeatureSessionHandle | None = None,
        max_length: int | None = None,
    ) -> EncodedFeatureHandle:
        """编码文本，返回受控特征句柄。"""
        raise NotImplementedError

    def resolve_feature(self, handle: EncodedFeatureHandle) -> ResolvedFeaturePayload:
        """按句柄取回 runtime 内部使用的特征载荷。"""
        raise NotImplementedError

    def release_session(self, session: FeatureSessionHandle) -> None:
        """释放会话相关缓存。"""
        return None


@dataclass(slots=True)
class SharedQwenBackboneProvider:
    """共享 Qwen 骨干 provider 的最小抽象。

    provider 只负责骨干实例访问和生命周期管理，不承载 locate / understand
    这类业务入口。
    """

    provider_name: str
    model_path: str | None = None

    def get_backbone(self) -> SharedVisionLanguageBackbone:
        """返回当前 provider 管理的共享骨干。"""
        raise NotImplementedError

    def close(self) -> None:
        """释放 provider 持有的底层资源。"""
        return None
