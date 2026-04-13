"""Train adapter / embedded locate 的底层 runtime 骨架。"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.common import ImageRef
from msagent.core.contracts.types import NormalizedBox
from msagent.infra.backbones import (
    EncodedFeatureHandle,
    FeatureSessionHandle,
    SharedQwenBackboneProvider,
    SharedVisionLanguageBackbone,
)


@dataclass(slots=True)
class EmbeddedLocateBridgeHint:
    """runtime 输出给 adapter 的弱桥接提示。"""

    hint_type: str
    reason: str


@dataclass(slots=True)
class EmbeddedLocatePoint:
    """embedded locate 产出的归一化点提示。"""

    x: float
    y: float
    confidence: float | None = None
    reason: str | None = None


@dataclass(slots=True)
class EmbeddedLocatePrediction:
    """embedded locate runtime 的受控输出。"""

    runtime_name: str
    points: list[EmbeddedLocatePoint] = field(default_factory=list)
    coarse_box: NormalizedBox | None = None
    bridge_hints: list[EmbeddedLocateBridgeHint] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    @property
    def top_confidence(self) -> float | None:
        confidences = [point.confidence for point in self.points if point.confidence is not None]
        if not confidences:
            return None
        return max(confidences)


@dataclass(slots=True)
class TrainAdapterRuntimeRequest:
    """传给 train adapter runtime 的结构化输入。"""

    task_id: str
    image_ref: ImageRef
    query_text: str
    focus_terms: list[str] = field(default_factory=list)
    options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SharedBackboneFeatureContext:
    """shared backbone 预编码后的运行时上下文。"""

    backbone: SharedVisionLanguageBackbone
    session: FeatureSessionHandle
    image_features: EncodedFeatureHandle
    text_features: EncodedFeatureHandle
    instruction_text: str


@dataclass(slots=True)
class TrainAdapterRuntime:
    """train adapter runtime 的统一骨架。"""

    runtime_name: str

    def predict_embedded_locate(
        self,
        request: TrainAdapterRuntimeRequest,
    ) -> EmbeddedLocatePrediction:
        """执行 embedded locate 推理。"""
        raise NotImplementedError


@dataclass(slots=True)
class SharedBackboneTrainAdapterRuntime(TrainAdapterRuntime):
    """依赖共享骨干的 train adapter runtime 基类。

    真正的 embedded predictor 复杂性继续压在 runtime 内部；
    adapter 只依赖这个受控输出，不触碰骨干实现细节。
    """

    backbone_provider: SharedQwenBackboneProvider
    instruction_prefix: str = "Locate the referent described as"

    def build_instruction(self, request: TrainAdapterRuntimeRequest) -> str:
        """构造运行时消费的文本指令。"""
        focus_terms = ", ".join(request.focus_terms)
        if focus_terms:
            return (
                f"{self.instruction_prefix} '{request.query_text}'. "
                f"Focus on: {focus_terms}."
            )
        return f"{self.instruction_prefix} '{request.query_text}'."

    def prepare_feature_context(
        self,
        request: TrainAdapterRuntimeRequest,
    ) -> SharedBackboneFeatureContext:
        """通过共享 provider 准备图像与文本特征句柄。"""
        backbone = self.backbone_provider.get_backbone()
        session = backbone.open_feature_session(
            task_id=request.task_id,
            metadata={"runtime_name": self.runtime_name},
        )
        instruction_text = self.build_instruction(request)
        return SharedBackboneFeatureContext(
            backbone=backbone,
            session=session,
            image_features=backbone.encode_image(request.image_ref, session=session),
            text_features=backbone.encode_text(instruction_text, session=session),
            instruction_text=instruction_text,
        )
