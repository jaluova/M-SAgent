"""共享 Qwen backbone 的最小真实实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from msagent.core.contracts.common import ImageRef
from msagent.infra.backbones import (
    EncodedFeatureHandle,
    FeatureSessionHandle,
    ResolvedFeaturePayload,
    SharedQwenBackboneProvider,
    SharedVisionLanguageBackbone,
)


@dataclass(slots=True, kw_only=True)
class QwenSharedVisionLanguageBackbone(SharedVisionLanguageBackbone):
    """对 Qwen2.5-VL 编码能力的受控封装。"""

    qwen_model: object = field(repr=False)
    qwen_processor: object = field(repr=False)
    merge_size: int = 1
    visual_dim: int | None = None
    text_dim: int | None = None
    _feature_payloads: dict[str, ResolvedFeaturePayload] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _session_feature_ids: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_loaded_components(
        cls,
        *,
        qwen_model: object,
        qwen_processor: object,
        backbone_name: str,
    ) -> "QwenSharedVisionLanguageBackbone":
        tokenizer = getattr(qwen_processor, "tokenizer", None)
        merge_size = getattr(getattr(qwen_processor, "image_processor", None), "merge_size", 1)
        return cls(
            backbone_name=backbone_name,
            tokenizer=tokenizer,
            device=str(cls._module_device(qwen_model)),
            dtype=cls._resolve_dtype_name(qwen_model),
            qwen_model=qwen_model,
            qwen_processor=qwen_processor,
            merge_size=merge_size,
            visual_dim=cls._resolve_hidden_size(
                getattr(qwen_model, "config", None),
                getattr(getattr(qwen_model, "visual", None), "config", None),
                getattr(getattr(qwen_model, "config", None), "vision_config", None),
            ),
            text_dim=cls._resolve_hidden_size(
                getattr(qwen_model, "config", None),
                getattr(getattr(qwen_model, "config", None), "text_config", None),
            ),
        )

    @staticmethod
    def _resolve_hidden_size(*configs: object) -> int | None:
        for config in configs:
            if config is None:
                continue
            for attr in ("hidden_size", "embed_dim", "out_hidden_size", "d_model"):
                value = getattr(config, attr, None)
                if value is not None:
                    return int(value)
        return None

    @staticmethod
    def _module_device(module: object) -> object:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return "cpu"

    @staticmethod
    def _resolve_dtype_name(module: object) -> str:
        try:
            return str(next(module.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            return "float32"

    def visual_device(self) -> object:
        if hasattr(self.qwen_model, "visual"):
            return self._module_device(self.qwen_model.visual)
        if hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "visual"):
            return self._module_device(self.qwen_model.model.visual)
        return self._module_device(self.qwen_model)

    def text_device(self) -> object:
        if hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "embed_tokens"):
            return self._module_device(self.qwen_model.model.embed_tokens)
        if hasattr(self.qwen_model, "get_input_embeddings"):
            return self._module_device(self.qwen_model.get_input_embeddings())
        return self._module_device(self.qwen_model)

    def open_feature_session(
        self,
        *,
        task_id: str,
        metadata: dict[str, object] | None = None,
    ) -> FeatureSessionHandle:
        handle = super(QwenSharedVisionLanguageBackbone, self).open_feature_session(
            task_id=task_id,
            metadata=metadata,
        )
        self._session_feature_ids.setdefault(handle.session_id, [])
        return handle

    def encode_image(
        self,
        image_ref: ImageRef,
        *,
        session: FeatureSessionHandle | None = None,
    ) -> EncodedFeatureHandle:
        import torch
        from PIL import Image

        image_path = self._resolve_image_path(image_ref)
        with Image.open(image_path) as image:
            pil_images = [image.convert("RGB")]
        image_inputs = self.qwen_processor.image_processor(
            pil_images,
            return_tensors="pt",
        )
        pixel_values = image_inputs["pixel_values"].to(self.visual_device())
        image_grid_thw = image_inputs["image_grid_thw"].to(self.visual_device())

        with torch.no_grad():
            if hasattr(self.qwen_model, "get_image_features"):
                outputs = self.qwen_model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                feature_tensor = self._coerce_visual_features(getattr(outputs, "pooler_output", outputs))
            elif hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "get_image_features"):
                outputs = self.qwen_model.model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                feature_tensor = self._coerce_visual_features(getattr(outputs, "pooler_output", outputs))
            elif hasattr(self.qwen_model, "visual"):
                outputs = self.qwen_model.visual(pixel_values, image_grid_thw)
                feature_tensor = self._pack_visual_outputs(outputs, image_grid_thw)
            else:
                raise AttributeError("Qwen model does not expose a supported visual encoder")

        return self._register_feature(
            feature_kind="image",
            tensor=feature_tensor,
            session=session,
            metadata={"uri": image_ref.uri},
        )

    def encode_text(
        self,
        text: str,
        *,
        session: FeatureSessionHandle | None = None,
        max_length: int | None = None,
    ) -> EncodedFeatureHandle:
        import torch

        if self.tokenizer is None:
            raise RuntimeError("Shared Qwen backbone does not expose a tokenizer")
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length or 512,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.text_device())
        attention_mask = encoding["attention_mask"].to(self.text_device())

        with torch.no_grad():
            if hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "embed_tokens"):
                tensor = self.qwen_model.model.embed_tokens(input_ids)
            elif hasattr(self.qwen_model, "get_input_embeddings"):
                tensor = self.qwen_model.get_input_embeddings()(input_ids)
            else:
                raise AttributeError("Qwen model does not expose a supported text embedding layer")

        return self._register_feature(
            feature_kind="text",
            tensor=tensor,
            attention_mask=attention_mask,
            session=session,
            metadata={"text": text},
        )

    def resolve_feature(self, handle: EncodedFeatureHandle) -> ResolvedFeaturePayload:
        try:
            return self._feature_payloads[handle.feature_id]
        except KeyError as exc:
            raise KeyError(f"Unknown feature handle: {handle.feature_id}") from exc

    def release_session(self, session: FeatureSessionHandle) -> None:
        feature_ids = self._session_feature_ids.pop(session.session_id, [])
        for feature_id in feature_ids:
            self._feature_payloads.pop(feature_id, None)

    def _register_feature(
        self,
        *,
        feature_kind: str,
        tensor: object,
        session: FeatureSessionHandle | None,
        attention_mask: object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EncodedFeatureHandle:
        shape = getattr(tensor, "shape", ())
        token_count = int(shape[1]) if len(shape) >= 2 else None
        hidden_dim = int(shape[-1]) if len(shape) >= 1 else None
        feature_id = uuid4().hex
        handle = EncodedFeatureHandle(
            feature_id=feature_id,
            feature_kind=feature_kind,
            backbone_name=self.backbone_name,
            session_id=session.session_id if session else None,
            token_count=token_count,
            hidden_dim=hidden_dim,
            metadata=dict(metadata or {}),
        )
        self._feature_payloads[feature_id] = ResolvedFeaturePayload(
            tensor=tensor,
            attention_mask=attention_mask,
            metadata=dict(metadata or {}),
        )
        if session is not None:
            self._session_feature_ids.setdefault(session.session_id, []).append(feature_id)
        return handle

    def _pack_visual_outputs(self, outputs: object, image_grid_thw: object) -> object:
        token_counts = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ) // (self.merge_size**2)
        token_counts = token_counts.tolist()

        chunks = []
        start = 0
        for count in token_counts:
            end = start + count
            chunks.append(outputs[start:end])
            start = end

        max_tokens = max(token_counts)
        padded = outputs.new_zeros((len(chunks), max_tokens, outputs.shape[-1]))
        for idx, chunk in enumerate(chunks):
            padded[idx, : chunk.shape[0]] = chunk
        return padded

    @staticmethod
    def _pad_feature_chunks(chunks: list[object]) -> object:
        if not chunks:
            raise ValueError("No visual feature chunks were returned by Qwen")
        max_tokens = max(chunk.shape[0] for chunk in chunks)
        hidden_dim = chunks[0].shape[-1]
        padded = chunks[0].new_zeros((len(chunks), max_tokens, hidden_dim))
        for idx, chunk in enumerate(chunks):
            padded[idx, : chunk.shape[0]] = chunk
        return padded

    def _coerce_visual_features(self, outputs: object) -> object:
        import torch

        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                return outputs
            if outputs.ndim == 2:
                return outputs.unsqueeze(0)
            raise ValueError(f"Unsupported visual tensor shape: {tuple(outputs.shape)}")

        if isinstance(outputs, (list, tuple)):
            chunks: list[object] = []
            for chunk in outputs:
                if not isinstance(chunk, torch.Tensor):
                    raise TypeError(f"Unsupported visual feature chunk type: {type(chunk)!r}")
                if chunk.ndim == 1:
                    chunk = chunk.unsqueeze(0)
                chunks.append(chunk)
            return self._pad_feature_chunks(chunks)

        raise TypeError(f"Unsupported visual feature container: {type(outputs)!r}")

    @staticmethod
    def _resolve_image_path(image_ref: ImageRef) -> Path:
        parsed = urlparse(image_ref.uri)
        if parsed.scheme in ("", "file"):
            candidate = Path(parsed.path if parsed.scheme == "file" else image_ref.uri).expanduser()
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"ImageRef does not resolve to a local file: {image_ref.uri}")


@dataclass(slots=True, kw_only=True)
class QwenSharedBackboneProvider(SharedQwenBackboneProvider):
    """共享 Qwen backbone provider 的真实实现。"""

    backbone_name: str = "qwen2.5-vl-shared-backbone"
    device_map: str | None = "auto"
    torch_dtype: str | None = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    loaded_model: object | None = field(default=None, repr=False)
    loaded_processor: object | None = field(default=None, repr=False)
    _backbone: QwenSharedVisionLanguageBackbone | None = field(default=None, init=False, repr=False)

    def get_backbone(self) -> SharedVisionLanguageBackbone:
        if self._backbone is None:
            model = self.loaded_model
            processor = self.loaded_processor
            if model is None or processor is None:
                model, processor = self._load_components()
            self._backbone = QwenSharedVisionLanguageBackbone.from_loaded_components(
                qwen_model=model,
                qwen_processor=processor,
                backbone_name=self.backbone_name,
            )
        return self._backbone

    def close(self) -> None:
        self._backbone = None
        self.loaded_model = None
        self.loaded_processor = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return None

    def _load_components(self) -> tuple[object, object]:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not self.model_path:
            raise ValueError("QwenSharedBackboneProvider requires model_path to load components")

        model_kwargs: dict[str, object] = {"trust_remote_code": self.trust_remote_code}
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        torch_dtype = self._resolve_torch_dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.loaded_model = model
        self.loaded_processor = processor
        return model, processor

    def _resolve_torch_dtype(self) -> object | None:
        if not self.torch_dtype or self.torch_dtype == "auto":
            return self.torch_dtype

        import torch

        try:
            return getattr(torch, self.torch_dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype: {self.torch_dtype}") from exc


__all__ = [
    "QwenSharedBackboneProvider",
    "QwenSharedVisionLanguageBackbone",
]
