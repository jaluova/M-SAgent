"""Train adapter / embedded locate 的底层 runtime。"""

from __future__ import annotations

from contextlib import contextmanager
import json
from dataclasses import dataclass, field
from pathlib import Path

from msagent.core.contracts.common import ImageRef
from msagent.core.contracts.types import NormalizedBox
from msagent.infra.backbones import (
    EncodedFeatureHandle,
    FeatureSessionHandle,
    SharedQwenBackboneProvider,
    SharedVisionLanguageBackbone,
)

DEFAULT_EMBEDDED_GRIDGROUND_INSTRUCTION_TEMPLATE = (
    "Given the grid coordinate system, locate the referent described as "
    "'{query_text}' in the image and predict the most likely target points."
)
DEFAULT_EMBEDDED_GRIDGROUND_FOCUS_TERMS_TEMPLATE: str | None = None


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

    def reset(self) -> None:
        """重置 runtime 内部缓存。"""
        return None

    def close(self) -> None:
        """释放 runtime 持有的底层资源。"""
        self.reset()


@dataclass(slots=True)
class SharedBackboneTrainAdapterRuntime(TrainAdapterRuntime):
    """依赖共享骨干的 train adapter runtime 基类。

    真正的 embedded predictor 复杂性继续压在 runtime 内部；
    adapter 只依赖这个受控输出，不触碰骨干实现细节。
    """

    backbone_provider: SharedQwenBackboneProvider
    instruction_prefix: str = "Given the grid coordinate system, locate the referent described as"
    instruction_suffix: str = "in the image and predict the most likely target points."

    def build_instruction(self, request: TrainAdapterRuntimeRequest) -> str:
        """构造运行时消费的文本指令。"""
        base_instruction = (
            f"{self.instruction_prefix} '{request.query_text}' "
            f"{self.instruction_suffix}"
        )
        focus_terms = ", ".join(request.focus_terms)
        if focus_terms:
            return f"{base_instruction} Focus on: {focus_terms}."
        return base_instruction

    def resolve_text_max_length(
        self,
        request: TrainAdapterRuntimeRequest,
    ) -> int | None:
        """返回文本编码阶段使用的最大长度。"""
        del request
        return None

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
        try:
            instruction_text = self.build_instruction(request)
            image_features = backbone.encode_image(request.image_ref, session=session)
            text_features = backbone.encode_text(
                instruction_text,
                session=session,
                max_length=self.resolve_text_max_length(request),
            )
            return SharedBackboneFeatureContext(
                backbone=backbone,
                session=session,
                image_features=image_features,
                text_features=text_features,
                instruction_text=instruction_text,
            )
        except Exception:
            backbone.release_session(session)
            raise

    @contextmanager
    def feature_context(
        self,
        request: TrainAdapterRuntimeRequest,
    ):
        """以显式生命周期包装共享骨干特征上下文。"""
        context = self.prepare_feature_context(request)
        try:
            yield context
        finally:
            context.backbone.release_session(context.session)


@dataclass(slots=True)
class EmbeddedGridGroundRuntimeConfig:
    """embedded GridGround runtime 的最小配置。"""

    adapter_type: str = "lightweight"
    visual_dim: int = 768
    grid_feature_dim: int = 256
    hidden_dim: int = 256
    num_heads: int = 4
    num_grid_tokens: int = 25
    num_output_points: int = 4
    dropout: float = 0.1
    grid_size: int = 11
    max_length: int = 512
    abs_threshold: float = 0.50
    rel_ratio: float = 0.75
    min_k: int = 1
    max_k: int = 3
    min_point_confidence: float = 0.0
    box_margin_ratio: float = 0.12
    instruction_template: str = DEFAULT_EMBEDDED_GRIDGROUND_INSTRUCTION_TEMPLATE
    focus_terms_template: str | None = DEFAULT_EMBEDDED_GRIDGROUND_FOCUS_TERMS_TEMPLATE

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("EmbeddedGridGroundRuntimeConfig.max_length must be > 0")

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        abs_threshold: float = 0.50,
        rel_ratio: float = 0.75,
        min_k: int = 1,
        max_k: int = 3,
        min_point_confidence: float = 0.0,
        box_margin_ratio: float = 0.12,
    ) -> "EmbeddedGridGroundRuntimeConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = payload.get("model", {})
        data = payload.get("data", {})
        runtime = payload.get("runtime", {})
        defaults = cls()
        return cls(
            adapter_type=model.get("adapter_type", defaults.adapter_type),
            visual_dim=int(model.get("visual_dim", defaults.visual_dim)),
            grid_feature_dim=int(model.get("grid_feature_dim", defaults.grid_feature_dim)),
            hidden_dim=int(model.get("hidden_dim", defaults.hidden_dim)),
            num_heads=int(model.get("num_heads", defaults.num_heads)),
            num_grid_tokens=int(model.get("num_grid_tokens", defaults.num_grid_tokens)),
            num_output_points=int(model.get("num_output_points", defaults.num_output_points)),
            dropout=float(model.get("dropout", defaults.dropout)),
            grid_size=int(model.get("grid_size", defaults.grid_size)),
            max_length=int(data.get("max_length", defaults.max_length)),
            abs_threshold=abs_threshold,
            rel_ratio=rel_ratio,
            min_k=min_k,
            max_k=max_k,
            min_point_confidence=min_point_confidence,
            box_margin_ratio=box_margin_ratio,
            instruction_template=runtime.get(
                "instruction_template",
                DEFAULT_EMBEDDED_GRIDGROUND_INSTRUCTION_TEMPLATE,
            ),
            focus_terms_template=runtime.get(
                "focus_terms_template",
                DEFAULT_EMBEDDED_GRIDGROUND_FOCUS_TERMS_TEMPLATE,
            ),
        )


@dataclass(slots=True, kw_only=True)
class EmbeddedGridGroundTrainAdapterRuntime(SharedBackboneTrainAdapterRuntime):
    """shared Qwen + embedded locate 的最小真实 runtime。"""

    adapter_path: str
    config_path: str
    runtime_config: EmbeddedGridGroundRuntimeConfig
    _adapter_module: object | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_files(
        cls,
        *,
        runtime_name: str,
        backbone_provider: SharedQwenBackboneProvider,
        adapter_path: str | Path,
        config_path: str | Path,
        abs_threshold: float = 0.50,
        rel_ratio: float = 0.75,
        min_k: int = 1,
        max_k: int = 3,
        min_point_confidence: float = 0.0,
        box_margin_ratio: float = 0.12,
    ) -> "EmbeddedGridGroundTrainAdapterRuntime":
        config_path = str(Path(config_path).expanduser())
        return cls(
            runtime_name=runtime_name,
            backbone_provider=backbone_provider,
            adapter_path=str(Path(adapter_path).expanduser()),
            config_path=config_path,
            runtime_config=EmbeddedGridGroundRuntimeConfig.from_json_file(
                config_path,
                abs_threshold=abs_threshold,
                rel_ratio=rel_ratio,
                min_k=min_k,
                max_k=max_k,
                min_point_confidence=min_point_confidence,
                box_margin_ratio=box_margin_ratio,
            ),
        )

    def predict_embedded_locate(
        self,
        request: TrainAdapterRuntimeRequest,
    ) -> EmbeddedLocatePrediction:
        import torch

        with self.feature_context(request) as context:
            adapter = self._get_or_load_adapter(context.backbone)
            image_payload = context.backbone.resolve_feature(context.image_features)
            text_payload = context.backbone.resolve_feature(context.text_features)
            image_tensor, grid_image_tensor = self._preprocess_image(request.image_ref)

            adapter_device = next(adapter.parameters()).device
            adapter_dtype = next(adapter.parameters()).dtype
            visual_features = image_payload.tensor.to(device=adapter_device, dtype=adapter_dtype)
            text_features = text_payload.tensor.to(device=adapter_device, dtype=adapter_dtype)
            attention_mask = (
                text_payload.attention_mask.to(device=adapter_device)
                if text_payload.attention_mask is not None
                else None
            )
            image_tensor = image_tensor.to(device=adapter_device, dtype=adapter_dtype)
            grid_image_tensor = grid_image_tensor.to(device=adapter_device, dtype=adapter_dtype)

            with torch.no_grad():
                enhanced = adapter(
                    image_tensor,
                    grid_image_tensor,
                    visual_features,
                    text_features=text_features,
                    text_attention_mask=attention_mask,
                )
                grid_logits = adapter.predict_grid_logits(
                    enhanced,
                    text_features=text_features,
                    attention_mask=attention_mask,
                )
                selected = adapter.decode_grid_logits_dynamic(
                    grid_logits,
                    abs_threshold=self._resolve_float_option(
                        request.options,
                        "abs_threshold",
                        self.runtime_config.abs_threshold,
                    ),
                    rel_ratio=self._resolve_float_option(
                        request.options,
                        "rel_ratio",
                        self.runtime_config.rel_ratio,
                    ),
                    min_k=self._resolve_int_option(
                        request.options,
                        "min_k",
                        self.runtime_config.min_k,
                    ),
                    max_k=self._resolve_int_option(
                        request.options,
                        "max_k",
                        self.runtime_config.max_k,
                    ),
                )

            selected_points = selected["selected_points"][0].detach().cpu()
            selected_scores = selected["selected_scores"][0].detach().cpu()
            point_conf_threshold = self._resolve_float_option(
                request.options,
                "min_point_confidence",
                self.runtime_config.min_point_confidence,
            )
            points: list[EmbeddedLocatePoint] = []
            for idx, (point, score) in enumerate(zip(selected_points, selected_scores), start=1):
                confidence = round(float(score), 4)
                if confidence < point_conf_threshold:
                    continue
                points.append(
                    EmbeddedLocatePoint(
                        x=round(float(point[0]), 4),
                        y=round(float(point[1]), 4),
                        confidence=confidence,
                        reason=f"grid_peak_{idx}",
                    )
                )

            diagnostics = [
                f"selected_k={len(points)}",
                f"feature_session={context.session.session_id}",
            ]
            if not points:
                diagnostics.append("no_points_after_runtime_filter")
                return EmbeddedLocatePrediction(
                    runtime_name=self.runtime_name,
                    diagnostics=diagnostics,
                    metadata={
                        "adapter_path": self.adapter_path,
                        "config_path": self.config_path,
                        "backbone": context.backbone.backbone_name,
                    },
                    limitations=["runtime produced no point above confidence threshold"],
                )

            coarse_box = self._build_coarse_box(points)
            limitations: list[str] = []
            bridge_hints: list[EmbeddedLocateBridgeHint] = []
            if coarse_box is not None:
                limitations.append("coarse_box derived from selected point envelope")
                bridge_hints.append(
                    EmbeddedLocateBridgeHint(
                        hint_type="prefer_box_plus_points",
                        reason="embedded runtime produced both point priors and a derived coarse box",
                    )
                )
            return EmbeddedLocatePrediction(
                runtime_name=self.runtime_name,
                points=points,
                coarse_box=coarse_box,
                bridge_hints=bridge_hints,
                diagnostics=diagnostics,
                metadata={
                    "adapter_path": self.adapter_path,
                    "config_path": self.config_path,
                    "adapter_type": self.runtime_config.adapter_type,
                    "backbone": context.backbone.backbone_name,
                    "device": str(adapter_device),
                    "instruction_text": context.instruction_text,
                },
                limitations=limitations,
            )

    def build_instruction(self, request: TrainAdapterRuntimeRequest) -> str:
        instruction_text = self.runtime_config.instruction_template.format(
            query_text=request.query_text,
        )
        focus_terms = ", ".join(request.focus_terms)
        focus_terms_template = self.runtime_config.focus_terms_template
        if focus_terms and focus_terms_template:
            return f"{instruction_text} {focus_terms_template.format(focus_terms=focus_terms)}"
        return instruction_text

    def resolve_text_max_length(
        self,
        request: TrainAdapterRuntimeRequest,
    ) -> int | None:
        del request
        return self.runtime_config.max_length

    def reset(self) -> None:
        adapter = self._adapter_module
        self._adapter_module = None
        if adapter is None:
            return None

        try:
            adapter.to(device="cpu")
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return None

    def _get_or_load_adapter(self, backbone: SharedVisionLanguageBackbone) -> object:
        from msagent.infra.runtime.embedded_gridground_adapter import (
            CoordinateAdapter,
            LightweightCoordinateAdapter,
        )

        if self._adapter_module is None:
            self._align_runtime_config_with_backbone(backbone)
            adapter_kwargs = dict(
                visual_dim=self.runtime_config.visual_dim,
                grid_feature_dim=self.runtime_config.grid_feature_dim,
                hidden_dim=self.runtime_config.hidden_dim,
                num_heads=self.runtime_config.num_heads,
                num_grid_tokens=self.runtime_config.num_grid_tokens,
                num_output_points=self.runtime_config.num_output_points,
                dropout=self.runtime_config.dropout,
                grid_size=self.runtime_config.grid_size,
            )
            adapter_cls = (
                LightweightCoordinateAdapter
                if self.runtime_config.adapter_type == "lightweight"
                else CoordinateAdapter
            )
            prototype_adapter = adapter_cls(**adapter_kwargs)
            target_keys = set(prototype_adapter.state_dict().keys())
            self._adapter_module = self._load_adapter_module(
                adapter_cls=adapter_cls,
                adapter_kwargs=adapter_kwargs,
                target_keys=target_keys,
                prototype_adapter=prototype_adapter,
            )

        visual_device = backbone.visual_device() if hasattr(backbone, "visual_device") else backbone.device
        self._adapter_module.to(device=visual_device)
        return self._adapter_module

    def _load_adapter_module(
        self,
        *,
        adapter_cls: type[object],
        adapter_kwargs: dict[str, object],
        target_keys: set[str],
        prototype_adapter: object | None = None,
    ) -> object:
        errors: list[str] = []
        compact_checkpoint_path = self._compact_checkpoint_path()
        for index, checkpoint_path in enumerate(self._iter_candidate_checkpoint_paths()):
            adapter = (
                prototype_adapter
                if index == 0 and prototype_adapter is not None
                else adapter_cls(**adapter_kwargs)
            )
            try:
                state_dict = self._load_adapter_state_dict(
                    target_keys=target_keys,
                    checkpoint_path=checkpoint_path,
                )
                missing_keys, unexpected_keys = adapter.load_state_dict(state_dict, strict=False)
            except Exception as exc:
                errors.append(f"{checkpoint_path}: {exc}")
                continue
            if missing_keys:
                errors.append(f"{checkpoint_path}: Embedded runtime missing keys: {missing_keys}")
                continue
            if unexpected_keys:
                errors.append(
                    f"{checkpoint_path}: Embedded runtime unexpected keys: {unexpected_keys}"
                )
                continue
            if checkpoint_path != compact_checkpoint_path:
                try:
                    self._torch_save_checkpoint(compact_checkpoint_path, state_dict)
                except Exception:
                    pass
            adapter.eval()
            return adapter

        details = "; ".join(errors) if errors else "no checkpoint candidates were loadable"
        raise RuntimeError(f"Failed to load embedded runtime adapter checkpoint: {details}")

    def _load_adapter_state_dict(
        self,
        *,
        target_keys: set[str],
        checkpoint_path: Path | None = None,
    ) -> dict[str, object]:
        errors: list[str] = []
        checkpoint_paths = (
            [checkpoint_path]
            if checkpoint_path is not None
            else list(self._iter_candidate_checkpoint_paths())
        )

        for candidate_path in checkpoint_paths:
            try:
                checkpoint = self._torch_load_checkpoint(candidate_path)
                normalized = self._extract_adapter_state_dict(checkpoint, target_keys=target_keys)
            except Exception as exc:
                errors.append(f"{candidate_path}: {exc}")
                continue
            if not normalized:
                errors.append(
                    f"{candidate_path}: No adapter weights matched the embedded runtime checkpoint"
                )
                continue
            return normalized

        detail = "; ".join(errors) if errors else "no checkpoint candidates were attempted"
        raise RuntimeError(f"Failed to extract embedded runtime adapter weights: {detail}")

    def _iter_candidate_checkpoint_paths(self) -> tuple[Path, ...]:
        compact_checkpoint_path = self._compact_checkpoint_path()
        adapter_checkpoint_path = Path(self.adapter_path)
        candidates: list[Path] = []
        if compact_checkpoint_path.is_file():
            candidates.append(compact_checkpoint_path)
        if (
            adapter_checkpoint_path != compact_checkpoint_path
            and adapter_checkpoint_path.is_file()
        ):
            candidates.append(adapter_checkpoint_path)
        if not candidates:
            candidates.append(self._resolve_checkpoint_path())
        return tuple(candidates)

    def _resolve_checkpoint_path(self) -> Path:
        compact_checkpoint_path = self._compact_checkpoint_path()
        if compact_checkpoint_path.is_file():
            return compact_checkpoint_path
        return Path(self.adapter_path)

    def _compact_checkpoint_path(self) -> Path:
        adapter_path = Path(self.adapter_path)
        suffix = "".join(adapter_path.suffixes) or ".pth"
        if suffix == ".adapter_only.pth":
            return adapter_path
        stem = adapter_path.name
        if suffix:
            stem = stem[: -len(suffix)]
        return adapter_path.with_name(f"{stem}.adapter_only.pth")

    @staticmethod
    def _torch_load_checkpoint(checkpoint_path: Path) -> object:
        import torch

        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            try:
                return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(checkpoint_path, map_location="cpu")

    @classmethod
    def _extract_adapter_state_dict(
        cls,
        checkpoint: object,
        *,
        target_keys: set[str],
    ) -> dict[str, object]:
        best_match: dict[str, object] = {}
        saw_mapping_candidate = False
        for state_dict in cls._iter_state_dict_candidates(checkpoint):
            saw_mapping_candidate = True
            normalized: dict[str, object] = {}
            for key, value in state_dict.items():
                if not isinstance(key, str):
                    continue
                normalized_key = cls._normalize_state_key(key)
                if normalized_key in target_keys:
                    normalized[normalized_key] = value
            if len(normalized) > len(best_match):
                best_match = normalized
            if len(best_match) == len(target_keys):
                break

        if not saw_mapping_candidate:
            raise RuntimeError("GridGround checkpoint does not contain a loadable state_dict")
        return best_match

    @classmethod
    def _iter_state_dict_candidates(cls, checkpoint: object):
        if not isinstance(checkpoint, dict):
            return

        container_keys = (
            "model_state_dict",
            "state_dict",
            "adapter_state_dict",
            "adapter",
            "model",
            "module",
        )
        stack: list[dict[str, object]] = [checkpoint]
        visited: set[int] = set()
        while stack:
            candidate = stack.pop()
            candidate_id = id(candidate)
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            yield candidate
            for key in container_keys:
                nested = candidate.get(key)
                if isinstance(nested, dict):
                    stack.append(nested)

    @staticmethod
    def _torch_save_checkpoint(checkpoint_path: Path, payload: dict[str, object]) -> None:
        import torch

        torch.save(payload, checkpoint_path)

    @staticmethod
    def _normalize_state_key(key: str) -> str:
        prefixes = (
            "module.adapter.",
            "adapter.",
            "module.",
            "model.adapter.",
            "model.",
        )
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
                    changed = True
        return normalized

    def _align_runtime_config_with_backbone(self, backbone: SharedVisionLanguageBackbone) -> None:
        backbone_dim = getattr(backbone, "text_dim", None) or getattr(backbone, "visual_dim", None)
        if backbone_dim and int(backbone_dim) != int(self.runtime_config.visual_dim):
            self.runtime_config.visual_dim = int(backbone_dim)

    def _preprocess_image(self, image_ref: ImageRef) -> tuple[object, object]:
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        image_path = self._resolve_image_path(image_ref)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_tensor = transform(rgb_image).unsqueeze(0)
            grid_image_tensor = transform(self._build_grid_image(rgb_image)).unsqueeze(0)
        return image_tensor, grid_image_tensor

    def _build_grid_image(self, image: Image.Image) -> Image.Image:
        from PIL import Image, ImageDraw

        width, height = image.size
        border_size = 28
        grid_image = Image.new("RGB", (width + border_size * 2, height + border_size * 2), "white")
        grid_image.paste(image, (border_size, border_size))
        draw = ImageDraw.Draw(grid_image)
        grid_font = self._load_font(15, bold=False)
        for index in range(self.runtime_config.grid_size):
            x = border_size + index * (width / max(self.runtime_config.grid_size - 1, 1))
            y = border_size + index * (height / max(self.runtime_config.grid_size - 1, 1))
            draw.line([(x, border_size), (x, border_size + height)], fill="black", width=1)
            draw.line([(border_size, y), (border_size + width, y)], fill="black", width=1)
            label = str(index)
            bbox = draw.textbbox((0, 0), label, font=grid_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((x - text_w / 2, border_size - text_h - 5), label, fill="black", font=grid_font)
            draw.text((border_size - text_w - 5, y - text_h / 2), label, fill="black", font=grid_font)
        return grid_image

    @staticmethod
    def _load_font(size: int, *, bold: bool) -> object:
        from PIL import ImageFont

        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
            if bold
            else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        for path in candidates:
            candidate = Path(path)
            if not candidate.exists():
                continue
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _build_coarse_box(
        self,
        points: list[EmbeddedLocatePoint],
    ) -> NormalizedBox | None:
        if not points:
            return None
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        margin = max(
            self.runtime_config.box_margin_ratio,
            1.0 / max(self.runtime_config.grid_size - 1, 1),
        )
        return NormalizedBox(
            x1=max(0.0, min(xs) - margin),
            y1=max(0.0, min(ys) - margin),
            x2=min(1.0, max(xs) + margin),
            y2=min(1.0, max(ys) + margin),
        )

    @staticmethod
    def _resolve_image_path(image_ref: ImageRef) -> Path:
        uri = image_ref.uri
        if uri.startswith("file://"):
            return Path(uri[7:]).expanduser()
        return Path(uri).expanduser()

    @staticmethod
    def _resolve_float_option(
        options: dict[str, object],
        key: str,
        default: float,
    ) -> float:
        value = options.get(key)
        if value is None:
            return float(default)
        return float(value)

    @staticmethod
    def _resolve_int_option(
        options: dict[str, object],
        key: str,
        default: int,
    ) -> int:
        value = options.get(key)
        if value is None:
            return int(default)
        return int(value)
