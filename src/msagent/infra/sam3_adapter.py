"""可选真实 SAM3 adapter 装配。"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
import sys
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

from msagent.core.contracts.adapter_requests import SegmentAdapterRequest
from msagent.core.contracts.common import ArtifactKind
from msagent.core.contracts.types import (
    NormalizedBox,
    PromptPackage,
    SegmentationCandidate,
    SegmentationResult,
    SegmentationStatus,
)
from msagent.infra.adapters import ArtifactStore, SAMAdapter
from msagent.infra.mask_artifact import MaskArtifact


@dataclass(slots=True)
class RealSAM3AdapterConfig:
    """真实 SAM3 adapter 的最小配置。"""

    sam_model_path: str
    checkpoint_path: str
    backend_name: str = "sam3-real"
    bpe_path: str | None = None


@dataclass(slots=True)
class RealSAM3AdapterBundle:
    """把真实 SAM3 adapter 与其资源打包为受控装配单元。"""

    sam_adapter: "RealSAM3Adapter"
    runtime: "_LoadedSAM3Runtime"

    def close(self) -> None:
        self.runtime.close()


@dataclass(slots=True)
class _SAM3MaskPrediction:
    """infra 内部使用的单个 mask 预测结果。"""

    mask_bitmap: list[list[bool]]
    score: float | None = None


@dataclass(slots=True)
class _SAM3RuntimePrediction:
    """infra 内部使用的 SAM3 运行时输出。"""

    prompt_mode: str
    masks: list[_SAM3MaskPrediction] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _SAM3ImportSpec:
    """真实 SAM3 代码导入方案。"""

    root_key: str
    sys_path_entry: str
    builder_module_name: str
    processor_module_name: str
    module_prefixes: tuple[str, ...]
    asset_root: Path


@dataclass(slots=True)
class _SAM3ImportState:
    """当前进程内活动中的 SAM3 导入状态。"""

    spec: _SAM3ImportSpec
    previous_modules: dict[str, object]
    inserted_sys_path: bool
    ref_count: int = 1


@dataclass(slots=True, kw_only=True)
class RealSAM3Adapter(SAMAdapter):
    """最小真实 SAM3 adapter。

    当前阶段优先承诺：
    - 消费标准 `SegmentAdapterRequest`
    - 在 infra 内部完成真实 SAM3 调用
    - 对上层只返回稳定的 `SegmentationResult`
    """

    artifact_store: ArtifactStore
    runtime: "_LoadedSAM3Runtime"
    segment_calls: int = 0

    def segment(self, request: SegmentAdapterRequest) -> SegmentationResult:
        self.segment_calls += 1

        try:
            image_path = _resolve_local_image_path(request.image_uri)
        except ValueError as exc:
            return SegmentationResult(
                segmentation_id=f"{request.task_id}-segmentation",
                status=SegmentationStatus.FAILED,
                result_summary="Real SAM3 adapter could not resolve the image path.",
                diagnostics=[
                    "sam3.failed",
                    f"reason={exc}",
                ],
            )

        if not image_path.is_file():
            return SegmentationResult(
                segmentation_id=f"{request.task_id}-segmentation",
                status=SegmentationStatus.FAILED,
                result_summary="Real SAM3 adapter requires a readable local image file.",
                diagnostics=[
                    "sam3.failed",
                    f"reason=image_not_found:{image_path}",
                ],
            )

        try:
            prediction = self.runtime.predict(
                image_path=image_path,
                prompt_package=request.prompt_package,
            )
        except Exception as exc:
            return SegmentationResult(
                segmentation_id=f"{request.task_id}-segmentation",
                status=SegmentationStatus.FAILED,
                result_summary="Real SAM3 runtime failed before producing a segmentation.",
                diagnostics=[
                    "sam3.failed",
                    "reason=runtime_exception",
                    f"exception_type={type(exc).__name__}",
                ],
            )

        ordered_masks = sorted(
            prediction.masks,
            key=lambda item: item.score if item.score is not None else -1.0,
            reverse=True,
        )
        ordered_masks = ordered_masks[: self._resolve_return_top_k(request.prompt_package, len(ordered_masks))]
        if not ordered_masks:
            return SegmentationResult(
                segmentation_id=f"{request.task_id}-segmentation",
                status=SegmentationStatus.EMPTY,
                result_summary="Real SAM3 adapter returned no usable mask candidate.",
                diagnostics=[
                    *prediction.diagnostics,
                    f"prompt_mode={prediction.prompt_mode}",
                    "candidate_count=0",
                ],
            )

        candidates: list[SegmentationCandidate] = []
        for index, mask_prediction in enumerate(ordered_masks, start=1):
            mask_payload = self._build_mask_payload(
                request=request,
                prediction=mask_prediction,
                prompt_mode=prediction.prompt_mode,
                candidate_index=index,
            )
            mask_ref = self.artifact_store.save_artifact(ArtifactKind.MASK, mask_payload)
            mask_ref.summary = (
                f"{self.backend_name}:{prediction.prompt_mode}:"
                f"{mask_payload.width}x{mask_payload.height}"
            )
            candidates.append(
                SegmentationCandidate(
                    candidate_id=f"{request.task_id}-segmentation-candidate-{index}",
                    mask_ref=mask_ref,
                    score=mask_prediction.score,
                    prompt_summary=request.prompt_package.text_prompts.normalized_text,
                    notes=[
                        "generated by RealSAM3Adapter",
                        f"prompt_mode={prediction.prompt_mode}",
                    ],
                )
            )

        primary_candidate_id = candidates[0].candidate_id
        return SegmentationResult(
            segmentation_id=f"{request.task_id}-segmentation",
            status=SegmentationStatus.READY,
            result_summary=(
                f"Real SAM3 adapter produced {len(candidates)} segmentation candidate(s) "
                f"via {prediction.prompt_mode} prompts."
            ),
            candidates=candidates,
            primary_candidate_id=primary_candidate_id,
            diagnostics=[
                *prediction.diagnostics,
                f"prompt_mode={prediction.prompt_mode}",
                f"candidate_count={len(candidates)}",
                f"backend={self.backend_name}",
            ],
        )

    def _resolve_return_top_k(self, prompt_package: PromptPackage, available: int) -> int:
        hints = prompt_package.execution_hints
        if hints is None or hints.return_top_k is None:
            return max(1, available)
        return max(1, min(available, hints.return_top_k))

    def _build_mask_payload(
        self,
        *,
        request: SegmentAdapterRequest,
        prediction: _SAM3MaskPrediction,
        prompt_mode: str,
        candidate_index: int,
    ) -> MaskArtifact:
        active_box = _infer_active_box(prediction.mask_bitmap)
        height = len(prediction.mask_bitmap)
        width = len(prediction.mask_bitmap[0]) if prediction.mask_bitmap else 0
        active_points = _collect_active_points(
            prediction.mask_bitmap,
            max_points=32,
        )
        pixel_area = sum(
            1
            for row in prediction.mask_bitmap
            for value in row
            if value
        )
        return MaskArtifact(
            mask_id=f"{request.task_id}-sam3-mask-{candidate_index}",
            width=width,
            height=height,
            active_box=active_box,
            mask_bitmap=prediction.mask_bitmap,
            active_points=active_points,
            label="sam3_mask",
            backend_name=self.backend_name,
            score=prediction.score,
            prompt_mode=prompt_mode,
            pixel_area=pixel_area,
        )


def build_real_sam3_adapter_bundle(
    config: RealSAM3AdapterConfig,
    *,
    artifact_store: ArtifactStore,
) -> RealSAM3AdapterBundle:
    """构造真实 SAM3 adapter bundle。"""
    runtime = _LoadedSAM3Runtime.from_config(config)
    return RealSAM3AdapterBundle(
        sam_adapter=RealSAM3Adapter(
            backend_name=config.backend_name,
            model_path=config.sam_model_path,
            checkpoint_path=config.checkpoint_path,
            artifact_store=artifact_store,
            runtime=runtime,
        ),
        runtime=runtime,
    )


@dataclass(slots=True)
class _LoadedSAM3Runtime:
    """真实 SAM3 模型与处理器的受控运行时。"""

    _active_import_state: ClassVar[_SAM3ImportState | None] = None

    config: RealSAM3AdapterConfig
    model: object
    processor: object
    import_state: _SAM3ImportState

    @classmethod
    def from_config(cls, config: RealSAM3AdapterConfig) -> "_LoadedSAM3Runtime":
        model_root = Path(config.sam_model_path).expanduser()
        checkpoint_path = Path(config.checkpoint_path).expanduser()
        if not model_root.is_dir():
            raise ValueError(
                "Real SAM3 adapter requires model_paths.sam_model_path to point to an "
                f"existing directory; got {model_root}."
            )
        if not checkpoint_path.is_file():
            raise ValueError(
                "Real SAM3 adapter requires model_paths.sam_checkpoint_path to point to "
                f"an existing checkpoint file; got {checkpoint_path}."
            )

        import_spec = _resolve_sam3_import_spec(model_root)
        import_state = cls._acquire_import_state(import_spec)

        try:
            model_builder = importlib.import_module(import_spec.builder_module_name)
            processor_module = importlib.import_module(import_spec.processor_module_name)
            build_sam3_image_model = getattr(model_builder, "build_sam3_image_model")
            sam3_processor_cls = getattr(processor_module, "Sam3Processor")

            bpe_path = (
                Path(config.bpe_path).expanduser()
                if config.bpe_path
                else _resolve_bpe_path(import_spec.asset_root)
            )
            model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                checkpoint_path=str(checkpoint_path),
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
            processor = sam3_processor_cls(model)
        except Exception:
            cls._release_import_state(import_state)
            raise

        return cls(
            config=config,
            model=model,
            processor=processor,
            import_state=import_state,
        )

    def predict(
        self,
        *,
        image_path: Path,
        prompt_package: PromptPackage,
    ) -> _SAM3RuntimePrediction:
        pil_image_module = importlib.import_module("PIL.Image")
        image = pil_image_module.open(image_path).convert("RGB")

        inference_state = self.processor.set_image(image)
        point_coords, point_labels = _extract_point_prompts(prompt_package)
        box_prompt = _extract_primary_box(prompt_package)
        use_spatial_prompt = bool(point_coords) or box_prompt is not None
        multimask = bool(
            prompt_package.execution_hints is not None
            and prompt_package.execution_hints.multimask
        )

        if use_spatial_prompt:
            masks, scores, _ = self.model.predict_inst(
                inference_state,
                point_coords=point_coords or None,
                point_labels=point_labels or None,
                box=box_prompt,
                # PromptPackage carries normalized coordinates in [0, 1].
                # The external SAM3 interactive predictor defaults to interpreting
                # prompts as absolute image pixels when normalize_coords=True.
                normalize_coords=False,
                multimask_output=multimask,
            )
            return _SAM3RuntimePrediction(
                prompt_mode="spatial",
                masks=_zip_masks_and_scores(masks, scores),
                diagnostics=["sam3.runtime=spatial_prompt"],
            )

        text_prompt = _select_text_prompt(prompt_package)
        if text_prompt:
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt,
            )
            masks = inference_state.get("masks")
            scores = inference_state.get("scores")
            return _SAM3RuntimePrediction(
                prompt_mode="text",
                masks=_zip_masks_and_scores(masks, scores),
                diagnostics=["sam3.runtime=text_prompt"],
            )

        return _SAM3RuntimePrediction(
            prompt_mode="missing",
            masks=[],
            diagnostics=[
                "sam3.runtime=missing_prompt",
                "reason=no_spatial_or_text_prompt",
            ],
        )

    def close(self) -> None:
        model = self.model
        self.model = None
        self.processor = None
        import_state = self.import_state
        self.import_state = None
        if model is not None and hasattr(model, "cpu"):
            model.cpu()
        if import_state is not None:
            self._release_import_state(import_state)
        try:
            torch = importlib.import_module("torch")
        except Exception:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @classmethod
    def _acquire_import_state(cls, import_spec: _SAM3ImportSpec) -> _SAM3ImportState:
        active_state = cls._active_import_state
        if active_state is not None:
            if active_state.spec.root_key != import_spec.root_key:
                raise RuntimeError(
                    "Real SAM3 adapter cannot load multiple distinct external SAM "
                    "code roots in the same process at once. Close the existing "
                    "bundle before loading a different sam_model_path."
                )
            active_state.ref_count += 1
            return active_state

        previous_modules = _snapshot_matching_modules(import_spec.module_prefixes)
        inserted_sys_path = False
        if import_spec.sys_path_entry not in sys.path:
            sys.path.insert(0, import_spec.sys_path_entry)
            inserted_sys_path = True

        try:
            _remove_matching_modules(import_spec.module_prefixes)
            importlib.invalidate_caches()
            for module_name in (
                import_spec.builder_module_name,
                import_spec.processor_module_name,
            ):
                importlib.import_module(module_name)
        except Exception:
            _remove_matching_modules(import_spec.module_prefixes)
            _restore_modules(previous_modules)
            if inserted_sys_path:
                _remove_sys_path_entry(import_spec.sys_path_entry)
            raise

        active_state = _SAM3ImportState(
            spec=import_spec,
            previous_modules=previous_modules,
            inserted_sys_path=inserted_sys_path,
        )
        cls._active_import_state = active_state
        return active_state

    @classmethod
    def _release_import_state(cls, import_state: _SAM3ImportState) -> None:
        active_state = cls._active_import_state
        if active_state is None:
            return
        if active_state is not import_state:
            return
        if import_state.ref_count > 1:
            import_state.ref_count -= 1
            return

        _remove_matching_modules(import_state.spec.module_prefixes)
        _restore_modules(import_state.previous_modules)
        if import_state.inserted_sys_path:
            _remove_sys_path_entry(import_state.spec.sys_path_entry)
        importlib.invalidate_caches()
        cls._active_import_state = None


def _resolve_sam3_import_spec(model_root: Path) -> _SAM3ImportSpec:
    nested_root = model_root / "sam3"
    nested_builder = nested_root / "model_builder.py"
    if nested_builder.is_file():
        return _SAM3ImportSpec(
            root_key=str(nested_root.resolve()),
            sys_path_entry=str(model_root.resolve()),
            builder_module_name="sam3.model_builder",
            processor_module_name="sam3.model.sam3_image_processor",
            module_prefixes=("sam3",),
            asset_root=nested_root,
        )
    raise ValueError(
        "Real SAM3 adapter currently supports only external code directories "
        "with a 'sam3/' package layout. Expected to find "
        f"{nested_builder}."
    )


def _resolve_bpe_path(import_root: Path) -> Path:
    bpe_path = import_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.is_file():
        raise ValueError(
            "Real SAM3 adapter could not locate the SAM3 BPE asset at "
            f"{bpe_path}."
        )
    return bpe_path


def _resolve_local_image_path(image_uri: str) -> Path:
    parsed = urlparse(image_uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser().resolve()
    if parsed.scheme not in ("", None):
        raise ValueError(f"unsupported_image_uri_scheme:{parsed.scheme}")
    return Path(image_uri).expanduser().resolve()


def _extract_point_prompts(prompt_package: PromptPackage) -> tuple[list[list[float]], list[int]]:
    points: list[list[float]] = []
    labels: list[int] = []
    for point in prompt_package.spatial_prompts.positive_points:
        points.append([point.x, point.y])
        labels.append(1)
    for point in prompt_package.spatial_prompts.negative_points:
        points.append([point.x, point.y])
        labels.append(0)
    return points, labels


def _extract_primary_box(prompt_package: PromptPackage) -> list[float] | None:
    boxes = prompt_package.spatial_prompts.boxes
    if not boxes:
        return None
    first_box = boxes[0]
    return [first_box.x1, first_box.y1, first_box.x2, first_box.y2]


def _select_text_prompt(prompt_package: PromptPackage) -> str | None:
    candidates = [
        prompt_package.text_prompts.rewritten_text,
        prompt_package.text_prompts.normalized_text,
        prompt_package.text_prompts.raw_text,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.strip():
            return candidate.strip()
    return None


def _zip_masks_and_scores(raw_masks: object, raw_scores: object) -> list[_SAM3MaskPrediction]:
    masks = _coerce_mask_collection(raw_masks)
    scores = _coerce_score_list(raw_scores)
    predictions: list[_SAM3MaskPrediction] = []
    for index, mask in enumerate(masks):
        score = scores[index] if index < len(scores) else None
        predictions.append(
            _SAM3MaskPrediction(
                mask_bitmap=mask,
                score=score,
            )
        )
    return predictions


def _coerce_score_list(raw_scores: object) -> list[float]:
    flattened = _coerce_scalar_sequence(raw_scores)
    return [float(value) for value in flattened]


def _coerce_mask_collection(raw_masks: object) -> list[list[list[bool]]]:
    data = _to_python_data(raw_masks)
    if not isinstance(data, list):
        return []
    if not data:
        return []
    if _looks_like_mask_2d(data):
        return [_coerce_single_mask(data)]
    masks: list[list[list[bool]]] = []
    for item in data:
        masks.append(_coerce_single_mask(_collapse_leading_singleton_dims(item)))
    return masks


def _coerce_single_mask(raw_mask: object) -> list[list[bool]]:
    data = _collapse_leading_singleton_dims(raw_mask)
    if not isinstance(data, list):
        raise TypeError("SAM3 mask payload must be list-like after conversion.")

    rows: list[list[bool]] = []
    for row in data:
        if not isinstance(row, list):
            raise TypeError("SAM3 mask row must be list-like after conversion.")
        rows.append([bool(value > 0) if isinstance(value, (int, float)) else bool(value) for value in row])
    return rows


def _collapse_leading_singleton_dims(raw_value: object) -> object:
    data = _to_python_data(raw_value)
    while (
        isinstance(data, list)
        and len(data) == 1
        and data
        and isinstance(data[0], list)
    ):
        data = data[0]
    return data


def _to_python_data(raw_value: object) -> object:
    data = raw_value
    for attribute in ("detach", "cpu"):
        if hasattr(data, attribute):
            data = getattr(data, attribute)()
    if hasattr(data, "numpy"):
        data = data.numpy()
    if hasattr(data, "tolist"):
        data = data.tolist()
    return data


def _coerce_scalar_sequence(raw_value: object) -> list[float]:
    data = _to_python_data(raw_value)
    if data is None:
        return []
    if isinstance(data, (int, float)):
        return [float(data)]
    if not isinstance(data, list):
        return []
    flattened: list[float] = []
    stack: list[Any] = list(data)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = list(item) + stack
            continue
        flattened.append(float(item))
    return flattened


def _looks_like_mask_2d(value: list[object]) -> bool:
    return bool(value) and isinstance(value[0], list) and (
        not value[0] or not isinstance(value[0][0], list)
    )


def _infer_active_box(mask_bitmap: list[list[bool]]) -> NormalizedBox:
    height = len(mask_bitmap)
    width = len(mask_bitmap[0]) if height else 0
    if height == 0 or width == 0:
        return NormalizedBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0)

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for y, row in enumerate(mask_bitmap):
        for x, active in enumerate(row):
            if not active:
                continue
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    if max_x < 0 or max_y < 0:
        return NormalizedBox(x1=0.0, y1=0.0, x2=0.0, y2=0.0)

    return NormalizedBox(
        x1=min_x / width,
        y1=min_y / height,
        x2=min(1.0, (max_x + 1) / width),
        y2=min(1.0, (max_y + 1) / height),
    )


def _collect_active_points(
    mask_bitmap: list[list[bool]],
    *,
    max_points: int,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for y, row in enumerate(mask_bitmap):
        for x, active in enumerate(row):
            if not active:
                continue
            points.append((x, y))
            if len(points) >= max_points:
                return points
    return points


def _snapshot_matching_modules(prefixes: tuple[str, ...]) -> dict[str, object]:
    return {
        module_name: module
        for module_name, module in sys.modules.items()
        if _matches_any_prefix(module_name, prefixes)
    }


def _remove_matching_modules(prefixes: tuple[str, ...]) -> None:
    _remove_module_names(
        module_name
        for module_name in tuple(sys.modules)
        if _matches_any_prefix(module_name, prefixes)
    )


def _remove_module_names(module_names: object) -> None:
    for module_name in tuple(module_names):
        sys.modules.pop(module_name, None)


def _restore_modules(previous_modules: dict[str, object]) -> None:
    for module_name, module in previous_modules.items():
        sys.modules[module_name] = module


def _remove_sys_path_entry(entry: str) -> None:
    try:
        sys.path.remove(entry)
    except ValueError:
        return


def _matches_any_prefix(module_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in prefixes
    )
