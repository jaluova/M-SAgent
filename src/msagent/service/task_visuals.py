"""CLI task artifact visualization helpers."""

from __future__ import annotations

import importlib
from pathlib import Path

from msagent.core.contracts.common import ArtifactKind, ArtifactRef
from msagent.core.contracts.types import (
    BoxPrompt,
    EvaluationResult,
    PointPrompt,
    PromptPackage,
    ProposalCandidate,
    ProposalResult,
    SegmentationCandidate,
    SegmentationResult,
)
from msagent.core.task.models import RunTask
from msagent.infra.adapters import ArtifactStore
from msagent.infra.mask_artifact import MaskArtifact


def render_task_visuals(
    task: RunTask,
    *,
    artifact_store: ArtifactStore,
    output_dir: str | Path,
) -> list[Path]:
    """Render attempt-level overlays and the final accepted mask overlay."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = Path(task.request.image_ref.uri).expanduser().resolve()

    created_files: list[Path] = []
    for attempt in task.attempt_history:
        suffix = f"attempt_{attempt.attempt_index:02d}"
        proposal = _load_optional_artifact(
            artifact_store,
            attempt.proposal_ref,
            ProposalResult,
        )
        if proposal is not None:
            proposal_path = output_path / f"{suffix}_proposal_overlay.png"
            _render_proposal_overlay(
                image_path=image_path,
                proposal=proposal,
                output_path=proposal_path,
            )
            created_files.append(proposal_path)

        prompt_package = _load_optional_artifact(
            artifact_store,
            attempt.prompt_package_ref,
            PromptPackage,
        )
        if prompt_package is not None:
            prompt_path = output_path / f"{suffix}_prompt_overlay.png"
            _render_prompt_overlay(
                image_path=image_path,
                prompt_package=prompt_package,
                output_path=prompt_path,
            )
            created_files.append(prompt_path)

        segmentation = _load_optional_artifact(
            artifact_store,
            attempt.segmentation_ref,
            SegmentationResult,
        )
        if segmentation is not None:
            segmentation_path = output_path / f"{suffix}_mask_overlay.png"
            mask_bitmap_path = output_path / f"{suffix}_mask_bitmap.png"
            mask_payload = _load_primary_mask_payload(
                artifact_store,
                segmentation=segmentation,
            )
            if mask_payload is not None:
                _render_mask_bitmap(
                    mask_payload=mask_payload,
                    output_path=mask_bitmap_path,
                )
                created_files.append(mask_bitmap_path)
                _render_mask_overlay(
                    image_path=image_path,
                    mask_payload=mask_payload,
                    output_path=segmentation_path,
                )
                created_files.append(segmentation_path)

    final_mask_payload = _load_optional_artifact(
        artifact_store,
        task.result.final_mask_ref,
        MaskArtifact,
    )
    if final_mask_payload is not None:
        final_bitmap_output = output_path / "final_mask_bitmap.png"
        final_output = output_path / "final_mask_overlay.png"
        _render_mask_bitmap(
            mask_payload=final_mask_payload,
            output_path=final_bitmap_output,
        )
        created_files.append(final_bitmap_output)
        _render_mask_overlay(
            image_path=image_path,
            mask_payload=final_mask_payload,
            output_path=final_output,
        )
        created_files.append(final_output)

    return created_files


def render_latest_artifact_visuals(
    *,
    image_path: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Render overlays from the latest artifacts found under an artifact root."""
    artifact_store = _build_artifact_store(artifact_root)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    resolved_image_path = Path(image_path).expanduser().resolve()
    latest_proposal_ref = _latest_artifact_ref(artifact_root, ArtifactKind.PROPOSAL_RESULT)
    latest_prompt_ref = _latest_artifact_ref(artifact_root, ArtifactKind.PROMPT_PACKAGE)
    latest_segmentation_ref = _latest_artifact_ref(artifact_root, ArtifactKind.SEGMENTATION_RESULT)
    latest_evaluation_ref = _latest_artifact_ref(artifact_root, ArtifactKind.EVALUATION_RESULT)

    proposal = _load_optional_artifact(artifact_store, latest_proposal_ref, ProposalResult)
    if proposal is not None:
        proposal_path = output_path / "latest_proposal_overlay.png"
        _render_proposal_overlay(
            image_path=resolved_image_path,
            proposal=proposal,
            output_path=proposal_path,
        )
        created_files.append(proposal_path)

    prompt_package = _load_optional_artifact(artifact_store, latest_prompt_ref, PromptPackage)
    if prompt_package is not None:
        prompt_path = output_path / "latest_prompt_overlay.png"
        _render_prompt_overlay(
            image_path=resolved_image_path,
            prompt_package=prompt_package,
            output_path=prompt_path,
        )
        created_files.append(prompt_path)

    segmentation = _load_optional_artifact(
        artifact_store,
        latest_segmentation_ref,
        SegmentationResult,
    )
    if segmentation is not None:
        latest_mask_payload = _load_primary_mask_payload(
            artifact_store,
            segmentation=segmentation,
        )
        if latest_mask_payload is not None:
            bitmap_path = output_path / "latest_mask_bitmap.png"
            segmentation_path = output_path / "latest_mask_overlay.png"
            _render_mask_bitmap(
                mask_payload=latest_mask_payload,
                output_path=bitmap_path,
            )
            created_files.append(bitmap_path)
            _render_mask_overlay(
                image_path=resolved_image_path,
                mask_payload=latest_mask_payload,
                output_path=segmentation_path,
            )
            created_files.append(segmentation_path)

    evaluation = _load_optional_artifact(
        artifact_store,
        latest_evaluation_ref,
        EvaluationResult,
    )
    if evaluation is not None and evaluation.accepted_mask_ref is not None:
        final_mask_payload = _load_optional_artifact(
            artifact_store,
            evaluation.accepted_mask_ref,
            MaskArtifact,
        )
        if final_mask_payload is not None:
            final_bitmap_output = output_path / "latest_final_mask_bitmap.png"
            final_output = output_path / "latest_final_mask_overlay.png"
            _render_mask_bitmap(
                mask_payload=final_mask_payload,
                output_path=final_bitmap_output,
            )
            created_files.append(final_bitmap_output)
            _render_mask_overlay(
                image_path=resolved_image_path,
                mask_payload=final_mask_payload,
                output_path=final_output,
            )
            created_files.append(final_output)

    return created_files


def _build_artifact_store(artifact_root: str | Path) -> ArtifactStore:
    from msagent.infra.local_artifact_store import LocalFileArtifactStore

    return LocalFileArtifactStore(str(Path(artifact_root).expanduser().resolve()))


def _latest_artifact_ref(
    artifact_root: str | Path,
    artifact_kind: ArtifactKind,
) -> ArtifactRef | None:
    artifact_dir = Path(artifact_root).expanduser().resolve() / artifact_kind.value
    if not artifact_dir.is_dir():
        return None

    latest_path: Path | None = None
    latest_index = -1
    for candidate in artifact_dir.glob("*.json"):
        suffix = candidate.stem.rsplit("-", 1)
        if len(suffix) != 2 or not suffix[1].isdigit():
            continue
        index = int(suffix[1])
        if index > latest_index:
            latest_index = index
            latest_path = candidate

    if latest_path is None:
        return None
    return ArtifactRef(
        artifact_id=latest_path.stem,
        artifact_type=artifact_kind,
    )


def _load_primary_mask_payload(
    artifact_store: ArtifactStore,
    *,
    segmentation: SegmentationResult,
) -> MaskArtifact | None:
    candidate = _select_primary_segmentation_candidate(segmentation)
    if candidate is None:
        return None
    return _load_optional_artifact(
        artifact_store,
        candidate.mask_ref,
        MaskArtifact,
    )


def _load_optional_artifact(
    artifact_store: ArtifactStore,
    artifact_ref: ArtifactRef | None,
    expected_type: type[object],
) -> object | None:
    if artifact_ref is None:
        return None
    try:
        return artifact_store.load_artifact(artifact_ref, expected_type)
    except Exception:
        return None


def _select_primary_proposal_candidate(proposal: ProposalResult) -> ProposalCandidate | None:
    if proposal.primary_candidate_id is not None:
        for candidate in proposal.candidates:
            if candidate.candidate_id == proposal.primary_candidate_id:
                return candidate
    return proposal.candidates[0] if proposal.candidates else None


def _select_primary_segmentation_candidate(
    segmentation: SegmentationResult,
) -> SegmentationCandidate | None:
    if segmentation.primary_candidate_id is not None:
        for candidate in segmentation.candidates:
            if candidate.candidate_id == segmentation.primary_candidate_id:
                return candidate
    return segmentation.candidates[0] if segmentation.candidates else None


def _render_proposal_overlay(
    *,
    image_path: Path,
    proposal: ProposalResult,
    output_path: Path,
) -> None:
    image, draw = _open_canvas(image_path)
    candidate = _select_primary_proposal_candidate(proposal)
    if candidate is not None:
        if candidate.region_box is not None:
            _draw_normalized_box(
                draw,
                image.size,
                candidate.region_box.x1,
                candidate.region_box.y1,
                candidate.region_box.x2,
                candidate.region_box.y2,
                outline=(255, 166, 0, 255),
                fill=(255, 166, 0, 40),
            )
        _draw_points(
            draw,
            image.size,
            candidate.positive_point_hints,
            fill=(34, 197, 94, 255),
        )
        _draw_points(
            draw,
            image.size,
            candidate.negative_point_hints,
            fill=(239, 68, 68, 255),
        )
    image.save(output_path)


def _render_prompt_overlay(
    *,
    image_path: Path,
    prompt_package: PromptPackage,
    output_path: Path,
) -> None:
    image, draw = _open_canvas(image_path)
    for box in prompt_package.spatial_prompts.boxes:
        _draw_box_prompt(
            draw,
            image.size,
            box,
            outline=(59, 130, 246, 255),
            fill=(59, 130, 246, 35),
        )
    _draw_points(
        draw,
        image.size,
        prompt_package.spatial_prompts.positive_points,
        fill=(34, 197, 94, 255),
    )
    _draw_points(
        draw,
        image.size,
        prompt_package.spatial_prompts.negative_points,
        fill=(239, 68, 68, 255),
    )
    image.save(output_path)


def _render_mask_overlay(
    *,
    image_path: Path,
    mask_payload: MaskArtifact,
    output_path: Path,
) -> None:
    image, draw = _open_canvas(image_path)
    if mask_payload.mask_bitmap:
        image = _apply_dense_mask_overlay(
            image=image,
            mask_payload=mask_payload,
            fill=(14, 165, 233, 108),
        )
        image_draw_module = importlib.import_module("PIL.ImageDraw")
        draw = image_draw_module.Draw(image, "RGBA")
    _draw_normalized_box(
        draw,
        image.size,
        mask_payload.active_box.x1,
        mask_payload.active_box.y1,
        mask_payload.active_box.x2,
        mask_payload.active_box.y2,
        outline=(14, 165, 233, 255),
        fill=(14, 165, 233, 25),
    )
    if mask_payload.mask_bitmap:
        _draw_mask_outline(
            draw,
            image.size,
            mask_payload,
            fill=(255, 255, 255, 200),
        )
    else:
        _draw_pixel_points(
            draw,
            image.size,
            mask_payload,
            fill=(6, 182, 212, 255),
        )
    image.save(output_path)


def _render_mask_bitmap(
    *,
    mask_payload: MaskArtifact,
    output_path: Path,
) -> None:
    image_module = importlib.import_module("PIL.Image")
    width = max(1, mask_payload.width)
    height = max(1, mask_payload.height)
    bitmap_image = image_module.new("L", (width, height), 0)
    if mask_payload.mask_bitmap:
        pixels = [
            255 if value else 0
            for row in mask_payload.mask_bitmap
            for value in row
        ]
        bitmap_image.putdata(pixels)
    bitmap_image.save(output_path)


def _open_canvas(image_path: Path) -> tuple[object, object]:
    image_module = importlib.import_module("PIL.Image")
    image_draw_module = importlib.import_module("PIL.ImageDraw")
    image = image_module.open(image_path).convert("RGBA")
    draw = image_draw_module.Draw(image, "RGBA")
    return image, draw


def _apply_dense_mask_overlay(
    *,
    image: object,
    mask_payload: MaskArtifact,
    fill: tuple[int, int, int, int],
) -> object:
    image_module = importlib.import_module("PIL.Image")
    width = max(1, mask_payload.width)
    height = max(1, mask_payload.height)
    mask_image = image_module.new("L", (width, height), 0)
    pixels = [
        255 if value else 0
        for row in mask_payload.mask_bitmap
        for value in row
    ]
    mask_image.putdata(pixels)
    mask_image = mask_image.resize(image.size, image_module.Resampling.NEAREST)

    overlay = image_module.new("RGBA", image.size, fill)
    return image_module.composite(overlay, image, mask_image)


def _draw_box_prompt(
    draw: object,
    image_size: tuple[int, int],
    box: BoxPrompt,
    *,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
) -> None:
    _draw_normalized_box(
        draw,
        image_size,
        box.x1,
        box.y1,
        box.x2,
        box.y2,
        outline=outline,
        fill=fill,
    )


def _draw_normalized_box(
    draw: object,
    image_size: tuple[int, int],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    outline: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
) -> None:
    width, height = image_size
    line_width = max(2, min(width, height) // 180)
    draw.rectangle(
        (
            x1 * width,
            y1 * height,
            x2 * width,
            y2 * height,
        ),
        outline=outline,
        fill=fill,
        width=line_width,
    )


def _draw_points(
    draw: object,
    image_size: tuple[int, int],
    points: list[PointPrompt],
    *,
    fill: tuple[int, int, int, int],
) -> None:
    width, height = image_size
    radius = max(4, min(width, height) // 120)
    for point in points:
        x = point.x * width
        y = point.y * height
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=(255, 255, 255, 255),
            width=max(1, radius // 3),
        )


def _draw_pixel_points(
    draw: object,
    image_size: tuple[int, int],
    mask_payload: MaskArtifact,
    *,
    fill: tuple[int, int, int, int],
) -> None:
    image_width, image_height = image_size
    if mask_payload.width <= 0 or mask_payload.height <= 0:
        return
    radius = max(2, min(image_width, image_height) // 180)
    scale_x = image_width / mask_payload.width
    scale_y = image_height / mask_payload.height
    for x, y in mask_payload.active_points:
        center_x = (x + 0.5) * scale_x
        center_y = (y + 0.5) * scale_y
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=fill,
            outline=(255, 255, 255, 210),
            width=1,
        )


def _draw_mask_outline(
    draw: object,
    image_size: tuple[int, int],
    mask_payload: MaskArtifact,
    *,
    fill: tuple[int, int, int, int],
) -> None:
    image_width, image_height = image_size
    if mask_payload.width <= 0 or mask_payload.height <= 0:
        return
    radius = max(1, min(image_width, image_height) // 300)
    scale_x = image_width / mask_payload.width
    scale_y = image_height / mask_payload.height
    height = len(mask_payload.mask_bitmap)
    width = len(mask_payload.mask_bitmap[0]) if height else 0
    for y in range(height):
        for x in range(width):
            if not mask_payload.mask_bitmap[y][x]:
                continue
            if _is_interior_mask_pixel(mask_payload.mask_bitmap, x=x, y=y):
                continue
            center_x = (x + 0.5) * scale_x
            center_y = (y + 0.5) * scale_y
            draw.ellipse(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                fill=fill,
            )


def _is_interior_mask_pixel(mask_bitmap: list[list[bool]], *, x: int, y: int) -> bool:
    height = len(mask_bitmap)
    width = len(mask_bitmap[0]) if height else 0
    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            return False
        if not mask_bitmap[ny][nx]:
            return False
    return True
