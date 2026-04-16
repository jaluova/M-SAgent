"""本地 artifact store 使用的结构化 mask 对象。"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.types import NormalizedBox


@dataclass(slots=True)
class MaskArtifact:
    """统一的结构化 mask 产物。"""

    mask_id: str
    width: int
    height: int
    active_box: NormalizedBox
    mask_bitmap: list[list[bool]] = field(default_factory=list)
    active_points: list[tuple[int, int]] = field(default_factory=list)
    label: str = "mask"
    backend_name: str | None = None
    score: float | None = None
    prompt_mode: str | None = None
    pixel_area: int | None = None
