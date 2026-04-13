"""mock vertical slice 使用的本地产物类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.contracts.types import NormalizedBox


@dataclass(slots=True)
class MockMask:
    """本地 mock vertical slice 使用的结构化 mask 对象。"""

    mask_id: str
    width: int
    height: int
    active_box: NormalizedBox
    active_points: list[tuple[int, int]] = field(default_factory=list)
    label: str = "mock_mask"
