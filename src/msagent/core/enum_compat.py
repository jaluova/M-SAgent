"""兼容不同 Python 版本的枚举类型。"""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10 fallback for enum.StrEnum."""

