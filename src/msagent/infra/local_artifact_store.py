"""本地文件版 artifact store。"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from msagent.core.contracts.common import ArtifactKind, ArtifactRef
from msagent.core.contracts.types import (
    EvaluationResult,
    PromptPackage,
    ProposalResult,
    QueryUnderstandingResult,
    SegmentationResult,
)
from msagent.infra.adapters import (
    ArtifactStore,
    LoadedArtifactT,
)
from msagent.infra.mask_artifact import MaskArtifact


def _is_union_type(type_hint: object) -> bool:
    origin = get_origin(type_hint)
    return origin is UnionType or origin is None and isinstance(type_hint, UnionType)


class LocalFileArtifactStore(ArtifactStore):
    """把结构化对象保存到本地 JSON 文件中。"""

    _index_filename = "_index.json"
    _default_payload_registry: dict[ArtifactKind, type[object]] = {
        ArtifactKind.QUERY_UNDERSTANDING_RESULT: QueryUnderstandingResult,
        ArtifactKind.PROPOSAL_RESULT: ProposalResult,
        ArtifactKind.PROMPT_PACKAGE: PromptPackage,
        ArtifactKind.SEGMENTATION_RESULT: SegmentationResult,
        ArtifactKind.EVALUATION_RESULT: EvaluationResult,
        ArtifactKind.MASK: MaskArtifact,
    }

    def __init__(
        self,
        root_uri: str,
        payload_registry: dict[ArtifactKind, type[object]] | None = None,
    ) -> None:
        super().__init__(root_uri=root_uri)
        self.root_path = Path(root_uri)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_path / self._index_filename
        self.payload_registry = payload_registry or dict(self._default_payload_registry)
        if not self.index_path.exists():
            self.index_path.write_text(
                json.dumps({"next_id": 1}, indent=2),
                encoding="utf-8",
            )

    def save_artifact(self, artifact_type: ArtifactKind, payload: object) -> ArtifactRef:
        expected_type = self._get_payload_type(artifact_type)
        if type(payload) is not expected_type:
            raise TypeError(
                f"Artifact payload type mismatch for {artifact_type.value}: "
                f"expected {expected_type.__name__}, got {type(payload).__name__}"
            )

        artifact_id = self._next_artifact_id(artifact_type)
        artifact_dir = self.root_path / artifact_type.value
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{artifact_id}.json"

        record = {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type.value,
            "payload_type": expected_type.__qualname__,
            "saved_at": datetime.now().isoformat(),
            "payload": self._serialize(payload),
        }
        artifact_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            summary=f"{artifact_type.value}:{artifact_id}",
        )

    def load_artifact(
        self,
        artifact_ref: ArtifactRef,
        expected_type: type[LoadedArtifactT],
    ) -> LoadedArtifactT:
        artifact_path = self.root_path / artifact_ref.artifact_type.value / (
            f"{artifact_ref.artifact_id}.json"
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        record = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_kind = artifact_ref.artifact_type
        registered_type = self._get_payload_type(artifact_kind)
        if expected_type is not registered_type:
            raise TypeError(
                f"Artifact expected type mismatch for {artifact_kind.value}: "
                f"registered {registered_type.__name__}, got {expected_type.__name__}"
            )
        if record["artifact_type"] != artifact_kind.value:
            raise TypeError(
                "Artifact record kind mismatch: "
                f"ref={artifact_kind.value}, record={record['artifact_type']}"
            )
        if record["payload_type"] != registered_type.__qualname__:
            raise TypeError(
                "Artifact payload type mismatch in record: "
                f"expected {registered_type.__qualname__}, got {record['payload_type']}"
            )
        payload = self._deserialize(record["payload"], expected_type)
        if not isinstance(payload, expected_type):
            raise TypeError(
                "Loaded artifact type mismatch: "
                f"expected {expected_type.__name__}, got {type(payload).__name__}"
            )
        return payload

    def _get_payload_type(self, artifact_kind: ArtifactKind) -> type[object]:
        try:
            return self.payload_registry[artifact_kind]
        except KeyError as exc:
            raise KeyError(f"Unregistered artifact kind: {artifact_kind.value}") from exc

    def _next_artifact_id(self, artifact_type: ArtifactKind) -> str:
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        next_id = int(index["next_id"])
        index["next_id"] = next_id + 1
        self.index_path.write_text(
            json.dumps(index, indent=2),
            encoding="utf-8",
        )
        return f"{artifact_type.value}-{next_id:04d}"

    def _serialize(self, value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return {"__enum__": type(value).__name__, "value": value.value}
        if isinstance(value, datetime):
            return {"__datetime__": value.isoformat()}
        if isinstance(value, tuple):
            return {"__tuple__": [self._serialize(item) for item in value]}
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: self._serialize(item) for key, item in value.items()}
        if is_dataclass(value):
            return {
                "__dataclass__": type(value).__qualname__,
                "fields": {
                    field.name: self._serialize(getattr(value, field.name))
                    for field in fields(value)
                },
            }
        raise TypeError(f"Unsupported artifact payload type: {type(value)!r}")

    def _deserialize(self, raw: object, type_hint: object) -> object:
        if raw is None:
            return None

        origin = get_origin(type_hint)
        if origin in (list, list[str].__origin__):
            item_type = get_args(type_hint)[0]
            return [self._deserialize(item, item_type) for item in raw]  # type: ignore[arg-type]

        if origin is tuple:
            item_types = get_args(type_hint)
            raw_items = raw["__tuple__"] if isinstance(raw, dict) else raw
            return tuple(
                self._deserialize(item, item_types[index])
                for index, item in enumerate(raw_items)
            )

        if origin in (dict, dict[str, object].__origin__):
            args = get_args(type_hint)
            value_type = args[1] if len(args) == 2 else object
            return {
                str(key): self._deserialize(value, value_type)
                for key, value in raw.items()  # type: ignore[union-attr]
            }

        if origin in (UnionType, None) and (
            isinstance(type_hint, UnionType) or origin is UnionType
        ):
            for candidate_type in get_args(type_hint):
                if candidate_type is type(None):
                    continue
                try:
                    return self._deserialize(raw, candidate_type)
                except (TypeError, ValueError, KeyError):
                    continue
            raise TypeError(f"Cannot deserialize value {raw!r} to {type_hint!r}")

        if isinstance(type_hint, type):
            if issubclass(type_hint, Enum):
                enum_value = raw["value"] if isinstance(raw, dict) else raw
                return type_hint(enum_value)
            if type_hint is datetime:
                iso_value = raw["__datetime__"] if isinstance(raw, dict) else raw
                return datetime.fromisoformat(iso_value)
            if is_dataclass(type_hint):
                field_payload = raw["fields"] if isinstance(raw, dict) else raw
                type_hints = get_type_hints(type_hint)
                kwargs = {
                    field.name: self._deserialize(
                        field_payload[field.name],
                        type_hints[field.name],
                    )
                    for field in fields(type_hint)
                    if field.name in field_payload
                }
                return type_hint(**kwargs)
            if type_hint in (str, int, float, bool):
                if not isinstance(raw, type_hint):
                    raise TypeError(f"Expected {type_hint!r}, got {type(raw)!r}")
                return raw
            if type_hint is object:
                return raw

        if _is_union_type(type_hint):
            for candidate_type in get_args(type_hint):
                if candidate_type is type(None):
                    continue
                try:
                    return self._deserialize(raw, candidate_type)
                except (TypeError, ValueError, KeyError):
                    continue
            raise TypeError(f"Cannot deserialize value {raw!r} to {type_hint!r}")

        return raw
