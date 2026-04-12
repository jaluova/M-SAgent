from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class LocalizationResult:
    absolute_points: List[List[float]]
    normalized_points: List[List[float]]
    scores: List[float]
    selection_mode: str
    selected_k: int
    source: str
    annotated_image: Optional[Image.Image] = None

    @classmethod
    def from_train_adapter_payload(cls, payload: Dict, annotated_image: Optional[Image.Image] = None):
        absolute_points = [
            [float(x), float(y)]
            for x, y in payload.get("absolute_points", [])
        ]
        normalized_points = [
            [float(x), float(y)]
            for x, y in payload.get("normalized_points", [])
        ]
        scores = [float(score) for score in payload.get("scores", [])]
        selected_k = int(payload.get("selected_k", len(absolute_points)))
        return cls(
            absolute_points=absolute_points,
            normalized_points=normalized_points,
            scores=scores,
            selection_mode=str(payload.get("selection_mode", "fixed_topk")),
            selected_k=selected_k,
            source="train_adapter",
            annotated_image=annotated_image,
        )

    def top_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    def has_usable_points(self, min_score: float = 0.0) -> bool:
        return bool(self.absolute_points) and self.selected_k > 0 and self.top_score() >= min_score

    def as_dict(self, include_annotated_image: bool = False) -> Dict:
        payload = {
            "absolute_points": [list(point) for point in self.absolute_points],
            "normalized_points": [list(point) for point in self.normalized_points],
            "scores": list(self.scores),
            "selection_mode": self.selection_mode,
            "selected_k": self.selected_k,
            "source": self.source,
        }
        if include_annotated_image and self.annotated_image is not None:
            payload["annotated_image"] = self.annotated_image
        return payload
