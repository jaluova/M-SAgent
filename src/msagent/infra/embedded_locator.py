"""embedded locate 的 locator adapter 实现。"""

from __future__ import annotations

from dataclasses import dataclass

from msagent.core.contracts.adapter_requests import LocateAdapterRequest
from msagent.core.contracts.types import (
    PointHint,
    ProposalBridgeHint,
    ProposalCandidate,
    ProposalResult,
    ProposalRoute,
    ProposalStatus,
)
from msagent.infra.adapters import LocatorAdapter
from msagent.infra.runtime.train_adapter_runtime import (
    EmbeddedLocatePoint,
    EmbeddedLocatePrediction,
    TrainAdapterRuntime,
    TrainAdapterRuntimeRequest,
)


@dataclass(slots=True, kw_only=True)
class EmbeddedLocatorAdapter(LocatorAdapter):
    """通过 embedded runtime 构建 proposal 的 locator adapter。"""

    runtime: TrainAdapterRuntime

    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        image_ref = request.resolved_image_ref()
        if image_ref is None:
            return ProposalResult(
                proposal_id=f"{request.task_id}-proposal-embedded-locate",
                route=ProposalRoute.LOCATE,
                status=ProposalStatus.FAILED,
                proposal_summary="Embedded locator requires an image reference.",
                diagnostics=["missing_image_ref"],
            )

        runtime_prediction = self.runtime.predict_embedded_locate(
            TrainAdapterRuntimeRequest(
                task_id=request.task_id,
                image_ref=image_ref,
                query_text=request.understanding.normalized_query,
                focus_terms=list(request.understanding.focus_terms),
                options=dict(request.options),
            )
        )
        return self._map_prediction_to_proposal(request, runtime_prediction)

    def _map_prediction_to_proposal(
        self,
        request: LocateAdapterRequest,
        prediction: EmbeddedLocatePrediction,
    ) -> ProposalResult:
        if not prediction.points and prediction.coarse_box is None:
            return ProposalResult(
                proposal_id=f"{request.task_id}-proposal-embedded-locate",
                route=ProposalRoute.LOCATE,
                status=ProposalStatus.EMPTY,
                proposal_summary="Embedded locate runtime returned no usable spatial prior.",
                matched_text_clues=[request.understanding.normalized_query],
                diagnostics=list(prediction.diagnostics),
            )

        positive_point_hints = [
            self._map_point_hint(point)
            for point in prediction.points
        ]
        candidate = ProposalCandidate(
            candidate_id=f"{request.task_id}-embedded-candidate-1",
            rank=1,
            confidence=prediction.top_confidence,
            matched_clues=self._collect_matched_clues(request),
            region_box=prediction.coarse_box,
            positive_point_hints=positive_point_hints,
            rationale=(
                f"Embedded runtime '{prediction.runtime_name}' produced "
                f"{len(positive_point_hints)} point priors."
            ),
            limitations=list(prediction.limitations),
        )
        diagnostics = list(prediction.diagnostics)
        if prediction.metadata:
            diagnostics.append(
                "runtime_metadata="
                + ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(prediction.metadata.items())
                )
            )

        bridge_hints = [
            ProposalBridgeHint(hint_type=hint.hint_type, reason=hint.reason)
            for hint in prediction.bridge_hints
        ]
        if prediction.coarse_box is not None and not any(
            hint.hint_type == "prefer_box"
            for hint in bridge_hints
        ):
            bridge_hints.append(
                ProposalBridgeHint(
                    hint_type="prefer_box",
                    reason="embedded locate runtime provided a coarse region box",
                )
            )
        if positive_point_hints and not any(
            hint.hint_type == "prefer_positive_points"
            for hint in bridge_hints
        ):
            bridge_hints.append(
                ProposalBridgeHint(
                    hint_type="prefer_positive_points",
                    reason="embedded locate runtime provided positive point priors",
                )
            )

        summary_parts = [f"Embedded locator returned {len(positive_point_hints)} point priors"]
        if prediction.coarse_box is not None:
            summary_parts.append("with a coarse region box")
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-embedded-locate",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.READY,
            proposal_summary=" ".join(summary_parts) + ".",
            candidates=[candidate],
            primary_candidate_id=candidate.candidate_id,
            matched_text_clues=self._collect_matched_clues(request),
            bridge_hints=bridge_hints,
            diagnostics=diagnostics,
        )

    def _collect_matched_clues(self, request: LocateAdapterRequest) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for clue in [request.understanding.normalized_query, *request.understanding.focus_terms]:
            if not clue or clue in seen:
                continue
            ordered.append(clue)
            seen.add(clue)
        return ordered

    def _map_point_hint(self, point: EmbeddedLocatePoint) -> PointHint:
        return PointHint(
            x=point.x,
            y=point.y,
            confidence=point.confidence,
            reason=point.reason,
        )
