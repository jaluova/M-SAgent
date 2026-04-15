"""embedded locator 的独立本地评测闭环。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from msagent.core.contracts.common import ImageRef
from msagent.core.contracts.types import (
    ImplicitnessLevel,
    ProposalRoute,
    ProposalStatus,
    QueryUnderstandingResult,
    ReferentNumber,
    TargetType,
)
from msagent.core.enum_compat import StrEnum
from msagent.infra.adapters import LocatorAdapter
from msagent.infra.embedded_locator import EmbeddedLocatorAdapter
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.runtime.train_adapter_runtime import EmbeddedGridGroundRuntimeConfig
from msagent.modules.proposal_engine import (
    DefaultProposalEngineModule,
    LocateProposalRouteHandler,
    ProposalEngineModuleInput,
)

EMBEDDED_LOCATOR_RUNTIME_OPTION_KEYS = (
    "abs_threshold",
    "rel_ratio",
    "min_k",
    "max_k",
    "min_point_confidence",
)
_INT_RUNTIME_OPTION_KEYS = frozenset({"min_k", "max_k"})
_FLOAT_RUNTIME_OPTION_KEYS = frozenset(
    {"abs_threshold", "rel_ratio", "min_point_confidence"}
)


class EmbeddedLocatorFailureCategory(StrEnum):
    """评测层固定失败分类。"""

    NO_POINTS_AFTER_FILTER = "no_points_after_filter"
    RUNTIME_FAILED = "runtime_failed"
    EMPTY_PROPOSAL = "empty_proposal"
    LOW_CONFIDENCE_OR_SPARSE_POINTS = "low_confidence_or_sparse_points"
    COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR = "coarse_box_only_or_unusable_prior"


@dataclass(slots=True)
class EmbeddedLocatorManifestSample:
    """单个弱标注样例。"""

    sample_id: str
    image_path: str
    query_text: str
    should_locate: bool
    acceptable_point_standard: str
    acceptable_box_standard: str | None = None
    remark: str = ""
    minimum_points: int = 1
    minimum_top_confidence: float | None = None
    require_coarse_box: bool = False

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("Embedded locator manifest sample_id must be non-empty")
        if not self.image_path:
            raise ValueError(
                f"Embedded locator manifest sample {self.sample_id!r} image_path must be non-empty"
            )
        if not self.query_text:
            raise ValueError(
                f"Embedded locator manifest sample {self.sample_id!r} query_text must be non-empty"
            )
        if not self.acceptable_point_standard:
            raise ValueError(
                "Embedded locator manifest sample "
                f"{self.sample_id!r} acceptable_point_standard must be non-empty"
            )
        if not self.remark:
            raise ValueError(
                f"Embedded locator manifest sample {self.sample_id!r} remark must be non-empty"
            )
        if self.minimum_points < 0:
            raise ValueError(
                f"Embedded locator manifest sample {self.sample_id!r} minimum_points must be >= 0"
            )
        if self.minimum_top_confidence is not None:
            if not 0.0 <= self.minimum_top_confidence <= 1.0:
                raise ValueError(
                    "Embedded locator manifest sample "
                    f"{self.sample_id!r} minimum_top_confidence must be within [0, 1]"
                )
        if self.require_coarse_box and not self.acceptable_box_standard:
            raise ValueError(
                "Embedded locator manifest sample "
                f"{self.sample_id!r} require_coarse_box requires acceptable_box_standard"
            )


@dataclass(slots=True)
class EmbeddedLocatorManifest:
    """评测 manifest。"""

    manifest_id: str
    manifest_path: str
    base_dir: str
    samples: list[EmbeddedLocatorManifestSample]
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.manifest_id:
            raise ValueError("Embedded locator manifest_id must be non-empty")
        if not self.samples:
            raise ValueError("Embedded locator manifest must contain at least one sample")
        seen_ids: set[str] = set()
        for sample in self.samples:
            if sample.sample_id in seen_ids:
                raise ValueError(
                    f"Embedded locator manifest contains duplicate sample_id {sample.sample_id!r}"
                )
            seen_ids.add(sample.sample_id)

    def resolve_image_path(self, sample: EmbeddedLocatorManifestSample) -> Path:
        return (Path(self.base_dir) / sample.image_path).expanduser().resolve()


@dataclass(slots=True)
class EmbeddedLocatorParameterGroup:
    """单组 embedded locator runtime 参数。"""

    label: str
    runtime_options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Embedded locator parameter group label must be non-empty")
        self.runtime_options = _normalize_runtime_options(self.runtime_options)


@dataclass(slots=True)
class EmbeddedLocatorSampleEvaluationResult:
    """单样例评测结果。"""

    sample_id: str
    image_path: str
    query_text: str
    should_locate: bool
    proposal_status: str
    point_count: int
    top_confidence: float | None
    has_coarse_box: bool
    diagnostics: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    runtime_options_snapshot: dict[str, object] = field(default_factory=dict)
    runtime_option_overrides: dict[str, object] = field(default_factory=dict)
    failure_category: str | None = None
    passed: bool = False
    proposal_summary: str | None = None


@dataclass(slots=True)
class EmbeddedLocatorRunSummary:
    """单组参数运行摘要。"""

    total_samples: int
    ready_count: int
    empty_count: int
    failed_count: int
    passed_sample_count: int
    failure_category_counts: dict[str, int] = field(default_factory=dict)
    representative_samples_by_failure_category: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddedLocatorRunReport:
    """单组参数完整结果。"""

    parameter_group: EmbeddedLocatorParameterGroup
    effective_runtime_options: dict[str, object]
    samples: list[EmbeddedLocatorSampleEvaluationResult]
    summary: EmbeddedLocatorRunSummary


@dataclass(slots=True)
class EmbeddedLocatorSweepComparisonEntry:
    """sweep 对比中的单组概览。"""

    label: str
    runtime_options: dict[str, object]
    passed_sample_count: int
    runtime_option_overrides: dict[str, object] = field(default_factory=dict)
    failure_category_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddedLocatorSweepComparison:
    """参数 sweep 对比输出。"""

    entries: list[EmbeddedLocatorSweepComparisonEntry]
    best_parameter_group_label: str


@dataclass(slots=True)
class EmbeddedLocatorPrioritySample:
    """固定化审查产物中的优先样例。"""

    sample_id: str
    image_path: str
    query_text: str
    failure_category: str
    rationale: str


@dataclass(slots=True)
class EmbeddedLocatorReviewArtifact:
    """每轮评测后的固定结构审查摘要。"""

    best_parameter_group_label: str
    best_parameter_group_options: dict[str, object]
    best_parameter_group_overrides: dict[str, object]
    most_common_failure_type: str | None
    priority_samples: list[EmbeddedLocatorPrioritySample]
    suggested_next_iteration: str


@dataclass(slots=True)
class EmbeddedLocatorEvaluationReport:
    """评测总报告。"""

    manifest_id: str
    manifest_path: str
    run_mode: str
    runs: list[EmbeddedLocatorRunReport]
    comparison: EmbeddedLocatorSweepComparison
    review: EmbeddedLocatorReviewArtifact


def load_embedded_locator_manifest(path: str | Path) -> EmbeddedLocatorManifest:
    """读取并校验 embedded locator 弱标注 manifest。"""
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_payload = payload.get("samples")
    if not isinstance(samples_payload, list):
        raise ValueError("Embedded locator manifest 'samples' must be a list")

    samples: list[EmbeddedLocatorManifestSample] = []
    for index, item in enumerate(samples_payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                "Embedded locator manifest sample must be an object, "
                f"got {type(item).__name__} at index {index}"
            )
        sample = EmbeddedLocatorManifestSample(
            sample_id=_read_required_str(item, "id"),
            image_path=_read_required_str(item, "image_path"),
            query_text=_read_required_str(item, "query_text"),
            should_locate=_read_required_bool(item, "should_locate"),
            acceptable_point_standard=_read_required_str(
                item, "acceptable_point_standard"
            ),
            acceptable_box_standard=_read_optional_str(item, "acceptable_box_standard"),
            remark=_read_required_str(item, "remark"),
            minimum_points=int(item.get("minimum_points", 1)),
            minimum_top_confidence=_read_optional_float(
                item, "minimum_top_confidence"
            ),
            require_coarse_box=_read_optional_bool(item, "require_coarse_box", False),
        )
        resolved_path = (manifest_path.parent / sample.image_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise ValueError(
                "Embedded locator manifest sample "
                f"{sample.sample_id!r} points to a missing image: {resolved_path}"
            )
        samples.append(sample)

    notes = _read_str_list(payload.get("notes", []), field_name="notes")
    manifest = EmbeddedLocatorManifest(
        manifest_id=_read_required_str(payload, "manifest_id"),
        manifest_path=str(manifest_path),
        base_dir=str(manifest_path.parent),
        samples=samples,
        notes=notes,
    )
    return manifest


def load_parameter_groups_payload(payload: object) -> list[EmbeddedLocatorParameterGroup]:
    """把 JSON payload 转成参数组。"""
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("Embedded locator parameter sweep payload must be a non-empty list")

    groups: list[EmbeddedLocatorParameterGroup] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                "Embedded locator parameter group must be an object, "
                f"got {type(item).__name__} at index {index}"
            )
        label = item.get("label") or item.get("name") or f"group_{index}"
        options = item.get("runtime_options", item.get("options", {}))
        if not isinstance(options, dict):
            raise ValueError(
                "Embedded locator parameter group runtime options must be an object "
                f"for {label!r}"
            )
        groups.append(
            EmbeddedLocatorParameterGroup(
                label=str(label),
                runtime_options=dict(options),
            )
        )
    return groups


def classify_embedded_locator_failure(
    *,
    proposal_status: str,
    diagnostics: list[str],
    point_count: int,
    has_coarse_box: bool,
    top_confidence: float | None,
    minimum_points: int = 1,
    minimum_top_confidence: float | None = None,
    require_coarse_box: bool = False,
) -> str | None:
    """基于受控输出做最小失败分类。"""
    normalized_status = str(proposal_status).lower()
    normalized_diagnostics = [str(item) for item in diagnostics]

    if normalized_status == ProposalStatus.FAILED.value:
        return EmbeddedLocatorFailureCategory.RUNTIME_FAILED.value
    if normalized_status == ProposalStatus.EMPTY.value:
        if _diagnostics_contain(
            normalized_diagnostics,
            ("no_points_after_runtime_filter", "no_points_after_filter"),
        ):
            return EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value
        return EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value
    if point_count <= 0:
        if has_coarse_box or require_coarse_box or _diagnostics_contain(
            normalized_diagnostics,
            ("coarse_box_only", "unusable_prior", "prefer_box"),
        ):
            return (
                EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value
            )
        return EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value
    if require_coarse_box and not has_coarse_box:
        return EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value
    if minimum_top_confidence is not None:
        if top_confidence is None or top_confidence < float(minimum_top_confidence):
            return EmbeddedLocatorFailureCategory.LOW_CONFIDENCE_OR_SPARSE_POINTS.value
    if point_count < max(int(minimum_points), 0):
        return EmbeddedLocatorFailureCategory.LOW_CONFIDENCE_OR_SPARSE_POINTS.value
    if _diagnostics_contain(normalized_diagnostics, ("unusable_prior",)):
        return EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value
    return None


class EmbeddedLocatorEvaluationHarness:
    """复用现有 proposal chain 的 embedded locator 评测器。"""

    def __init__(self, proposal_module: DefaultProposalEngineModule) -> None:
        self.proposal_module = proposal_module
        self._default_runtime_options = _default_runtime_options()

    @classmethod
    def from_locator_adapter(
        cls,
        *,
        locator_adapter: LocatorAdapter,
        artifact_root: str | Path,
    ) -> "EmbeddedLocatorEvaluationHarness":
        store = LocalFileArtifactStore(str(Path(artifact_root).expanduser()))
        return cls(
            DefaultProposalEngineModule(
                route_handlers={
                    ProposalRoute.LOCATE: LocateProposalRouteHandler(
                        locator_adapter=locator_adapter
                    )
                },
                artifact_store=store,
            )
        )

    def evaluate(
        self,
        *,
        manifest: EmbeddedLocatorManifest,
        parameter_groups: list[EmbeddedLocatorParameterGroup],
    ) -> EmbeddedLocatorEvaluationReport:
        if not parameter_groups:
            raise ValueError("Embedded locator evaluation requires at least one parameter group")

        run_reports = [
            self._evaluate_parameter_group(manifest=manifest, parameter_group=group)
            for group in parameter_groups
        ]
        comparison = self._build_sweep_comparison(run_reports)
        best_run = self._select_best_run(run_reports)
        review = self._build_review(best_run)
        run_mode = "sweep" if len(run_reports) > 1 else "single"
        return EmbeddedLocatorEvaluationReport(
            manifest_id=manifest.manifest_id,
            manifest_path=manifest.manifest_path,
            run_mode=run_mode,
            runs=run_reports,
            comparison=comparison,
            review=review,
        )

    def _evaluate_parameter_group(
        self,
        *,
        manifest: EmbeddedLocatorManifest,
        parameter_group: EmbeddedLocatorParameterGroup,
    ) -> EmbeddedLocatorRunReport:
        effective_runtime_options = self._resolve_effective_runtime_options(
            parameter_group
        )
        sample_results = [
            self._evaluate_sample(
                manifest=manifest,
                sample=sample,
                parameter_group=parameter_group,
                effective_runtime_options=effective_runtime_options,
            )
            for sample in manifest.samples
        ]
        return EmbeddedLocatorRunReport(
            parameter_group=parameter_group,
            effective_runtime_options=effective_runtime_options,
            samples=sample_results,
            summary=self._build_summary(sample_results),
        )

    def _evaluate_sample(
        self,
        *,
        manifest: EmbeddedLocatorManifest,
        sample: EmbeddedLocatorManifestSample,
        parameter_group: EmbeddedLocatorParameterGroup,
        effective_runtime_options: dict[str, object],
    ) -> EmbeddedLocatorSampleEvaluationResult:
        image_path = manifest.resolve_image_path(sample)
        try:
            output = self.proposal_module.run(
                ProposalEngineModuleInput(
                    task_id=self._task_id_for_sample(parameter_group.label, sample.sample_id),
                    attempt_index=1,
                    understanding=_build_minimal_understanding(sample.query_text),
                    image_ref=ImageRef(uri=str(image_path)),
                    preferred_route=ProposalRoute.LOCATE,
                    module_options=dict(parameter_group.runtime_options),
                )
            )
            proposal = output.primary_payload
            if proposal is None:
                diagnostics = ["proposal_engine_returned_no_payload"]
                limitations = ["proposal output missing primary payload"]
                proposal_status = ProposalStatus.FAILED.value
                point_count = 0
                top_confidence = None
                has_coarse_box = False
                proposal_summary = None
            else:
                primary_candidate = _resolve_primary_candidate(proposal)
                diagnostics = _sanitize_proposal_diagnostics(proposal.diagnostics)
                limitations = (
                    list(primary_candidate.limitations) if primary_candidate is not None else []
                )
                proposal_status = proposal.status.value
                point_count = (
                    len(primary_candidate.positive_point_hints)
                    if primary_candidate is not None
                    else 0
                )
                top_confidence = (
                    round(float(primary_candidate.confidence), 4)
                    if primary_candidate is not None
                    and primary_candidate.confidence is not None
                    else None
                )
                has_coarse_box = (
                    primary_candidate.region_box is not None
                    if primary_candidate is not None
                    else False
                )
                proposal_summary = proposal.proposal_summary
        except Exception as exc:
            diagnostics = [f"evaluation_exception={type(exc).__name__}: {exc}"]
            limitations = ["locator runtime raised before emitting ProposalResult"]
            proposal_status = ProposalStatus.FAILED.value
            point_count = 0
            top_confidence = None
            has_coarse_box = False
            proposal_summary = None

        failure_category = classify_embedded_locator_failure(
            proposal_status=proposal_status,
            diagnostics=diagnostics,
            point_count=point_count,
            has_coarse_box=has_coarse_box,
            top_confidence=top_confidence,
            minimum_points=sample.minimum_points,
            minimum_top_confidence=sample.minimum_top_confidence,
            require_coarse_box=sample.require_coarse_box,
        )
        passed = _sample_passed_expectation(
            should_locate=sample.should_locate,
            proposal_status=proposal_status,
            failure_category=failure_category,
        )
        return EmbeddedLocatorSampleEvaluationResult(
            sample_id=sample.sample_id,
            image_path=str(image_path),
            query_text=sample.query_text,
            should_locate=sample.should_locate,
            proposal_status=proposal_status,
            point_count=point_count,
            top_confidence=top_confidence,
            has_coarse_box=has_coarse_box,
            diagnostics=diagnostics,
            limitations=limitations,
            runtime_options_snapshot=dict(effective_runtime_options),
            runtime_option_overrides=dict(parameter_group.runtime_options),
            failure_category=failure_category,
            passed=passed,
            proposal_summary=proposal_summary,
        )

    def _build_summary(
        self,
        sample_results: list[EmbeddedLocatorSampleEvaluationResult],
    ) -> EmbeddedLocatorRunSummary:
        ready_count = sum(
            1 for item in sample_results if item.proposal_status == ProposalStatus.READY.value
        )
        empty_count = sum(
            1 for item in sample_results if item.proposal_status == ProposalStatus.EMPTY.value
        )
        failed_count = sum(
            1 for item in sample_results if item.proposal_status == ProposalStatus.FAILED.value
        )
        passed_sample_count = sum(1 for item in sample_results if item.passed)
        failure_counts = {
            category.value: 0 for category in EmbeddedLocatorFailureCategory
        }
        representative_samples: dict[str, str] = {}
        for item in sample_results:
            if item.failure_category is None:
                continue
            failure_counts[item.failure_category] += 1
            representative_samples.setdefault(item.failure_category, item.sample_id)
        return EmbeddedLocatorRunSummary(
            total_samples=len(sample_results),
            ready_count=ready_count,
            empty_count=empty_count,
            failed_count=failed_count,
            passed_sample_count=passed_sample_count,
            failure_category_counts=failure_counts,
            representative_samples_by_failure_category=representative_samples,
        )

    def _build_sweep_comparison(
        self,
        run_reports: list[EmbeddedLocatorRunReport],
    ) -> EmbeddedLocatorSweepComparison:
        entries = [
            EmbeddedLocatorSweepComparisonEntry(
                label=run.parameter_group.label,
                runtime_options=dict(run.effective_runtime_options),
                runtime_option_overrides=dict(run.parameter_group.runtime_options),
                passed_sample_count=run.summary.passed_sample_count,
                failure_category_counts=dict(run.summary.failure_category_counts),
            )
            for run in run_reports
        ]
        best_run = self._select_best_run(run_reports)
        return EmbeddedLocatorSweepComparison(
            entries=entries,
            best_parameter_group_label=best_run.parameter_group.label,
        )

    def _build_review(
        self,
        best_run: EmbeddedLocatorRunReport,
    ) -> EmbeddedLocatorReviewArtifact:
        most_common_failure_type = _most_common_failure_type(
            best_run.summary.failure_category_counts
        )
        priority_samples = [
            EmbeddedLocatorPrioritySample(
                sample_id=item.sample_id,
                image_path=item.image_path,
                query_text=item.query_text,
                failure_category=item.failure_category or "unknown",
                rationale=_priority_sample_rationale(item),
            )
            for item in _select_priority_samples(best_run.samples)[:3]
        ]
        return EmbeddedLocatorReviewArtifact(
            best_parameter_group_label=best_run.parameter_group.label,
            best_parameter_group_options=dict(best_run.effective_runtime_options),
            best_parameter_group_overrides=dict(best_run.parameter_group.runtime_options),
            most_common_failure_type=most_common_failure_type,
            priority_samples=priority_samples,
            suggested_next_iteration=_suggest_next_iteration_direction(
                most_common_failure_type
            ),
        )

    def _select_best_run(
        self,
        run_reports: list[EmbeddedLocatorRunReport],
    ) -> EmbeddedLocatorRunReport:
        return max(
            run_reports,
            key=lambda run: (
                run.summary.passed_sample_count,
                _negative_pass_count(run.samples),
                -_unexpected_ready_on_negative_count(run.samples),
                -run.summary.failed_count,
                -_failure_severity_score(run.summary.failure_category_counts),
                run.parameter_group.label,
            ),
        )

    def _resolve_effective_runtime_options(
        self,
        parameter_group: EmbeddedLocatorParameterGroup,
    ) -> dict[str, object]:
        effective_options = dict(self._default_runtime_options)
        locator_adapter = self._resolve_locator_adapter()
        runtime_defaults = _extract_runtime_option_defaults(locator_adapter)
        if runtime_defaults:
            effective_options.update(runtime_defaults)
        effective_options.update(parameter_group.runtime_options)
        return _normalize_runtime_options(effective_options)

    def _resolve_locator_adapter(self) -> LocatorAdapter | None:
        handler = self.proposal_module.route_handlers.get(ProposalRoute.LOCATE)
        if isinstance(handler, LocateProposalRouteHandler):
            return handler.locator_adapter
        return None

    @staticmethod
    def _task_id_for_sample(parameter_label: str, sample_id: str) -> str:
        return "embedded-locator-eval-" + _sanitize_identifier(
            f"{parameter_label}-{sample_id}"
        )


def write_embedded_locator_evaluation_report(
    report: EmbeddedLocatorEvaluationReport,
    path: str | Path,
) -> Path:
    """把结构化评测结果写入 JSON 文件。"""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _read_required_str(payload: dict[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Embedded locator manifest field {field_name!r} must be a non-empty string"
        )
    return value.strip()


def _read_optional_str(payload: dict[str, object], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Embedded locator manifest field {field_name!r} must be a string when provided"
        )
    stripped = value.strip()
    return stripped or None


def _read_required_bool(payload: dict[str, object], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(
            f"Embedded locator manifest field {field_name!r} must be a boolean"
        )
    return value


def _read_optional_bool(
    payload: dict[str, object],
    field_name: str,
    default: bool,
) -> bool:
    value = payload.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(
            f"Embedded locator manifest field {field_name!r} must be a boolean"
        )
    return value


def _read_optional_float(payload: dict[str, object], field_name: str) -> float | None:
    value = payload.get(field_name)
    if value is None:
        return None
    return float(value)


def _read_str_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(
            f"Embedded locator manifest field {field_name!r} must be a list of strings"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"Embedded locator manifest field {field_name!r} must contain only strings"
            )
        items.append(item)
    return items


def _normalize_runtime_options(options: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in options.items():
        if key not in EMBEDDED_LOCATOR_RUNTIME_OPTION_KEYS:
            raise ValueError(f"Unsupported embedded locator runtime option: {key}")
        if key in _INT_RUNTIME_OPTION_KEYS:
            normalized[key] = int(value)
        elif key in _FLOAT_RUNTIME_OPTION_KEYS:
            normalized[key] = float(value)
        else:
            normalized[key] = value
    return normalized


def _default_runtime_options() -> dict[str, object]:
    defaults = EmbeddedGridGroundRuntimeConfig()
    return {
        key: getattr(defaults, key)
        for key in EMBEDDED_LOCATOR_RUNTIME_OPTION_KEYS
    }


def _build_minimal_understanding(query_text: str) -> QueryUnderstandingResult:
    focus_terms = _split_focus_terms(query_text)
    return QueryUnderstandingResult(
        understanding_id=f"eval-understanding-{_sanitize_identifier(query_text)}",
        normalized_query=query_text,
        target_summary=query_text,
        target_type=TargetType.UNKNOWN,
        implicitness=ImplicitnessLevel.EXPLICIT,
        canonical_referent_text=query_text,
        referent_number=ReferentNumber.UNKNOWN,
        focus_terms=focus_terms,
        attribute_clues=list(focus_terms),
    )


def _split_focus_terms(query_text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    normalized_query = query_text.replace("_", " ").replace("-", " ")
    for raw_term in normalized_query.split():
        term = raw_term.strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered


def _resolve_primary_candidate(proposal) -> object | None:
    if not proposal.candidates:
        return None
    if proposal.primary_candidate_id:
        for candidate in proposal.candidates:
            if candidate.candidate_id == proposal.primary_candidate_id:
                return candidate
    return proposal.candidates[0]


def _extract_runtime_option_defaults(
    locator_adapter: LocatorAdapter | None,
) -> dict[str, object]:
    if not isinstance(locator_adapter, EmbeddedLocatorAdapter):
        return {}
    runtime = locator_adapter.runtime
    runtime_config = getattr(runtime, "runtime_config", None)
    if runtime_config is None:
        return {}
    defaults: dict[str, object] = {}
    for key in EMBEDDED_LOCATOR_RUNTIME_OPTION_KEYS:
        if hasattr(runtime_config, key):
            defaults[key] = getattr(runtime_config, key)
    return _normalize_runtime_options(defaults) if defaults else {}


def _sanitize_proposal_diagnostics(diagnostics: list[str]) -> list[str]:
    sanitized: list[str] = []
    for item in diagnostics:
        normalized = str(item).strip()
        if not normalized:
            continue
        if normalized.startswith("selected_k="):
            sanitized.append(normalized)
            continue
        if normalized.startswith("embedded_locator."):
            sanitized.append(normalized)
            continue
        if normalized.startswith("reason="):
            sanitized.append(normalized)
            continue
        if normalized.startswith("evaluation_exception="):
            sanitized.append(normalized)
            continue
        if normalized in {
            "no_points_after_runtime_filter",
            "no_points_after_filter",
            "coarse_box_only",
            "unusable_prior",
            "proposal_engine_returned_no_payload",
        }:
            sanitized.append(normalized)
    return sanitized


def _diagnostics_contain(diagnostics: list[str], markers: tuple[str, ...]) -> bool:
    for diagnostic in diagnostics:
        for marker in markers:
            if marker in diagnostic:
                return True
    return False


def _sample_passed_expectation(
    *,
    should_locate: bool,
    proposal_status: str,
    failure_category: str | None,
) -> bool:
    if should_locate:
        return (
            proposal_status == ProposalStatus.READY.value
            and failure_category is None
        )
    return (
        proposal_status == ProposalStatus.EMPTY.value
        and failure_category
        in {
            EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value,
            EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value,
        }
    )


def _sanitize_identifier(text: str) -> str:
    chars: list[str] = []
    for char in text.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "sample"


def _most_common_failure_type(failure_counts: dict[str, int]) -> str | None:
    ranked = [
        (count, category)
        for category, count in failure_counts.items()
        if count > 0
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][1]


def _failure_severity_score(failure_counts: dict[str, int]) -> int:
    severity_weights = {
        EmbeddedLocatorFailureCategory.RUNTIME_FAILED.value: 5,
        EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value: 4,
        EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value: 3,
        EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value: 2,
        EmbeddedLocatorFailureCategory.LOW_CONFIDENCE_OR_SPARSE_POINTS.value: 1,
    }
    total = 0
    for category, count in failure_counts.items():
        total += severity_weights.get(category, 0) * count
    return total


def _negative_pass_count(
    sample_results: list[EmbeddedLocatorSampleEvaluationResult],
) -> int:
    return sum(
        1 for item in sample_results if not item.should_locate and item.passed
    )


def _unexpected_ready_on_negative_count(
    sample_results: list[EmbeddedLocatorSampleEvaluationResult],
) -> int:
    return sum(
        1
        for item in sample_results
        if not item.should_locate and item.proposal_status == ProposalStatus.READY.value
    )


def _select_priority_samples(
    sample_results: list[EmbeddedLocatorSampleEvaluationResult],
) -> list[EmbeddedLocatorSampleEvaluationResult]:
    severity_rank = {
        EmbeddedLocatorFailureCategory.RUNTIME_FAILED.value: 0,
        EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value: 1,
        EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value: 2,
        EmbeddedLocatorFailureCategory.LOW_CONFIDENCE_OR_SPARSE_POINTS.value: 3,
        EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value: 4,
    }
    failing = [item for item in sample_results if item.failure_category is not None]
    failing.sort(
        key=lambda item: (
            severity_rank.get(item.failure_category or "", 99),
            item.sample_id,
        )
    )
    return failing


def _priority_sample_rationale(
    sample_result: EmbeddedLocatorSampleEvaluationResult,
) -> str:
    if sample_result.failure_category == EmbeddedLocatorFailureCategory.RUNTIME_FAILED.value:
        return "Runtime failed before a usable proposal was produced."
    if (
        sample_result.failure_category
        == EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value
    ):
        return "Runtime reached filtering but removed all points."
    if sample_result.failure_category == EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value:
        return "Proposal stayed empty even though the sample is in the baseline."
    if (
        sample_result.failure_category
        == EmbeddedLocatorFailureCategory.LOW_CONFIDENCE_OR_SPARSE_POINTS.value
    ):
        return "Proposal is present but still too weak on confidence or point density."
    if (
        sample_result.failure_category
        == EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value
    ):
        return "Only a coarse box or unusable prior survived the proposal stage."
    return "Review this sample before the next tuning round."


def _suggest_next_iteration_direction(most_common_failure_type: str | None) -> str:
    if most_common_failure_type == EmbeddedLocatorFailureCategory.RUNTIME_FAILED.value:
        return (
            "先稳住 runtime 装配与异常路径，再做效果调参；优先检查 checkpoint、"
            "config 和骨干 session 生命周期。"
        )
    if (
        most_common_failure_type
        == EmbeddedLocatorFailureCategory.NO_POINTS_AFTER_FILTER.value
    ):
        return (
            "下一轮先围绕已有 runtime 选项放松过滤：对比更低的 abs_threshold、"
            "min_point_confidence，以及略大的 max_k。"
        )
    if most_common_failure_type == EmbeddedLocatorFailureCategory.EMPTY_PROPOSAL.value:
        return (
            "下一轮优先核对空提案样例的 query 与 proposal mapping，确认 runtime "
            "至少能稳定产出点或 coarse box。"
        )
    if (
        most_common_failure_type
        == EmbeddedLocatorFailureCategory.LOW_CONFIDENCE_OR_SPARSE_POINTS.value
    ):
        return (
            "下一轮重点对比置信度与点稀疏问题：在不新增算法开关的前提下，调小 "
            "abs_threshold / rel_ratio，并比较 min_k、max_k 的召回变化。"
        )
    if (
        most_common_failure_type
        == EmbeddedLocatorFailureCategory.COARSE_BOX_ONLY_OR_UNUSABLE_PRIOR.value
    ):
        return (
            "下一轮优先修 prior 可用性，关注只剩 coarse box 的样例，确认正点是否在"
            "过滤阶段被提前抹掉。"
        )
    return "当前最佳参数组已通过现有弱标注基线；下一轮可以增加更难样例或收紧弱标准。"
