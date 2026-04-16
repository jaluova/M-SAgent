"""service 层默认组合根。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from msagent.core.config.settings import MSAgentSettings
from msagent.core.contracts.types import ProposalRoute
from msagent.core.policies.retry_policy import RetryPolicy
from msagent.infra.adapters import LLMAdapter, LocatorAdapter, SAMAdapter
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.qwen_llm_adapter import (
    RealQwenLLMAdapterBundle,
    RealQwenLLMAdapterConfig,
    build_real_qwen_llm_adapter_bundle,
)
from msagent.infra.sam3_adapter import (
    RealSAM3AdapterBundle,
    RealSAM3AdapterConfig,
    build_real_sam3_adapter_bundle,
)
from msagent.infra.runtime.factory import (
    EmbeddedLocatorRuntimeBundle,
    EmbeddedLocatorRuntimeFactoryConfig,
    build_embedded_locator_runtime_bundle,
)
from msagent.modules.evaluator import LLMEvaluatorModule
from msagent.modules.prompt_bridge import RuleBasedPromptBridgeModule
from msagent.modules.proposal_engine import DefaultProposalEngineModule, LocateProposalRouteHandler
from msagent.modules.query_understanding import LLMQueryUnderstandingModule
from msagent.modules.segmenter import SAMSegmenterModule
from msagent.orchestrator.orchestrator import Orchestrator, OrchestratorDependencies
from msagent.orchestrator.orchestrator import OrchestrationResult
from msagent.service.api import APIResponse, APIService
from msagent.service.api_transport import APIHandler, create_fastapi_app
from msagent.service.cli import CLIService
from msagent.service.cli import CLIRequest
from msagent.service.demo_report import (
    build_demo_task_report,
    render_demo_task_report_markdown,
)


@dataclass(slots=True)
class _DefaultCoreAssembly:
    """默认 service 组合根共享的核心依赖。"""

    artifact_store: LocalFileArtifactStore
    locator_adapter: LocatorAdapter
    llm_adapter: LLMAdapter
    sam_adapter: SAMAdapter
    runtime_bundle: EmbeddedLocatorRuntimeBundle | None
    llm_bundle: RealQwenLLMAdapterBundle | None
    sam_bundle: RealSAM3AdapterBundle | None
    diagnostics: list[str]
    orchestrator: Orchestrator


@dataclass(slots=True)
class CLIServiceAssembly:
    """默认 CLI 装配产物。"""

    service: CLIService
    artifact_store: LocalFileArtifactStore
    locator_adapter: LocatorAdapter
    llm_adapter: LLMAdapter
    sam_adapter: SAMAdapter
    runtime_bundle: EmbeddedLocatorRuntimeBundle | None = None
    llm_bundle: RealQwenLLMAdapterBundle | None = None
    sam_bundle: RealSAM3AdapterBundle | None = None
    diagnostics: list[str] = field(default_factory=list)
    _closed: bool = False

    def run(self, request: CLIRequest) -> OrchestrationResult:
        """执行一次默认 CLI 任务，并对称释放装配阶段持有的资源。"""
        try:
            result = self.service.run(request)
            self._write_task_report_if_requested(request, result)
            return result
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.sam_bundle is not None:
            self.sam_bundle.close()
        if self.llm_bundle is not None:
            self.llm_bundle.close()
        if self.runtime_bundle is not None:
            self.runtime_bundle.close()

    def _write_task_report_if_requested(
        self,
        request: CLIRequest,
        result: OrchestrationResult,
    ) -> None:
        output_dir = request.output_dir
        if output_dir is None or not output_dir.strip():
            return

        report = build_demo_task_report(
            result,
            artifact_store=self.artifact_store,
        )
        markdown = render_demo_task_report_markdown(report)
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "task_report.md").write_text(markdown, encoding="utf-8")


@dataclass(slots=True)
class APIServiceAssembly:
    """默认 API 装配产物。"""

    service: APIService
    handler: APIHandler
    artifact_store: LocalFileArtifactStore
    host: str
    port: int
    enable_debug_features: bool = False
    diagnostics: list[str] = field(default_factory=list)
    _runtime_bundle: EmbeddedLocatorRuntimeBundle | None = None
    _llm_bundle: RealQwenLLMAdapterBundle | None = None
    _sam_bundle: RealSAM3AdapterBundle | None = None
    _closed: bool = False

    def handle(self, payload: dict[str, object]) -> APIResponse:
        """执行一次 transport payload -> APIResponse 的标准入口调用。"""
        return self.handler.handle_run(payload)

    def create_app(self) -> object:
        """按需创建 FastAPI app。"""
        return create_fastapi_app(
            self.handler,
            on_shutdown=self.close,
            enable_debug_features=self.enable_debug_features,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sam_bundle is not None:
            self._sam_bundle.close()
        if self._llm_bundle is not None:
            self._llm_bundle.close()
        if self._runtime_bundle is not None:
            self._runtime_bundle.close()


def build_default_cli_service(
    settings: MSAgentSettings | None = None,
    *,
    locator_adapter: LocatorAdapter | None = None,
    llm_adapter: LLMAdapter | None = None,
    sam_adapter: SAMAdapter | None = None,
) -> CLIServiceAssembly:
    """构造默认 CLI service。"""
    resolved_settings = settings or MSAgentSettings.from_env()
    core = _build_default_core_assembly(
        resolved_settings,
        locator_adapter=locator_adapter,
        llm_adapter=llm_adapter,
        sam_adapter=sam_adapter,
        caller_name="build_default_cli_service",
    )
    return CLIServiceAssembly(
        service=CLIService(
            orchestrator=core.orchestrator,
            default_max_attempts=resolved_settings.runtime.max_attempts,
        ),
        artifact_store=core.artifact_store,
        locator_adapter=core.locator_adapter,
        llm_adapter=core.llm_adapter,
        sam_adapter=core.sam_adapter,
        runtime_bundle=core.runtime_bundle,
        llm_bundle=core.llm_bundle,
        sam_bundle=core.sam_bundle,
        diagnostics=core.diagnostics,
    )


def build_default_api_service(
    settings: MSAgentSettings | None = None,
    *,
    locator_adapter: LocatorAdapter | None = None,
    llm_adapter: LLMAdapter | None = None,
    sam_adapter: SAMAdapter | None = None,
) -> APIServiceAssembly:
    """构造默认 API service 与 transport handler。"""
    resolved_settings = settings or MSAgentSettings.from_env()
    if not resolved_settings.service.enable_api:
        raise ValueError(
            "build_default_api_service requires service.enable_api=True."
        )

    core = _build_default_core_assembly(
        resolved_settings,
        locator_adapter=locator_adapter,
        llm_adapter=llm_adapter,
        sam_adapter=sam_adapter,
        caller_name="build_default_api_service",
    )
    service = APIService(
        orchestrator=core.orchestrator,
        artifact_store=core.artifact_store,
        include_debug_report=resolved_settings.service.enable_debug_features,
        default_max_attempts=resolved_settings.runtime.max_attempts,
    )
    return APIServiceAssembly(
        service=service,
        handler=APIHandler(service=service),
        artifact_store=core.artifact_store,
        host=resolved_settings.service.host,
        port=resolved_settings.service.port,
        enable_debug_features=resolved_settings.service.enable_debug_features,
        diagnostics=core.diagnostics,
        _runtime_bundle=core.runtime_bundle,
        _llm_bundle=core.llm_bundle,
        _sam_bundle=core.sam_bundle,
    )


def _build_default_core_assembly(
    settings: MSAgentSettings,
    *,
    locator_adapter: LocatorAdapter | None,
    llm_adapter: LLMAdapter | None,
    sam_adapter: SAMAdapter | None,
    caller_name: str,
) -> _DefaultCoreAssembly:
    if settings.runtime.default_route is not ProposalRoute.LOCATE:
        raise ValueError(
            f"{caller_name} only supports runtime.default_route=LOCATE; "
            f"got {settings.runtime.default_route.value!r}."
        )

    artifact_store = LocalFileArtifactStore(settings.runtime.artifact_root)
    locator, runtime_bundle, diagnostics = _build_default_locator_adapter(
        settings,
        locator_adapter=locator_adapter,
    )
    llm_bundle: RealQwenLLMAdapterBundle | None = None
    sam_bundle: RealSAM3AdapterBundle | None = None
    try:
        sam, sam_bundle, sam_diagnostics = _build_default_sam_adapter(
            settings,
            sam_adapter=sam_adapter,
            artifact_store=artifact_store,
        )
        diagnostics.extend(sam_diagnostics)
        llm, llm_bundle = _build_default_llm_adapter(
            settings,
            llm_adapter=llm_adapter,
            runtime_bundle=runtime_bundle,
        )
        orchestrator = Orchestrator(
            OrchestratorDependencies(
                query_understanding_module=LLMQueryUnderstandingModule(
                    llm_adapter=llm,
                    artifact_store=artifact_store,
                ),
                proposal_engine_module=DefaultProposalEngineModule(
                    route_handlers={
                        ProposalRoute.LOCATE: LocateProposalRouteHandler(
                            locator_adapter=locator
                        )
                    },
                    artifact_store=artifact_store,
                ),
                prompt_bridge_module=RuleBasedPromptBridgeModule(artifact_store=artifact_store),
                segmenter_module=SAMSegmenterModule(
                    sam_adapter=sam,
                    artifact_store=artifact_store,
                ),
                evaluator_module=LLMEvaluatorModule(
                    llm_adapter=llm,
                    artifact_store=artifact_store,
                ),
                retry_policy=RetryPolicy(
                    default_route=settings.runtime.default_route,
                ),
            )
        )
    except Exception:
        if llm_bundle is not None:
            llm_bundle.close()
        if sam_bundle is not None:
            sam_bundle.close()
        if runtime_bundle is not None:
            runtime_bundle.close()
        raise
    return _DefaultCoreAssembly(
        artifact_store=artifact_store,
        locator_adapter=locator,
        llm_adapter=llm,
        sam_adapter=sam,
        runtime_bundle=runtime_bundle,
        llm_bundle=llm_bundle,
        sam_bundle=sam_bundle,
        diagnostics=diagnostics,
        orchestrator=orchestrator,
    )


def _build_default_locator_adapter(
    settings: MSAgentSettings,
    *,
    locator_adapter: LocatorAdapter | None,
) -> tuple[LocatorAdapter, EmbeddedLocatorRuntimeBundle | None, list[str]]:
    if locator_adapter is not None:
        return locator_adapter, None, ["embedded_locator_runtime=custom"]

    model_paths = settings.model_paths
    if model_paths.has_partial_embedded_locator_runtime():
        raise ValueError(
            "Embedded locator runtime configuration is partial; "
            "qwen_model_path, embedded_locator_adapter_path, and "
            "embedded_locator_config_path must be provided together."
        )

    if not model_paths.has_embedded_locator_runtime():
        raise ValueError(
            "Embedded locator runtime is not configured; qwen_model_path, "
            "embedded_locator_adapter_path, and embedded_locator_config_path "
            "must be provided together, or pass locator_adapter explicitly."
        )

    runtime_bundle = build_embedded_locator_runtime_bundle(
        EmbeddedLocatorRuntimeFactoryConfig(
            qwen_model_path=model_paths.qwen_model_path or "",
            adapter_path=model_paths.embedded_locator_adapter_path or "",
            config_path=model_paths.embedded_locator_config_path or "",
        )
    )
    return (
        runtime_bundle.locator_adapter,
        runtime_bundle,
        ["embedded_locator_runtime=enabled"],
    )


def _build_default_sam_adapter(
    settings: MSAgentSettings,
    *,
    sam_adapter: SAMAdapter | None,
    artifact_store: LocalFileArtifactStore,
) -> tuple[SAMAdapter, RealSAM3AdapterBundle | None, list[str]]:
    if sam_adapter is not None:
        return sam_adapter, None, ["sam_runtime=custom"]

    model_paths = settings.model_paths
    if model_paths.has_partial_real_sam_runtime():
        raise ValueError(
            "SAM3 runtime configuration is partial; "
            "sam_model_path and sam_checkpoint_path must be provided together."
        )

    if not model_paths.has_real_sam_runtime():
        raise ValueError(
            "SAM3 runtime is not configured; sam_model_path and "
            "sam_checkpoint_path must be provided together, or pass "
            "sam_adapter explicitly."
        )

    sam_bundle = build_real_sam3_adapter_bundle(
        RealSAM3AdapterConfig(
            sam_model_path=model_paths.sam_model_path or "",
            checkpoint_path=model_paths.sam_checkpoint_path or "",
            bpe_path=model_paths.sam_bpe_path,
        ),
        artifact_store=artifact_store,
    )
    return sam_bundle.sam_adapter, sam_bundle, ["sam_runtime=enabled"]


def _build_default_llm_adapter(
    settings: MSAgentSettings,
    *,
    llm_adapter: LLMAdapter | None,
    runtime_bundle: EmbeddedLocatorRuntimeBundle | None,
) -> tuple[LLMAdapter, RealQwenLLMAdapterBundle | None]:
    if llm_adapter is not None:
        return llm_adapter, None

    if not settings.service.enable_real_llm:
        raise ValueError(
            "Real LLM adapter is disabled; set service.enable_real_llm=True "
            "or pass llm_adapter explicitly."
        )

    model_path = settings.model_paths.qwen_model_path
    if model_path is None or not model_path.strip():
        raise ValueError(
            "Real LLM adapter requires model_paths.qwen_model_path to be configured."
        )

    shared_provider = runtime_bundle.backbone_provider if runtime_bundle is not None else None
    llm_bundle = build_real_qwen_llm_adapter_bundle(
        RealQwenLLMAdapterConfig(
            qwen_model_path=model_path,
        ),
        shared_backbone_provider=shared_provider,
    )
    return llm_bundle.llm_adapter, llm_bundle
