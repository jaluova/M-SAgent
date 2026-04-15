"""service 层默认组合根。"""

from __future__ import annotations

from dataclasses import dataclass, field

from msagent.core.config.settings import MSAgentSettings
from msagent.core.contracts.types import EvaluationVerdict, ProposalRoute
from msagent.core.policies.retry_policy import RetryPolicy
from msagent.infra.adapters import LLMAdapter, LocatorAdapter, SAMAdapter
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.mock_adapters import MockLLMAdapter, MockLocatorAdapter, MockSAMAdapter
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


@dataclass(slots=True)
class _DefaultCoreAssembly:
    """默认 service 组合根共享的核心依赖。"""

    artifact_store: LocalFileArtifactStore
    locator_adapter: LocatorAdapter
    llm_adapter: LLMAdapter
    sam_adapter: SAMAdapter
    runtime_bundle: EmbeddedLocatorRuntimeBundle | None
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
    diagnostics: list[str] = field(default_factory=list)
    _closed: bool = False

    def run(self, request: CLIRequest) -> OrchestrationResult:
        """执行一次默认 CLI 任务，并对称释放装配阶段持有的资源。"""
        try:
            return self.service.run(request)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.runtime_bundle is not None:
            self.runtime_bundle.close()


@dataclass(slots=True)
class APIServiceAssembly:
    """默认 API 装配产物。"""

    service: APIService
    handler: APIHandler
    artifact_store: LocalFileArtifactStore
    host: str
    port: int
    diagnostics: list[str] = field(default_factory=list)
    _runtime_bundle: EmbeddedLocatorRuntimeBundle | None = None
    _closed: bool = False

    def handle(self, payload: dict[str, object]) -> APIResponse:
        """执行一次 transport payload -> APIResponse 的标准入口调用。"""
        return self.handler.handle_run(payload)

    def create_app(self) -> object:
        """按需创建 FastAPI app。"""
        return create_fastapi_app(self.handler, on_shutdown=self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runtime_bundle is not None:
            self._runtime_bundle.close()


def build_default_cli_service(
    settings: MSAgentSettings | None = None,
    *,
    llm_adapter: LLMAdapter | None = None,
    sam_adapter: SAMAdapter | None = None,
) -> CLIServiceAssembly:
    """构造默认 CLI service，并在配置完整时接入真实 embedded locator。"""
    resolved_settings = settings or MSAgentSettings()
    core = _build_default_core_assembly(
        resolved_settings,
        llm_adapter=llm_adapter,
        sam_adapter=sam_adapter,
        caller_name="build_default_cli_service",
    )
    return CLIServiceAssembly(
        service=CLIService(orchestrator=core.orchestrator),
        artifact_store=core.artifact_store,
        locator_adapter=core.locator_adapter,
        llm_adapter=core.llm_adapter,
        sam_adapter=core.sam_adapter,
        runtime_bundle=core.runtime_bundle,
        diagnostics=core.diagnostics,
    )


def build_default_api_service(
    settings: MSAgentSettings | None = None,
    *,
    llm_adapter: LLMAdapter | None = None,
    sam_adapter: SAMAdapter | None = None,
) -> APIServiceAssembly:
    """构造默认 API service 与 transport handler。"""
    resolved_settings = settings or MSAgentSettings()
    if not resolved_settings.service.enable_api:
        raise ValueError(
            "build_default_api_service requires service.enable_api=True."
        )

    core = _build_default_core_assembly(
        resolved_settings,
        llm_adapter=llm_adapter,
        sam_adapter=sam_adapter,
        caller_name="build_default_api_service",
    )
    service = APIService(orchestrator=core.orchestrator)
    return APIServiceAssembly(
        service=service,
        handler=APIHandler(service=service),
        artifact_store=core.artifact_store,
        host=resolved_settings.service.host,
        port=resolved_settings.service.port,
        diagnostics=core.diagnostics,
        _runtime_bundle=core.runtime_bundle,
    )


def _build_default_core_assembly(
    settings: MSAgentSettings,
    *,
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
    llm = llm_adapter or MockLLMAdapter(
        backend_name="mock-llm",
        evaluation_verdict_sequence=(EvaluationVerdict.ACCEPT,),
    )
    sam = sam_adapter or MockSAMAdapter(
        backend_name="mock-sam",
        artifact_store=artifact_store,
        model_path=settings.model_paths.sam_model_path,
        checkpoint_path=settings.model_paths.sam_checkpoint_path,
    )
    locator_adapter, runtime_bundle, diagnostics = _build_default_locator_adapter(settings)
    orchestrator = Orchestrator(
        OrchestratorDependencies(
            query_understanding_module=LLMQueryUnderstandingModule(
                llm_adapter=llm,
                artifact_store=artifact_store,
            ),
            proposal_engine_module=DefaultProposalEngineModule(
                route_handlers={
                    ProposalRoute.LOCATE: LocateProposalRouteHandler(
                        locator_adapter=locator_adapter
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
    return _DefaultCoreAssembly(
        artifact_store=artifact_store,
        locator_adapter=locator_adapter,
        llm_adapter=llm,
        sam_adapter=sam,
        runtime_bundle=runtime_bundle,
        diagnostics=diagnostics,
        orchestrator=orchestrator,
    )


def _build_default_locator_adapter(
    settings: MSAgentSettings,
) -> tuple[LocatorAdapter, EmbeddedLocatorRuntimeBundle | None, list[str]]:
    model_paths = settings.model_paths
    if model_paths.has_partial_embedded_locator_runtime():
        raise ValueError(
            "Embedded locator runtime configuration is partial; "
            "qwen_model_path, embedded_locator_adapter_path, and "
            "embedded_locator_config_path must be provided together."
        )

    if not model_paths.has_embedded_locator_runtime():
        return (
            MockLocatorAdapter(
                backend_name="mock-locator",
                model_path=model_paths.locator_model_path,
            ),
            None,
            ["embedded_locator_runtime=disabled"],
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
