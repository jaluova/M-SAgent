from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.contracts.adapter_requests import LocateAdapterRequest
from msagent.core.contracts.common import ArtifactKind, ImageRef, ModuleStatus
from msagent.core.contracts.types import (
    EvaluationVerdict,
    ImplicitnessLevel,
    NormalizedBox,
    ProposalCandidate,
    ProposalResult,
    ProposalRoute,
    ProposalStatus,
    QueryUnderstandingResult,
    ReferentNumber,
    TargetType,
)
from msagent.core.policies.retry_policy import RetryPolicy
from msagent.core.task.enums import StopReason, TaskStage, TaskStatus
from msagent.infra.backbones import (
    EncodedFeatureHandle,
    FeatureSessionHandle,
    SharedQwenBackboneProvider,
    SharedVisionLanguageBackbone,
)
from msagent.infra.adapters import LocatorAdapter
from msagent.infra.embedded_locator import EmbeddedLocatorAdapter
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.mock_adapters import MockLLMAdapter, MockSAMAdapter
from msagent.infra.runtime.train_adapter_runtime import (
    EmbeddedGridGroundRuntimeConfig,
    EmbeddedGridGroundTrainAdapterRuntime,
    EmbeddedLocateBridgeHint,
    EmbeddedLocatePoint,
    EmbeddedLocatePrediction,
    SharedBackboneTrainAdapterRuntime,
    TrainAdapterRuntimeRequest,
)
from msagent.modules.evaluator import LLMEvaluatorModule
from msagent.modules.prompt_bridge import RuleBasedPromptBridgeModule
from msagent.modules.proposal_engine import (
    DefaultProposalEngineModule,
    LocateProposalRouteHandler,
    ProposalEngineModuleInput,
)
from msagent.modules.query_understanding import LLMQueryUnderstandingModule
from msagent.modules.segmenter import SAMSegmenterModule
from msagent.orchestrator.orchestrator import Orchestrator, OrchestratorDependencies
from msagent.service.cli import CLIRequest, CLIService


def make_understanding() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        understanding_id="u-embedded",
        normalized_query="the small red cup",
        target_summary="a small red cup",
        target_type=TargetType.OBJECT,
        implicitness=ImplicitnessLevel.EXPLICIT,
        canonical_referent_text="red cup",
        referent_number=ReferentNumber.SINGLE,
        focus_terms=["small", "red", "cup"],
        attribute_clues=["small", "red"],
    )


class FakeSharedBackbone(SharedVisionLanguageBackbone):
    def __init__(self) -> None:
        super().__init__(
            backbone_name="fake-shared-qwen",
            tokenizer={"kind": "fake-tokenizer"},
            device="cpu",
            dtype="float32",
        )
        self.session_handles: list[FeatureSessionHandle] = []
        self.image_calls: list[tuple[ImageRef, str | None]] = []
        self.text_calls: list[tuple[str, str | None]] = []
        self.text_max_lengths: list[int | None] = []
        self.released_sessions: list[str] = []

    def open_feature_session(
        self,
        *,
        task_id: str,
        metadata: dict[str, object] | None = None,
    ) -> FeatureSessionHandle:
        handle = super().open_feature_session(task_id=task_id, metadata=metadata)
        self.session_handles.append(handle)
        return handle

    def encode_image(
        self,
        image_ref: ImageRef,
        *,
        session: FeatureSessionHandle | None = None,
    ) -> EncodedFeatureHandle:
        self.image_calls.append((image_ref, session.session_id if session else None))
        return EncodedFeatureHandle(
            feature_id=f"image::{len(self.image_calls)}",
            feature_kind="image",
            backbone_name=self.backbone_name,
            session_id=session.session_id if session else None,
            token_count=256,
            hidden_dim=1024,
            metadata={"uri": image_ref.uri},
        )

    def encode_text(
        self,
        text: str,
        *,
        session: FeatureSessionHandle | None = None,
        max_length: int | None = None,
    ) -> EncodedFeatureHandle:
        self.text_calls.append((text, session.session_id if session else None))
        self.text_max_lengths.append(max_length)
        return EncodedFeatureHandle(
            feature_id=f"text::{len(self.text_calls)}",
            feature_kind="text",
            backbone_name=self.backbone_name,
            session_id=session.session_id if session else None,
            token_count=24,
            hidden_dim=1024,
            metadata={"text": text},
        )

    def release_session(self, session: FeatureSessionHandle) -> None:
        self.released_sessions.append(session.session_id)


class FakeSharedQwenProvider(SharedQwenBackboneProvider):
    def __init__(self) -> None:
        super().__init__(
            provider_name="fake-shared-qwen-provider",
            model_path="/models/fake-qwen",
        )
        self.backbone = FakeSharedBackbone()
        self.get_backbone_calls = 0
        self.closed = False

    def get_backbone(self) -> SharedVisionLanguageBackbone:
        self.get_backbone_calls += 1
        return self.backbone

    def close(self) -> None:
        self.closed = True


class FakeEmbeddedTrainAdapterRuntime(SharedBackboneTrainAdapterRuntime):
    def __init__(self, backbone_provider: SharedQwenBackboneProvider) -> None:
        super().__init__(
            runtime_name="fake-embedded-train-adapter",
            backbone_provider=backbone_provider,
        )
        self.last_request: TrainAdapterRuntimeRequest | None = None
        self.last_instruction_text: str | None = None

    def predict_embedded_locate(
        self,
        request: TrainAdapterRuntimeRequest,
    ) -> EmbeddedLocatePrediction:
        self.last_request = request
        context = self.prepare_feature_context(request)
        self.last_instruction_text = context.instruction_text
        return EmbeddedLocatePrediction(
            runtime_name=self.runtime_name,
            points=[
                EmbeddedLocatePoint(x=0.36, y=0.47, confidence=0.93, reason="peak_1"),
                EmbeddedLocatePoint(x=0.44, y=0.58, confidence=0.81, reason="peak_2"),
            ],
            coarse_box=NormalizedBox(x1=0.22, y1=0.19, x2=0.63, y2=0.78),
            bridge_hints=[
                EmbeddedLocateBridgeHint(
                    hint_type="prefer_box_plus_points",
                    reason="runtime produced both coarse box and point priors",
                )
            ],
            diagnostics=[f"session={context.session.session_id}"],
            metadata={
                "backbone": context.backbone.backbone_name,
                "device": context.backbone.device,
            },
            limitations=["fake_runtime_for_tests"],
        )


class ConfiguredFakeEmbeddedTrainAdapterRuntime(FakeEmbeddedTrainAdapterRuntime):
    def __init__(self, backbone_provider: SharedQwenBackboneProvider, *, max_length: int) -> None:
        super().__init__(backbone_provider)
        self.max_length = max_length

    def resolve_text_max_length(self, request: TrainAdapterRuntimeRequest) -> int | None:
        del request
        return self.max_length


class FailingTextSharedBackbone(FakeSharedBackbone):
    def encode_text(
        self,
        text: str,
        *,
        session: FeatureSessionHandle | None = None,
        max_length: int | None = None,
    ) -> EncodedFeatureHandle:
        self.text_calls.append((text, session.session_id if session else None))
        self.text_max_lengths.append(max_length)
        raise RuntimeError("text_encode_failed")


class FailingTextSharedQwenProvider(FakeSharedQwenProvider):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FailingTextSharedBackbone()


class DiagnosticEmptyLocatorAdapter(LocatorAdapter):
    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-empty",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.EMPTY,
            proposal_summary="embedded locator produced no candidates",
            diagnostics=["embedded_locator.empty", "reason=no_points_after_runtime_filter"],
        )


class DiagnosticFailedLocatorAdapter(LocatorAdapter):
    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-failed",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.FAILED,
            proposal_summary="embedded locator failed to initialize runtime",
            diagnostics=["embedded_locator.failed", "reason=runtime_not_ready"],
        )


class LegacyUriOnlyLocatorAdapter(LocatorAdapter):
    def __init__(self, *, backend_name: str) -> None:
        super().__init__(backend_name=backend_name)
        self.last_image_uri: str | None = None

    def locate(self, request: LocateAdapterRequest) -> ProposalResult:
        self.last_image_uri = request.image_uri
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-legacy-uri",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.READY,
            proposal_summary="legacy locator consumed image_uri only",
            candidates=[
                ProposalCandidate(
                    candidate_id="legacy-uri-candidate",
                    rank=1,
                    confidence=0.88,
                    region_box=NormalizedBox(x1=0.2, y1=0.2, x2=0.7, y2=0.8),
                )
            ],
            primary_candidate_id="legacy-uri-candidate",
            diagnostics=[f"legacy_image_uri={request.image_uri}"],
        )


def build_cli_service_with_embedded_locator(
    artifact_root: Path,
    locator_adapter: LocatorAdapter,
) -> tuple[CLIService, LocalFileArtifactStore]:
    store = LocalFileArtifactStore(str(artifact_root))
    llm_adapter = MockLLMAdapter(
        backend_name="mock-llm",
        evaluation_verdict_sequence=(EvaluationVerdict.ACCEPT,),
    )
    orchestrator = Orchestrator(
        OrchestratorDependencies(
            query_understanding_module=LLMQueryUnderstandingModule(
                llm_adapter=llm_adapter,
                artifact_store=store,
            ),
            proposal_engine_module=DefaultProposalEngineModule(
                route_handlers={
                    ProposalRoute.LOCATE: LocateProposalRouteHandler(
                        locator_adapter=locator_adapter
                    )
                },
                artifact_store=store,
            ),
            prompt_bridge_module=RuleBasedPromptBridgeModule(artifact_store=store),
            segmenter_module=SAMSegmenterModule(
                sam_adapter=MockSAMAdapter(
                    backend_name="mock-sam",
                    artifact_store=store,
                ),
                artifact_store=store,
            ),
            evaluator_module=LLMEvaluatorModule(
                llm_adapter=llm_adapter,
                artifact_store=store,
            ),
            retry_policy=RetryPolicy(),
        )
    )
    return CLIService(orchestrator=orchestrator), store


class EmbeddedLocateIntegrationTests(unittest.TestCase):
    def test_provider_boundary_is_runtime_only(self) -> None:
        provider = FakeSharedQwenProvider()

        backbone = provider.get_backbone()

        self.assertIs(backbone, provider.backbone)
        self.assertEqual(provider.get_backbone_calls, 1)
        self.assertTrue(hasattr(backbone, "encode_image"))
        self.assertTrue(hasattr(backbone, "encode_text"))
        self.assertTrue(hasattr(backbone, "open_feature_session"))
        self.assertFalse(hasattr(provider, "locate"))
        self.assertFalse(hasattr(provider, "run_query_understanding"))

    def test_locate_request_prefers_image_ref_but_keeps_image_uri_compat(self) -> None:
        understanding = make_understanding()
        request = LocateAdapterRequest(
            task_id="task-embedded",
            understanding=understanding,
            image_ref=ImageRef(uri="/tmp/from_ref.png", image_id="img-1"),
            image_uri="/tmp/from_ref.png",
        )
        legacy_only_request = LocateAdapterRequest(
            task_id="task-embedded-legacy",
            understanding=understanding,
            image_uri="/tmp/legacy_only.png",
        )

        resolved = request.resolved_image_ref()
        legacy_resolved = legacy_only_request.resolved_image_ref()

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.uri, "/tmp/from_ref.png")
        self.assertEqual(request.resolved_image_uri(), "/tmp/from_ref.png")
        self.assertIsNotNone(legacy_resolved)
        assert legacy_resolved is not None
        self.assertEqual(legacy_resolved.uri, "/tmp/legacy_only.png")
        self.assertEqual(request.image_uri, "/tmp/from_ref.png")

    def test_locate_request_rejects_conflicting_image_sources(self) -> None:
        with self.assertRaises(ValueError):
            LocateAdapterRequest(
                task_id="task-embedded-conflict",
                understanding=make_understanding(),
                image_ref=ImageRef(uri="/tmp/from_ref.png", image_id="img-1"),
                image_uri="/tmp/from_uri.png",
            )

    def test_embedded_locator_maps_runtime_output_to_proposal(self) -> None:
        provider = FakeSharedQwenProvider()
        runtime = FakeEmbeddedTrainAdapterRuntime(provider)
        adapter = EmbeddedLocatorAdapter(
            backend_name="embedded-locator",
            runtime=runtime,
        )

        proposal = adapter.locate(
            LocateAdapterRequest(
                task_id="task-proposal",
                understanding=make_understanding(),
                image_ref=ImageRef(uri="/tmp/input.png", image_id="image-1"),
            )
        )

        self.assertIs(proposal.status, ProposalStatus.READY)
        self.assertEqual(proposal.route, ProposalRoute.LOCATE)
        self.assertEqual(len(proposal.candidates), 1)
        candidate = proposal.candidates[0]
        self.assertEqual(candidate.region_box, NormalizedBox(0.22, 0.19, 0.63, 0.78))
        self.assertEqual(len(candidate.positive_point_hints), 2)
        self.assertIn("fake_runtime_for_tests", candidate.limitations)
        self.assertTrue(
            any(hint.hint_type == "prefer_box_plus_points" for hint in proposal.bridge_hints)
        )
        self.assertTrue(
            any(message.startswith("runtime_metadata=") for message in proposal.diagnostics)
        )
        self.assertIsNotNone(runtime.last_request)
        assert runtime.last_request is not None
        self.assertEqual(runtime.last_request.image_ref.uri, "/tmp/input.png")
        self.assertEqual(
            runtime.last_instruction_text,
            (
                "Given the grid coordinate system, locate the referent described as "
                "'the small red cup' in the image and predict the most likely target points. "
                "Focus on: small, red, cup."
            ),
        )

    def test_prepare_feature_context_releases_session_on_encode_failure(self) -> None:
        provider = FailingTextSharedQwenProvider()
        runtime = FakeEmbeddedTrainAdapterRuntime(provider)

        with self.assertRaisesRegex(RuntimeError, "text_encode_failed"):
            runtime.prepare_feature_context(
                TrainAdapterRuntimeRequest(
                    task_id="task-prepare-failure",
                    image_ref=ImageRef(uri="/tmp/input.png"),
                    query_text="the small red cup",
                    focus_terms=["small", "red", "cup"],
                )
            )

        self.assertEqual(len(provider.backbone.session_handles), 1)
        self.assertEqual(
            provider.backbone.released_sessions,
            [provider.backbone.session_handles[0].session_id],
        )

    def test_prepare_feature_context_passes_runtime_max_length_to_backbone(self) -> None:
        provider = FakeSharedQwenProvider()
        runtime = ConfiguredFakeEmbeddedTrainAdapterRuntime(provider, max_length=384)

        context = runtime.prepare_feature_context(
            TrainAdapterRuntimeRequest(
                task_id="task-max-length",
                image_ref=ImageRef(uri="/tmp/input.png"),
                query_text="the small red cup",
                focus_terms=["small", "red", "cup"],
            )
        )

        self.assertEqual(provider.backbone.text_max_lengths, [384])
        self.assertEqual(
            context.instruction_text,
            (
                "Given the grid coordinate system, locate the referent described as "
                "'the small red cup' in the image and predict the most likely target points. "
                "Focus on: small, red, cup."
            ),
        )
        provider.backbone.release_session(context.session)

    def test_checkpoint_loader_prefers_adapter_only_sidecar(self) -> None:
        provider = FakeSharedQwenProvider()
        runtime = EmbeddedGridGroundTrainAdapterRuntime(
            runtime_name="embedded-runtime-checkpoint-test",
            backbone_provider=provider,
            adapter_path="/tmp/best_model.pth",
            config_path="/tmp/config.json",
            runtime_config=EmbeddedGridGroundRuntimeConfig(max_length=384),
        )

        with mock.patch(
            "pathlib.Path.is_file",
            autospec=True,
            side_effect=lambda path: str(path).endswith(".adapter_only.pth"),
        ), mock.patch(
            "msagent.infra.runtime.train_adapter_runtime.EmbeddedGridGroundTrainAdapterRuntime._torch_load_checkpoint",
            return_value={"adapter.layer": "value"},
        ) as mocked_torch_load:
            result = runtime._load_adapter_state_dict(target_keys={"layer"})

        self.assertEqual(result, {"layer": "value"})
        load_path = mocked_torch_load.call_args.args[0]
        self.assertEqual(str(load_path), "/tmp/best_model.adapter_only.pth")

    def test_checkpoint_loader_caches_filtered_adapter_state(self) -> None:
        provider = FakeSharedQwenProvider()
        runtime = EmbeddedGridGroundTrainAdapterRuntime(
            runtime_name="embedded-runtime-checkpoint-test",
            backbone_provider=provider,
            adapter_path="/tmp/best_model.pth",
            config_path="/tmp/config.json",
            runtime_config=EmbeddedGridGroundRuntimeConfig(max_length=384),
        )

        with mock.patch(
            "pathlib.Path.is_file",
            autospec=True,
            return_value=False,
        ), mock.patch(
            "msagent.infra.runtime.train_adapter_runtime.EmbeddedGridGroundTrainAdapterRuntime._torch_load_checkpoint",
            return_value={"model_state_dict": {"module.adapter.layer": "value", "ignored": "noise"}},
        ) as mocked_torch_load, mock.patch(
            "msagent.infra.runtime.train_adapter_runtime.EmbeddedGridGroundTrainAdapterRuntime._torch_save_checkpoint"
        ) as mocked_torch_save:
            result = runtime._load_adapter_state_dict(target_keys={"layer"})

        self.assertEqual(result, {"layer": "value"})
        mocked_torch_load.assert_called_once()
        mocked_torch_save.assert_called_once()
        save_path, saved_state_dict = mocked_torch_save.call_args.args
        self.assertEqual(saved_state_dict, {"layer": "value"})
        self.assertEqual(str(save_path), "/tmp/best_model.adapter_only.pth")

    def test_proposal_engine_runs_through_embedded_locator(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))
            provider = FakeSharedQwenProvider()
            runtime = FakeEmbeddedTrainAdapterRuntime(provider)
            module = DefaultProposalEngineModule(
                route_handlers={
                    ProposalRoute.LOCATE: LocateProposalRouteHandler(
                        locator_adapter=EmbeddedLocatorAdapter(
                            backend_name="embedded-locator",
                            runtime=runtime,
                        )
                    )
                },
                artifact_store=store,
            )

            output = module.run(
                ProposalEngineModuleInput(
                    task_id="task-proposal-engine",
                    attempt_index=1,
                    understanding=make_understanding(),
                    image_ref=ImageRef(uri="/tmp/proposal-engine.png"),
                    preferred_route=ProposalRoute.LOCATE,
                )
            )

            self.assertIs(output.status, ModuleStatus.SUCCESS)
            self.assertIsNotNone(output.primary_payload)
            proposal = output.primary_payload
            assert proposal is not None
            self.assertIs(proposal.status, ProposalStatus.READY)
            self.assertIsNotNone(output.artifact_ref)
            assert output.artifact_ref is not None
            self.assertIs(output.artifact_ref.artifact_type, ArtifactKind.PROPOSAL_RESULT)
            self.assertEqual(provider.get_backbone_calls, 1)

    def test_proposal_engine_surfaces_payload_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))
            module = DefaultProposalEngineModule(
                route_handlers={
                    ProposalRoute.LOCATE: LocateProposalRouteHandler(
                        locator_adapter=DiagnosticEmptyLocatorAdapter(
                            backend_name="diagnostic-empty-locator"
                        )
                    )
                },
                artifact_store=store,
            )

            output = module.run(
                ProposalEngineModuleInput(
                    task_id="task-proposal-empty",
                    attempt_index=1,
                    understanding=make_understanding(),
                    image_ref=ImageRef(uri="/tmp/proposal-empty.png"),
                    preferred_route=ProposalRoute.LOCATE,
                )
            )

            self.assertIs(output.status, ModuleStatus.EMPTY)
            self.assertEqual(
                [diagnostic.message for diagnostic in output.diagnostics],
                ["embedded_locator.empty", "reason=no_points_after_runtime_filter"],
            )
            self.assertTrue(
                all(diagnostic.level == "warning" for diagnostic in output.diagnostics)
            )

    def test_legacy_locator_can_still_consume_image_uri(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))
            locator = LegacyUriOnlyLocatorAdapter(backend_name="legacy-uri-locator")
            module = DefaultProposalEngineModule(
                route_handlers={
                    ProposalRoute.LOCATE: LocateProposalRouteHandler(
                        locator_adapter=locator
                    )
                },
                artifact_store=store,
            )

            output = module.run(
                ProposalEngineModuleInput(
                    task_id="task-legacy-uri",
                    attempt_index=1,
                    understanding=make_understanding(),
                    image_ref=ImageRef(uri="/tmp/legacy-uri-input.png", image_id="legacy-image"),
                    preferred_route=ProposalRoute.LOCATE,
                )
            )

            self.assertIs(output.status, ModuleStatus.SUCCESS)
            self.assertEqual(locator.last_image_uri, "/tmp/legacy-uri-input.png")
            self.assertIsNotNone(output.primary_payload)
            proposal = output.primary_payload
            assert proposal is not None
            self.assertEqual(
                proposal.diagnostics,
                ["legacy_image_uri=/tmp/legacy-uri-input.png"],
            )

    def test_orchestrator_end_to_end_stays_closed_with_embedded_locator(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"fake-image")

            provider = FakeSharedQwenProvider()
            runtime = FakeEmbeddedTrainAdapterRuntime(provider)
            locator_adapter = EmbeddedLocatorAdapter(
                backend_name="embedded-locator",
                runtime=runtime,
            )
            cli_service, store = build_cli_service_with_embedded_locator(
                tmp_path / "artifacts",
                locator_adapter=locator_adapter,
            )

            result = cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the small red cup",
                    max_attempts=2,
                )
            )

            task = result.task
            self.assertIs(task.runtime.status, TaskStatus.SUCCEEDED)
            self.assertIs(task.runtime.stage, TaskStage.FINISHED)
            self.assertIs(task.result.stop_reason, StopReason.ACCEPTED)
            self.assertEqual(len(task.attempt_history), 1)
            attempt = task.attempt_history[0]
            self.assertIsNotNone(attempt.finished_at)
            self.assertIsNotNone(attempt.proposal_ref)
            self.assertIsNotNone(attempt.segmentation_ref)
            self.assertIsNotNone(attempt.evaluation_ref)
            self.assertGreaterEqual(len(task.artifacts.artifact_refs), 5)

            assert attempt.proposal_ref is not None
            proposal = store.load_artifact(attempt.proposal_ref, ProposalResult)
            self.assertEqual(proposal.primary_candidate_id, f"{task.identity.task_id}-embedded-candidate-1")
            self.assertEqual(provider.get_backbone_calls, 1)
            self.assertEqual(len(provider.backbone.image_calls), 1)
            self.assertEqual(len(provider.backbone.text_calls), 1)

    def test_orchestrator_records_proposal_diagnostics_on_empty(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"fake-image")

            cli_service, _ = build_cli_service_with_embedded_locator(
                tmp_path / "artifacts",
                locator_adapter=DiagnosticEmptyLocatorAdapter(
                    backend_name="diagnostic-empty-locator"
                ),
            )

            result = cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the small red cup",
                    max_attempts=1,
                )
            )

            task = result.task
            self.assertIs(task.runtime.status, TaskStatus.FAILED)
            self.assertIs(task.runtime.stage, TaskStage.FINISHED)
            self.assertIs(task.result.stop_reason, StopReason.EMPTY_PROPOSAL)
            self.assertEqual(
                task.result.failure_summary,
                "embedded_locator.empty; reason=no_points_after_runtime_filter",
            )
            self.assertIn(
                "embedded_locator.empty; reason=no_points_after_runtime_filter",
                task.attempt_history[0].notes,
            )

    def test_orchestrator_records_proposal_diagnostics_on_failed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"fake-image")

            cli_service, _ = build_cli_service_with_embedded_locator(
                tmp_path / "artifacts",
                locator_adapter=DiagnosticFailedLocatorAdapter(
                    backend_name="diagnostic-failed-locator"
                ),
            )

            result = cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the small red cup",
                    max_attempts=1,
                )
            )

            task = result.task
            self.assertIs(task.runtime.status, TaskStatus.FAILED)
            self.assertIs(task.runtime.stage, TaskStage.FINISHED)
            self.assertIs(task.result.stop_reason, StopReason.EMPTY_PROPOSAL)
            self.assertEqual(
                task.result.failure_summary,
                "embedded_locator.failed; reason=runtime_not_ready",
            )
            self.assertIn(
                "embedded_locator.failed; reason=runtime_not_ready",
                task.attempt_history[0].notes,
            )


if __name__ == "__main__":
    unittest.main()
