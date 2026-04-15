from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.config.settings import MSAgentSettings
from msagent.core.contracts.types import ProposalRoute
from msagent.core.task.enums import TaskStatus
from msagent.infra.mock_adapters import MockLocatorAdapter
from msagent.service import build_default_cli_service
from msagent.service.cli import CLIRequest


class RecordingEmbeddedLocatorAdapter(MockLocatorAdapter):
    def __init__(self) -> None:
        super().__init__(backend_name="embedded-locator")
        self.locate_calls = 0

    def locate(self, request):
        self.locate_calls += 1
        return super().locate(request)


class FakeEmbeddedLocatorRuntimeBundle:
    def __init__(self) -> None:
        self.locator_adapter = RecordingEmbeddedLocatorAdapter()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ServiceAssemblyTests(unittest.TestCase):
    def test_default_cli_service_uses_embedded_locator_when_runtime_is_fully_configured(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"mock-image")

            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            settings.model_paths.qwen_model_path = "/models/qwen"
            settings.model_paths.embedded_locator_adapter_path = "/models/locator.ckpt"
            settings.model_paths.embedded_locator_config_path = "/models/runtime.json"

            fake_bundle = FakeEmbeddedLocatorRuntimeBundle()
            with patch(
                "msagent.service.assembly.build_embedded_locator_runtime_bundle",
                return_value=fake_bundle,
            ) as build_bundle:
                assembly = build_default_cli_service(settings)
                try:
                    result = assembly.service.run(
                        CLIRequest(
                            image_path=str(image_path),
                            query_text="the red cup",
                        )
                    )
                finally:
                    assembly.close()

        build_bundle.assert_called_once()
        self.assertIs(assembly.locator_adapter, fake_bundle.locator_adapter)
        self.assertIs(assembly.runtime_bundle, fake_bundle)
        self.assertEqual(assembly.diagnostics, ["embedded_locator_runtime=enabled"])
        self.assertEqual(fake_bundle.locator_adapter.locate_calls, 1)
        self.assertTrue(fake_bundle.closed)
        self.assertIs(result.task.runtime.status, TaskStatus.SUCCEEDED)

    def test_default_cli_service_falls_back_to_mock_locator_when_runtime_is_unconfigured(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")

            assembly = build_default_cli_service(settings)
            try:
                self.assertIsInstance(assembly.locator_adapter, MockLocatorAdapter)
                self.assertIsNone(assembly.runtime_bundle)
                self.assertEqual(assembly.diagnostics, ["embedded_locator_runtime=disabled"])
            finally:
                assembly.close()

    def test_default_cli_service_rejects_partial_embedded_runtime_configuration(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")
            settings.model_paths.qwen_model_path = "/models/qwen"

            with self.assertRaisesRegex(ValueError, "partial"):
                build_default_cli_service(settings)

    def test_cli_service_assembly_run_closes_runtime_bundle_on_success(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"mock-image")

            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            settings.model_paths.qwen_model_path = "/models/qwen"
            settings.model_paths.embedded_locator_adapter_path = "/models/locator.ckpt"
            settings.model_paths.embedded_locator_config_path = "/models/runtime.json"

            fake_bundle = FakeEmbeddedLocatorRuntimeBundle()
            with patch(
                "msagent.service.assembly.build_embedded_locator_runtime_bundle",
                return_value=fake_bundle,
            ):
                assembly = build_default_cli_service(settings)
                result = assembly.run(
                    CLIRequest(
                        image_path=str(image_path),
                        query_text="the red cup",
                    )
                )

        self.assertTrue(fake_bundle.closed)
        self.assertIs(result.task.runtime.status, TaskStatus.SUCCEEDED)

    def test_cli_service_assembly_run_closes_runtime_bundle_on_failure(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")
            settings.model_paths.qwen_model_path = "/models/qwen"
            settings.model_paths.embedded_locator_adapter_path = "/models/locator.ckpt"
            settings.model_paths.embedded_locator_config_path = "/models/runtime.json"

            fake_bundle = FakeEmbeddedLocatorRuntimeBundle()
            with patch(
                "msagent.service.assembly.build_embedded_locator_runtime_bundle",
                return_value=fake_bundle,
            ):
                assembly = build_default_cli_service(settings)
                with patch.object(
                    assembly.service,
                    "run",
                    side_effect=RuntimeError("boom"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        assembly.run(
                            CLIRequest(
                                image_path=str(Path(tmp_dir) / "input.png"),
                                query_text="the red cup",
                            )
                        )

        self.assertTrue(fake_bundle.closed)

    def test_default_cli_service_propagates_supported_default_route_to_retry_policy(
        self,
    ) -> None:
        settings = MSAgentSettings()
        settings.runtime.default_route = ProposalRoute.LOCATE

        assembly = build_default_cli_service(settings)
        try:
            retry_policy = assembly.service.orchestrator.dependencies.retry_policy
            self.assertIs(retry_policy.default_route, ProposalRoute.LOCATE)
        finally:
            assembly.close()

    def test_default_cli_service_rejects_non_locate_default_route(self) -> None:
        settings = MSAgentSettings()
        settings.runtime.default_route = ProposalRoute.REWRITE

        with self.assertRaisesRegex(ValueError, "default_route"):
            build_default_cli_service(settings)
