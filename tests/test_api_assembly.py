from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.config.settings import MSAgentSettings
from msagent.core.contracts.types import ProposalResult, ProposalRoute, ProposalStatus
from msagent.infra.adapters import LocatorAdapter
from msagent.infra.mock_adapters import MockLocatorAdapter
from msagent.service import build_default_api_service
from msagent.service.api import APIResponse


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


class EmptyLocatorAdapter(MockLocatorAdapter):
    def locate(self, request):
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-empty",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.EMPTY,
            proposal_summary="no candidate produced",
            candidates=[],
            primary_candidate_id=None,
        )


class FailingLocatorAdapter(LocatorAdapter):
    def locate(self, request):
        raise RuntimeError("provider=embedded-locator session=session-123 crashed")


class FakeFastAPIApp:
    def __init__(self, title: str, lifespan=None) -> None:
        self.title = title
        self.lifespan = lifespan

    def get(self, path: str):
        def register(func):
            return func

        return register

    def post(self, path: str):
        def register(func):
            return func

        return register

    async def run_lifespan(self) -> None:
        if self.lifespan is None:
            return
        async with self.lifespan(self):
            pass


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class APIAssemblyTests(unittest.TestCase):
    def test_default_api_service_uses_embedded_locator_when_runtime_is_fully_configured(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            settings.service.host = "0.0.0.0"
            settings.service.port = 9010
            settings.model_paths.qwen_model_path = "/models/qwen"
            settings.model_paths.embedded_locator_adapter_path = "/models/locator.ckpt"
            settings.model_paths.embedded_locator_config_path = "/models/runtime.json"

            fake_bundle = FakeEmbeddedLocatorRuntimeBundle()
            with patch(
                "msagent.service.assembly.build_embedded_locator_runtime_bundle",
                return_value=fake_bundle,
            ) as build_bundle:
                assembly = build_default_api_service(settings)
                try:
                    response = assembly.handle(
                        {
                            "image_uri": "file:///tmp/input.png",
                            "query_text": "the red cup",
                        }
                    )
                finally:
                    assembly.close()

        build_bundle.assert_called_once()
        self.assertIsInstance(response, APIResponse)
        self.assertEqual(response.status, "succeeded")
        self.assertEqual(response.summary, "Task completed successfully.")
        self.assertEqual(assembly.host, "0.0.0.0")
        self.assertEqual(assembly.port, 9010)
        self.assertEqual(assembly.diagnostics, ["embedded_locator_runtime=enabled"])
        self.assertEqual(fake_bundle.locator_adapter.locate_calls, 1)
        self.assertTrue(fake_bundle.closed)
        self.assertFalse(hasattr(assembly, "runtime_bundle"))
        self.assertFalse(hasattr(assembly, "locator_adapter"))

    def test_default_api_service_rejects_when_api_is_disabled(self) -> None:
        settings = MSAgentSettings()
        settings.service.enable_api = False

        with self.assertRaisesRegex(ValueError, "enable_api"):
            build_default_api_service(settings)

    def test_default_api_service_rejects_partial_embedded_runtime_configuration(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")
            settings.model_paths.qwen_model_path = "/models/qwen"

            with self.assertRaisesRegex(ValueError, "partial"):
                build_default_api_service(settings)

    def test_default_api_service_rejects_non_locate_default_route(self) -> None:
        settings = MSAgentSettings()
        settings.runtime.default_route = ProposalRoute.REWRITE

        with self.assertRaisesRegex(ValueError, "default_route"):
            build_default_api_service(settings)

    def test_api_handler_rejects_invalid_transport_payload(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")

            assembly = build_default_api_service(settings)
            try:
                with self.assertRaisesRegex(ValueError, "request_metadata"):
                    assembly.handle(
                        {
                            "image_uri": "file:///tmp/input.png",
                            "query_text": "the red cup",
                            "request_metadata": ["not", "an", "object"],
                        }
                    )
            finally:
                assembly.close()

    def test_api_handler_success_path_returns_api_response(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")

            assembly = build_default_api_service(settings)
            try:
                response = assembly.handle(
                    {
                        "image_uri": "file:///tmp/input.png",
                        "query_text": "the red cup",
                    }
                )
            finally:
                assembly.close()

        self.assertIsInstance(response, APIResponse)
        self.assertEqual(response.status, "succeeded")
        self.assertEqual(response.summary, "Task completed successfully.")

    def test_api_handler_failure_path_returns_api_response(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")
            with patch(
                "msagent.service.assembly._build_default_locator_adapter",
                return_value=(
                    EmptyLocatorAdapter(backend_name="empty-locator"),
                    None,
                    ["embedded_locator_runtime=disabled"],
                ),
            ):
                assembly = build_default_api_service(settings)
                try:
                    response = assembly.handle(
                        {
                            "image_uri": "file:///tmp/input.png",
                            "query_text": "the red cup",
                        }
                    )
                finally:
                    assembly.close()

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.summary, "Task could not produce a usable result.")
        self.assertEqual(response.result_refs, [])

    def test_api_handler_exception_path_returns_api_response(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")
            with patch(
                "msagent.service.assembly._build_default_locator_adapter",
                return_value=(
                    FailingLocatorAdapter(backend_name="failing-locator"),
                    None,
                    ["embedded_locator_runtime=disabled"],
                ),
            ):
                assembly = build_default_api_service(settings)
                try:
                    response = assembly.handle(
                        {
                            "image_uri": "file:///tmp/input.png",
                            "query_text": "the red cup",
                        }
                    )
                finally:
                    assembly.close()

        self.assertEqual(response.status, "failed")
        self.assertEqual(response.summary, "Task failed due to an internal error.")
        self.assertEqual(response.result_refs, [])
        self.assertNotIn("session=session-123", response.summary)
        self.assertNotIn("embedded-locator", response.summary)

    def test_api_assembly_create_app_requires_fastapi_dependency(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(Path(tmp_dir) / "artifacts")

            assembly = build_default_api_service(settings)
            try:
                with patch.dict(sys.modules, {"fastapi": None}):
                    with self.assertRaisesRegex(RuntimeError, "FastAPI"):
                        assembly.create_app()
            finally:
                assembly.close()

    def test_api_assembly_create_app_registers_shutdown_close_hook(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            settings.model_paths.qwen_model_path = "/models/qwen"
            settings.model_paths.embedded_locator_adapter_path = "/models/locator.ckpt"
            settings.model_paths.embedded_locator_config_path = "/models/runtime.json"

            fake_bundle = FakeEmbeddedLocatorRuntimeBundle()
            fake_fastapi = ModuleType("fastapi")
            fake_fastapi.FastAPI = FakeFastAPIApp
            fake_fastapi.HTTPException = FakeHTTPException

            with patch(
                "msagent.service.assembly.build_embedded_locator_runtime_bundle",
                return_value=fake_bundle,
            ):
                with patch.dict(sys.modules, {"fastapi": fake_fastapi}):
                    assembly = build_default_api_service(settings)
                    try:
                        app = assembly.create_app()
                        self.assertFalse(fake_bundle.closed)
                        import asyncio

                        asyncio.run(app.run_lifespan())
                    finally:
                        assembly.close()

        self.assertTrue(fake_bundle.closed)


if __name__ == "__main__":
    unittest.main()
