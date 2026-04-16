from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
HAS_HTTPX = importlib.util.find_spec("httpx") is not None

if HAS_FASTAPI and HAS_HTTPX:
    from fastapi.testclient import TestClient

from msagent.core.config.settings import MSAgentSettings
from msagent.core.contracts.types import ProposalResult, ProposalRoute, ProposalStatus
from msagent.service import build_default_api_service
from tests.support.deterministic_adapters import (
    DeterministicLocatorAdapter,
    make_deterministic_service_adapters,
)


class FakeEmbeddedLocatorRuntimeBundle:
    def __init__(self) -> None:
        self.locator_adapter = DeterministicLocatorAdapter(backend_name="embedded-locator")
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class EmptyLocatorAdapter(DeterministicLocatorAdapter):
    def locate(self, request):
        return ProposalResult(
            proposal_id=f"{request.task_id}-proposal-empty",
            route=ProposalRoute.LOCATE,
            status=ProposalStatus.EMPTY,
            proposal_summary="no candidate produced",
            candidates=[],
            primary_candidate_id=None,
        )


@unittest.skipUnless(
    HAS_FASTAPI and HAS_HTTPX,
    "fastapi and httpx are required for real API integration tests.",
)
class RealFastAPIIntegrationTests(unittest.TestCase):
    def test_real_fastapi_health_success_and_shutdown(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            settings.model_paths.qwen_model_path = "/models/qwen"
            settings.model_paths.embedded_locator_adapter_path = "/models/locator.ckpt"
            settings.model_paths.embedded_locator_config_path = "/models/runtime.json"

            fake_bundle = FakeEmbeddedLocatorRuntimeBundle()
            _, llm_adapter, sam_adapter = make_deterministic_service_adapters(
                tmp_path / "artifacts"
            )
            with patch(
                "msagent.service.assembly.build_embedded_locator_runtime_bundle",
                return_value=fake_bundle,
            ):
                assembly = build_default_api_service(
                    settings,
                    llm_adapter=llm_adapter,
                    sam_adapter=sam_adapter,
                )
                try:
                    app = assembly.create_app()
                    with TestClient(app) as client:
                        health_response = client.get("/health")
                        self.assertEqual(health_response.status_code, 200)
                        self.assertEqual(health_response.json(), {"status": "ok"})

                        run_response = client.post(
                            "/tasks/run",
                            json={
                                "image_uri": "file:///tmp/input.png",
                                "query_text": "the red cup",
                            },
                        )
                        self.assertEqual(run_response.status_code, 200)
                        self.assertEqual(
                            run_response.json()["summary"],
                            "Task completed successfully.",
                        )
                finally:
                    assembly.close()

        self.assertGreaterEqual(fake_bundle.close_calls, 1)

    def test_real_fastapi_serves_visualizer_and_local_image_preview(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            settings.service.enable_debug_features = True
            locator_adapter, llm_adapter, sam_adapter = make_deterministic_service_adapters(
                tmp_path / "artifacts"
            )

            assembly = build_default_api_service(
                settings,
                locator_adapter=locator_adapter,
                llm_adapter=llm_adapter,
                sam_adapter=sam_adapter,
            )
            try:
                app = assembly.create_app()
                with TestClient(app) as client:
                    visualizer_response = client.get("/")
                    self.assertEqual(visualizer_response.status_code, 200)
                    self.assertIn("M-SAgent Visualizer", visualizer_response.text)

                    image_path = tmp_path / "preview.png"
                    image_path.write_bytes(
                        base64.b64decode(
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
                            "/w8AAn8B9Wn0jQAAAABJRU5ErkJggg=="
                        )
                    )
                    preview_response = client.get(
                        "/debug/local-file",
                        params={"path": str(image_path)},
                    )
                    self.assertEqual(preview_response.status_code, 200)
                    self.assertEqual(preview_response.headers["content-type"], "image/png")

                    text_path = tmp_path / "note.txt"
                    text_path.write_text("not an image", encoding="utf-8")
                    bad_preview_response = client.get(
                        "/debug/local-file",
                        params={"path": str(text_path)},
                    )
                    self.assertEqual(bad_preview_response.status_code, 400)
            finally:
                assembly.close()

    def test_real_fastapi_hides_debug_routes_when_debug_features_are_disabled(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            locator_adapter, llm_adapter, sam_adapter = make_deterministic_service_adapters(
                tmp_path / "artifacts"
            )

            assembly = build_default_api_service(
                settings,
                locator_adapter=locator_adapter,
                llm_adapter=llm_adapter,
                sam_adapter=sam_adapter,
            )
            try:
                app = assembly.create_app()
                with TestClient(app) as client:
                    self.assertEqual(client.get("/").status_code, 404)
                    self.assertEqual(client.get("/debug/local-file").status_code, 404)
            finally:
                assembly.close()

    def test_real_fastapi_failure_and_bad_request_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            _, llm_adapter, sam_adapter = make_deterministic_service_adapters(
                tmp_path / "artifacts"
            )

            with patch(
                "msagent.service.assembly._build_default_locator_adapter",
                return_value=(
                    EmptyLocatorAdapter(backend_name="empty-locator"),
                    None,
                    ["embedded_locator_runtime=custom"],
                ),
            ):
                assembly = build_default_api_service(
                    settings,
                    llm_adapter=llm_adapter,
                    sam_adapter=sam_adapter,
                )
                try:
                    app = assembly.create_app()
                    with TestClient(app) as client:
                        failed_response = client.post(
                            "/tasks/run",
                            json={
                                "image_uri": "file:///tmp/input.png",
                                "query_text": "the red cup",
                            },
                        )
                        self.assertEqual(failed_response.status_code, 200)
                        self.assertEqual(
                            failed_response.json()["summary"],
                            "Task could not produce a usable result.",
                        )

                        bad_request_response = client.post(
                            "/tasks/run",
                            json={
                                "image_uri": "file:///tmp/input.png",
                                "query_text": "the red cup",
                                "request_metadata": ["bad"],
                            },
                        )
                        self.assertEqual(bad_request_response.status_code, 400)
                finally:
                    assembly.close()


if __name__ == "__main__":
    unittest.main()
