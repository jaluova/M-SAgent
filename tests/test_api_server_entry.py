from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.config.settings import MSAgentSettings
from msagent.service.api_server import run_default_api_server


class FakeAssembly:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.host = host
        self.port = port
        self.app = object()
        self.closed = 0
        self.create_app_calls = 0

    def create_app(self) -> object:
        self.create_app_calls += 1
        return self.app

    def close(self) -> None:
        self.closed += 1


class APIServerEntryTests(unittest.TestCase):
    def test_run_default_api_server_uses_assembly_host_and_port(self) -> None:
        settings = MSAgentSettings()
        settings.service.host = "0.0.0.0"
        settings.service.port = 9010

        fake_assembly = FakeAssembly(host="0.0.0.0", port=9010)
        uvicorn_module = ModuleType("uvicorn")
        uvicorn_module.run = lambda *args, **kwargs: None

        with patch(
            "msagent.service.api_server.build_default_api_service",
            return_value=fake_assembly,
        ) as build_assembly:
            with patch.dict(sys.modules, {"uvicorn": uvicorn_module}):
                with patch.object(uvicorn_module, "run") as run_server:
                    run_default_api_server(settings)

        build_assembly.assert_called_once_with(settings)
        run_server.assert_called_once_with(
            fake_assembly.app,
            host="0.0.0.0",
            port=9010,
        )
        self.assertEqual(fake_assembly.create_app_calls, 1)
        self.assertEqual(fake_assembly.closed, 1)

    def test_run_default_api_server_closes_assembly_when_server_fails(self) -> None:
        settings = MSAgentSettings()
        fake_assembly = FakeAssembly()
        uvicorn_module = ModuleType("uvicorn")
        uvicorn_module.run = lambda *args, **kwargs: None

        with patch(
            "msagent.service.api_server.build_default_api_service",
            return_value=fake_assembly,
        ):
            with patch.dict(sys.modules, {"uvicorn": uvicorn_module}):
                with patch.object(
                    uvicorn_module,
                    "run",
                    side_effect=RuntimeError("server boom"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "server boom"):
                        run_default_api_server(settings)

        self.assertEqual(fake_assembly.closed, 1)

    def test_run_default_api_server_rejects_when_api_is_disabled(self) -> None:
        settings = MSAgentSettings()
        settings.service.enable_api = False

        with self.assertRaisesRegex(ValueError, "enable_api"):
            run_default_api_server(settings)

    def test_run_default_api_server_requires_uvicorn_dependency(self) -> None:
        settings = MSAgentSettings()

        with patch.dict(sys.modules, {"uvicorn": None}):
            with self.assertRaisesRegex(RuntimeError, "uvicorn"):
                run_default_api_server(settings)


if __name__ == "__main__":
    unittest.main()
