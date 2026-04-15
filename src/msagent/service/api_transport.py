"""API transport adapter.

本模块负责把 Web/HTTP 层 payload 映射为 APIRequest / APIResponse，
不直接接触 RunTask 账本或 orchestrator 内部对象。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass

from msagent.service.api import APIRequest, APIResponse, APIService


@dataclass(slots=True)
class APIHandler:
    """薄 API transport handler。"""

    service: APIService

    def build_request(self, payload: Mapping[str, object]) -> APIRequest:
        """把 transport payload 转换为 APIRequest。"""
        image_uri = self._require_str(payload, "image_uri")
        query_text = self._require_str(payload, "query_text")

        max_attempts_object = payload.get("max_attempts", 3)
        if not isinstance(max_attempts_object, int) or isinstance(max_attempts_object, bool):
            raise ValueError("Field 'max_attempts' must be an integer.")

        request_metadata_object = payload.get("request_metadata", {})
        if not isinstance(request_metadata_object, dict):
            raise ValueError("Field 'request_metadata' must be an object.")

        return APIRequest(
            image_uri=image_uri,
            query_text=query_text,
            max_attempts=max_attempts_object,
            request_metadata=dict(request_metadata_object),
        )

    def handle_run(self, payload: Mapping[str, object]) -> APIResponse:
        """执行一次标准 API 入口调用。"""
        request = self.build_request(payload)
        result = self.service.run(request)
        return self.service.to_response(result)

    @staticmethod
    def _require_str(payload: Mapping[str, object], field_name: str) -> str:
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{field_name}' must be a non-empty string.")
        return value


def create_fastapi_app(
    handler: APIHandler,
    *,
    on_shutdown: Callable[[], None] | None = None,
) -> object:
    """创建 FastAPI app。

    这里把 FastAPI 视为外层 transport adapter 依赖；核心 service 层不依赖它。
    """

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised in environments without fastapi
        raise RuntimeError(
            "FastAPI is not installed; install fastapi to create the API app."
        ) from exc

    if on_shutdown is None:
        app = FastAPI(title="M-SAgent API")
    else:
        @asynccontextmanager
        async def lifespan(_: object):
            try:
                yield
            finally:
                on_shutdown()

        app = FastAPI(title="M-SAgent API", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tasks/run")
    def run_task(payload: dict[str, object]) -> dict[str, object]:
        try:
            return asdict(handler.handle_run(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
