"""最小 API 启动入口。"""

from __future__ import annotations

from msagent.core.config.settings import MSAgentSettings
from msagent.service.assembly import build_default_api_service


def run_default_api_server(settings: MSAgentSettings | None = None) -> None:
    """使用默认组合根启动 API 服务。"""
    resolved_settings = settings or MSAgentSettings.from_env()
    if not resolved_settings.service.enable_api:
        raise ValueError("run_default_api_server requires service.enable_api=True.")

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is not installed; install uvicorn to run the API server."
        ) from exc

    assembly = build_default_api_service(resolved_settings)
    try:
        app = assembly.create_app()
        uvicorn.run(
            app,
            host=assembly.host,
            port=assembly.port,
        )
    finally:
        assembly.close()


def main() -> int:
    """模块启动入口。"""
    run_default_api_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
