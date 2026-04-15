"""对外暴露的薄服务入口。"""

from msagent.service.assembly import (
    APIServiceAssembly,
    CLIServiceAssembly,
    build_default_api_service,
    build_default_cli_service,
)
from msagent.service.api_server import run_default_api_server

__all__ = [
    "APIServiceAssembly",
    "CLIServiceAssembly",
    "build_default_api_service",
    "build_default_cli_service",
    "run_default_api_server",
]
