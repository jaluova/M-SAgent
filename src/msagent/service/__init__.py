"""对外暴露的薄服务入口。"""

from msagent.service.assembly import (
    APIServiceAssembly,
    CLIServiceAssembly,
    build_default_api_service,
    build_default_cli_service,
)
from msagent.service.api_server import run_default_api_server
from msagent.service.demo_report import (
    DemoAttemptSummary,
    DemoTaskReport,
    build_demo_task_report,
    render_demo_task_report_markdown,
)

__all__ = [
    "APIServiceAssembly",
    "CLIServiceAssembly",
    "DemoAttemptSummary",
    "DemoTaskReport",
    "build_default_api_service",
    "build_default_cli_service",
    "build_demo_task_report",
    "render_demo_task_report_markdown",
    "run_default_api_server",
]
