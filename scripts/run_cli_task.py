#!/usr/bin/env python3
"""运行一条本地 CLI 任务，可按环境变量或命令行切换 real Qwen。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.config.settings import MSAgentSettings  # noqa: E402
from msagent.service import build_default_cli_service  # noqa: E402
from msagent.service.cli import CLIRequest  # noqa: E402
from msagent.service.task_visuals import render_task_visuals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one local M-SAgent CLI task. "
            "Configure M_SAGENT_QWEN_MODEL_PATH, GRIDGROUND_* and SAM3 paths "
            "before running the default real pipeline."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language query text.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Override runtime.max_attempts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for task_report.md.",
    )
    parser.add_argument(
        "--artifact-root",
        default=None,
        help="Optional artifact root override.",
    )
    parser.add_argument(
        "--render-visuals",
        action="store_true",
        help="Render proposal/prompt/mask overlays into output_dir after the run.",
    )
    parser.add_argument(
        "--enable-real-llm",
        action="store_true",
        help="Force real Qwen LLM adapter on for this run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = MSAgentSettings.from_env()

    if args.artifact_root is not None:
        settings.runtime.artifact_root = args.artifact_root
    if args.max_attempts is not None:
        settings.runtime.max_attempts = args.max_attempts
    if args.enable_real_llm:
        settings.service.enable_real_llm = True
    if args.render_visuals and args.output_dir is None:
        raise ValueError("--render-visuals requires --output-dir.")

    assembly = build_default_cli_service(settings)
    try:
        result = assembly.run(
            CLIRequest(
                image_path=args.image,
                query_text=args.query,
                max_attempts=args.max_attempts or settings.runtime.max_attempts,
                output_dir=args.output_dir,
            )
        )
        summary = {
            "task_id": result.task.identity.task_id,
            "task_status": result.task.runtime.status.value,
            "final_verdict": (
                result.task.result.final_verdict.value
                if result.task.result.final_verdict is not None
                else None
            ),
            "stop_reason": (
                result.task.result.stop_reason.value
                if result.task.result.stop_reason is not None
                else None
            ),
            "llm_backend": assembly.llm_adapter.backend_name,
            "locator_backend": assembly.locator_adapter.backend_name,
            "sam_backend": assembly.sam_adapter.backend_name,
            "diagnostics": assembly.diagnostics,
            "artifact_root": settings.runtime.artifact_root,
        }
        if args.render_visuals:
            visual_paths = render_task_visuals(
                result.task,
                artifact_store=assembly.artifact_store,
                output_dir=args.output_dir,
            )
            summary["visual_outputs"] = [str(path) for path in visual_paths]
        if args.output_dir is not None:
            summary["report_path"] = str(
                Path(args.output_dir).expanduser().resolve() / "task_report.md"
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        assembly.close()


if __name__ == "__main__":
    raise SystemExit(main())
