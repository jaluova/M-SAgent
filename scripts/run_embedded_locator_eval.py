#!/usr/bin/env python3
"""独立 embedded locator 本地评测入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.evals.embedded_locator import (  # noqa: E402
    EmbeddedLocatorEvaluationHarness,
    EmbeddedLocatorParameterGroup,
    load_embedded_locator_manifest,
    load_parameter_groups_payload,
    write_embedded_locator_evaluation_report,
)
from msagent.infra.runtime.factory import (  # noqa: E402
    EmbeddedLocatorRuntimeFactoryConfig,
    build_embedded_locator_runtime_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated embedded locator evaluation without touching API/CLI."
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "embedded_locator_eval_manifest.json"),
        help="Path to the weak-label manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "embedded_locator_eval"),
        help="Directory for the structured evaluation report and local artifacts.",
    )
    parser.add_argument(
        "--qwen-model-path",
        default=os.environ.get("M_SAGENT_QWEN_MODEL_PATH"),
        help="Qwen model path. Defaults to M_SAGENT_QWEN_MODEL_PATH.",
    )
    parser.add_argument(
        "--gridground-config-path",
        default=os.environ.get("GRIDGROUND_CONFIG_PATH"),
        help="GridGround config path. Defaults to GRIDGROUND_CONFIG_PATH.",
    )
    parser.add_argument(
        "--gridground-adapter-path",
        default=os.environ.get("GRIDGROUND_ADAPTER_PATH"),
        help="GridGround adapter checkpoint path. Defaults to GRIDGROUND_ADAPTER_PATH.",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("MSAGENT_QWEN_DEVICE_MAP", "auto"),
        help="Backbone device map.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=os.environ.get("MSAGENT_QWEN_TORCH_DTYPE", "auto"),
        help="Backbone torch dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("MSAGENT_QWEN_ATTN_IMPL"),
        help="Optional attention implementation override.",
    )
    parser.add_argument(
        "--options-json",
        default=None,
        help="Inline JSON object or JSON file path for a single parameter group.",
    )
    parser.add_argument(
        "--sweep-json",
        default=None,
        help="Inline JSON list or JSON file path for a parameter sweep.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_embedded_locator_manifest(args.manifest)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    qwen_model_path = _require_path(args.qwen_model_path, "qwen-model-path")
    config_path = _require_path(args.gridground_config_path, "gridground-config-path")
    adapter_path = _require_path(args.gridground_adapter_path, "gridground-adapter-path")

    if args.sweep_json:
        parameter_groups = load_parameter_groups_payload(_load_json_value(args.sweep_json))
    else:
        single_options = _load_json_value(args.options_json) if args.options_json else {}
        if not isinstance(single_options, dict):
            raise ValueError("--options-json must decode to a JSON object")
        parameter_groups = [
            EmbeddedLocatorParameterGroup(
                label="default",
                runtime_options=dict(single_options),
            )
        ]

    bundle = build_embedded_locator_runtime_bundle(
        EmbeddedLocatorRuntimeFactoryConfig(
            qwen_model_path=qwen_model_path,
            adapter_path=adapter_path,
            config_path=config_path,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            runtime_name="embedded-gridground-eval-runtime",
            locator_backend_name="embedded-locator-eval",
        )
    )
    try:
        harness = EmbeddedLocatorEvaluationHarness.from_locator_adapter(
            locator_adapter=bundle.locator_adapter,
            artifact_root=output_dir / "store",
        )
        report = harness.evaluate(
            manifest=manifest,
            parameter_groups=parameter_groups,
        )
    finally:
        bundle.close()

    report_path = write_embedded_locator_evaluation_report(
        report,
        output_dir / "embedded_locator_eval_report.json",
    )
    print(report_path)
    return 0


def _load_json_value(raw_value: str) -> object:
    candidate_path = Path(raw_value).expanduser()
    if candidate_path.is_file():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    return json.loads(raw_value)


def _require_path(value: str | None, field_name: str) -> str:
    if not value:
        raise ValueError(f"Missing required path for --{field_name}")
    return str(Path(value).expanduser())


if __name__ == "__main__":
    raise SystemExit(main())
