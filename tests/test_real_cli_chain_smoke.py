from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.config.settings import MSAgentSettings
from msagent.core.contracts.types import EvaluationVerdict, SegmentationResult, SegmentationStatus
from msagent.core.task.enums import TaskStatus
from msagent.infra.embedded_locator import EmbeddedLocatorAdapter
from msagent.infra.mock_adapters import MockLLMAdapter
from msagent.infra.mock_artifacts import MockMask
from msagent.infra.sam3_adapter import RealSAM3Adapter
from msagent.service import build_default_cli_service
from msagent.service.cli import CLIRequest


@unittest.skipUnless(
    os.environ.get("MSAGENT_ENABLE_REAL_CLI_SMOKE") == "1",
    "real CLI chain smoke test is opt-in",
)
class RealCLIChainSmokeTests(unittest.TestCase):
    def test_default_cli_service_runs_real_locator_and_real_sam_and_writes_task_report(self) -> None:
        qwen_model_path = self._require_env("M_SAGENT_QWEN_MODEL_PATH")
        gridground_config_path = self._require_env("GRIDGROUND_CONFIG_PATH")
        gridground_adapter_path = self._require_env("GRIDGROUND_ADAPTER_PATH")
        sam_model_path = self._require_env("M_SAGENT_SAM3_MODEL_PATH")
        sam_checkpoint_path = self._require_env("M_SAGENT_SAM3_CHECKPOINT_PATH")
        image_path = Path(
            os.environ.get(
                "MSAGENT_REAL_CLI_IMAGE",
                os.environ.get(
                    "MSAGENT_REAL_E2E_IMAGE",
                    os.environ.get(
                        "MSAGENT_REAL_SMOKE_IMAGE",
                        str(ROOT / "old/example/truck.jpg"),
                    ),
                ),
            )
        ).expanduser()
        query_text = os.environ.get(
            "MSAGENT_REAL_CLI_QUERY",
            os.environ.get(
                "MSAGENT_REAL_E2E_QUERY",
                os.environ.get("MSAGENT_REAL_SMOKE_QUERY", "truck"),
            ),
        )

        if not image_path.is_file():
            self.fail(f"real CLI smoke image does not exist: {image_path}")

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifact_root = tmp_path / "artifacts"
            output_dir = tmp_path / "output"

            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(artifact_root)
            settings.model_paths.qwen_model_path = qwen_model_path
            settings.model_paths.embedded_locator_config_path = gridground_config_path
            settings.model_paths.embedded_locator_adapter_path = gridground_adapter_path
            settings.model_paths.sam_model_path = sam_model_path
            settings.model_paths.sam_checkpoint_path = sam_checkpoint_path

            assembly = build_default_cli_service(
                settings,
                llm_adapter=MockLLMAdapter(
                    backend_name="mock-llm-real-cli-smoke",
                    evaluation_verdict_sequence=(EvaluationVerdict.ACCEPT,),
                ),
            )

            self.assertIsInstance(assembly.locator_adapter, EmbeddedLocatorAdapter)
            self.assertIsInstance(assembly.sam_adapter, RealSAM3Adapter)

            result = assembly.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text=query_text,
                    max_attempts=1,
                    output_dir=str(output_dir),
                )
            )

            self.assertEqual(
                assembly.diagnostics,
                ["embedded_locator_runtime=enabled", "sam_runtime=enabled"],
            )
            self.assertIs(result.task.runtime.status, TaskStatus.SUCCEEDED)
            self.assertIs(result.task.result.final_verdict, EvaluationVerdict.ACCEPT)
            self.assertGreaterEqual(assembly.sam_adapter.segment_calls, 1)

            attempt = result.task.attempt_history[0]
            self.assertIsNotNone(attempt.proposal_ref)
            self.assertIsNotNone(attempt.segmentation_ref)
            self.assertIsNotNone(attempt.evaluation_ref)

            segmentation_ref = attempt.segmentation_ref
            assert segmentation_ref is not None
            segmentation = assembly.artifact_store.load_artifact(
                segmentation_ref,
                SegmentationResult,
            )
            self.assertIs(segmentation.status, SegmentationStatus.READY)
            self.assertGreaterEqual(len(segmentation.candidates), 1)
            self.assertTrue(
                any(message.startswith("backend=sam3-real") for message in segmentation.diagnostics)
            )

            primary_mask_ref = result.task.result.final_mask_ref
            self.assertIsNotNone(primary_mask_ref)
            assert primary_mask_ref is not None
            mask_payload = assembly.artifact_store.load_artifact(primary_mask_ref, MockMask)
            self.assertEqual(mask_payload.label, "sam3_mask")
            self.assertEqual(mask_payload.backend_name, "sam3-real")
            self.assertIsNotNone(mask_payload.pixel_area)
            self.assertGreater(mask_payload.pixel_area or 0, 0)

            report_path = output_dir / "task_report.md"
            self.assertTrue(report_path.is_file())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("# M-SAgent Demo Report", report_text)
            self.assertIn("Embedded locator returned", report_text)
            self.assertIn("Real SAM3 adapter produced", report_text)
            self.assertIn("Mock evaluator accepted", report_text)
            self.assertIn(result.task.identity.task_id, report_text)

    def _require_env(self, name: str) -> str:
        value = os.environ.get(name)
        if value is None or not value.strip():
            self.fail(
                "real CLI chain smoke test requires the environment variable "
                f"{name} to be set when MSAGENT_ENABLE_REAL_CLI_SMOKE=1."
            )
        return value


if __name__ == "__main__":
    unittest.main()
