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
from msagent.infra.mock_adapters import MockLLMAdapter
from msagent.infra.mock_artifacts import MockMask
from msagent.infra.sam3_adapter import RealSAM3Adapter
from msagent.service import build_default_cli_service
from msagent.service.cli import CLIRequest


@unittest.skipUnless(
    os.environ.get("MSAGENT_ENABLE_REAL_E2E_SMOKE") == "1",
    "real end-to-end smoke test is opt-in",
)
class RealEndToEndSmokeTests(unittest.TestCase):
    def test_orchestrator_main_chain_reaches_real_sam_segmentation(self) -> None:
        sam_model_path = self._require_env("M_SAGENT_SAM3_MODEL_PATH")
        sam_checkpoint_path = self._require_env("M_SAGENT_SAM3_CHECKPOINT_PATH")
        image_path = Path(self._require_env("MSAGENT_REAL_E2E_IMAGE")).expanduser()
        query_text = os.environ.get("MSAGENT_REAL_E2E_QUERY", "truck")

        if not image_path.is_file():
            self.fail(f"real E2E smoke image does not exist: {image_path}")

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings = MSAgentSettings()
            settings.runtime.artifact_root = str(tmp_path / "artifacts")
            settings.model_paths.sam_model_path = sam_model_path
            settings.model_paths.sam_checkpoint_path = sam_checkpoint_path

            assembly = build_default_cli_service(
                settings,
                llm_adapter=MockLLMAdapter(
                    backend_name="mock-llm-real-sam-smoke",
                    evaluation_verdict_sequence=(EvaluationVerdict.ACCEPT,),
                ),
            )
            try:
                self.assertIsInstance(assembly.sam_adapter, RealSAM3Adapter)
                result = assembly.service.run(
                    CLIRequest(
                        image_path=str(image_path),
                        query_text=query_text,
                        max_attempts=1,
                    )
                )

                self.assertEqual(
                    assembly.diagnostics,
                    ["embedded_locator_runtime=disabled", "sam_runtime=enabled"],
                )
                self.assertIs(result.task.runtime.status, TaskStatus.SUCCEEDED)
                self.assertIs(result.task.result.final_verdict, EvaluationVerdict.ACCEPT)
                self.assertGreaterEqual(assembly.sam_adapter.segment_calls, 1)

                attempt = result.task.attempt_history[0]
                self.assertIsNotNone(attempt.query_understanding_ref)
                self.assertIsNotNone(attempt.proposal_ref)
                self.assertIsNotNone(attempt.prompt_package_ref)
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
                    any(message.startswith("backend=") for message in segmentation.diagnostics)
                )

                primary_mask_ref = result.task.result.final_mask_ref
                self.assertIsNotNone(primary_mask_ref)
                assert primary_mask_ref is not None
                mask_payload = assembly.artifact_store.load_artifact(primary_mask_ref, MockMask)
                self.assertEqual(mask_payload.label, "sam3_mask")
                self.assertEqual(mask_payload.backend_name, "sam3-real")
                self.assertIsNotNone(mask_payload.pixel_area)
                self.assertGreater(mask_payload.pixel_area or 0, 0)
            finally:
                assembly.close()

    def _require_env(self, name: str) -> str:
        value = os.environ.get(name)
        if value is None or not value.strip():
            self.fail(
                "real end-to-end smoke test requires the environment variable "
                f"{name} to be set when MSAGENT_ENABLE_REAL_E2E_SMOKE=1."
            )
        return value


if __name__ == "__main__":
    unittest.main()
