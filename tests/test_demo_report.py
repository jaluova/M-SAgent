from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.orchestrator.orchestrator import OrchestrationResult
from msagent.service.demo_report import (
    build_demo_task_report,
    render_demo_task_report_markdown,
)
from tests.test_minimal_vertical_slice import build_cli_service
from msagent.service.cli import CLIRequest


class DemoReportTests(unittest.TestCase):
    def test_demo_report_builds_from_happy_path_task(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"test-image")
            cli_service, store = build_cli_service(tmp_path / "artifacts")

            result = cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the red cup",
                    max_attempts=2,
                )
            )

            report = build_demo_task_report(
                result,
                artifact_store=store,
            )
            markdown = render_demo_task_report_markdown(report)

        self.assertEqual(report.task_id, result.task.identity.task_id)
        self.assertEqual(report.status, "succeeded")
        self.assertEqual(report.attempt_count, 1)
        self.assertEqual(len(report.attempts), 1)
        self.assertEqual(report.attempts[0].route, "locate")
        self.assertIn("status=ready, route=locate", report.attempts[0].proposal_summary or "")
        self.assertIn("verdict=accept", report.attempts[0].evaluation_summary or "")
        self.assertEqual(report.load_warnings, [])
        self.assertIn("# M-SAgent Demo Report", markdown)
        self.assertIn("## Attempts", markdown)
        self.assertIn("### Attempt 1", markdown)
        self.assertIn("the red cup", markdown)

    def test_demo_report_records_load_warning_for_missing_artifact(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"test-image")
            cli_service, store = build_cli_service(tmp_path / "artifacts")

            result = cli_service.run(
                CLIRequest(
                    image_path=str(image_path),
                    query_text="the red cup",
                    max_attempts=2,
                )
            )
            attempt = result.task.attempt_history[0]
            segmentation_ref = attempt.segmentation_ref
            assert segmentation_ref is not None
            missing_artifact_path = (
                Path(store.root_uri)
                / segmentation_ref.artifact_type.value
                / f"{segmentation_ref.artifact_id}.json"
            )
            missing_artifact_path.unlink()

            report = build_demo_task_report(
                OrchestrationResult(task=result.task),
                artifact_store=store,
            )
            markdown = render_demo_task_report_markdown(report)

        self.assertEqual(len(report.load_warnings), 1)
        self.assertIn(segmentation_ref.artifact_id, report.load_warnings[0])
        self.assertIn("## Load Warnings", markdown)


if __name__ == "__main__":
    unittest.main()
