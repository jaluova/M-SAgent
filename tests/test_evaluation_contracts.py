import unittest

from mllm_processor import MLLMProcessor
from pipeline import MLLMSAMPipeline


class FailingImage:
    def save(self, *_args, **_kwargs):
        raise RuntimeError("boom")


class EvaluationContractTests(unittest.TestCase):
    def test_segmentation_evaluation_returns_reject_tuple_on_exception(self):
        processor = MLLMProcessor.__new__(MLLMProcessor)
        processor.device = "cpu"
        processor.model = None
        processor.processor = None

        verdict, rejected_indices = processor.segmentation_evaluation(
            FailingImage(),
            FailingImage(),
            "truck",
        )

        self.assertEqual(verdict, "Reject")
        self.assertEqual(rejected_indices, [])

    def test_pipeline_normalizes_legacy_string_result(self):
        pipeline = MLLMSAMPipeline.__new__(MLLMSAMPipeline)

        verdict, rejected_indices = pipeline._normalize_segmentation_evaluation_result("")
        self.assertEqual(verdict, "Reject")
        self.assertEqual(rejected_indices, [])

    def test_pipeline_preserves_tuple_result(self):
        pipeline = MLLMSAMPipeline.__new__(MLLMSAMPipeline)

        verdict, rejected_indices = pipeline._normalize_segmentation_evaluation_result(
            ("Accept", [1, 3])
        )
        self.assertEqual(verdict, "Accept")
        self.assertEqual(rejected_indices, [1, 3])
