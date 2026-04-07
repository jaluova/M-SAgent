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

    def test_parse_response_accepts_json_body_tool_format(self):
        processor = MLLMProcessor.__new__(MLLMProcessor)

        parsed = processor._parse_response(
            '<tool>{"name": "object_locator", "parameters": {"points": [[2, 2]], "labels": [1]}}</tool>'
        )

        self.assertEqual(parsed["name"], "object_locator")
        self.assertEqual(parsed["parameters"]["points"], [[2, 2]])

    def test_parse_response_accepts_attribute_style_tool_format(self):
        processor = MLLMProcessor.__new__(MLLMProcessor)

        parsed = processor._parse_response(
            '<tool name="object_locator" parameters=\'{"points": [[2, 2], [4, 2]], "labels": [1, 1]}\'></tool>'
        )

        self.assertEqual(parsed["name"], "object_locator")
        self.assertEqual(parsed["parameters"]["labels"], [1, 1])
