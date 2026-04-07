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
            zoomed_image=FailingImage(),
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

    def test_parse_response_accepts_html_escaped_attribute_parameters(self):
        processor = MLLMProcessor.__new__(MLLMProcessor)

        parsed = processor._parse_response(
            '<tool name="image_enhancer" parameters="{&quot;rectangular area&quot;: [[2, 2], [4, 4]]}"></tool>'
        )

        self.assertEqual(parsed["name"], "image_enhancer")
        self.assertEqual(parsed["parameters"]["rectangular area"], [[2, 2], [4, 4]])

    def test_parse_response_accepts_direct_tool_tag_attributes(self):
        processor = MLLMProcessor.__new__(MLLMProcessor)

        parsed = processor._parse_response(
            '<object_locator points="[[4, 2], [4, 3], [5, 2]]" labels="[1, 1, 1]"></object_locator>'
        )

        self.assertEqual(parsed["name"], "object_locator")
        self.assertEqual(parsed["parameters"]["points"], [[4, 2], [4, 3], [5, 2]])
        self.assertEqual(parsed["parameters"]["labels"], [1, 1, 1])

    def test_build_user_prompt_includes_history_details_and_overlay_context(self):
        processor = MLLMProcessor.__new__(MLLMProcessor)

        prompt = processor._build_user_prompt(
            "truck",
            tool_history=[
                {
                    "tool": "object_locator",
                    "verdict": "Reject",
                    "iteration": 1,
                    "score": 0.8123,
                    "note": "mask rejected by evaluation",
                }
            ],
            has_mask_overlay=True,
        )

        self.assertIn("score=0.812", prompt)
        self.assertIn("note=mask rejected by evaluation", prompt)
        self.assertIn("current accepted-mask overlay image", prompt)

    def test_build_check_prompt_normalizes_query_and_requires_full_person_mask(self):
        processor = MLLMProcessor.__new__(MLLMProcessor)

        prompt = processor._build_check_user_prompt("old_man")

        self.assertIn("Normalized natural-language form of the query: old man", prompt)
        self.assertIn("Judge the returned mask itself", prompt)
        self.assertIn("whole visible person", prompt)
