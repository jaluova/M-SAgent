import unittest
from unittest.mock import patch

from PIL import Image

from config import Config
from tools.object_locator import ObjectLocator
from utils.localization import LocalizationResult


class FakeTrainAdapterClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def localize(self, image, query):
        self.calls.append((image.size, query))
        if self.error:
            raise self.error
        return self.result


class FakeSAMProcessor:
    def __init__(self):
        self.calls = []

    def segment_with_points(self, image, points, labels, multimask_output=True):
        self.calls.append({
            "image_size": image.size,
            "points": points,
            "labels": labels,
            "multimask_output": multimask_output,
        })
        return {
            "success": True,
            "results": [],
            "best_result": {
                "mask": [[True, False], [False, True]],
                "score": 0.91,
            },
        }

    def apply_mask_to_image(self, image, mask):
        return image.copy()


class ObjectLocatorTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (500, 300), color="white")
        self.sam = FakeSAMProcessor()

    def test_localization_result_parses_train_adapter_payload(self):
        payload = {
            "absolute_points": [[25, 30], [125.5, 60]],
            "normalized_points": [[0.05, 0.1], [0.251, 0.2]],
            "scores": [0.87, 0.61],
            "selection_mode": "dynamic_topk",
            "selected_k": 2,
        }
        result = LocalizationResult.from_train_adapter_payload(payload)
        self.assertEqual(result.source, "train_adapter")
        self.assertEqual(result.selected_k, 2)
        self.assertEqual(result.absolute_points[1], [125.5, 60.0])
        self.assertAlmostEqual(result.top_score(), 0.87)

    def test_grid_points_are_mapped_to_pixel_points(self):
        locator = ObjectLocator(None, train_adapter_client=FakeTrainAdapterClient())
        points, labels = locator._grid_points_to_pixel_points(
            [[0, 0], [5, 5], [2, 1]],
            [1, 0, 1],
            self.image.size,
        )
        self.assertEqual(labels, [1, 0, 1])
        self.assertEqual(points[0], [0.0, 0.0])
        self.assertEqual(points[1], [500.0, 300.0])
        self.assertEqual(points[2], [200.0, 60.0])

    def test_train_adapter_success_is_used_before_fallback(self):
        localization = LocalizationResult(
            absolute_points=[[20.0, 30.0], [40.0, 60.0], [80.0, 90.0]],
            normalized_points=[[0.04, 0.1], [0.08, 0.2], [0.16, 0.3]],
            scores=[0.91, 0.77, 0.51],
            selection_mode="dynamic_topk",
            selected_k=3,
            source="train_adapter",
        )
        locator = ObjectLocator(None, train_adapter_client=FakeTrainAdapterClient(result=localization))

        with patch.object(Config, "TRAIN_ADAPTER_ENABLED", True), \
             patch.object(Config, "TRAIN_ADAPTER_MIN_SCORE_FOR_USE", 0.35):
            result = locator.locate_referent(
                {"points": [[5, 5]], "labels": [0]},
                self.image,
                self.sam,
                query="target object",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["localization"]["source"], "train_adapter")
        self.assertEqual(self.sam.calls[0]["labels"], [1, 1, 1])
        self.assertEqual(len(self.sam.calls[0]["points"]), 3)

    def test_low_confidence_train_adapter_falls_back_to_hints(self):
        localization = LocalizationResult(
            absolute_points=[[30.0, 50.0]],
            normalized_points=[[0.06, 0.1667]],
            scores=[0.2],
            selection_mode="dynamic_topk",
            selected_k=1,
            source="train_adapter",
        )
        locator = ObjectLocator(None, train_adapter_client=FakeTrainAdapterClient(result=localization))

        with patch.object(Config, "TRAIN_ADAPTER_ENABLED", True), \
             patch.object(Config, "TRAIN_ADAPTER_MIN_SCORE_FOR_USE", 0.35):
            result = locator.locate_referent(
                {"points": [[1, 2], [3, 4]], "labels": [1, 0]},
                self.image,
                self.sam,
                query="target object",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["localization"]["source"], "mllm_hints")
        self.assertIn("low-confidence", result["fallback_reason"])
        self.assertEqual(self.sam.calls[0]["labels"], [1, 0])

    def test_service_failure_falls_back_to_hints(self):
        locator = ObjectLocator(
            None,
            train_adapter_client=FakeTrainAdapterClient(error=RuntimeError("boom")),
        )

        with patch.object(Config, "TRAIN_ADAPTER_ENABLED", True):
            result = locator.locate_referent(
                {"points": [[2, 2]], "labels": [1]},
                self.image,
                self.sam,
                query="target object",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["localization"]["source"], "mllm_hints")
        self.assertIn("TrainAdapter unavailable", result["fallback_reason"])

    def test_predict_counts_for_multiple_train_adapter_outputs(self):
        for count in (1, 3, 6):
            sam = FakeSAMProcessor()
            points = [[float(i * 10), float(i * 15)] for i in range(1, count + 1)]
            localization = LocalizationResult(
                absolute_points=points,
                normalized_points=[[point[0] / 500.0, point[1] / 300.0] for point in points],
                scores=[0.9] * count,
                selection_mode="dynamic_topk",
                selected_k=count,
                source="train_adapter",
            )
            locator = ObjectLocator(None, train_adapter_client=FakeTrainAdapterClient(result=localization))

            with patch.object(Config, "TRAIN_ADAPTER_ENABLED", True), \
                 patch.object(Config, "TRAIN_ADAPTER_MIN_SCORE_FOR_USE", 0.35):
                result = locator.locate_referent({}, self.image, sam, query="target object")

            self.assertTrue(result["success"])
            self.assertEqual(len(sam.calls[0]["points"]), count)
            self.assertEqual(sam.calls[0]["labels"], [1] * count)

    def test_disabled_train_adapter_keeps_fallback_behavior(self):
        locator = ObjectLocator(None, train_adapter_client=FakeTrainAdapterClient())

        with patch.object(Config, "TRAIN_ADAPTER_ENABLED", False):
            result = locator.locate_referent(
                {"points": [[4, 1]], "labels": [1]},
                self.image,
                self.sam,
                query="target object",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["localization"]["source"], "mllm_hints")
        self.assertEqual(self.sam.calls[0]["labels"], [1])


if __name__ == "__main__":
    unittest.main()
