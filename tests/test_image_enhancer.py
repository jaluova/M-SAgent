import unittest

import numpy as np
from PIL import Image

from tools.image_enhancer import ImageEnhancer


class FakeSAMProcessor:
    def __init__(self, mask, score=0.9):
        self.mask = mask
        self.score = score

    def segment_with_text(self, image, text_prompt=None, multimask_output=True):
        return {
            "success": True,
            "results": [
                {
                    "mask": self.mask,
                    "score": self.score,
                }
            ],
            "best_result": {
                "mask": self.mask,
                "score": self.score,
            },
        }

    def apply_mask_to_image(self, image, mask):
        return image.copy()


class ImageEnhancerTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (500, 300), color="white")
        self.enhancer = ImageEnhancer()

    def test_rejects_mask_touching_internal_crop_edges(self):
        mask = np.ones((64, 64), dtype=np.float32)
        sam = FakeSAMProcessor(mask)

        result = self.enhancer.enhance_image(
            self.image,
            "truck",
            sam,
            {"rectangular area": [[2, 2], [4, 4]]},
        )

        self.assertFalse(result["success"])
        self.assertIn("truncates", result["message"])

    def test_accepts_mask_with_margin_inside_crop(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:54, 12:52] = 1.0
        sam = FakeSAMProcessor(mask)

        result = self.enhancer.enhance_image(
            self.image,
            "truck",
            sam,
            {"rectangular area": [[2, 2], [4, 4]]},
        )

        self.assertTrue(result["success"])
        self.assertGreater(result["best_result"]["mask_area"], 0)


if __name__ == "__main__":
    unittest.main()
