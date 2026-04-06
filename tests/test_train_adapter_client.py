import json
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tools.train_adapter_client import TrainAdapterClient


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TrainAdapterClientTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (64, 32), color="white")
        self.config = SimpleNamespace(
            TRAIN_ADAPTER_URL="http://127.0.0.1:8765",
            TRAIN_ADAPTER_TIMEOUT=5.0,
            TRAIN_ADAPTER_TIMEOUT_RETRY=12.0,
            TRAIN_ADAPTER_RETRY_ON_TIMEOUT=True,
            TRAIN_ADAPTER_DYNAMIC_TOPK=True,
            TRAIN_ADAPTER_ABS_THRESHOLD=0.35,
            TRAIN_ADAPTER_REL_RATIO=0.75,
            TRAIN_ADAPTER_MIN_K=1,
            TRAIN_ADAPTER_MAX_K=6,
        )

    def test_describe_service_reports_device(self):
        client = TrainAdapterClient(config=self.config)
        with patch(
            "tools.train_adapter_client.request.urlopen",
            return_value=FakeHTTPResponse(
                {"ok": True, "device": "cpu", "output_mode": "grid_logits", "adapter_type": "lightweight"}
            ),
        ):
            status = client.describe_service()

        self.assertTrue(status["ok"])
        self.assertEqual(status["device"], "cpu")
        self.assertEqual(status["adapter_type"], "lightweight")

    def test_localize_retries_once_after_timeout(self):
        client = TrainAdapterClient(config=self.config)
        response_payload = {
            "ok": True,
            "device": "cpu",
            "result": {
                "absolute_points": [[16, 8]],
                "normalized_points": [[0.25, 0.25]],
                "scores": [0.92],
                "selection_mode": "dynamic_topk",
                "selected_k": 1,
            },
        }

        with patch(
            "tools.train_adapter_client.request.urlopen",
            side_effect=[socket.timeout("slow request"), FakeHTTPResponse(response_payload)],
        ):
            localization, metadata = client.localize_with_metadata(self.image, "truck")

        self.assertEqual(localization.absolute_points, [[16.0, 8.0]])
        self.assertTrue(metadata["retry_used"])
        self.assertEqual(metadata["service_device"], "cpu")
        self.assertTrue(metadata["slow_path"])
        self.assertEqual(len(metadata["attempts"]), 2)
        self.assertFalse(metadata["attempts"][0]["ok"])
        self.assertTrue(metadata["attempts"][1]["ok"])


if __name__ == "__main__":
    unittest.main()
