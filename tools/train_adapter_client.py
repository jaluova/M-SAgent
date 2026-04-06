import base64
import io
import json
import socket
from urllib import error, request

from PIL import Image

from config import Config
from utils.localization import LocalizationResult


class TrainAdapterClientError(RuntimeError):
    """Raised when the TrainAdapter service cannot provide a localization result."""


class TrainAdapterClient:
    def __init__(self, base_url=None, timeout=None, config=Config):
        self.config = config
        self.base_url = (base_url or config.TRAIN_ADAPTER_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.TRAIN_ADAPTER_TIMEOUT

    @staticmethod
    def _image_to_base64(image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _image_from_base64(encoded):
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    def healthcheck(self):
        url = f"{self.base_url}/health"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, socket.timeout, TimeoutError) as exc:
            raise TrainAdapterClientError(f"healthcheck failed: {exc}") from exc

        if not payload.get("ok"):
            raise TrainAdapterClientError(f"healthcheck failed: {payload}")
        return payload

    def localize(self, image, query):
        url = f"{self.base_url}/predict"
        payload = {
            "image_base64": self._image_to_base64(image.convert("RGB")),
            "query": query,
            "use_dynamic_topk": bool(self.config.TRAIN_ADAPTER_DYNAMIC_TOPK),
            "abs_threshold": float(self.config.TRAIN_ADAPTER_ABS_THRESHOLD),
            "rel_ratio": float(self.config.TRAIN_ADAPTER_REL_RATIO),
            "min_k": int(self.config.TRAIN_ADAPTER_MIN_K),
            "max_k": int(self.config.TRAIN_ADAPTER_MAX_K),
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response_payload = json.loads(resp.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, socket.timeout, TimeoutError) as exc:
            raise TrainAdapterClientError(f"predict failed: {exc}") from exc

        if not response_payload.get("ok"):
            raise TrainAdapterClientError(response_payload.get("error", "unknown predict error"))

        result_payload = dict(response_payload.get("result") or {})
        annotated_image = None
        encoded_image = result_payload.pop("annotated_image_base64", None)
        if encoded_image:
            annotated_image = self._image_from_base64(encoded_image)

        return LocalizationResult.from_train_adapter_payload(
            result_payload,
            annotated_image=annotated_image,
        )
