import base64
import io
import json
import socket
import time
from urllib import error, request

from PIL import Image

from config import Config
from utils.localization import LocalizationResult


class TrainAdapterClientError(RuntimeError):
    """Raised when the TrainAdapter service cannot provide a localization result."""

    def __init__(self, message, kind="unknown"):
        super().__init__(message)
        self.kind = kind


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

    def _request_json(self, req, timeout):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise TrainAdapterClientError(
                f"http {exc.code}: {detail or exc.reason}",
                kind="http",
            ) from exc
        except error.URLError as exc:
            kind = "timeout" if isinstance(exc.reason, socket.timeout) else "connection"
            raise TrainAdapterClientError(f"request failed: {exc}", kind=kind) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise TrainAdapterClientError(f"request timed out: {exc}", kind="timeout") from exc

    def healthcheck(self):
        url = f"{self.base_url}/health"
        req = request.Request(url, method="GET")
        payload = self._request_json(req, timeout=self.timeout)

        if not payload.get("ok"):
            raise TrainAdapterClientError(f"healthcheck failed: {payload}")
        return payload

    def describe_service(self):
        started = time.perf_counter()
        try:
            payload = self.healthcheck()
        except TrainAdapterClientError as exc:
            return {
                "ok": False,
                "error": str(exc),
                "error_kind": exc.kind,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
            }

        return {
            "ok": True,
            "device": str(payload.get("device", "")),
            "output_mode": payload.get("output_mode"),
            "adapter_type": payload.get("adapter_type"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }

    def localize(self, image, query):
        localization, _ = self.localize_with_metadata(image, query)
        return localization

    def localize_with_metadata(self, image, query):
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
        timeouts = [float(self.timeout)]
        retry_timeout = float(max(self.config.TRAIN_ADAPTER_TIMEOUT_RETRY, self.timeout))
        if self.config.TRAIN_ADAPTER_RETRY_ON_TIMEOUT and retry_timeout > self.timeout:
            timeouts.append(retry_timeout)

        last_exc = None
        attempts = []
        response_payload = None
        for attempt_index, timeout in enumerate(timeouts, start=1):
            started = time.perf_counter()
            try:
                response_payload = self._request_json(req, timeout=timeout)
                attempts.append({
                    "attempt": attempt_index,
                    "timeout_s": timeout,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "ok": True,
                })
                break
            except TrainAdapterClientError as exc:
                attempts.append({
                    "attempt": attempt_index,
                    "timeout_s": timeout,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "ok": False,
                    "error": str(exc),
                    "error_kind": exc.kind,
                })
                last_exc = exc
                if exc.kind != "timeout" or attempt_index == len(timeouts):
                    raise TrainAdapterClientError(f"predict failed: {exc}", kind=exc.kind) from exc
                print(
                    f"TrainAdapter request timed out after {timeout:.1f}s; "
                    f"retrying once with {retry_timeout:.1f}s timeout"
                )
        if response_payload is None and last_exc is not None:
            raise TrainAdapterClientError(f"predict failed: {last_exc}", kind=last_exc.kind) from last_exc

        if not response_payload.get("ok"):
            raise TrainAdapterClientError(response_payload.get("error", "unknown predict error"), kind="server")

        result_payload = dict(response_payload.get("result") or {})
        annotated_image = None
        encoded_image = result_payload.pop("annotated_image_base64", None)
        if encoded_image:
            annotated_image = self._image_from_base64(encoded_image)

        localization = LocalizationResult.from_train_adapter_payload(
            result_payload,
            annotated_image=annotated_image,
        )
        service_device = str(response_payload.get("device", "")).lower()
        metadata = {
            "service_device": service_device or "unknown",
            "slow_path": service_device == "cpu",
            "attempts": attempts,
            "retry_used": len(attempts) > 1,
        }
        return localization, metadata
