import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import Config
from sam_processor import SAMProcessor
from server.config import ServerConfig
from tools.train_adapter_client import TrainAdapterClient, TrainAdapterClientError


@dataclass
class JobRecord:
    job_id: str
    text: str
    max_iterations: int
    work_dir: Path
    image_path: Path
    status: str = "queued"
    position: int | None = None
    error: str | None = None
    current_iteration: int = 0
    current_tool: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None


class JobManager:
    def __init__(self):
        ServerConfig.setup_dirs()
        self._jobs: dict[str, JobRecord] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._state_lock = threading.Lock()
        self._resource_lock = threading.Lock()
        self._sam: SAMProcessor | None = None
        self._train_adapter_client = TrainAdapterClient()

    async def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self):
        if self._worker_task is None:
            return

        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    def get_health(self) -> dict[str, Any]:
        service_status = self._train_adapter_client.describe_service()
        detail = None

        if service_status.get("ok"):
            detail = (
                f"TrainAdapter ready: device={service_status.get('device', 'unknown')}, "
                f"adapter={service_status.get('adapter_type', 'unknown')}"
            )
        else:
            detail = service_status.get("error", "TrainAdapter unavailable")

        return {
            "ok": bool(service_status.get("ok")),
            "gpu": str(service_status.get("device") or Config.DEVICE),
            "queueSize": self._queue.qsize(),
            "modelLoaded": self._sam.is_available() if self._sam is not None else False,
            "detail": detail,
        }

    async def create_job(self, image_bytes: bytes, filename: str, text: str, max_iterations: int) -> JobRecord:
        job_id = uuid.uuid4().hex
        work_dir = ServerConfig.JOBS_DIR / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        image_path = work_dir / self._normalize_filename(filename or "input.png")
        image_path.write_bytes(image_bytes)

        record = JobRecord(
            job_id=job_id,
            text=text.strip(),
            max_iterations=max(1, min(5, int(max_iterations))),
            work_dir=work_dir,
            image_path=image_path,
            position=self._queue.qsize() + 1,
        )
        record.events.append({"type": "queued", "position": record.position})

        with self._state_lock:
            self._jobs[job_id] = record

        await self._queue.put(job_id)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._state_lock:
            return self._jobs.get(job_id)

    def snapshot_job(self, job_id: str) -> dict[str, Any] | None:
        with self._state_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None

            return {
                "jobId": record.job_id,
                "status": record.status,
                "position": record.position,
                "error": record.error,
                "result": record.result,
                "currentIteration": record.current_iteration,
                "currentTool": record.current_tool,
                "events": list(record.events),
            }

    def delete_job(self, job_id: str) -> bool:
        with self._state_lock:
            record = self._jobs.pop(job_id, None)

        if record is None:
            return False

        if record.work_dir.exists():
            for path in sorted(record.work_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()
            record.work_dir.rmdir()

        return True

    async def _worker_loop(self):
        while True:
            job_id = await self._queue.get()
            try:
                await asyncio.to_thread(self._run_job_sync, job_id)
            finally:
                self._queue.task_done()

    def _run_job_sync(self, job_id: str):
        job = self.get_job(job_id)
        if job is None:
            return

        try:
            self._update_job(job_id, status="running", position=None)
            self._append_event(job_id, {"type": "started"})
            self._update_job(job_id, current_iteration=1)
            self._append_event(
                job_id,
                {
                    "type": "iteration_start",
                    "iteration": 1,
                    "maxIterations": job.max_iterations,
                },
            )
            self._update_job(job_id, current_tool="object_locator")
            self._append_event(
                job_id,
                {
                    "type": "tool_selected",
                    "iteration": 1,
                    "tool": "object_locator",
                    "params": {"query": job.text},
                },
            )

            image = Image.open(job.image_path).convert("RGB")
            localization, _ = self._train_adapter_client.localize_with_metadata(image, job.text)
            if not localization.absolute_points:
                raise TrainAdapterClientError("TrainAdapter returned no usable points", kind="server")

            check_image_name = "localization_overlay.jpg"
            if localization.annotated_image is not None:
                self._save_preview_image(localization.annotated_image, job.work_dir / check_image_name)

            labels = [1] * len(localization.absolute_points)
            sam = self._get_sam()
            segmentation = sam.segment_with_points(
                image,
                points=localization.absolute_points,
                labels=labels,
                multimask_output=True,
            )
            best_result = segmentation.get("best_result")
            if not segmentation.get("success") or best_result is None:
                raise RuntimeError(segmentation.get("message", "SAM segmentation failed"))

            mask = np.asarray(best_result["mask"], dtype=np.float32)
            mask_png_path = job.work_dir / "mask.png"
            mask_npy_path = job.work_dir / "mask.npy"
            result_image_path = job.work_dir / "result.png"
            result_preview_path = job.work_dir / "result_preview.jpg"
            self._save_mask_png(mask, mask_png_path)
            np.save(mask_npy_path, mask)
            result_image = sam.apply_mask_to_image(image, mask)
            result_image.save(result_image_path)
            self._save_preview_image(result_image, result_preview_path)

            check_image_url = (
                f"/api/jobs/{job_id}/images/{check_image_name}"
                if (job.work_dir / check_image_name).exists()
                else f"/api/jobs/{job_id}/result"
            )
            score = float(best_result.get("score", 0.0))
            self._append_event(
                job_id,
                {
                    "type": "segmentation_result",
                    "iteration": 1,
                    "tool": "object_locator",
                    "score": score,
                    "checkImageUrl": check_image_url,
                },
            )
            self._append_event(
                job_id,
                {
                    "type": "evaluation",
                    "iteration": 1,
                    "verdict": "Accept",
                    "rejectedIndices": [],
                },
            )

            result_payload = {
                "jobId": job_id,
                "success": True,
                "bestScore": score,
                "iterations": 1,
                "maskCount": 1,
                "resultImageUrl": f"/api/jobs/{job_id}/result",
                "resultPreviewUrl": f"/api/jobs/{job_id}/images/result_preview.jpg",
                "maskUrl": f"/api/jobs/{job_id}/mask?format=png",
            }
            self._update_job(job_id, status="complete", result=result_payload)
            self._append_event(job_id, {"type": "complete", "result": result_payload})
        except Exception as exc:
            self._update_job(job_id, status="failed", error=str(exc))
            self._append_event(job_id, {"type": "error", "message": str(exc)})

    def _append_event(self, job_id: str, event: dict[str, Any]):
        with self._state_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.events.append(event)

    def _get_sam(self) -> SAMProcessor:
        with self._resource_lock:
            if self._sam is None:
                self._sam = SAMProcessor()
            return self._sam

    def _update_job(self, job_id: str, **updates: Any):
        with self._state_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            for key, value in updates.items():
                setattr(record, key, value)

    def list_events(self, job_id: str) -> list[dict[str, Any]] | None:
        with self._state_lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            return list(record.events)

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        cleaned = Path(filename).name or "input.png"
        return cleaned.replace(" ", "_")

    @staticmethod
    def _save_mask_png(mask: np.ndarray, output_path: Path):
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        Image.fromarray(binary_mask, mode="L").save(output_path)

    @staticmethod
    def _save_preview_image(image: Image.Image, output_path: Path, max_size: int = 1600, quality: int = 82):
        preview = image.copy()
        preview.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        if preview.mode not in {"RGB", "L"}:
            preview = preview.convert("RGB")
        elif preview.mode == "L":
            preview = preview.convert("RGB")

        preview.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )

    def resolve_job_file(self, job_id: str, filename: str) -> Path | None:
        job = self.get_job(job_id)
        if job is None:
            return None

        candidate = (job.work_dir / filename).resolve()
        try:
            candidate.relative_to(job.work_dir.resolve())
        except ValueError:
            return None

        return candidate if candidate.exists() else None
