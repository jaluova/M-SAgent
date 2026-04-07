import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import Config
from pipeline import MLLMSAMPipeline
from sam_processor import SAMProcessor
from server.config import ServerConfig
from tools.train_adapter_client import TrainAdapterClient


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
    SUPPORTED_EVENT_TOOLS = {"object_locator", "concept_generator", "image_enhancer", "report_no_mask"}

    def __init__(self):
        ServerConfig.setup_dirs()
        self._jobs: dict[str, JobRecord] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._state_lock = threading.Lock()
        self._resource_lock = threading.Lock()
        self._sam: SAMProcessor | None = None
        self._pipeline: MLLMSAMPipeline | None = None
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
            self._update_job(job_id, status="running", position=None, error=None, current_tool=None)
            self._append_event(job_id, {"type": "started"})
            pipeline = self._get_pipeline()
            pipeline.set_event_callback(lambda event: self._handle_pipeline_event(job_id, event))

            original_output_dir = Config.OUTPUT_DIR
            original_tool_calls_log = Config.TOOL_CALLS_LOG
            try:
                Config.OUTPUT_DIR = job.work_dir
                Config.TOOL_CALLS_LOG = job.work_dir / "tool_calls"
                Config.setup_dirs()

                pipeline_result = pipeline.run(
                    str(job.image_path),
                    job.text,
                    max_iterations=job.max_iterations,
                )
            finally:
                pipeline.set_event_callback(None)
                Config.OUTPUT_DIR = original_output_dir
                Config.TOOL_CALLS_LOG = original_tool_calls_log

            result_payload = self._materialize_pipeline_result(job, pipeline_result)
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

    def _get_pipeline(self) -> MLLMSAMPipeline:
        with self._resource_lock:
            if self._pipeline is None:
                self._pipeline = MLLMSAMPipeline()
                self._sam = self._pipeline.sam
            return self._pipeline

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

        if "A" in preview.getbands():
            background = Image.new("RGB", preview.size, (255, 255, 255))
            background.paste(preview, mask=preview.getchannel("A"))
            preview = background
        elif preview.mode not in {"RGB", "L"}:
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

    def _handle_pipeline_event(self, job_id: str, event: dict[str, Any]):
        event_payload = dict(event)
        event_type = str(event_payload.get("type") or "")
        iteration = int(event_payload.get("iteration") or 0)
        tool = event_payload.get("tool")

        if event_type == "iteration_start":
            self._update_job(job_id, current_iteration=iteration)
        elif event_type == "tool_selected":
            current_tool = tool if tool in self.SUPPORTED_EVENT_TOOLS else None
            self._update_job(job_id, current_iteration=iteration, current_tool=current_tool)
        elif event_type == "segmentation_result":
            check_image_path = event_payload.pop("check_image_path", None)
            if check_image_path:
                event_payload["checkImageUrl"] = f"/api/jobs/{job_id}/images/{Path(check_image_path).name}"
            current_tool = tool if tool in self.SUPPORTED_EVENT_TOOLS else None
            self._update_job(job_id, current_iteration=iteration, current_tool=current_tool)
        elif event_type == "evaluation":
            rejected = event_payload.get("rejectedIndices")
            if not isinstance(rejected, list):
                event_payload["rejectedIndices"] = []

        self._append_event(job_id, event_payload)

    def _materialize_pipeline_result(self, job: JobRecord, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        if not pipeline_result.get("success"):
            raise RuntimeError(pipeline_result.get("message", "Pipeline did not produce an accepted mask"))

        accepted_masks = pipeline_result.get("accepted_masks") or []
        mask_arrays = [
            np.asarray(item.get("mask"), dtype=np.float32)
            for item in accepted_masks
            if isinstance(item, dict) and item.get("mask") is not None
        ]
        if not mask_arrays:
            raise RuntimeError("Pipeline returned success without any accepted masks")

        result_image_source = Path(str(pipeline_result.get("final_image_path", ""))).expanduser()
        if not result_image_source.exists():
            raise RuntimeError(f"Final image not found: {result_image_source}")

        result_image = Image.open(result_image_source).convert("RGBA")
        result_image_path = job.work_dir / "result.png"
        result_preview_path = job.work_dir / "result_preview.jpg"
        result_image.save(result_image_path)
        self._save_preview_image(result_image, result_preview_path)

        mask_stack = np.stack(mask_arrays) if len(mask_arrays) > 1 else mask_arrays[0]
        union_mask = np.maximum.reduce([(mask > 0.5).astype(np.float32) for mask in mask_arrays])
        mask_png_path = job.work_dir / "mask.png"
        mask_npy_path = job.work_dir / "mask.npy"
        self._save_mask_png(union_mask, mask_png_path)
        np.save(mask_npy_path, mask_stack)

        best_score = float(pipeline_result.get("best_score", 0.0))
        iterations = int(pipeline_result.get("iterations", job.max_iterations))
        result_payload = {
            "jobId": job.job_id,
            "success": True,
            "bestScore": best_score,
            "iterations": iterations,
            "maskCount": len(mask_arrays),
            "resultImageUrl": f"/api/jobs/{job.job_id}/result",
            "resultPreviewUrl": f"/api/jobs/{job.job_id}/images/result_preview.jpg",
            "maskUrl": f"/api/jobs/{job.job_id}/mask?format=png",
        }

        for artifact in job.work_dir.glob("final_result_*"):
            if artifact.resolve() != result_image_path.resolve():
                artifact.unlink(missing_ok=True)
        for artifact in job.work_dir.glob("final_mask_*"):
            artifact.unlink(missing_ok=True)

        return result_payload
