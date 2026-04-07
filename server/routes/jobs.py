from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from server.config import ServerConfig
from server.models import JobCreateResponseModel, JobStatusResponseModel

router = APIRouter()


@router.post("/jobs", response_model=JobCreateResponseModel)
async def create_job(
    request: Request,
    image: UploadFile = File(...),
    text: str = Form(...),
    max_iter: int = Form(3),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    image_bytes = await image.read()
    max_bytes = ServerConfig.MAX_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")

    manager = request.app.state.job_manager
    job = await manager.create_job(
        image_bytes=image_bytes,
        filename=image.filename or "input.png",
        text=text,
        max_iterations=max_iter,
    )
    return {
        "jobId": job.job_id,
        "status": job.status,
        "position": job.position,
    }


@router.get("/jobs/{job_id}", response_model=JobStatusResponseModel)
async def get_job(job_id: str, request: Request):
    manager = request.app.state.job_manager
    snapshot = manager.snapshot_job(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return snapshot


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, request: Request):
    manager = request.app.state.job_manager
    if not manager.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


@router.get("/jobs/{job_id}/result")
async def get_result_image(job_id: str, request: Request):
    manager = request.app.state.job_manager
    result_path = manager.resolve_job_file(job_id, "result.png")
    if result_path is None:
        raise HTTPException(status_code=404, detail="Result image not found")
    return FileResponse(result_path, media_type="image/png", filename=f"{job_id}-result.png")


@router.get("/jobs/{job_id}/mask")
async def get_mask(job_id: str, request: Request, format: str = Query("png")):
    manager = request.app.state.job_manager
    normalized = format.lower()
    filename = "mask.npy" if normalized == "npy" else "mask.png"
    media_type = "application/octet-stream" if normalized == "npy" else "image/png"
    mask_path = manager.resolve_job_file(job_id, filename)
    if mask_path is None:
        raise HTTPException(status_code=404, detail="Mask not found")
    return FileResponse(mask_path, media_type=media_type, filename=f"{job_id}-{filename}")


@router.get("/jobs/{job_id}/images/{name}")
async def get_intermediate_image(job_id: str, name: str, request: Request):
    manager = request.app.state.job_manager
    image_path = manager.resolve_job_file(job_id, name)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/png", filename=name)
