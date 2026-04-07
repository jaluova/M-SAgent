from typing import Any, Literal

from pydantic import BaseModel, Field


JobStatusLiteral = Literal["idle", "uploading", "queued", "running", "complete", "failed"]
ToolNameLiteral = Literal["object_locator", "concept_generator", "image_enhancer", "report_no_mask"]


class JobResultModel(BaseModel):
    job_id: str = Field(alias="jobId")
    success: bool
    best_score: float = Field(alias="bestScore")
    iterations: int
    mask_count: int = Field(alias="maskCount")
    result_image_url: str = Field(alias="resultImageUrl")
    result_preview_url: str | None = Field(default=None, alias="resultPreviewUrl")
    mask_url: str = Field(alias="maskUrl")

    model_config = {"populate_by_name": True}


class JobCreateResponseModel(BaseModel):
    job_id: str = Field(alias="jobId")
    status: JobStatusLiteral
    position: int | None = None

    model_config = {"populate_by_name": True}


class JobStatusResponseModel(BaseModel):
    job_id: str = Field(alias="jobId")
    status: JobStatusLiteral
    position: int | None = None
    error: str | None = None
    result: JobResultModel | None = None
    current_iteration: int = Field(alias="currentIteration")
    current_tool: ToolNameLiteral | None = Field(default=None, alias="currentTool")
    events: list[dict[str, Any]]

    model_config = {"populate_by_name": True}


class HealthResponseModel(BaseModel):
    ok: bool
    gpu: str | None = None
    queue_size: int | None = Field(default=None, alias="queueSize")
    model_loaded: bool | None = Field(default=None, alias="modelLoaded")
    detail: str | None = None

    model_config = {"populate_by_name": True}
