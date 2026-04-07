from fastapi import APIRouter, Request

from server.models import HealthResponseModel

router = APIRouter()


@router.get("/health", response_model=HealthResponseModel)
async def get_health(request: Request):
    manager = request.app.state.job_manager
    return manager.get_health()
