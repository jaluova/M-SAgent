from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import ServerConfig
from server.job_manager import JobManager
from server.routes.health import router as health_router
from server.routes.jobs import router as jobs_router
from server.routes.ws import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = JobManager()
    app.state.job_manager = manager
    await manager.start()
    try:
        yield
    finally:
        await manager.stop()


app = FastAPI(title="M-SAgent Adapter API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")
app.include_router(ws_router, prefix="/api")
