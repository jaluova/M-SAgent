from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

frontend_dist_dir = ServerConfig.FRONTEND_DIST_DIR
frontend_assets_dir = frontend_dist_dir / "assets"
frontend_index_path = frontend_dist_dir / "index.html"

if frontend_assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=frontend_assets_dir), name="assets")


def _serve_spa_index() -> FileResponse:
    if not frontend_index_path.exists():
        raise RuntimeError(
            f"Frontend build not found at {frontend_index_path}. "
            "Build the frontend before serving static assets."
        )
    return FileResponse(frontend_index_path)


@app.get("/")
async def serve_frontend_root():
    return _serve_spa_index()


@app.get("/{full_path:path}")
async def serve_frontend_app(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    candidate = (frontend_dist_dir / Path(full_path)).resolve()
    try:
        candidate.relative_to(frontend_dist_dir.resolve())
    except ValueError:
        return _serve_spa_index()

    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)

    return _serve_spa_index()
