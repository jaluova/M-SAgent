import os
from pathlib import Path

from config import Config


class ServerConfig:
    HOST = os.environ.get("M_SAGENT_SERVER_HOST", "0.0.0.0")
    PORT = int(os.environ.get("M_SAGENT_SERVER_PORT", "8000"))
    MAX_UPLOAD_MB = int(os.environ.get("M_SAGENT_SERVER_MAX_UPLOAD_MB", "30"))
    WS_POLL_INTERVAL_MS = int(os.environ.get("M_SAGENT_SERVER_WS_POLL_INTERVAL_MS", "400"))
    JOBS_DIR = Path(
        os.environ.get(
            "M_SAGENT_SERVER_JOBS_DIR",
            Config.BASE_DIR / "remote_artifacts" / "server_jobs",
        )
    ).expanduser()
    FRONTEND_DIST_DIR = Path(
        os.environ.get(
            "M_SAGENT_FRONTEND_DIST_DIR",
            Config.BASE_DIR / "frontend" / "dist",
        )
    ).expanduser()
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.environ.get(
            "M_SAGENT_SERVER_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]

    @classmethod
    def setup_dirs(cls):
        cls.JOBS_DIR.mkdir(parents=True, exist_ok=True)
