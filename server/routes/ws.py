import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.config import ServerConfig

router = APIRouter()


@router.websocket("/jobs/{job_id}/ws")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    manager = websocket.app.state.job_manager
    sent_index = 0
    closed = False

    try:
        while True:
            snapshot = manager.snapshot_job(job_id)
            if snapshot is None:
                await websocket.send_json({"type": "error", "message": "Job not found"})
                break
            events = manager.list_events(job_id) or []
            while sent_index < len(events):
                await websocket.send_json(events[sent_index])
                sent_index += 1

            if snapshot["status"] in {"complete", "failed"} and sent_index >= len(events):
                break

            await asyncio.sleep(ServerConfig.WS_POLL_INTERVAL_MS / 1000.0)
    except WebSocketDisconnect:
        closed = True
        return
    finally:
        if not closed:
            await websocket.close()
