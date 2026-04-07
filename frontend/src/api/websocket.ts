import { createWebSocketUrl } from './client'
import { type ProgressEvent, normalizeProgressEvent } from '../types/job'

interface JobWebSocketHandlers {
  readonly onEvent: (event: ProgressEvent) => void
  readonly onError: (error: Error) => void
}

export function connectJobWebSocket(
  jobId: string,
  handlers: JobWebSocketHandlers,
): () => void {
  const socket = new WebSocket(createWebSocketUrl(`/api/jobs/${jobId}/ws`))

  socket.onmessage = (messageEvent) => {
    try {
      const payload = JSON.parse(messageEvent.data as string)
      const event = normalizeProgressEvent(payload, jobId)

      if (event) {
        handlers.onEvent(event)
      }
    } catch (error) {
      handlers.onError(
        error instanceof Error
          ? error
          : new Error('无法解析 WebSocket 事件'),
      )
    }
  }

  socket.onerror = () => {
    handlers.onError(new Error('WebSocket 连接异常，已切换到轮询状态。'))
  }

  return () => {
    if (
      socket.readyState === WebSocket.OPEN ||
      socket.readyState === WebSocket.CONNECTING
    ) {
      socket.close()
    }
  }
}
