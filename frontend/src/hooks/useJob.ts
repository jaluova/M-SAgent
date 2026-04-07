import { useEffect, useEffectEvent, useState } from 'react'
import { getHealth, getJob } from '../api/jobs'
import { connectJobWebSocket } from '../api/websocket'
import { type HealthStatus } from '../types/job'
import { useJobStore } from '../stores/job-store'

export function useJob() {
  const store = useJobStore()
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const refreshHealth = useEffectEvent(async () => {
    try {
      const response = await getHealth()
      setHealth(response)
      setHealthError(null)
    } catch (error) {
      setHealth(null)
      setHealthError(
        error instanceof Error ? error.message : '无法连接到后端服务。',
      )
    }
  })

  const refreshJob = useEffectEvent(async () => {
    if (!store.jobId) {
      return
    }

    try {
      const snapshot = await getJob(store.jobId)
      store.hydrateJobStatus(snapshot)
    } catch {
      // Polling is a silent fallback when WS is unavailable.
    }
  })

  useEffect(() => {
    void refreshHealth()
    const timer = window.setInterval(() => {
      void refreshHealth()
    }, 20000)

    return () => window.clearInterval(timer)
  }, [refreshHealth])

  useEffect(() => {
    if (!store.jobId || !['queued', 'running'].includes(store.jobStatus)) {
      return
    }

    const disconnect = connectJobWebSocket(store.jobId, {
      onEvent: (event) => {
        store.handleEvent(event)
      },
      onError: () => {
        void refreshJob()
      },
    })

    return disconnect
  }, [store.jobId, store.jobStatus, store.handleEvent, refreshJob])

  useEffect(() => {
    if (!store.jobId || !['queued', 'running'].includes(store.jobStatus)) {
      return
    }

    const timer = window.setInterval(() => {
      void refreshJob()
    }, 6000)

    void refreshJob()

    return () => window.clearInterval(timer)
  }, [store.jobId, store.jobStatus, refreshJob])

  return {
    ...store,
    health,
    healthError,
  }
}
