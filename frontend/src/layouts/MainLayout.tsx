import { type ReactNode } from 'react'
import { StatusIndicator } from '../features/shared/StatusIndicator'
import { type HealthStatus } from '../types/job'

interface MainLayoutProps {
  readonly health: HealthStatus | null
  readonly healthError: string | null
  readonly leftColumn: ReactNode
  readonly rightColumn: ReactNode
}

export function MainLayout(props: MainLayoutProps) {
  const { health, healthError, leftColumn, rightColumn } = props

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__title">
          <p className="eyebrow">M-SAgent Frontend</p>
          <h1>Agentic Referring Segmentation Console</h1>
          <p>
            面向 Qwen2.5-VL + SAM3 的前端控制台，覆盖图片上传、任务排队、实时进度、
            结果可视化与下载这条完整链路。
          </p>
        </div>

        <div className="topbar__meta">
          <div className="health-card">
            <StatusIndicator
              status={healthError ? 'failed' : health?.ok ? 'complete' : 'queued'}
              label={
                healthError
                  ? '后端未连接'
                  : health?.ok
                    ? '服务正常'
                    : '等待服务'
              }
            />
            <div className="health-card__grid">
              <div>
                <span>GPU</span>
                <strong>{health?.gpu ?? 'N/A'}</strong>
              </div>
              <div>
                <span>队列长度</span>
                <strong>{health?.queueSize ?? '--'}</strong>
              </div>
              <div>
                <span>模型状态</span>
                <strong>
                  {health?.modelLoaded == null
                    ? '--'
                    : health.modelLoaded
                      ? '已加载'
                      : '未加载'}
                </strong>
              </div>
              <div>
                <span>说明</span>
                <strong>{healthError ?? health?.detail ?? '等待健康检查'}</strong>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="app-grid">
        <div className="stack">{leftColumn}</div>
        <div className="stack">{rightColumn}</div>
      </main>
    </div>
  )
}
