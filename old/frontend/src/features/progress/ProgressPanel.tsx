import {
  type IterationSnapshot,
  type JobLifecycleStatus,
  type ToolName,
} from '../../types/job'
import { IterationTimeline } from './IterationTimeline'
import { ToolBadge } from './ToolBadge'

const statusLabels: Record<JobLifecycleStatus, string> = {
  idle: '未开始',
  uploading: '上传中',
  queued: '排队中',
  running: '运行中',
  complete: '已完成',
  failed: '失败',
}

interface ProgressPanelProps {
  readonly jobStatus: JobLifecycleStatus
  readonly queuePosition: number | null
  readonly currentIteration: number
  readonly currentTool: ToolName | null
  readonly iterations: ReadonlyArray<IterationSnapshot>
}

export function ProgressPanel(props: ProgressPanelProps) {
  const {
    jobStatus,
    queuePosition,
    currentIteration,
    currentTool,
    iterations,
  } = props

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">过程</p>
          <h2>中间产物</h2>
        </div>
        <ToolBadge tool={currentTool} />
      </div>

      <div className="progress-summary">
        <div className="summary-card">
          <span>状态</span>
          <strong>{statusLabels[jobStatus]}</strong>
        </div>
        <div className="summary-card">
          <span>轮次</span>
          <strong>{currentIteration > 0 ? currentIteration : '--'}</strong>
        </div>
        <div className="summary-card">
          <span>排队</span>
          <strong>{queuePosition ?? '--'}</strong>
        </div>
      </div>

      <IterationTimeline iterations={iterations} />
    </section>
  )
}
