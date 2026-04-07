import {
  type IterationSnapshot,
  type JobLifecycleStatus,
  type ProgressEvent,
  type ToolName,
} from '../../types/job'
import { IterationTimeline } from './IterationTimeline'
import { AgentLogStream } from './AgentLogStream'
import { ToolBadge } from './ToolBadge'

interface ProgressPanelProps {
  readonly jobStatus: JobLifecycleStatus
  readonly queuePosition: number | null
  readonly currentIteration: number
  readonly currentTool: ToolName | null
  readonly iterations: ReadonlyArray<IterationSnapshot>
  readonly events: ReadonlyArray<ProgressEvent>
}

export function ProgressPanel(props: ProgressPanelProps) {
  const {
    jobStatus,
    queuePosition,
    currentIteration,
    currentTool,
    iterations,
    events,
  } = props

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">实时进度</p>
          <h2>Agent Loop 追踪</h2>
        </div>
        <ToolBadge tool={currentTool} />
      </div>

      <div className="progress-summary">
        <div className="summary-card">
          <span>任务状态</span>
          <strong>{jobStatus}</strong>
        </div>
        <div className="summary-card">
          <span>当前轮次</span>
          <strong>{currentIteration > 0 ? currentIteration : '--'}</strong>
        </div>
        <div className="summary-card">
          <span>队列位置</span>
          <strong>{queuePosition ?? '--'}</strong>
        </div>
      </div>

      <div className="stack">
        <div>
          <p className="eyebrow">Iterations</p>
          <IterationTimeline iterations={iterations} />
        </div>

        <div>
          <p className="eyebrow">Event Stream</p>
          <AgentLogStream events={events} />
        </div>
      </div>
    </section>
  )
}
