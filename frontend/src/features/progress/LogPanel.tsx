import { type ProgressEvent } from '../../types/job'
import { AgentLogStream } from './AgentLogStream'

interface LogPanelProps {
  readonly events: ReadonlyArray<ProgressEvent>
}

export function LogPanel(props: LogPanelProps) {
  const { events } = props

  return (
    <section className="panel panel--log">
      <div className="panel__header">
        <div>
          <p className="eyebrow">日志</p>
          <h2>运行日志</h2>
        </div>
      </div>

      <AgentLogStream events={events} />
    </section>
  )
}
