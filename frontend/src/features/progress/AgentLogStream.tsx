import { type ProgressEvent } from '../../types/job'

function describeEvent(event: ProgressEvent): string {
  switch (event.type) {
    case 'queued':
      return `进入队列 ${event.position}`
    case 'started':
      return '开始执行'
    case 'iteration_start':
      return `第 ${event.iteration} / ${event.maxIterations} 轮`
    case 'tool_selected':
      return `${event.iteration} 轮调用 ${event.tool}`
    case 'segmentation_result':
      return event.tool === 'image_enhancer'
        ? `${event.iteration} 轮完成分割`
        : `${event.iteration} 轮得分 ${event.score.toFixed(3)}`
    case 'evaluation':
      return `${event.iteration} 轮 ${event.verdict === 'Accept' ? '通过' : '拒绝'}`
    case 'complete':
      return `完成 ${event.result.bestScore.toFixed(3)}`
    case 'error':
      return event.message
    default:
      return '收到未知事件。'
  }
}

interface AgentLogStreamProps {
  readonly events: ReadonlyArray<ProgressEvent>
}

export function AgentLogStream(props: AgentLogStreamProps) {
  const { events } = props

  if (events.length === 0) {
    return <p className="muted">暂无日志</p>
  }

  return (
    <div className="log-list">
      {events.map((event, index) => (
        <div className="log-line" key={`${event.type}-${index}`}>
          <div className="log-line__index">{index + 1}</div>
          <div>{describeEvent(event)}</div>
        </div>
      ))}
    </div>
  )
}
