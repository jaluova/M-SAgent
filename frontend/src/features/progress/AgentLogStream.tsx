import { type ProgressEvent } from '../../types/job'

function describeEvent(event: ProgressEvent): string {
  switch (event.type) {
    case 'queued':
      return `任务已进入队列，当前位置 ${event.position}。`
    case 'started':
      return '任务开始执行，等待第一个 iteration。'
    case 'iteration_start':
      return `开始第 ${event.iteration} / ${event.maxIterations} 轮推理。`
    case 'tool_selected':
      return `第 ${event.iteration} 轮选择工具 ${event.tool}。`
    case 'segmentation_result':
      return `第 ${event.iteration} 轮完成分割，得分 ${event.score.toFixed(3)}。`
    case 'evaluation':
      return `第 ${event.iteration} 轮评估结论为 ${event.verdict}。`
    case 'complete':
      return `任务完成，最佳得分 ${event.result.bestScore.toFixed(3)}。`
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
    return <p className="muted">当前还没有日志事件。</p>
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
