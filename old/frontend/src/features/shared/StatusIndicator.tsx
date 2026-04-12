import { type JobLifecycleStatus } from '../../types/job'

const labels: Record<JobLifecycleStatus, string> = {
  idle: '未开始',
  uploading: '上传中',
  queued: '排队中',
  running: '运行中',
  complete: '已完成',
  failed: '失败',
}

interface StatusIndicatorProps {
  readonly status: JobLifecycleStatus
  readonly label?: string
}

export function StatusIndicator(props: StatusIndicatorProps) {
  const { status, label } = props

  return (
    <span className={`status-pill status-pill--${status}`}>
      {label ?? labels[status]}
    </span>
  )
}
