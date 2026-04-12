import { type JobResult } from '../../types/job'

interface ResultStatsProps {
  readonly result: JobResult
}

export function ResultStats(props: ResultStatsProps) {
  const { result } = props

  return (
    <div className="stats-grid">
      <div className="stat-card">
        <span>最佳分数</span>
        <strong>{result.bestScore.toFixed(3)}</strong>
      </div>
      <div className="stat-card">
        <span>迭代轮次</span>
        <strong>{result.iterations}</strong>
      </div>
      <div className="stat-card">
        <span>掩码数量</span>
        <strong>{result.maskCount}</strong>
      </div>
      <div className="stat-card">
        <span>任务结论</span>
        <strong>{result.success ? '成功' : '失败'}</strong>
      </div>
    </div>
  )
}
