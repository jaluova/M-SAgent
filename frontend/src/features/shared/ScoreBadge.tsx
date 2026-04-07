interface ScoreBadgeProps {
  readonly score: number | null
}

export function ScoreBadge(props: ScoreBadgeProps) {
  const { score } = props

  if (score === null) {
    return <span className="score-badge">等待评分</span>
  }

  const tone =
    score >= 0.75
      ? 'score-badge--good'
      : score >= 0.45
        ? 'score-badge--medium'
        : 'score-badge--low'

  return <span className={`score-badge ${tone}`}>Score {score.toFixed(3)}</span>
}
