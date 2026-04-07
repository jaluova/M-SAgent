interface ScoreBadgeProps {
  readonly score: number | null
  readonly hidden?: boolean
}

export function ScoreBadge(props: ScoreBadgeProps) {
  const { score, hidden = false } = props

  if (hidden || score === null) {
    return null
  }

  const tone =
    score >= 0.75
      ? 'score-badge--good'
      : score >= 0.45
        ? 'score-badge--medium'
        : 'score-badge--low'

  return <span className={`score-badge ${tone}`}>Score {score.toFixed(3)}</span>
}
