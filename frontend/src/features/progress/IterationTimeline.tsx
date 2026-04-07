import { type IterationSnapshot } from '../../types/job'
import { ScoreBadge } from '../shared/ScoreBadge'
import { ToolBadge } from './ToolBadge'

interface IterationTimelineProps {
  readonly iterations: ReadonlyArray<IterationSnapshot>
}

export function IterationTimeline(props: IterationTimelineProps) {
  const { iterations } = props

  if (iterations.length === 0) {
    return <p className="muted">暂无进度</p>
  }

  return (
    <div className="iteration-list">
      {iterations.map((item) => {
        const verdictTone =
          item.verdict === 'Accept'
            ? 'verdict-pill--accept'
            : item.verdict === 'Reject'
              ? 'verdict-pill--reject'
              : ''

        return (
          <article className="iteration-card" key={item.iteration}>
            <div className="iteration-card__header">
              <div>
                <p className="eyebrow">Iteration {item.iteration}</p>
              </div>
              <span className={`verdict-pill ${verdictTone}`}>
                {item.verdict === 'pending' ? '等待评估' : item.verdict}
              </span>
            </div>

            <div className="iteration-card__meta">
              <ToolBadge tool={item.tool} />
              <ScoreBadge score={item.score} hidden={item.tool === 'image_enhancer'} />
            </div>

            {item.checkImageUrl ? (
              <div className="iteration-card__preview">
                <img src={item.checkImageUrl} alt={`Iteration ${item.iteration}`} />
              </div>
            ) : null}
          </article>
        )
      })}
    </div>
  )
}
