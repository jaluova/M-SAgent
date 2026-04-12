import { type JobResult } from '../../types/job'
import { MaskOverlay } from './MaskOverlay'
import { ResultStats } from './ResultStats'
import { DownloadActions } from './DownloadActions'

interface ResultPanelProps {
  readonly result: JobResult
  readonly sourceImageUrl: string | null
}

export function ResultPanel(props: ResultPanelProps) {
  const { result, sourceImageUrl } = props

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">结果</p>
          <h2>输出</h2>
        </div>
      </div>

      <div className="result-grid">
        <MaskOverlay result={result} sourceImageUrl={sourceImageUrl} />
        <div className="stack">
          <ResultStats result={result} />
          <DownloadActions result={result} />
        </div>
      </div>
    </section>
  )
}
