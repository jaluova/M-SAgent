import { toApiUrl } from '../../api/client'
import { type JobResult } from '../../types/job'

interface DownloadActionsProps {
  readonly result: JobResult
}

export function DownloadActions(props: DownloadActionsProps) {
  const { result } = props

  return (
    <div className="download-list">
      <a className="download-link" href={result.resultImageUrl} target="_blank" rel="noreferrer">
        <div>
          <strong>下载结果图</strong>
          <span>JPEG / PNG 可视化结果</span>
        </div>
        <span>Open</span>
      </a>
      <a className="download-link" href={result.maskUrl} target="_blank" rel="noreferrer">
        <div>
          <strong>下载掩码 PNG</strong>
          <span>用于快速校验和展示</span>
        </div>
        <span>Open</span>
      </a>
      <a
        className="download-link"
        href={toApiUrl(`/api/jobs/${result.jobId}/mask?format=npy`)}
        target="_blank"
        rel="noreferrer"
      >
        <div>
          <strong>下载掩码 NPY</strong>
          <span>用于离线处理或进一步分析</span>
        </div>
        <span>Open</span>
      </a>
    </div>
  )
}
