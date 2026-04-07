import { type DragEventHandler, useRef, useState } from 'react'
import { ImagePreview } from './ImagePreview'
import { StatusIndicator } from '../shared/StatusIndicator'
import { type JobLifecycleStatus } from '../../types/job'

interface UploadPanelProps {
  readonly selectedFile: File | null
  readonly previewUrl: string | null
  readonly textQuery: string
  readonly maxIterations: number
  readonly jobStatus: JobLifecycleStatus
  readonly queuePosition: number | null
  readonly errorMessage: string | null
  readonly onFileChange: (file: File | null) => void
  readonly onTextChange: (value: string) => void
  readonly onMaxIterationsChange: (value: number) => void
  readonly onSubmit: () => Promise<void>
  readonly onReset: () => void
}

export function UploadPanel(props: UploadPanelProps) {
  const {
    selectedFile,
    previewUrl,
    textQuery,
    maxIterations,
    jobStatus,
    queuePosition,
    errorMessage,
    onFileChange,
    onTextChange,
    onMaxIterationsChange,
    onSubmit,
    onReset,
  } = props

  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const isBusy = ['uploading', 'queued', 'running'].includes(jobStatus)

  const pickFile = () => inputRef.current?.click()

  const handleDrop: DragEventHandler<HTMLDivElement> = (event) => {
    event.preventDefault()
    setIsDragging(false)

    const file = event.dataTransfer.files?.[0]
    if (file) {
      onFileChange(file)
    }
  }

  return (
    <section className="panel">
      <div className="panel__header">
        <div>
          <p className="eyebrow">输入区</p>
          <h2>上传图片并发起任务</h2>
        </div>
        <StatusIndicator status={jobStatus} />
      </div>

      <div
        className={`upload-dropzone ${isDragging ? 'is-dragging' : ''}`}
        onDragEnter={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          event.preventDefault()
          setIsDragging(false)
        }}
        onDrop={handleDrop}
      >
        <div className="upload-dropzone__cta">
          <div className="upload-dropzone__icon">IMG</div>
          <div>
            <strong>选择待分割图片</strong>
            <p className="muted">
              拖拽图片到这里，或点击按钮选择本地文件。推荐使用 example 目录中的样例图快速验证流程。
            </p>
          </div>
        </div>

        <div className="upload-dropzone__actions">
          <button className="button button--primary" type="button" onClick={pickFile}>
            选择图片
          </button>
          <button className="button button--secondary" type="button" onClick={onReset}>
            重置界面
          </button>
        </div>

        <input
          ref={inputRef}
          hidden
          accept="image/*"
          type="file"
          onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
        />
      </div>

      <div className="field-group">
        <ImagePreview previewUrl={previewUrl} file={selectedFile} />
      </div>

      <div className="field-group">
        <label htmlFor="query">Referring Expression</label>
        <textarea
          id="query"
          placeholder="例如：the red truck on the right"
          value={textQuery}
          onChange={(event) => onTextChange(event.target.value)}
        />
        <p className="helper-text">
          尽量描述颜色、位置、数量或属性，能帮助 MLLM 更快收敛。
        </p>
      </div>

      <div className="field-group">
        <div className="slider-row">
          <label htmlFor="maxIterations">最大迭代次数</label>
          <strong>{maxIterations}</strong>
        </div>
        <input
          id="maxIterations"
          className="slider-input"
          type="range"
          min={1}
          max={5}
          step={1}
          value={maxIterations}
          onChange={(event) => onMaxIterationsChange(Number(event.target.value))}
        />
      </div>

      {queuePosition ? (
        <div className="note-list">
          <div className="note">
            <strong>当前排队位置</strong>
            <span className="muted">队列第 {queuePosition} 位，前端会自动监听状态变化。</span>
          </div>
        </div>
      ) : null}

      {errorMessage ? <div className="alert">{errorMessage}</div> : null}

      <div className="upload-dropzone__actions">
        <button
          className="button button--primary"
          type="button"
          disabled={!selectedFile || !textQuery.trim() || isBusy}
          onClick={() => void onSubmit()}
        >
          {jobStatus === 'uploading' ? '上传中...' : '开始分割'}
        </button>
        <span className="helper-text">
          任务提交后会自动建立 WebSocket 连接，实时刷新每一轮推理结果。
        </span>
      </div>
    </section>
  )
}
