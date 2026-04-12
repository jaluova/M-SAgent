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
          <p className="eyebrow">输入</p>
          <h2>新任务</h2>
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
            <strong>选择图片</strong>
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
        <label htmlFor="query">指代表达</label>
        <textarea
          id="query"
          placeholder="例如：右侧那辆红色卡车"
          value={textQuery}
          onChange={(event) => onTextChange(event.target.value)}
        />
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
            <strong>排队中</strong>
            <span className="muted">第 {queuePosition} 位</span>
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
      </div>
    </section>
  )
}
