interface ImagePreviewProps {
  readonly previewUrl: string | null
  readonly file: File | null
}

function formatBytes(size: number): string {
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`
  }

  return `${(size / (1024 * 1024)).toFixed(2)} MB`
}

export function ImagePreview(props: ImagePreviewProps) {
  const { previewUrl, file } = props

  if (!previewUrl || !file) {
    return (
      <div className="note">
        <strong>尚未选择图片</strong>
        <span className="muted">支持点击上传或直接拖拽图片文件到上方区域。</span>
      </div>
    )
  }

  return (
    <div className="image-preview">
      <div className="image-preview__frame">
        <img src={previewUrl} alt={file.name} />
      </div>
      <div className="image-preview__meta">
        <div>
          <strong>{file.name}</strong>
          <div>{file.type || 'image/*'}</div>
        </div>
        <div>{formatBytes(file.size)}</div>
      </div>
    </div>
  )
}
