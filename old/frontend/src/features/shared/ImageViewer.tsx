import { useEffect, useState } from 'react'

interface ImageViewerProps {
  readonly baseSrc: string | null
  readonly overlaySrc?: string | null
  readonly overlayOpacity?: number
  readonly alt: string
}

export function ImageViewer(props: ImageViewerProps) {
  const { baseSrc, overlaySrc, overlayOpacity = 0.55, alt } = props
  const [zoom, setZoom] = useState(1)
  const [overlayVisible, setOverlayVisible] = useState(Boolean(overlaySrc))

  const effectiveBaseSrc = baseSrc ?? overlaySrc ?? null
  const transform = `scale(${zoom})`

  useEffect(() => {
    setOverlayVisible(Boolean(overlaySrc))
  }, [overlaySrc])

  if (!effectiveBaseSrc) {
    return (
      <div className="viewer">
        <div className="viewer__frame" />
        <p className="muted">暂无可展示的图像。</p>
      </div>
    )
  }

  return (
    <div className="viewer">
      <div className="viewer__frame">
        <img
          className="viewer__image"
          src={effectiveBaseSrc}
          alt={alt}
          style={{ transform }}
        />
        {overlaySrc && overlayVisible ? (
          <img
            className="viewer__overlay"
            src={overlaySrc}
            alt=""
            style={{ opacity: overlayOpacity, transform }}
            onError={() => setOverlayVisible(false)}
          />
        ) : null}
      </div>

      <div className="viewer__controls">
        <button
          className="button button--secondary"
          type="button"
          onClick={() => setZoom((value) => Math.max(1, value - 0.25))}
        >
          缩小
        </button>
        <button
          className="button button--secondary"
          type="button"
          onClick={() => setZoom((value) => Math.min(3, value + 0.25))}
        >
          放大
        </button>
        <button
          className="button button--ghost"
          type="button"
          onClick={() => setZoom(1)}
        >
          重置缩放
        </button>
      </div>
    </div>
  )
}
