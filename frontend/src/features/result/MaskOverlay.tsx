import { useState } from 'react'
import { type JobResult } from '../../types/job'
import { ImageViewer } from '../shared/ImageViewer'

interface MaskOverlayProps {
  readonly result: JobResult
  readonly sourceImageUrl: string | null
}

export function MaskOverlay(props: MaskOverlayProps) {
  const { result, sourceImageUrl } = props
  const [overlayOpacity, setOverlayOpacity] = useState(0.55)
  const [mode, setMode] = useState<'mask' | 'result'>('mask')

  const overlaySrc = mode === 'mask' ? result.maskUrl : null
  const resultPreviewSrc = result.resultPreviewUrl ?? result.resultImageUrl
  const maskBaseSrc = resultPreviewSrc ?? sourceImageUrl
  const baseSrc = mode === 'mask' ? maskBaseSrc : resultPreviewSrc

  return (
    <div className="stack">
      <div className="viewer__controls">
        <button
          className={`button ${mode === 'mask' ? 'button--primary' : 'button--secondary'}`}
          type="button"
          onClick={() => setMode('mask')}
        >
          掩码叠加
        </button>
        <button
          className={`button ${mode === 'result' ? 'button--primary' : 'button--secondary'}`}
          type="button"
          onClick={() => setMode('result')}
        >
          结果图预览
        </button>
      </div>

      <div className="slider-row">
        <label htmlFor="overlayOpacity">叠加透明度</label>
        <strong>{Math.round(overlayOpacity * 100)}%</strong>
      </div>
      <input
        id="overlayOpacity"
        className="slider-input"
        type="range"
        min={0.1}
        max={0.95}
        step={0.05}
        value={overlayOpacity}
        onChange={(event) => setOverlayOpacity(Number(event.target.value))}
      />

      <ImageViewer
        alt="segmentation result"
        baseSrc={baseSrc}
        overlaySrc={overlaySrc}
        overlayOpacity={overlayOpacity}
      />
    </div>
  )
}
