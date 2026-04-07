import { useEffect, useState } from 'react'
import { UploadPanel } from './features/upload/UploadPanel'
import { ProgressPanel } from './features/progress/ProgressPanel'
import { LogPanel } from './features/progress/LogPanel'
import { ResultPanel } from './features/result/ResultPanel'
import { MainLayout } from './layouts/MainLayout'
import { useJob } from './hooks/useJob'

const PREVIEW_MAX_DIMENSION = 1600

function preloadImage(src: string | null) {
  if (!src) {
    return
  }

  const image = new Image()
  image.decoding = 'async'
  image.src = src
}

function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.decoding = 'async'
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error(`Failed to load image: ${src}`))
    image.src = src
  })
}

async function createPreviewUrl(
  sourceUrl: string,
  fileType: string,
): Promise<string | null> {
  try {
    const image = await loadImageElement(sourceUrl)
    const { naturalWidth, naturalHeight } = image

    if (!naturalWidth || !naturalHeight) {
      return null
    }

    const scale = Math.min(
      1,
      PREVIEW_MAX_DIMENSION / Math.max(naturalWidth, naturalHeight),
    )

    if (scale >= 0.999) {
      return null
    }

    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(naturalWidth * scale))
    canvas.height = Math.max(1, Math.round(naturalHeight * scale))

    const context = canvas.getContext('2d')
    if (!context) {
      return null
    }

    context.drawImage(image, 0, 0, canvas.width, canvas.height)

    const previewBlob = await new Promise<Blob | null>((resolve) => {
      const preferredType = fileType === 'image/png' ? 'image/png' : 'image/jpeg'
      const quality = preferredType === 'image/png' ? undefined : 0.9
      canvas.toBlob(resolve, preferredType, quality)
    })

    if (!previewBlob) {
      return null
    }

    return URL.createObjectURL(previewBlob)
  } catch {
    return null
  }
}

function App() {
  const {
    jobStatus,
    errorMessage,
    queuePosition,
    textQuery,
    maxIterations,
    events,
    currentIteration,
    currentTool,
    iterations,
    result,
    setTextQuery,
    setMaxIterations,
    submitJob,
    reset,
  } = useJob()

  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [sourceImageUrl, setSourceImageUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null)
      setSourceImageUrl(null)
      return
    }

    let revoked = false
    const originalUrl = URL.createObjectURL(selectedFile)
    let currentPreviewUrl: string | null = null

    setSourceImageUrl(originalUrl)

    void createPreviewUrl(originalUrl, selectedFile.type).then((nextUrl) => {
      if (revoked) {
        if (nextUrl) {
          URL.revokeObjectURL(nextUrl)
        }
        return
      }

      currentPreviewUrl = nextUrl
      setPreviewUrl(nextUrl ?? originalUrl)
    })

    return () => {
      revoked = true
      URL.revokeObjectURL(originalUrl)
      if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl)
      }
    }
  }, [selectedFile])

  useEffect(() => {
    if (!result) {
      return
    }

    preloadImage(result.maskUrl)
    preloadImage(result.resultPreviewUrl ?? result.resultImageUrl)
  }, [result])

  const handleSubmit = async () => {
    if (!selectedFile) {
      return
    }

    await submitJob(selectedFile)
  }

  const handleReset = () => {
    setSelectedFile(null)
    reset()
  }

  const showProgress =
    jobStatus !== 'idle' ||
    iterations.length > 0 ||
    events.length > 0 ||
    Boolean(errorMessage)

  return (
    <MainLayout
      leftColumn={
        <UploadPanel
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          textQuery={textQuery}
          maxIterations={maxIterations}
          jobStatus={jobStatus}
          queuePosition={queuePosition}
          errorMessage={errorMessage}
          onFileChange={setSelectedFile}
          onTextChange={setTextQuery}
          onMaxIterationsChange={setMaxIterations}
          onSubmit={handleSubmit}
          onReset={handleReset}
        />
      }
      centerColumn={
        showProgress ? (
          <ProgressPanel
            jobStatus={jobStatus}
            queuePosition={queuePosition}
            currentIteration={currentIteration}
            currentTool={currentTool}
            iterations={iterations}
          />
        ) : (
          <section className="panel panel--empty">
            <h2>中间产物会显示在这里</h2>
          </section>
        )
      }
      rightColumn={<LogPanel events={events} />}
      bottomSection={
        result ? (
          <ResultPanel result={result} sourceImageUrl={sourceImageUrl} />
        ) : (
          <section className="panel panel--empty">
            <h2>可视化成果会显示在这里</h2>
          </section>
        )
      }
    />
  )
}

export default App
