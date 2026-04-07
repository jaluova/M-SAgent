import { useEffect, useState } from 'react'
import { UploadPanel } from './features/upload/UploadPanel'
import { ProgressPanel } from './features/progress/ProgressPanel'
import { ResultPanel } from './features/result/ResultPanel'
import { MainLayout } from './layouts/MainLayout'
import { useJob } from './hooks/useJob'

function preloadImage(src: string | null) {
  if (!src) {
    return
  }

  const image = new Image()
  image.decoding = 'async'
  image.src = src
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

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null)
      return
    }

    const objectUrl = URL.createObjectURL(selectedFile)
    setPreviewUrl(objectUrl)

    return () => URL.revokeObjectURL(objectUrl)
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
      rightColumn={
        <>
          {showProgress ? (
            <ProgressPanel
              jobStatus={jobStatus}
              queuePosition={queuePosition}
              currentIteration={currentIteration}
              currentTool={currentTool}
              iterations={iterations}
              events={events}
            />
          ) : (
            <section className="panel panel--empty">
              <h2>等待任务</h2>
            </section>
          )}

          {result ? (
            <ResultPanel result={result} sourceImageUrl={previewUrl} />
          ) : (
            <section className="panel panel--empty">
              <h2>结果会显示在这里</h2>
            </section>
          )}
        </>
      }
    />
  )
}

export default App
