import { useEffect, useState } from 'react'
import { UploadPanel } from './features/upload/UploadPanel'
import { ProgressPanel } from './features/progress/ProgressPanel'
import { ResultPanel } from './features/result/ResultPanel'
import { StatusIndicator } from './features/shared/StatusIndicator'
import { MainLayout } from './layouts/MainLayout'
import { useJob } from './hooks/useJob'

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
    health,
    healthError,
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
      health={health}
      healthError={healthError}
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
          <section className="panel panel--compact">
            <div className="panel__header">
              <div>
                <p className="eyebrow">任务状态</p>
                <h2>运行看板</h2>
              </div>
              <StatusIndicator status={jobStatus} />
            </div>
            <p className="muted">
              上传图片并输入 referring expression 后，前端会通过 REST 创建任务，再通过
              WebSocket 持续接收 Agent Loop 的进度更新。
            </p>
          </section>

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
              <p className="eyebrow">等待任务</p>
              <h2>这里会实时展示推理进度</h2>
              <p className="muted">
                当任务开始后，你会看到每轮迭代选择的工具、分割评分、评估结论以及最终结果。
              </p>
            </section>
          )}

          {result ? (
            <ResultPanel result={result} sourceImageUrl={previewUrl} />
          ) : (
            <section className="panel panel--empty">
              <p className="eyebrow">结果区域</p>
              <h2>完成后在这里查看掩码与结果图</h2>
              <p className="muted">
                结果面板会提供叠加预览、指标摘要，以及结果图和掩码的下载入口。
              </p>
            </section>
          )}
        </>
      }
    />
  )
}

export default App
