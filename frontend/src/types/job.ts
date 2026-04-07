import { toApiUrl } from '../api/client'

export type ToolName =
  | 'object_locator'
  | 'concept_generator'
  | 'image_enhancer'
  | 'report_no_mask'

export type Verdict = 'Accept' | 'Reject' | 'pending'

export type JobLifecycleStatus =
  | 'idle'
  | 'uploading'
  | 'queued'
  | 'running'
  | 'complete'
  | 'failed'

export interface JobResult {
  readonly jobId: string
  readonly success: boolean
  readonly bestScore: number
  readonly iterations: number
  readonly maskCount: number
  readonly resultImageUrl: string
  readonly maskUrl: string
}

export type ProgressEvent =
  | { readonly type: 'queued'; readonly position: number }
  | { readonly type: 'started' }
  | {
      readonly type: 'iteration_start'
      readonly iteration: number
      readonly maxIterations: number
    }
  | {
      readonly type: 'tool_selected'
      readonly iteration: number
      readonly tool: ToolName
      readonly params: Record<string, unknown>
    }
  | {
      readonly type: 'segmentation_result'
      readonly iteration: number
      readonly tool: ToolName
      readonly score: number
      readonly checkImageUrl: string
    }
  | {
      readonly type: 'evaluation'
      readonly iteration: number
      readonly verdict: 'Accept' | 'Reject'
      readonly rejectedIndices: readonly number[]
    }
  | { readonly type: 'complete'; readonly result: JobResult }
  | { readonly type: 'error'; readonly message: string }

export interface IterationSnapshot {
  readonly iteration: number
  readonly tool: ToolName | null
  readonly score: number | null
  readonly verdict: Verdict
  readonly checkImageUrl: string | null
}

export interface JobCreateResponse {
  readonly jobId: string
  readonly status: JobLifecycleStatus
  readonly position: number | null
}

export interface JobStatusResponse {
  readonly jobId: string
  readonly status: JobLifecycleStatus
  readonly position: number | null
  readonly error: string | null
  readonly result: JobResult | null
  readonly currentIteration: number
  readonly currentTool: ToolName | null
  readonly events: ReadonlyArray<ProgressEvent>
}

export interface HealthStatus {
  readonly ok: boolean
  readonly gpu: string | null
  readonly queueSize: number | null
  readonly modelLoaded: boolean | null
  readonly detail: string | null
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {}
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback
}

function asOptionalString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null
}

function resolveUrl(url: unknown, fallbackPath: string): string {
  const candidate = asString(url)

  if (!candidate) {
    return toApiUrl(fallbackPath)
  }

  if (/^https?:\/\//.test(candidate)) {
    return candidate
  }

  return toApiUrl(candidate.startsWith("/") ? candidate : `/${candidate}`)
}

export function normalizeToolName(value: unknown): ToolName | null {
  if (
    value === 'object_locator' ||
    value === 'concept_generator' ||
    value === 'image_enhancer' ||
    value === 'report_no_mask'
  ) {
    return value
  }

  return null
}

export function normalizeJobStatus(value: unknown): JobLifecycleStatus {
  switch (value) {
    case 'uploading':
      return 'uploading'
    case 'queued':
    case 'pending':
      return 'queued'
    case 'running':
    case 'started':
    case 'processing':
      return 'running'
    case 'complete':
    case 'completed':
    case 'done':
      return 'complete'
    case 'failed':
    case 'error':
      return 'failed'
    default:
      return 'idle'
  }
}

export function buildJobResult(jobId: string, raw: unknown): JobResult {
  const record = asRecord(raw)

  return {
    jobId,
    success: Boolean(record.success ?? true),
    bestScore: asNumber(record.bestScore ?? record.best_score),
    iterations: asNumber(record.iterations),
    maskCount: asNumber(record.maskCount ?? record.mask_count),
    resultImageUrl: resolveUrl(
      record.resultImageUrl ?? record.result_image_url,
      `/api/jobs/${jobId}/result`,
    ),
    maskUrl: resolveUrl(
      record.maskUrl ?? record.mask_url,
      `/api/jobs/${jobId}/mask?format=png`,
    ),
  }
}

export function normalizeProgressEvent(
  raw: unknown,
  jobId: string,
): ProgressEvent | null {
  const record = asRecord(raw)
  const type = asString(record.type)
  const iteration = asNumber(record.iteration)
  const tool = normalizeToolName(record.tool)

  switch (type) {
    case 'queued':
      return {
        type: 'queued',
        position: asNumber(record.position, 1),
      }
    case 'started':
      return { type: 'started' }
    case 'iteration_start':
      return {
        type: 'iteration_start',
        iteration,
        maxIterations: asNumber(record.maxIterations ?? record.max_iterations, 5),
      }
    case 'tool_selected':
      if (!tool) {
        return null
      }
      return {
        type: 'tool_selected',
        iteration,
        tool,
        params: asRecord(record.params),
      }
    case 'segmentation_result':
      if (!tool) {
        return null
      }
      return {
        type: 'segmentation_result',
        iteration,
        tool,
        score: asNumber(record.score),
        checkImageUrl: resolveUrl(
          record.checkImageUrl ?? record.check_image_url,
          `/api/jobs/${jobId}/images/check-${iteration}.jpg`,
        ),
      }
    case 'evaluation': {
      const rejected = record.rejectedIndices ?? record.rejected_indices

      return {
        type: 'evaluation',
        iteration,
        verdict: record.verdict === 'Accept' ? 'Accept' : 'Reject',
        rejectedIndices: Array.isArray(rejected)
          ? rejected.filter((value): value is number => typeof value === 'number')
          : [],
      }
    }
    case 'complete':
      return {
        type: 'complete',
        result: buildJobResult(jobId, record.result ?? record),
      }
    case 'error':
      return {
        type: 'error',
        message: asString(record.message, '任务执行失败，请检查后端日志。'),
      }
    default:
      return null
  }
}

export function normalizeJobCreateResponse(raw: unknown): JobCreateResponse {
  const record = asRecord(raw)
  return {
    jobId: asString(record.jobId ?? record.job_id),
    status: normalizeJobStatus(record.status),
    position:
      typeof (record.position ?? record.queue_position) === 'number'
        ? asNumber(record.position ?? record.queue_position)
        : null,
  }
}

export function normalizeJobStatusResponse(raw: unknown): JobStatusResponse {
  const record = asRecord(raw)
  const jobId = asString(record.jobId ?? record.job_id)
  const status = normalizeJobStatus(record.status)
  const rawEvents = Array.isArray(record.events) ? record.events : []

  return {
    jobId,
    status,
    position:
      typeof (record.position ?? record.queue_position) === 'number'
        ? asNumber(record.position ?? record.queue_position)
        : null,
    error: asOptionalString(record.error ?? record.message),
    result:
      status === 'complete' || record.result
        ? buildJobResult(jobId, record.result ?? record)
        : null,
    currentIteration: asNumber(
      record.currentIteration ?? record.current_iteration,
    ),
    currentTool: normalizeToolName(record.currentTool ?? record.current_tool),
    events: rawEvents
      .map((event) => normalizeProgressEvent(event, jobId))
      .filter((event): event is ProgressEvent => event !== null),
  }
}

export function normalizeHealthStatus(raw: unknown): HealthStatus {
  const record = asRecord(raw)
  return {
    ok: Boolean(record.ok ?? record.healthy ?? false),
    gpu: asOptionalString(record.gpu ?? record.device),
    queueSize:
      typeof (record.queueSize ?? record.queue_size) === 'number'
        ? asNumber(record.queueSize ?? record.queue_size)
        : null,
    modelLoaded:
      typeof (record.modelLoaded ?? record.model_loaded) === 'boolean'
        ? Boolean(record.modelLoaded ?? record.model_loaded)
        : null,
    detail: asOptionalString(record.detail ?? record.message),
  }
}
