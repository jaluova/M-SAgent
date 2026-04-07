import { create } from 'zustand'
import { createJob } from '../api/jobs'
import {
  type IterationSnapshot,
  type JobCreateResponse,
  type JobLifecycleStatus,
  type JobResult,
  type JobStatusResponse,
  type ProgressEvent,
  type ToolName,
} from '../types/job'

interface JobStore {
  readonly jobStatus: JobLifecycleStatus
  readonly jobId: string | null
  readonly textQuery: string
  readonly maxIterations: number
  readonly events: ReadonlyArray<ProgressEvent>
  readonly currentIteration: number
  readonly currentTool: ToolName | null
  readonly iterations: ReadonlyArray<IterationSnapshot>
  readonly result: JobResult | null
  readonly errorMessage: string | null
  readonly queuePosition: number | null
  setTextQuery: (query: string) => void
  setMaxIterations: (value: number) => void
  submitJob: (image: File) => Promise<void>
  handleEvent: (event: ProgressEvent) => void
  hydrateJobStatus: (snapshot: JobStatusResponse) => void
  reset: () => void
}

interface StoreStateShape {
  readonly jobStatus: JobLifecycleStatus
  readonly jobId: string | null
  readonly textQuery: string
  readonly maxIterations: number
  readonly events: ReadonlyArray<ProgressEvent>
  readonly currentIteration: number
  readonly currentTool: ToolName | null
  readonly iterations: ReadonlyArray<IterationSnapshot>
  readonly result: JobResult | null
  readonly errorMessage: string | null
  readonly queuePosition: number | null
}

const initialState: StoreStateShape = {
  jobStatus: 'idle',
  jobId: null,
  textQuery: '',
  maxIterations: 3,
  events: [],
  currentIteration: 0,
  currentTool: null,
  iterations: [],
  result: null,
  errorMessage: null,
  queuePosition: null,
}

function ensureIteration(
  iterations: ReadonlyArray<IterationSnapshot>,
  iteration: number,
): IterationSnapshot {
  const existing = iterations.find((item) => item.iteration === iteration)

  return (
    existing ?? {
      iteration,
      tool: null,
      score: null,
      verdict: 'pending',
      checkImageUrl: null,
    }
  )
}

function upsertIteration(
  iterations: ReadonlyArray<IterationSnapshot>,
  snapshot: IterationSnapshot,
): ReadonlyArray<IterationSnapshot> {
  const next = [...iterations]
  const index = next.findIndex((item) => item.iteration === snapshot.iteration)

  if (index >= 0) {
    next[index] = snapshot
  } else {
    next.push(snapshot)
  }

  return next.sort((left, right) => left.iteration - right.iteration)
}

function applyQueuedResponse(
  state: StoreStateShape,
  response: JobCreateResponse,
): StoreStateShape {
  const events =
    typeof response.position === 'number'
      ? [...state.events, { type: 'queued', position: response.position } as const]
      : state.events

  return {
    ...state,
    jobId: response.jobId,
    jobStatus: response.status === 'idle' ? 'queued' : response.status,
    queuePosition: response.position,
    errorMessage: null,
    result: null,
    events,
  }
}

function applyEvent(
  state: StoreStateShape,
  event: ProgressEvent,
  appendEvent = true,
): StoreStateShape {
  const nextEvents = appendEvent ? [...state.events, event] : state.events

  switch (event.type) {
    case 'queued':
      return {
        ...state,
        events: nextEvents,
        jobStatus: 'queued',
        queuePosition: event.position,
      }
    case 'started':
      return {
        ...state,
        events: nextEvents,
        jobStatus: 'running',
        queuePosition: null,
      }
    case 'iteration_start':
      return {
        ...state,
        events: nextEvents,
        jobStatus: 'running',
        currentIteration: event.iteration,
      }
    case 'tool_selected': {
      const base = ensureIteration(state.iterations, event.iteration)
      return {
        ...state,
        events: nextEvents,
        jobStatus: 'running',
        currentIteration: event.iteration,
        currentTool: event.tool,
        iterations: upsertIteration(state.iterations, {
          ...base,
          tool: event.tool,
        }),
      }
    }
    case 'segmentation_result': {
      const base = ensureIteration(state.iterations, event.iteration)
      return {
        ...state,
        events: nextEvents,
        currentIteration: event.iteration,
        currentTool: event.tool,
        iterations: upsertIteration(state.iterations, {
          ...base,
          tool: event.tool,
          score: event.score,
          checkImageUrl: event.checkImageUrl,
        }),
      }
    }
    case 'evaluation': {
      const base = ensureIteration(state.iterations, event.iteration)
      return {
        ...state,
        events: nextEvents,
        iterations: upsertIteration(state.iterations, {
          ...base,
          verdict: event.verdict,
        }),
      }
    }
    case 'complete':
      return {
        ...state,
        events: nextEvents,
        jobStatus: 'complete',
        queuePosition: null,
        currentIteration: event.result.iterations,
        result: event.result,
      }
    case 'error':
      return {
        ...state,
        events: nextEvents,
        jobStatus: 'failed',
        errorMessage: event.message,
      }
    default:
      return state
  }
}

export const useJobStore = create<JobStore>((set, get) => ({
  ...initialState,

  setTextQuery: (query) => set({ textQuery: query }),

  setMaxIterations: (value) =>
    set({
      maxIterations: Math.max(1, Math.min(5, Math.round(value))),
    }),

  submitJob: async (image) => {
    const { textQuery, maxIterations } = get()
    const normalizedQuery = textQuery.trim()

    if (!normalizedQuery) {
      set({
        jobStatus: 'failed',
        errorMessage: '请输入需要分割的目标描述。',
      })
      return
    }

    set({
      ...initialState,
      textQuery: normalizedQuery,
      maxIterations,
      jobStatus: 'uploading',
    })

    try {
      const response = await createJob({
        image,
        text: normalizedQuery,
        maxIterations,
      })

      set((state) => applyQueuedResponse(state, response))
    } catch (error) {
      set({
        ...initialState,
        textQuery: normalizedQuery,
        maxIterations,
        jobStatus: 'failed',
        errorMessage:
          error instanceof Error ? error.message : '创建任务失败，请稍后重试。',
      })
    }
  },

  handleEvent: (event) => set((state) => applyEvent(state, event, true)),

  hydrateJobStatus: (snapshot) =>
    set((state) => {
      let nextState: StoreStateShape = {
        ...state,
        jobId: snapshot.jobId || state.jobId,
        jobStatus: snapshot.status,
        queuePosition: snapshot.position,
        events: snapshot.events.length > 0 ? snapshot.events : state.events,
        currentIteration:
          snapshot.currentIteration > 0
            ? snapshot.currentIteration
            : state.currentIteration,
        currentTool: snapshot.currentTool ?? state.currentTool,
        errorMessage: snapshot.error ?? state.errorMessage,
        result: snapshot.result ?? state.result,
        iterations: snapshot.events.length > 0 ? [] : state.iterations,
      }

      for (const event of snapshot.events) {
        nextState = applyEvent(nextState, event, false)
      }

      return nextState
    }),

  reset: () => set(initialState),
}))
