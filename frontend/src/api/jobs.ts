import { request } from './client'
import {
  type HealthStatus,
  type JobCreateResponse,
  type JobStatusResponse,
  normalizeHealthStatus,
  normalizeJobCreateResponse,
  normalizeJobStatusResponse,
} from '../types/job'

interface CreateJobOptions {
  readonly image: File
  readonly text: string
  readonly maxIterations: number
}

export async function createJob(
  options: CreateJobOptions,
): Promise<JobCreateResponse> {
  const formData = new FormData()
  formData.append('image', options.image)
  formData.append('text', options.text)
  formData.append('max_iter', String(options.maxIterations))

  const response = await request<unknown>('/api/jobs', {
    method: 'POST',
    body: formData,
  })

  return normalizeJobCreateResponse(response)
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  const response = await request<unknown>(`/api/jobs/${jobId}`)
  return normalizeJobStatusResponse(response)
}

export async function deleteJob(jobId: string): Promise<void> {
  await request(`/api/jobs/${jobId}`, {
    method: 'DELETE',
  })
}

export async function getHealth(): Promise<HealthStatus> {
  const response = await request<unknown>('/api/health')
  return normalizeHealthStatus(response)
}
