import type { ApiErrorPayload } from './types'

export class ApiClientError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly requestId: string,
  ) {
    super(message)
    this.name = 'ApiClientError'
  }
}

export function isApiErrorPayload(value: unknown): value is ApiErrorPayload {
  if (!value || typeof value !== 'object') return false
  const candidate = value as Record<string, unknown>
  return (
    typeof candidate.code === 'string' &&
    typeof candidate.message === 'string' &&
    typeof candidate.request_id === 'string'
  )
}

export function toApiClientError(error: unknown): ApiClientError {
  if (error instanceof ApiClientError) return error
  return new ApiClientError(
    0,
    'UNEXPECTED_CLIENT_ERROR',
    'The operator interface could not complete the request.',
    '-',
  )
}
