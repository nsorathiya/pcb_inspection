import { ApiClientError, isApiErrorPayload } from './errors'
import type { HealthResponse } from './types'

export const REQUEST_ID_HEADER = 'X-Request-ID'
export const REQUEST_TIMEOUT_MS = 30_000

export interface ApiResponse<T> {
  data: T
  requestId: string
}

export function apiBaseUrl(environmentValue = import.meta.env.VITE_API_BASE_URL): string {
  const value = environmentValue?.trim() ?? ''
  return value === '/' ? '' : value.replace(/\/+$/, '')
}

export function resolveApiUrl(path: string, base = apiBaseUrl()): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`
  return `${base}${normalizedPath}`
}

async function readJson(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) return null
  try {
    return await response.json()
  } catch {
    return null
  }
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<ApiResponse<T>> {
  const requestId = crypto.randomUUID()
  const headers = new Headers(init.headers)
  headers.set(REQUEST_ID_HEADER, requestId)
  headers.set('Accept', 'application/json')
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  const controller = new AbortController()
  let timedOut = false
  const abortFromCaller = () => controller.abort()
  if (init.signal?.aborted) controller.abort()
  else init.signal?.addEventListener('abort', abortFromCaller, { once: true })
  const timeoutId = globalThis.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, REQUEST_TIMEOUT_MS)

  try {
    const response = await fetch(resolveApiUrl(path), {
      ...init,
      headers,
      signal: controller.signal,
    })
    const payload = await readJson(response)
    const responseRequestId =
      response.headers.get(REQUEST_ID_HEADER) ??
      (isApiErrorPayload(payload) ? payload.request_id : requestId)

    if (!response.ok) {
      if (isApiErrorPayload(payload)) {
        throw new ApiClientError(
          response.status,
          payload.code,
          payload.message,
          responseRequestId,
        )
      }
      throw new ApiClientError(
        response.status,
        `HTTP_${response.status}`,
        'The backend returned an unexpected error response.',
        responseRequestId,
      )
    }

    return { data: payload as T, requestId: responseRequestId }
  } catch (error) {
    if (error instanceof ApiClientError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiClientError(
        0,
        timedOut ? 'REQUEST_TIMEOUT' : 'REQUEST_ABORTED',
        timedOut ? 'The backend request timed out. Try again.' : 'The request was cancelled.',
        requestId,
      )
    }
    throw new ApiClientError(
      0,
      'BACKEND_UNAVAILABLE',
      'The backend is unavailable. Check the API service and try again.',
      requestId,
    )
  } finally {
    globalThis.clearTimeout(timeoutId)
    init.signal?.removeEventListener('abort', abortFromCaller)
  }
}

export function getHealth(signal?: AbortSignal): Promise<ApiResponse<HealthResponse>> {
  return apiRequest<HealthResponse>('/api/v1/health', { signal })
}
