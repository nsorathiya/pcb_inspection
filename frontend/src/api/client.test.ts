import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  apiBaseUrl,
  apiRequest,
  REQUEST_ID_HEADER,
  REQUEST_TIMEOUT_MS,
  resolveApiUrl,
} from './client'

describe('typed API client', () => {
  beforeEach(() => {
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'frontend-request-id') })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('supports same-origin and configured base URLs', () => {
    expect(apiBaseUrl(undefined)).toBe('')
    expect(apiBaseUrl(' / ')).toBe('')
    expect(apiBaseUrl('https://aoi.example.test/backend///')).toBe('https://aoi.example.test/backend')
    expect(resolveApiUrl('/api/v1/health', '')).toBe('/api/v1/health')
    expect(resolveApiUrl('api/v1/health', 'https://aoi.test')).toBe('https://aoi.test/api/v1/health')
  })

  it('generates a request ID and preserves the backend response header', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'ok' }), {
      status: 200,
      headers: { 'content-type': 'application/json', [REQUEST_ID_HEADER]: 'backend-request-id' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    const response = await apiRequest<{ status: string }>('/api/v1/health')
    const request = fetchMock.mock.calls[0]?.[1] as RequestInit
    expect(new Headers(request.headers).get(REQUEST_ID_HEADER)).toBe('frontend-request-id')
    expect(response.requestId).toBe('backend-request-id')
  })

  it('parses the structured backend error contract', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      code: 'INSPECTION_NOT_FOUND', message: 'Inspection was not found.', request_id: 'backend-error-id',
    }), { status: 404, headers: { 'content-type': 'application/json' } })))

    await expect(apiRequest('/api/v1/inspections/missing')).rejects.toEqual(
      expect.objectContaining({
        status: 404,
        code: 'INSPECTION_NOT_FOUND',
        message: 'Inspection was not found.',
        requestId: 'backend-error-id',
      }),
    )
  })

  it('maps network failures without exposing raw exceptions', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('private network detail')))
    await expect(apiRequest('/api/v1/health')).rejects.toEqual(
      expect.objectContaining({ code: 'BACKEND_UNAVAILABLE', requestId: 'frontend-request-id' }),
    )
  })

  it('maps an elapsed client timeout without exposing an abort exception', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn().mockImplementation((_url, init: RequestInit) => (
      new Promise((_resolve, reject) => {
        init.signal?.addEventListener('abort', () => {
          reject(new DOMException('private abort detail', 'AbortError'))
        })
      })
    )))

    const request = apiRequest('/api/v1/health')
    const rejection = expect(request).rejects.toEqual(expect.objectContaining({
      code: 'REQUEST_TIMEOUT',
      requestId: 'frontend-request-id',
    }))
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS)
    await rejection
  })
})
