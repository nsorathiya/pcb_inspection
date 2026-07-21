import { apiRequest, type ApiResponse } from './client'
import type { DemoWorkspaceResponse } from './types'

export function getDemoWorkspace(
  signal?: AbortSignal,
): Promise<ApiResponse<DemoWorkspaceResponse>> {
  return apiRequest('/api/v1/development/demo-workspace', { signal })
}

export function loadDemoWorkspace(
  signal?: AbortSignal,
): Promise<ApiResponse<DemoWorkspaceResponse>> {
  return apiRequest('/api/v1/development/demo-workspace/load', {
    method: 'POST',
    signal,
  })
}
