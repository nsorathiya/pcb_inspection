import { apiRequest, resolveApiUrl, type ApiResponse } from './client'
import type { EngineeringSampleResponse, EngineeringViewResponse } from './types'

export function getEngineeringView(
  inspectionId: string,
  signal?: AbortSignal,
): Promise<ApiResponse<EngineeringViewResponse>> {
  return apiRequest(`/api/v1/inspections/${inspectionId}/engineering-view`, { signal })
}

export function getEngineeringSample(
  inspectionId: string,
  coordinates: { rgbX: number; rgbY: number; heightX: number; heightY: number },
  signal?: AbortSignal,
): Promise<ApiResponse<EngineeringSampleResponse>> {
  const query = new URLSearchParams({
    rgb_x: String(coordinates.rgbX),
    rgb_y: String(coordinates.rgbY),
    height_x: String(coordinates.heightX),
    height_y: String(coordinates.heightY),
  })
  return apiRequest(`/api/v1/inspections/${inspectionId}/engineering-view/sample?${query}`, { signal })
}

export function engineeringPreviewUrl(inspectionId: string, kind: 'rgb' | 'height'): string {
  return resolveApiUrl(`/api/v1/inspections/${inspectionId}/engineering-view/${kind}-preview`)
}
