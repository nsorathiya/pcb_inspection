import { apiRequest, resolveApiUrl, type ApiResponse } from './client'
import type { EngineeringHeightRoiResponse, EngineeringSampleResponse, EngineeringViewResponse } from './types'
import type { HeightPreviewPalette } from '../utils/engineeringSession'

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

export function engineeringPreviewUrl(
  inspectionId: string,
  kind: 'rgb' | 'height',
  options?: {
    palette: HeightPreviewPalette
    displayMin: number | null
    displayMax: number | null
    showInvalid: boolean
  },
): string {
  const path = `/api/v1/inspections/${inspectionId}/engineering-view/${kind}-preview`
  if (kind === 'rgb' || !options) return resolveApiUrl(path)
  const query = new URLSearchParams({
    palette: options.palette,
    show_invalid: String(options.showInvalid),
  })
  if (options.displayMin !== null && options.displayMax !== null) {
    query.set('display_min', String(options.displayMin))
    query.set('display_max', String(options.displayMax))
  }
  return resolveApiUrl(`${path}?${query}`)
}

export function getEngineeringHeightRoi(
  inspectionId: string,
  roi: { x: number; y: number; width: number; height: number },
  signal?: AbortSignal,
): Promise<ApiResponse<EngineeringHeightRoiResponse>> {
  const query = new URLSearchParams({
    x: String(roi.x),
    y: String(roi.y),
    width: String(roi.width),
    height: String(roi.height),
  })
  return apiRequest(`/api/v1/inspections/${inspectionId}/engineering-view/height-roi?${query}`, { signal })
}
