import { apiRequest, type ApiResponse } from './client'
import type {
  HistoryFilters,
  InspectionDetailResponse,
  InspectionHistoryResponse,
  InspectionIntakeResponse,
  InspectionProcessingResponse,
  InspectionValidationResponse,
  RecipeCatalogueItem,
} from './types'

export const VALIDATION_POLICY = {
  policy_id: 'development-native-rgb-height',
  policy_version: '1.0',
} as const

export const PROCESSING_POLICY = {
  preprocessing_policy_id: 'synthetic-paired-rgb-height',
  preprocessing_policy_version: '1.0',
  inference_policy_id: 'synthetic-deterministic-mock-inference',
  inference_policy_version: '1.0',
} as const

function appendQueryValue(parameters: URLSearchParams, key: string, value: unknown): void {
  if (value === undefined || value === null || value === '') return
  parameters.set(key, String(value))
}

export function getInspectionHistory(
  filters: HistoryFilters,
  cursor?: string,
  limit = 25,
  signal?: AbortSignal,
): Promise<ApiResponse<InspectionHistoryResponse>> {
  const parameters = new URLSearchParams({ limit: String(limit) })
  Object.entries(filters).forEach(([key, value]) => appendQueryValue(parameters, key, value))
  appendQueryValue(parameters, 'cursor', cursor)
  return apiRequest(`/api/v1/inspections?${parameters.toString()}`, { signal })
}

export interface NewInspectionInput {
  boardId: string
  recipe: RecipeCatalogueItem
  rgbImage: File
  heightMap: File
  lotId?: string
  operatorId?: string
  stationId?: string
}

export function buildInspectionFormData(input: NewInspectionInput): FormData {
  const form = new FormData()
  form.set('board_id', input.boardId)
  form.set('recipe_id', input.recipe.recipe_id)
  form.set('recipe_version', input.recipe.recipe_version)
  form.set('rgb_image', input.rgbImage)
  form.set('height_map', input.heightMap)
  const optionalValues = {
    lot_id: input.lotId,
    operator_id: input.operatorId,
    station_id: input.stationId,
  }
  Object.entries(optionalValues).forEach(([key, value]) => {
    const normalized = value?.trim()
    if (normalized) form.set(key, normalized)
  })
  return form
}

export function createInspection(
  input: NewInspectionInput,
  signal?: AbortSignal,
): Promise<ApiResponse<InspectionIntakeResponse>> {
  return apiRequest('/api/v1/inspections', {
    method: 'POST',
    body: buildInspectionFormData(input),
    signal,
  })
}

export function getInspection(
  inspectionId: string,
  signal?: AbortSignal,
): Promise<ApiResponse<InspectionDetailResponse>> {
  return apiRequest(`/api/v1/inspections/${inspectionId}`, { signal })
}

export function getValidation(
  inspectionId: string,
  signal?: AbortSignal,
): Promise<ApiResponse<InspectionValidationResponse>> {
  return apiRequest(`/api/v1/inspections/${inspectionId}/validation`, { signal })
}

export function runValidation(
  inspectionId: string,
  signal?: AbortSignal,
): Promise<ApiResponse<InspectionValidationResponse>> {
  return apiRequest(`/api/v1/inspections/${inspectionId}/validate`, {
    method: 'POST',
    body: JSON.stringify(VALIDATION_POLICY),
    signal,
  })
}

export function getProcessing(
  inspectionId: string,
  signal?: AbortSignal,
): Promise<ApiResponse<InspectionProcessingResponse>> {
  return apiRequest(`/api/v1/inspections/${inspectionId}/processing`, { signal })
}

export function runProcessing(
  inspectionId: string,
  signal?: AbortSignal,
): Promise<ApiResponse<InspectionProcessingResponse>> {
  return apiRequest(`/api/v1/inspections/${inspectionId}/process`, {
    method: 'POST',
    body: JSON.stringify(PROCESSING_POLICY),
    signal,
  })
}
