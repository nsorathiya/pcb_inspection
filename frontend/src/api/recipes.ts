import { apiRequest, type ApiResponse } from './client'
import type { RecipeCatalogueResponse, RecipeFilters } from './types'

export function getRecipes(
  filters: RecipeFilters = {},
  cursor?: string,
  limit = 25,
  signal?: AbortSignal,
): Promise<ApiResponse<RecipeCatalogueResponse>> {
  const parameters = new URLSearchParams({ limit: String(limit) })
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== '') parameters.set(key, value)
  })
  if (cursor) parameters.set('cursor', cursor)
  return apiRequest(`/api/v1/recipes?${parameters.toString()}`, { signal })
}
