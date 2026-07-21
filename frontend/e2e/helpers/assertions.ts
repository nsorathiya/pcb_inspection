import type { Page } from '@playwright/test'
import { expect } from '../fixtures/test'

export function collectKeys(value: unknown, result = new Set<string>()): Set<string> {
  if (Array.isArray(value)) value.forEach((item) => collectKeys(item, result))
  else if (value && typeof value === 'object') {
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      result.add(key)
      collectKeys(item, result)
    }
  }
  return result
}

export async function assertNoDuplicateIds(page: Page): Promise<void> {
  const duplicates = await page.locator('[id]').evaluateAll((elements) => {
    const counts = new Map<string, number>()
    elements.forEach((element) => {
      const id = (element as HTMLElement).id
      counts.set(id, (counts.get(id) ?? 0) + 1)
    })
    return [...counts.entries()].filter(([, count]) => count > 1)
  })
  expect(duplicates).toEqual([])
}
