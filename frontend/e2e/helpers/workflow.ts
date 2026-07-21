import type { Page, Response } from '@playwright/test'
import { expect } from '../fixtures/test'
import { scenarioFiles, type RuntimeState } from './runtime'

export interface IntakePayload {
  inspection_id: string
  status: string
  request_id: string
  artifacts: Array<{ artifact_type: string; sha256: string; byte_size: number }>
}

export async function createInspection(
  page: Page,
  state: RuntimeState,
  scenarioId: string,
  boardId: string,
): Promise<IntakePayload> {
  const scenario = scenarioFiles(state, scenarioId)
  if (scenario.record.expected_intake_outcome !== 'ACCEPTED') {
    throw new Error(`Scenario ${scenarioId} is not authoritative for accepted paired intake.`)
  }
  await page.goto(`${state.frontendUrl}/inspections/new`)
  await expect(page.getByRole('heading', { name: 'New Inspection' })).toBeVisible()
  const radios = page.getByRole('radio')
  await expect(radios).toHaveCount(2)
  await expect(page.getByText(/synthetic-e2e .* 1\.0/i)).toBeVisible()
  await expect(page.getByText(/synthetic-e2e .* 0\.9/i)).toBeVisible()
  await page.getByLabel(/Board ID/).fill(boardId)
  await page.getByRole('radio', { name: /synthetic-e2e .* 1\.0/i }).check()
  await page.getByLabel(/RGB image/).setInputFiles(scenario.rgb)
  await page.getByLabel(/Height \/ depth map/).setInputFiles(scenario.height)
  const responsePromise = page.waitForResponse((response) => (
    response.request().method() === 'POST' &&
    new URL(response.url()).pathname === '/api/v1/inspections'
  ))
  await page.getByRole('button', { name: 'Create inspection' }).click()
  const response = await responsePromise
  expect(response.status()).toBe(201)
  const payload = await response.json() as IntakePayload
  await expect(page).toHaveURL(new RegExp(`/inspections/${payload.inspection_id}$`))
  await expect(page.getByText(`Inspection ${payload.inspection_id} created with status RECEIVED.`)).toBeVisible()
  return payload
}

async function runAction(page: Page, name: string, suffix: string): Promise<Response> {
  const requests: string[] = []
  const observe = (request: import('@playwright/test').Request) => {
    if (request.method() === 'POST' && new URL(request.url()).pathname.endsWith(suffix)) {
      requests.push(request.url())
    }
  }
  page.on('request', observe)
  const responsePromise = page.waitForResponse((response) => (
    response.request().method() === 'POST' && new URL(response.url()).pathname.endsWith(suffix)
  ))
  const button = page.getByRole('button', { name })
  await button.dblclick({ delay: 10 })
  const response = await responsePromise
  await page.waitForTimeout(100)
  page.off('request', observe)
  expect(requests).toHaveLength(1)
  return response
}

export async function validateInspection(page: Page) {
  const response = await runAction(page, 'Run technical validation', '/validate')
  expect(response.status()).toBe(200)
  const payload = await response.json() as Record<string, unknown>
  await expect(page.getByText(String(payload.validation_outcome)).first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'Refresh' })).toBeEnabled()
  return payload
}

export async function processInspection(page: Page) {
  const response = await runAction(page, 'Run synthetic processing', '/process')
  expect(response.status()).toBe(200)
  const payload = await response.json() as Record<string, unknown>
  await expect(page.getByText(payload.processing_status === 'ERROR' ? 'TECHNICAL ERROR' : `MOCK ${payload.mock_decision}`).first()).toBeVisible()
  return payload
}
