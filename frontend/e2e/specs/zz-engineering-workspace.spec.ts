import { readFileSync } from 'node:fs'
import { test, expect } from '../fixtures/test'
import { collectKeys } from '../helpers/assertions'
import { snapshot } from '../helpers/control'
import { runtimeState } from '../helpers/runtime'

const state = runtimeState()
const demoPassId = '00000000-0000-4000-8000-000000000003'

test('loads the demo workspace and proves session-only engineering alignment and native pixel measurements', async ({ page, networkRecords }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto(state.frontendUrl)
  const loadResponsePromise = page.waitForResponse((response) => (
    response.request().method() === 'POST'
      && new URL(response.url()).pathname === '/api/v1/development/demo-workspace/load'
  ))
  await page.getByRole('button', { name: 'Load Demo Workspace' }).click()
  const demo = await (await loadResponsePromise).json() as {
    loaded: boolean
    production_approved: boolean
    inspections: Array<{
      key: string
      status: string
      validation_outcome: string
      processing_status: string | null
      mock_decision: string | null
      complete: boolean
    }>
  }
  expect(demo.loaded).toBe(true)
  expect(demo.production_approved).toBe(false)
  expect(demo.inspections).toEqual(expect.arrayContaining([
    expect.objectContaining({ key: 'mock_pass', status: 'PASS', mock_decision: 'PASS', complete: true }),
    expect.objectContaining({ key: 'mock_fail', status: 'FAIL', mock_decision: 'FAIL', complete: true }),
    expect.objectContaining({ key: 'mock_uncertain', status: 'UNCERTAIN', mock_decision: 'UNCERTAIN', complete: true }),
    expect.objectContaining({ key: 'technical_error', status: 'ERROR', processing_status: 'ERROR', complete: true }),
    expect.objectContaining({ key: 'validation_failure', status: 'VALIDATION_FAILED', validation_outcome: 'VALIDATION_FAILED', complete: true }),
  ]))

  const baseline = snapshot(state)
  const networkStart = networkRecords.length
  await page.goto(`${state.frontendUrl}/inspections/${demoPassId}/engineering-view`)
  await expect(page.getByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })).toBeVisible()
  await expect(page.getByText('Synthetic engineering data', { exact: true })).toBeVisible()

  for (const mode of ['RGB', 'Height', 'Side-by-side', 'Alpha overlay', 'Split comparison']) {
    await page.getByRole('button', { name: mode, exact: true }).click()
    await expect(page.getByTestId('vision-canvas')).toHaveAttribute('data-view-mode', mode)
  }

  await page.getByLabel('RGB X coordinate').fill('4')
  await page.getByLabel('RGB Y coordinate').fill('5')
  await page.getByLabel('Height X coordinate').fill('2')
  await page.getByLabel('Height Y coordinate').fill('2')
  await page.getByRole('button', { name: 'Sample native values' }).click()
  await expect(page.locator('.sample-result')).toBeVisible()

  await page.getByRole('button', { name: 'Zoom in' }).click()
  await expect(page.getByLabel('Zoom level')).toHaveText('125%')
  await page.getByRole('button', { name: 'Pan right' }).click()
  await expect(page.getByTestId('vision-canvas')).toHaveAttribute('data-pan', '0.05,0.00')
  await page.getByRole('button', { name: 'Reset', exact: true }).click()
  await expect(page.getByLabel('Zoom level')).toHaveText('100%')

  await page.getByLabel('Translation X (pixels)').fill('7')
  await page.getByLabel('Translation Y (pixels)').fill('-3')
  await page.getByLabel('Rotation (degrees)').fill('5')
  await page.getByLabel('Scale X').fill('1.1')
  await page.getByLabel('Scale Y').fill('0.9')
  await expect(page.getByTestId('matrix-0-2')).toHaveText('7')
  await expect(page.getByTestId('matrix-1-2')).toHaveText('-3')

  await page.getByRole('button', { name: 'Add RGB point' }).click()
  await page.getByRole('button', { name: 'Add height point' }).click()
  await expect(page.getByText(/Pixel residual:/)).toBeVisible()
  await expect(page.getByText(/Mean residual/).locator('..')).toContainText('px')
  await expect(page.getByText(/Maximum residual/).locator('..')).toContainText('px')
  await expect(page.getByRole('button', { name: 'Apply translation suggestion' })).toBeVisible()

  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export alignment JSON' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe(`inspection-${demoPassId}-development-alignment.json`)
  const downloadPath = await download.path()
  expect(downloadPath).not.toBeNull()
  const exported = JSON.parse(readFileSync(downloadPath as string, 'utf8')) as Record<string, unknown>
  expect(exported.contract_version).toBe('pcb-aoi-development-alignment/1.0')
  expect(exported.development_only).toBe(true)
  expect(exported.production_approved).toBe(false)
  expect(exported.units).toBe('pixels')
  expect(exported.correspondences).toHaveLength(1)
  const exportKeys = collectKeys(exported)
  expect(exportKeys.has('relative_path')).toBe(false)
  expect(exportKeys.has('path')).toBe(false)
  expect(exportKeys.has('request_id')).toBe(false)
  expect(exportKeys.has('confidence')).toBe(false)
  await download.delete()

  await page.getByRole('button', { name: 'RGB', exact: true }).click()
  await page.getByLabel('Measurement coordinate space').selectOption('RGB')
  await page.getByRole('button', { name: 'Point', exact: true }).click()
  let raster = await page.getByTestId('rgb-evidence-frame').locator('.engineering-raster-layer').boundingBox()
  expect(raster).not.toBeNull()
  await page.mouse.click(raster!.x + raster!.width * 0.25, raster!.y + raster!.height * 0.25)
  await expect(page.locator('[data-measurement-kind="POINT"]')).toBeVisible()

  await page.getByRole('button', { name: 'Line', exact: true }).click()
  await page.mouse.move(raster!.x + raster!.width * 0.2, raster!.y + raster!.height * 0.2)
  await page.mouse.down()
  await page.mouse.move(raster!.x + raster!.width * 0.7, raster!.y + raster!.height * 0.65)
  await page.mouse.up()
  await expect(page.locator('[data-measurement-kind="LINE"]')).toContainText('distance')

  await page.getByRole('button', { name: 'Height', exact: true }).click()
  await page.getByLabel('Measurement coordinate space').selectOption('HEIGHT')
  await page.getByRole('button', { name: 'Rectangle', exact: true }).click()
  raster = await page.getByTestId('height-evidence-frame').locator('.engineering-raster-layer').boundingBox()
  expect(raster).not.toBeNull()
  await page.mouse.move(raster!.x + raster!.width * 0.15, raster!.y + raster!.height * 0.15)
  await page.mouse.down()
  await page.mouse.move(raster!.x + raster!.width * 0.55, raster!.y + raster!.height * 0.55)
  await page.mouse.up()
  await expect(page.locator('[data-measurement-kind="RECTANGLE"]')).toContainText('area')
  await expect(page.locator('[data-measurement-kind="RECTANGLE"]')).toContainText('Native height min')

  await page.reload()
  await expect(page.getByLabel('Translation X (pixels)')).toHaveValue('0')
  await expect(page.getByText('No correspondence pairs in this browser session.')).toBeVisible()
  await expect(page.getByText('No pixel measurements in this browser session.')).toBeVisible()

  const after = snapshot(state)
  expect(after.database_fingerprint).toBe(baseline.database_fingerprint)
  expect(after.row_counts.audit_events).toBe(baseline.row_counts.audit_events)
  expect(after.artifact_integrity).toEqual(baseline.artifact_integrity)
  expect(after.runtime_files).toEqual(baseline.runtime_files)
  const engineeringTraffic = networkRecords.slice(networkStart).filter((item) => item.kind === 'request')
  expect(engineeringTraffic.some((item) => item.method === 'POST' || item.method === 'PUT' || item.method === 'PATCH' || item.method === 'DELETE')).toBe(false)
})
