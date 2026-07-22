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
  await expect(page.getByRole('region', { name: 'Engineering workspace guide' })).toBeVisible()
  await page.getByRole('button', { name: 'Dismiss guide' }).click()
  await expect(page.getByRole('region', { name: 'Engineering workspace guide' })).toBeHidden()
  await page.getByRole('button', { name: 'Show guide' }).click()
  await expect(page.getByRole('region', { name: 'Engineering workspace guide' })).toBeVisible()

  const viewModes = page.getByRole('group', { name: 'Vision view modes' })
  for (const mode of ['RGB', 'Height', 'Side-by-side', 'Alpha overlay', 'Split comparison']) {
    await viewModes.getByRole('button', { name: mode, exact: true }).click()
    await expect(page.getByTestId('vision-canvas')).toHaveAttribute('data-view-mode', mode)
  }

  await viewModes.getByRole('button', { name: 'Side-by-side', exact: true }).click()
  await page.getByRole('toolbar', { name: 'Engineering tools' }).getByRole('button', { name: /Sample/ }).click()
  let rgbRaster = await page.getByTestId('rgb-raster').boundingBox()
  let heightRaster = await page.getByTestId('height-raster').boundingBox()
  expect(rgbRaster).not.toBeNull()
  expect(heightRaster).not.toBeNull()
  await page.mouse.click(rgbRaster!.x + rgbRaster!.width * 0.25, rgbRaster!.y + rgbRaster!.height * 0.25)
  await expect(page.getByTestId('rgb-crosshair')).toBeVisible()
  const preservedRgbX = await page.getByLabel('RGB X coordinate').inputValue()
  const preservedRgbY = await page.getByLabel('RGB Y coordinate').inputValue()
  heightRaster = await page.getByTestId('height-raster').boundingBox()
  expect(heightRaster).not.toBeNull()
  await page.getByTestId('height-raster').click({ position: { x: heightRaster!.width * 0.6, y: heightRaster!.height * 0.2 } })
  await expect(page.getByTestId('height-crosshair')).toBeVisible()
  await expect(page.getByLabel('RGB X coordinate')).toHaveValue(preservedRgbX)
  await expect(page.getByLabel('RGB Y coordinate')).toHaveValue(preservedRgbY)
  await expect(page.locator('.sample-evidence-note')).toBeVisible()

  const heightX = Number(await page.getByLabel('Height X coordinate').inputValue())
  await page.keyboard.press('ArrowRight')
  await expect(page.getByLabel('Height X coordinate')).toHaveValue(String(heightX + 1))
  await page.keyboard.press('Shift+ArrowDown')
  await expect(page.getByRole('status', { name: 'Engineering workspace status' })).toContainText('Tool Sample')

  await page.keyboard.press('+')
  await expect(page.getByLabel('Zoom level')).toHaveText('125%')
  await page.keyboard.press('h')
  await expect(page.getByRole('toolbar', { name: 'Engineering tools' }).getByRole('button', { name: /Pan/ })).toHaveAttribute('aria-pressed', 'true')
  await page.getByRole('button', { name: 'Pan right' }).click()
  await expect(page.getByTestId('vision-canvas')).toHaveAttribute('data-pan', '0.05,0.00')
  await page.getByRole('button', { name: 'Reset view', exact: true }).click()
  await expect(page.getByLabel('Zoom level')).toHaveText('100%')

  await page.getByLabel('Translation X (pixels)').fill('7')
  await page.getByLabel('Translation Y (pixels)').fill('-3')
  await page.getByLabel('Rotation (degrees)').fill('5')
  await page.getByLabel('Scale X').fill('1.1')
  await page.getByLabel('Scale Y').fill('0.9')
  await expect(page.getByTestId('matrix-0-2')).toHaveText('7')
  await expect(page.getByTestId('matrix-1-2')).toHaveText('-3')

  await viewModes.getByRole('button', { name: 'Side-by-side', exact: true }).click()
  await page.keyboard.press('c')
  rgbRaster = await page.getByTestId('rgb-raster').boundingBox()
  heightRaster = await page.getByTestId('height-raster').boundingBox()
  await page.mouse.click(rgbRaster!.x + rgbRaster!.width * 0.35, rgbRaster!.y + rgbRaster!.height * 0.35)
  await page.mouse.click(heightRaster!.x + heightRaster!.width * 0.35, heightRaster!.y + heightRaster!.height * 0.35)
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

  await viewModes.getByRole('button', { name: 'RGB', exact: true }).click()
  await page.keyboard.press('l')
  let raster = await page.getByTestId('rgb-raster').boundingBox()
  expect(raster).not.toBeNull()
  await page.mouse.move(raster!.x + raster!.width * 0.2, raster!.y + raster!.height * 0.2)
  await page.mouse.down()
  await page.mouse.move(raster!.x + raster!.width * 0.7, raster!.y + raster!.height * 0.65)
  await page.mouse.up()
  await expect(page.locator('[data-measurement-kind="LINE"]')).toContainText('distance')
  await page.keyboard.press('Control+z')
  await expect(page.locator('[data-measurement-kind="LINE"]')).toBeHidden()
  await page.keyboard.press('Control+Shift+z')
  await expect(page.locator('[data-measurement-kind="LINE"]')).toBeVisible()

  await viewModes.getByRole('button', { name: 'Height', exact: true }).click()
  await page.keyboard.press('r')
  raster = await page.getByTestId('height-raster').boundingBox()
  expect(raster).not.toBeNull()
  await page.mouse.move(raster!.x + raster!.width * 0.15, raster!.y + raster!.height * 0.15)
  await page.mouse.down()
  await page.mouse.move(raster!.x + raster!.width * 0.55, raster!.y + raster!.height * 0.55)
  await page.mouse.up()
  await expect(page.locator('[data-measurement-kind="RECTANGLE"]')).toContainText('area')
  await expect(page.locator('[data-measurement-kind="RECTANGLE"]')).toContainText('Native height min')

  await page.reload()
  await expect(page.getByRole('toolbar', { name: 'Engineering tools' }).getByRole('button', { name: /Pointer/ })).toHaveAttribute('aria-pressed', 'true')
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
