import { createHash } from 'node:crypto'
import { readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import type { Page, TestInfo } from '@playwright/test'
import { test, expect, type NetworkRecord } from '../fixtures/test'
import { assertNoDuplicateIds } from '../helpers/assertions'
import { snapshot } from '../helpers/control'
import { runtimeState } from '../helpers/runtime'

const state = runtimeState()
const demoPassId = '00000000-0000-4000-8000-000000000003'

type MemoryObservation = {
  label: string
  jsHeapUsedBytes: number | null
  jsHeapTotalBytes: number | null
}

function sha256(file: string): string {
  return createHash('sha256').update(readFileSync(file)).digest('hex')
}

function fileIdentity(file: string) {
  const stat = statSync(file)
  return { sha256: sha256(file), byteSize: stat.size, mtimeMs: stat.mtimeMs }
}

async function ensureDemoWorkspace(page: Page): Promise<void> {
  const response = await page.request.post(`${state.backendUrl}/api/v1/development/demo-workspace/load`)
  expect(response.ok()).toBe(true)
}

async function observeMemory(page: Page, label: string): Promise<MemoryObservation> {
  try {
    const session = await page.context().newCDPSession(page)
    await session.send('Performance.enable')
    const result = await session.send('Performance.getMetrics')
    await session.detach()
    const metric = (name: string) => result.metrics.find((item) => item.name === name)?.value ?? null
    return {
      label,
      jsHeapUsedBytes: metric('JSHeapUsedSize'),
      jsHeapTotalBytes: metric('JSHeapTotalSize'),
    }
  } catch {
    return { label, jsHeapUsedBytes: null, jsHeapTotalBytes: null }
  }
}

async function attachJson(testInfo: TestInfo, name: string, value: unknown): Promise<void> {
  await testInfo.attach(name, {
    body: Buffer.from(JSON.stringify(value, null, 2)),
    contentType: 'application/json',
  })
}

function engineeringRequests(records: NetworkRecord[], start: number): NetworkRecord[] {
  return records.slice(start).filter((item) => new URL(item.url).pathname.includes('/engineering-view'))
}

test.describe.serial('engineering workspace Task 30D hardening', () => {
  test('verifies exact responsive viewports and core accessibility semantics', async ({ page }, testInfo) => {
    await ensureDemoWorkspace(page)
    const observations: Array<Record<string, unknown>> = []
    for (const viewport of [
      { name: 'desktop', width: 1440, height: 900 },
      { name: 'laptop', width: 1280, height: 720 },
      { name: 'compact-laptop', width: 1024, height: 768 },
      { name: 'tablet', width: 768, height: 1024 },
    ]) {
      await page.setViewportSize(viewport)
      await page.goto(`${state.frontendUrl}/inspections/${demoPassId}/engineering-view`)
      await expect(page.getByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })).toBeVisible()
      await expect(page.getByRole('note').filter({ hasText: 'Synthetic engineering data' })).toBeVisible()
      await expect(page.getByRole('navigation', { name: 'Operator navigation' })).toBeVisible()
      await expect(page.getByRole('navigation', { name: 'Engineering evidence navigator' })).toBeVisible()
      await expect(page.getByRole('region', { name: 'Vision canvas workspace' })).toBeVisible()
      await expect(page.getByRole('complementary', { name: 'Metadata and pixel inspector' })).toBeAttached()
      await page.getByRole('heading', { name: 'Persisted workflow record' }).scrollIntoViewIfNeeded()
      await expect(page.getByRole('heading', { name: 'Persisted workflow record' })).toBeVisible()
      await page.getByRole('heading', { name: 'Native height distribution' }).scrollIntoViewIfNeeded()
      await expect(page.getByRole('heading', { name: 'Native height distribution' })).toBeVisible()
      await page.getByRole('heading', { name: 'Session alignment' }).scrollIntoViewIfNeeded()
      await expect(page.getByRole('heading', { name: 'Session alignment' })).toBeVisible()
      await expect(page.getByRole('status', { name: 'Engineering workspace status' })).toBeAttached()
      await assertNoDuplicateIds(page)
      await expect(page.locator('main')).toHaveCount(1)

      const semantics = await page.evaluate(() => {
        const interactive = Array.from(document.querySelectorAll<HTMLElement>(
          'button, a[href], input, select, [role="button"][tabindex]',
        ))
        const unnamed = interactive.filter((element) => {
          const labelledBy = element.getAttribute('aria-labelledby')
          const labelledText = labelledBy
            ? labelledBy.split(/\s+/).map((id) => document.getElementById(id)?.textContent ?? '').join(' ')
            : ''
          const label = element instanceof HTMLInputElement || element instanceof HTMLSelectElement
            ? element.labels?.[0]?.textContent ?? ''
            : ''
          return ![
            element.getAttribute('aria-label'),
            labelledText,
            label,
            element.textContent,
            element.getAttribute('title'),
          ].some((value) => value?.trim())
        }).map((element) => element.outerHTML)
        const headings = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6'))
          .map((heading) => ({ level: Number(heading.tagName.slice(1)), text: heading.textContent?.trim() }))
        const skippedHeading = headings.some((heading, index) => (
          index > 0 && heading.level > headings[index - 1]!.level + 1
        ))
        return {
          pageOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          unnamed,
          headings,
          skippedHeading,
          histogramBins: document.querySelectorAll('[data-histogram-bin]').length,
        }
      })
      expect(semantics.pageOverflow).toBeLessThanOrEqual(1)
      expect(semantics.unnamed).toEqual([])
      expect(semantics.skippedHeading).toBe(false)
      expect(semantics.histogramBins).toBe(64)
      const firstMode = page.getByRole('group', { name: 'Vision view modes' }).getByRole('button', { name: 'RGB', exact: true })
      await firstMode.focus()
      const focusStyle = await firstMode.evaluate((element) => {
        const style = getComputedStyle(element)
        return { outlineStyle: style.outlineStyle, outlineWidth: style.outlineWidth }
      })
      expect(focusStyle.outlineStyle).not.toBe('none')
      expect(focusStyle.outlineWidth).not.toBe('0px')

      const localOverflow = await page.locator('.view-mode-toolbar,.canvas-toolbar,.engineering-tool-strip').evaluateAll((elements) => (
        elements.map((element) => {
          const html = element as HTMLElement
          return {
            overflowing: html.scrollWidth > html.clientWidth,
            overflowX: getComputedStyle(html).overflowX,
          }
        })
      ))
      expect(localOverflow.every((item) => !item.overflowing || ['auto', 'scroll'].includes(item.overflowX))).toBe(true)
      if (viewport.width === 768) {
        const touchHeights = await page.locator('.view-mode-toolbar button,.canvas-toolbar button,.engineering-tool-strip button').evaluateAll(
          (elements) => elements.map((element) => element.getBoundingClientRect().height),
        )
        expect(Math.min(...touchHeights)).toBeGreaterThanOrEqual(44)
      }
      observations.push({ viewport, semantics, localOverflow, focusStyle })
    }
    await attachJson(testInfo, 'responsive-accessibility-observations.json', observations)
  })

  test('completes the keyboard-only journey with bounded requests, read-only state, and memory diagnostics', async ({ page, networkRecords }, testInfo) => {
    await ensureDemoWorkspace(page)
    const baseline = snapshot(state)
    const networkStart = networkRecords.length
    const timings: Record<string, number> = {}
    const memory: MemoryObservation[] = []
    await page.setViewportSize({ width: 1280, height: 720 })
    await page.goto(state.frontendUrl)
    await expect(page.getByRole('heading', { name: 'Inspection History' })).toBeVisible()
    memory.push(await observeMemory(page, 'before-workspace'))

    const detailLink = page.getByRole('link', { name: demoPassId })
    await detailLink.focus()
    await page.keyboard.press('Enter')
    await expect(page.getByRole('heading', { name: 'Inspection Detail' })).toBeVisible()
    const engineeringLink = page.getByRole('link', { name: 'Open Engineering Workspace' })
    await engineeringLink.focus()
    const renderStart = Date.now()
    await page.keyboard.press('Enter')
    await expect(page.getByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })).toBeVisible()
    await expect(page.getByRole('img', { name: 'RGB evidence preview' })).toHaveJSProperty('complete', true)
    await expect(page.getByRole('img', { name: 'Height evidence preview' })).toHaveJSProperty('complete', true)
    timings.initialWorkspaceRenderMs = Date.now() - renderStart
    memory.push(await observeMemory(page, 'after-previews'))
    const requestsBeforePointerMovement = networkRecords.length
    for (let offset = 0; offset < 40; offset += 1) {
      await page.mouse.move(100 + offset, 100 + offset)
    }
    expect(networkRecords.length).toBe(requestsBeforePointerMovement)

    const modes = page.getByRole('group', { name: 'Vision view modes' })
    const viewChangeStart = Date.now()
    await modes.getByRole('button', { name: 'RGB', exact: true }).focus()
    await page.keyboard.press('Enter')
    await page.keyboard.press('Tab')
    expect(await page.evaluate(() => document.activeElement?.textContent?.trim())).toBe('Height')
    await page.keyboard.press('Enter')
    timings.viewModeChangeMs = Date.now() - viewChangeStart
    for (let iteration = 0; iteration < 3; iteration += 1) {
      for (const mode of ['RGB', 'Height', 'Side-by-side', 'Alpha overlay', 'Split comparison']) {
        const button = modes.getByRole('button', { name: mode, exact: true })
        await button.focus()
        await page.keyboard.press('Enter')
      }
    }
    memory.push(await observeMemory(page, 'after-view-switching'))

    const fillFromKeyboard = async (label: string, value: string) => {
      const input = page.getByLabel(label)
      await input.focus()
      await page.keyboard.press('Control+A')
      await page.keyboard.type(value)
    }
    await fillFromKeyboard('RGB X coordinate', '10')
    await fillFromKeyboard('RGB Y coordinate', '20')
    await fillFromKeyboard('Height X coordinate', '12')
    await fillFromKeyboard('Height Y coordinate', '22')
    await page.getByRole('group', { name: 'Active coordinate space' }).getByRole('button', { name: 'RGB', exact: true }).focus()
    await page.keyboard.press('Enter')
    await page.keyboard.press('ArrowRight')
    await expect(page.getByLabel('RGB X coordinate')).toHaveValue('11')
    const sampleStart = Date.now()
    await page.getByRole('button', { name: 'Sample values' }).focus()
    await page.keyboard.press('Enter')
    await expect(page.locator('.sample-evidence-note')).toBeVisible()
    timings.sampleRequestMs = Date.now() - sampleStart
    for (let iteration = 0; iteration < 3; iteration += 1) {
      await page.getByRole('button', { name: 'Sample values' }).focus()
      await page.keyboard.press('Enter')
    }

    const correspondenceTool = page.getByRole('toolbar', { name: 'Engineering tools' }).getByRole('button', { name: /Correspondence/ })
    await correspondenceTool.focus()
    await page.keyboard.press('Enter')
    for (const action of ['Use selected RGB point', 'Use selected height point', 'Add Pair']) {
      await page.getByRole('button', { name: action }).focus()
      await page.keyboard.press('Enter')
    }
    await expect(page.locator('.correspondence-list')).toContainText('P1')
    await fillFromKeyboard('Translation X (pixels)', '7')
    await fillFromKeyboard('Rotation (degrees)', '3')
    await page.keyboard.press('Control+z')
    await page.keyboard.press('Control+Shift+z')
    await expect(page.getByLabel('Rotation (degrees)')).toHaveValue('3')

    const lineTool = page.getByRole('toolbar', { name: 'Engineering tools' }).getByRole('button', { name: /Line/ })
    await lineTool.focus()
    await page.keyboard.press('Enter')
    await page.getByRole('group', { name: 'Active coordinate space' }).getByRole('button', { name: 'RGB', exact: true }).focus()
    await page.keyboard.press('Enter')
    await page.getByRole('button', { name: 'Set selected coordinate as measurement start' }).focus()
    await page.keyboard.press('Enter')
    await fillFromKeyboard('RGB X coordinate', '20')
    await fillFromKeyboard('RGB Y coordinate', '25')
    await page.getByRole('button', { name: 'Complete line at selected coordinate' }).focus()
    await page.keyboard.press('Enter')
    await expect(page.locator('[data-measurement-kind="LINE"]')).toBeVisible()

    const rectangleTool = page.getByRole('toolbar', { name: 'Engineering tools' }).getByRole('button', { name: /Rectangle/ })
    await rectangleTool.focus()
    await page.keyboard.press('Enter')
    const activeSpace = page.getByRole('group', { name: 'Active coordinate space' })
    await activeSpace.getByRole('button', { name: 'Height', exact: true }).focus()
    await page.keyboard.press('Enter')
    await page.getByRole('button', { name: 'Set selected coordinate as measurement start' }).focus()
    await page.keyboard.press('Enter')
    await fillFromKeyboard('Height X coordinate', '30')
    await fillFromKeyboard('Height Y coordinate', '40')
    const roiStart = Date.now()
    await page.getByRole('button', { name: 'Complete rectangle at selected coordinate' }).focus()
    await page.keyboard.press('Enter')
    await expect(page.locator('[data-measurement-kind="RECTANGLE"]')).toContainText('Native min')
    timings.roiRequestMs = Date.now() - roiStart

    for (let iteration = 0; iteration < 8; iteration += 1) {
      await page.keyboard.press('+')
      await page.getByRole('button', { name: 'Pan right' }).focus()
      await page.keyboard.press('Enter')
      await page.keyboard.press('-')
    }
    memory.push(await observeMemory(page, 'after-sample-alignment-roi-zoom-pan'))
    await page.getByRole('button', { name: 'Keyboard help' }).focus()
    await page.keyboard.press('Enter')
    await expect(page.getByRole('dialog', { name: 'Keyboard shortcuts' })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog', { name: 'Keyboard shortcuts' })).toBeHidden()

    page.once('dialog', (dialog) => dialog.accept())
    await page.getByRole('button', { name: 'Reset Engineering Session' }).focus()
    await page.keyboard.press('Enter')
    await expect(page.getByText('No pixel measurements in this browser session.')).toBeVisible()
    const resourceTimings = await page.evaluate(() => performance.getEntriesByType('resource')
      .filter((entry) => entry.name.includes('/engineering-view'))
      .map((entry) => ({ name: entry.name, durationMs: entry.duration })))
    await page.goto(state.frontendUrl)
    memory.push(await observeMemory(page, 'after-leaving-workspace'))

    const after = snapshot(state)
    expect(after.database_fingerprint).toBe(baseline.database_fingerprint)
    expect(after.row_counts).toEqual(baseline.row_counts)
    expect(after.artifact_integrity).toEqual(baseline.artifact_integrity)
    expect(after.fixture_integrity).toEqual(baseline.fixture_integrity)
    expect(after.fixture_control_files).toEqual(baseline.fixture_control_files)
    expect(after.runtime_files).toEqual(baseline.runtime_files)
    expect(after.report_files).toEqual([])

    const traffic = engineeringRequests(networkRecords, networkStart)
    const requests = traffic.filter((item) => item.kind === 'request')
    const paths = requests.map((item) => new URL(item.url).pathname)
    // Vite development runs React StrictMode's intentional mount/effect replay.
    // Bound the metadata GET to that initial replay; interactions must not add more.
    expect(paths.filter((item) => item.endsWith('/engineering-view')).length).toBeLessThanOrEqual(2)
    expect(paths.filter((item) => item.endsWith('/sample'))).toHaveLength(4)
    expect(paths.filter((item) => item.endsWith('/height-roi'))).toHaveLength(1)
    expect(requests.some((item) => ['POST', 'PUT', 'PATCH', 'DELETE'].includes(item.method))).toBe(false)
    expect(paths.filter((item) => item.endsWith('/rgb-preview')).length).toBeLessThanOrEqual(1)
    expect(paths.filter((item) => item.endsWith('/height-preview')).length).toBeLessThanOrEqual(8)
    const heapValues = memory.map((item) => item.jsHeapUsedBytes).filter((value): value is number => value !== null)
    if (heapValues.length > 1) {
      expect(Math.max(...heapValues) - Math.min(...heapValues)).toBeLessThan(256 * 1024 * 1024)
    }
    await attachJson(testInfo, 'keyboard-network-memory-diagnostics.json', {
      timings,
      memory,
      resourceTimings,
      engineeringRequests: requests,
      environment: {
        browser: state.browserChannel,
        viewport: { width: 1280, height: 720 },
        node: process.version,
        platform: process.platform,
      },
    })
    process.stdout.write(`\nENGINEERING_BROWSER_DIAGNOSTIC ${JSON.stringify({ timings, memory })}\n`)
  })

  test('loads temporary Full HD evidence and preserves source and database integrity', async ({ page, networkRecords }, testInfo) => {
    await ensureDemoWorkspace(page)
    const manifestPath = path.join(state.largeEvidenceRoot, 'manifest.json')
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
      files: Array<{ kind: string; name: string; sha256: string; byte_size: number }>
    }
    const rgbEntry = manifest.files.find((item) => item.kind === 'rgb')!
    const heightEntry = manifest.files.find((item) => item.kind === 'height')!
    const rgbFile = path.join(state.largeEvidenceRoot, rgbEntry.name)
    const heightFile = path.join(state.largeEvidenceRoot, heightEntry.name)
    const sourceBefore = {
      manifest: fileIdentity(manifestPath),
      rgb: fileIdentity(rgbFile),
      height: fileIdentity(heightFile),
    }
    const intake = await page.request.post(`${state.backendUrl}/api/v1/inspections`, {
      multipart: {
        board_id: 'E2E-LARGE-1920X1080',
        recipe_id: 'synthetic-e2e',
        recipe_version: '1.0',
        rgb_image: { name: rgbEntry.name, mimeType: 'image/png', buffer: readFileSync(rgbFile) },
        height_map: { name: heightEntry.name, mimeType: 'image/tiff', buffer: readFileSync(heightFile) },
      },
    })
    expect(intake.status()).toBe(201)
    const inspectionId = ((await intake.json()) as { inspection_id: string }).inspection_id
    const baseline = snapshot(state)
    const networkStart = networkRecords.length
    const largeLoadStart = Date.now()
    await page.setViewportSize({ width: 1440, height: 900 })
    await page.goto(`${state.frontendUrl}/inspections/${inspectionId}/engineering-view`)
    await expect(page.getByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })).toBeVisible()
    await expect(page.getByRole('img', { name: 'RGB evidence preview' })).toHaveJSProperty('naturalWidth', 1920)
    await expect(page.getByRole('img', { name: 'Height evidence preview' })).toHaveJSProperty('naturalWidth', 1920)
    const largePreviewLoadMs = Date.now() - largeLoadStart
    await expect(page.getByRole('status', { name: 'Engineering workspace status' })).toContainText('RGB 1920x1080')
    await expect(page.getByRole('status', { name: 'Engineering workspace status' })).toContainText('Height 1920x1080')
    await expect(page.locator('[data-histogram-bin]')).toHaveCount(64)
    await page.getByLabel('RGB X coordinate').fill('17')
    await page.getByLabel('RGB Y coordinate').fill('23')
    await page.getByLabel('Height X coordinate').fill('17')
    await page.getByLabel('Height Y coordinate').fill('23')
    await page.getByRole('button', { name: 'Sample values' }).click()
    await expect(page.locator('.coordinate-inspector').first().getByText('R', { exact: true }).locator('..')).toContainText('17')
    await expect(page.getByText('Native height', { exact: true }).locator('..')).toContainText('117')
    await page.getByRole('button', { name: 'Zoom in' }).click()
    await page.getByRole('button', { name: 'Pan right' }).click()
    await expect(page.getByLabel('Zoom level')).toHaveText('125%')
    await page.getByRole('group', { name: 'Vision view modes' }).getByRole('button', { name: 'Height', exact: true }).click()
    await page.getByRole('toolbar', { name: 'Engineering tools' }).getByRole('button', { name: /Rectangle/ }).click()
    await page.getByRole('group', { name: 'Active coordinate space' }).getByRole('button', { name: 'Height', exact: true }).click()
    await page.getByRole('button', { name: 'Set selected coordinate as measurement start' }).click()
    await page.getByLabel('Height X coordinate').fill('80')
    await page.getByLabel('Height Y coordinate').fill('80')
    await page.getByRole('button', { name: 'Complete rectangle at selected coordinate' }).click()
    await expect(page.locator('[data-measurement-kind="RECTANGLE"]')).toContainText('Native min')

    const after = snapshot(state)
    expect(after.database_fingerprint).toBe(baseline.database_fingerprint)
    expect(after.artifact_integrity).toEqual(baseline.artifact_integrity)
    expect(after.fixture_integrity).toEqual(baseline.fixture_integrity)
    expect(after.fixture_control_files).toEqual(baseline.fixture_control_files)
    expect(after.runtime_files).toEqual(baseline.runtime_files)
    expect(fileIdentity(rgbFile)).toEqual(sourceBefore.rgb)
    expect(fileIdentity(heightFile)).toEqual(sourceBefore.height)
    expect(fileIdentity(manifestPath)).toEqual(sourceBefore.manifest)
    const requests = engineeringRequests(networkRecords, networkStart).filter((item) => item.kind === 'request')
    expect(requests.some((item) => ['POST', 'PUT', 'PATCH', 'DELETE'].includes(item.method))).toBe(false)
    await attachJson(testInfo, 'large-browser-evidence-diagnostics.json', {
      dimensions: [1920, 1080],
      largePreviewLoadMs,
      sourceBefore,
      requests,
    })
    process.stdout.write(`\nENGINEERING_LARGE_BROWSER_DIAGNOSTIC ${JSON.stringify({ dimensions: [1920, 1080], largePreviewLoadMs })}\n`)
  })

  test('renders controlled engineering error states without mutating persisted state', async ({ page }) => {
    await ensureDemoWorkspace(page)
    const baseline = snapshot(state)
    const metadataPattern = `**/api/v1/inspections/${demoPassId}/engineering-view`
    await page.route(metadataPattern, (route) => route.fulfill({
      status: 404,
      contentType: 'application/json',
      headers: { 'X-Request-ID': 'viewer-disabled-request' },
      body: JSON.stringify({
        code: 'ENGINEERING_VIEWER_DISABLED',
        message: 'Engineering viewer is disabled.',
        request_id: 'viewer-disabled-request',
      }),
    }))
    await page.goto(`${state.frontendUrl}/inspections/${demoPassId}/engineering-view`)
    await expect(page.getByRole('alert')).toContainText('ENGINEERING_VIEWER_DISABLED')
    await expect(page.getByRole('alert')).toContainText('viewer-disabled-request')
    await page.unroute(metadataPattern)

    const missingId = 'ffffffff-ffff-4fff-8fff-ffffffffffff'
    await page.goto(`${state.frontendUrl}/inspections/${missingId}/engineering-view`)
    await expect(page.getByRole('alert')).toContainText('INSPECTION_NOT_FOUND')

    await page.route(metadataPattern, (route) => route.fulfill({
      status: 422,
      contentType: 'application/json',
      headers: { 'X-Request-ID': 'unsupported-evidence-request' },
      body: JSON.stringify({
        code: 'ENGINEERING_ARTIFACT_UNSUPPORTED',
        message: 'Controlled temporary evidence format is unsupported.',
        request_id: 'unsupported-evidence-request',
      }),
    }))
    await page.goto(`${state.frontendUrl}/inspections/${demoPassId}/engineering-view`)
    await expect(page.getByRole('alert')).toContainText('ENGINEERING_ARTIFACT_UNSUPPORTED')
    await expect(page.getByRole('alert')).toContainText('unsupported-evidence-request')
    await page.unroute(metadataPattern)

    await page.route('**/api/v1/**', (route) => route.abort('connectionrefused'))
    await page.goto(`${state.frontendUrl}/inspections/${demoPassId}/engineering-view`)
    await expect(page.getByRole('alert')).toContainText('BACKEND_UNAVAILABLE')
    await page.unroute('**/api/v1/**')
    expect(snapshot(state).database_fingerprint).toBe(baseline.database_fingerprint)
  })
})
