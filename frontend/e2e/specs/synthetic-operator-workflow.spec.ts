import { readFileSync, writeFileSync } from 'node:fs'
import { test, expect, type NetworkRecord } from '../fixtures/test'
import { assertNoDuplicateIds, collectKeys } from '../helpers/assertions'
import { queryProfiles, seedHistory, snapshot, tamperRgb, verifyReportEnvelope, type RuntimeSnapshot } from '../helpers/control'
import { successMarker } from '../helpers/paths'
import { runtimeState, scenarioFiles } from '../helpers/runtime'
import { createInspection, processInspection, validateInspection } from '../helpers/workflow'

const state = runtimeState()
const expectedIds = {
  pass: '00000000-0000-4000-8000-000000000003',
  fail: '00000000-0000-4000-8000-000000000001',
  uncertain: '00000000-0000-4000-8000-000000000002',
  technicalError: '00000000-0000-4000-8000-000000000004',
  validationFailure: '00000000-0000-4000-8000-000000000005',
  errorProbe: '00000000-0000-4000-8000-000000000006',
} as const

let passSnapshot: RuntimeSnapshot
let finalDemonstration: Record<string, unknown> = {}
let boundedQueryProfile: Record<string, unknown> = {}
const actionRequestIds = new Set<string>()

async function assertNoConfidenceValue(page: import('@playwright/test').Page) {
  await expect(page.getByText(/^Confidence$/i)).toHaveCount(0)
  await expect(page.getByText(/\b\d+(?:\.\d+)?\s*%/)).toHaveCount(0)
}

test.describe.serial('full synthetic operator workflow and release hardening', () => {
  test('runs the complete MOCK PASS workflow, persisted reload, audit, report, download, print, responsive and integrity journey', async ({ page, networkRecords, browser }, testInfo) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await page.goto(state.frontendUrl)
    await expect(page.getByRole('heading', { name: 'Inspection History' })).toBeVisible()
    await expect(page.getByText(/Development mode:/)).toBeVisible()
    await expect(page.getByText('Backend online')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'No inspections received yet' })).toBeVisible()
    await expect(page.locator('main')).toHaveAttribute('id', 'main-content')

    const intake = await createInspection(page, state, 'valid_rgb_png_height_tiff', 'E2E-PASS')
    expect(intake.inspection_id).toBe(expectedIds.pass)
    expect(intake.status).toBe('RECEIVED')
    await expect(page.getByText('RECEIVED').first()).toBeVisible()
    await expect(page.getByText('Not supplied')).toBeVisible()

    const validation = await validateInspection(page)
    expect(validation.validation_outcome).toBe('VALIDATION_PASSED')
    expect(validation.inspection_status).toBe('READY')
    await expect(page.getByText(/technically ready for preprocessing/i)).toBeVisible()
    await expect(page.getByText('RGB source')).toBeVisible()
    await expect(page.getByText('Height source')).toBeVisible()
    await expect(page.getByText(/^PCB PASS$/)).toHaveCount(0)

    const processing = await processInspection(page)
    expect(processing.mock_decision).toBe('PASS')
    expect(processing.inspection_status).toBe('PASS')
    expect(processing.processing_status).toBe('COMPLETED')
    expect(processing.defect_type).toBeNull()
    expect(processing.production_approved).toBe(false)
    await assertNoConfidenceValue(page)
    await expect(page.getByRole('button', { name: 'Run synthetic processing' })).toHaveCount(0)
    await expect(page.getByText(/Reprocessing is not supported/)).toBeVisible()

    for (const record of networkRecords.filter((item) => item.kind === 'request' && item.method === 'POST')) {
      expect(record.requestId).toMatch(/^[0-9a-f-]{36}$/)
      actionRequestIds.add(record.requestId as string)
    }
    expect(actionRequestIds.size).toBe(3)

    const beforeReload = snapshot(state)
    const postsBeforeReload = networkRecords.filter((item) => item.kind === 'request' && item.method === 'POST').length
    const processingResponsePromise = page.waitForResponse((response) => (
      response.request().method() === 'GET' && new URL(response.url()).pathname.endsWith('/processing')
    ))
    const auditResponsePromise = page.waitForResponse((response) => (
      response.request().method() === 'GET' && new URL(response.url()).pathname.endsWith('/audit')
    ))
    await page.reload()
    const persistedProcessing = await (await processingResponsePromise).json() as Record<string, unknown>
    const auditEnvelope = await (await auditResponsePromise).json() as Record<string, unknown>
    expect(persistedProcessing.processing_run_id).toBe(processing.processing_run_id)
    expect(persistedProcessing.preprocessing_id).toBe(processing.preprocessing_id)
    expect(persistedProcessing.inference_id).toBe(processing.inference_id)
    expect(persistedProcessing.mock_decision).toBe('PASS')
    expect(networkRecords.filter((item) => item.kind === 'request' && item.method === 'POST')).toHaveLength(postsBeforeReload)

    const actions = await page.locator('.audit-timeline code').allTextContents()
    expect(actions).toEqual([
      'INSPECTION_RECEIVED',
      'INSPECTION_VALIDATION_PASSED',
      'INSPECTION_PROCESSING_STARTED',
      'INSPECTION_MOCK_RESULT_PASS',
    ])
    const historicalRequestIds = (auditEnvelope.items as Array<Record<string, unknown>>)
      .map((item) => item.request_id).filter((value): value is string => typeof value === 'string')
    expect(new Set(historicalRequestIds)).toEqual(actionRequestIds)

    const reportResponsePromise = page.waitForResponse((response) => (
      response.request().method() === 'GET' && new URL(response.url()).pathname.endsWith('/report')
    ))
    await page.getByRole('link', { name: 'Open Development Report' }).click()
    const reportResponse = await reportResponsePromise
    const rawReportEnvelope = await reportResponse.text()
    const reportEnvelope = JSON.parse(rawReportEnvelope) as {
      report: Record<string, unknown>; report_sha256: string; request_id: string
    }
    await expect(page.getByRole('heading', { name: 'Development Report' })).toBeVisible()
    await expect(page.getByText(/Development-only .* nonproduction/)).toBeVisible()
    await expect(page.getByText(`Report SHA-256: ${reportEnvelope.report_sha256}`)).toBeVisible()
    expect(verifyReportEnvelope(state, rawReportEnvelope)).toBe(reportEnvelope.report_sha256)
    expect(JSON.stringify(reportEnvelope.report)).not.toContain(reportEnvelope.request_id)
    const reportKeys = collectKeys(reportEnvelope.report)
    expect(reportKeys.has('relative_path')).toBe(false)
    expect(reportKeys.has('confidence')).toBe(false)
    expect(reportEnvelope.report.contract_version).toBe('pcb-aoi-inspection-development-report/1.0')

    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Download JSON' }).click()
    const download = await downloadPromise
    expect(download.suggestedFilename()).toBe(`inspection-${expectedIds.pass}-development-report.json`)
    const downloadPath = await download.path()
    expect(downloadPath).not.toBeNull()
    const downloaded = JSON.parse(readFileSync(downloadPath as string, 'utf8')) as Record<string, unknown>
    expect(downloaded).toEqual(reportEnvelope.report)
    expect(downloaded.inspection_id).toBe(expectedIds.pass)
    await download.delete()

    await page.emulateMedia({ media: 'print' })
    await expect(page.locator('.primary-nav')).not.toBeVisible()
    await expect(page.locator('.report-toolbar')).not.toBeVisible()
    await expect(page.getByRole('heading', { name: 'Development Report' })).toBeVisible()
    await expect(page.getByText(expectedIds.pass).first()).toBeVisible()
    await expect(page.getByText(/Report SHA-256:/)).toBeVisible()
    await expect(page.getByText(/Development-only .* nonproduction/)).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Development limitations' })).toBeVisible()
    await page.emulateMedia({ media: 'screen' })

    for (const viewport of [
      { width: 1440, height: 900 },
      { width: 1280, height: 720 },
      { width: 768, height: 1024 },
    ]) {
      await page.setViewportSize(viewport)
      await expect(page.getByRole('heading', { name: 'Development Report' })).toBeVisible()
      await expect(page.getByText(/Development-only .* nonproduction/)).toBeVisible()
      await expect(page.getByRole('button', { name: 'Download JSON' })).toBeVisible()
    }
    await assertNoDuplicateIds(page)

    const afterReadOnlyViews = snapshot(state)
    expect(afterReadOnlyViews.database_fingerprint).toBe(beforeReload.database_fingerprint)
    expect(afterReadOnlyViews.row_counts.audit_events).toBe(beforeReload.row_counts.audit_events)
    expect(afterReadOnlyViews.report_files).toEqual([])
    expect(afterReadOnlyViews.schema_version).toBe(3)
    expect(afterReadOnlyViews.foreign_key_failures).toEqual([])
    expect(afterReadOnlyViews.fixture_tree_sha256).toBe(state.fixtureTreeSha256)
    expect(afterReadOnlyViews.fixture_files_verified).toBe(true)
    const passArtifacts = afterReadOnlyViews.artifact_integrity.filter((item) => item.inspection_id === expectedIds.pass)
    expect(passArtifacts).toHaveLength(2)
    for (const artifact of passArtifacts) {
      expect(artifact.contained).toBe(true)
      expect(artifact.exists).toBe(true)
      expect(artifact.actual_sha256).toBe(artifact.registered_sha256)
      expect(artifact.actual_byte_size).toBe(artifact.registered_byte_size)
    }
    passSnapshot = afterReadOnlyViews
    finalDemonstration = {
      browser: `${state.browserChannel} ${browser.version()}`,
      inspection_id: expectedIds.pass,
      inspection_status: processing.inspection_status,
      validation_outcome: validation.validation_outcome,
      processing_status: processing.processing_status,
      mock_decision: processing.mock_decision,
      report_sha256: reportEnvelope.report_sha256,
      audit_actions: actions,
      download_verified: true,
      print_media_verified: true,
    }
    await testInfo.attach('live-demonstration.json', {
      body: Buffer.from(JSON.stringify(finalDemonstration, null, 2)),
      contentType: 'application/json',
    })
  })

  test('browser-verifies deterministic MOCK FAIL and MOCK UNCERTAIN scenarios without confidence or reprocessing', async ({ page, networkRecords }) => {
    const cases = [
      { key: 'fail', board: 'E2E-FAIL', decision: 'FAIL', defect: 'misalignment' },
      { key: 'uncertain', board: 'E2E-UNCERTAIN', decision: 'UNCERTAIN', defect: null },
    ] as const
    for (const item of cases) {
      const intake = await createInspection(page, state, 'valid_rgb_png_height_tiff', item.board)
      expect(intake.inspection_id).toBe(expectedIds[item.key])
      const validation = await validateInspection(page)
      expect(validation.validation_outcome).toBe('VALIDATION_PASSED')
      const processing = await processInspection(page)
      expect(processing.mock_decision).toBe(item.decision)
      expect(processing.defect_type).toBe(item.defect)
      expect(processing.inspection_status).toBe(item.decision)
      if (item.decision === 'FAIL') {
        await expect(page.getByText(`Mock taxonomy label: ${item.defect}`)).toBeVisible()
      } else {
        await expect(page.getByText(/Mock taxonomy label:/)).toHaveCount(0)
      }
      await assertNoConfidenceValue(page)
      await expect(page.getByRole('button', { name: 'Run synthetic processing' })).toHaveCount(0)
      await expect(page.getByText(/Reprocessing is not supported/)).toBeVisible()
    }
    const actionIds = networkRecords
      .filter((item) => item.kind === 'request' && item.method === 'POST')
      .map((item) => item.requestId)
    expect(actionIds.every((value) => typeof value === 'string')).toBe(true)
    expect(new Set(actionIds).size).toBe(actionIds.length)
  })

  test('browser-verifies a technical preprocessing ERROR and a separate completed validation failure', async ({ page }) => {
    const errorIntake = await createInspection(page, state, 'valid_rgb_png_height_tiff', 'E2E-TECHNICAL-ERROR')
    expect(errorIntake.inspection_id).toBe(expectedIds.technicalError)
    expect((await validateInspection(page)).validation_outcome).toBe('VALIDATION_PASSED')
    tamperRgb(state, errorIntake.inspection_id)
    const technical = await processInspection(page)
    expect(technical.processing_status).toBe('ERROR')
    expect(technical.inspection_status).toBe('ERROR')
    expect(technical.preprocessing_outcome).toBe('PREPROCESSING_ERROR')
    expect(technical.mock_decision).toBeNull()
    expect(technical.inference).toBeNull()
    await expect(page.getByText('TECHNICAL ERROR').first()).toBeVisible()
    await expect(page.getByText(/^MOCK (PASS|FAIL|UNCERTAIN)$/)).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Run synthetic processing' })).toHaveCount(0)
    await assertNoConfidenceValue(page)

    const failureScenario = scenarioFiles(state, 'valid_different_dimensions').record
    expect(failureScenario.expected_technical_validation_outcome).toBe('VALIDATION_FAILED')
    const failedIntake = await createInspection(page, state, 'valid_different_dimensions', 'E2E-VALIDATION-FAILED')
    expect(failedIntake.inspection_id).toBe(expectedIds.validationFailure)
    const failedValidation = await validateInspection(page)
    expect(failedValidation.validation_outcome).toBe('VALIDATION_FAILED')
    await expect(page.getByText('DIMENSION_RELATIONSHIP_UNSUPPORTED')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Run synthetic processing' })).toHaveCount(0)
    await expect(page.getByText(/Processing is unavailable/)).toBeVisible()
    await expect(page.getByText(/^PCB FAIL$/)).toHaveCount(0)
    await assertNoConfidenceValue(page)

    const current = snapshot(state)
    expect(current.fixture_files_verified).toBe(true)
    const technicalArtifacts = current.artifact_integrity.filter((item) => item.inspection_id === expectedIds.technicalError)
    expect(technicalArtifacts.some((item) => item.actual_sha256 !== item.registered_sha256)).toBe(true)
    expect(current.report_files).toEqual([])
  })

  test('proves history pagination/filter/load behavior, read-only GETs, bounded queries, responsiveness and semantics', async ({ page, networkRecords }, testInfo) => {
    await page.goto(state.frontendUrl)
    await expect(page.locator('.history-table tbody tr')).toHaveCount(5)
    const firstPageIds = await page.locator('.history-table tbody .inspection-link').allTextContents()
    expect(firstPageIds[0]).toBe(expectedIds.validationFailure)
    expect(firstPageIds).toContain(expectedIds.pass)
    await expect(page.locator('.history-table thead th')).toHaveCount(7)
    await expect(page.getByText('Page 1')).toBeVisible()
    await expect(page.getByText(/total count/i)).toHaveCount(0)

    await page.getByLabel('Board ID').fill('E2E-PASS')
    await page.getByRole('button', { name: 'Apply filters' }).click()
    await expect(page.locator('.history-table tbody tr')).toHaveCount(1)
    await expect(page.getByText(expectedIds.pass)).toBeVisible()
    await page.getByRole('button', { name: 'Clear filters' }).click()

    seedHistory(state, 25)
    const beforeReads = snapshot(state)
    const recordStart = networkRecords.length
    await page.reload()
    await expect(page.locator('.history-table tbody tr')).toHaveCount(25)
    await expect(page.getByRole('button', { name: 'Next page' })).toBeEnabled()
    const historyRequests = networkRecords.slice(recordStart).filter((item) => item.kind === 'request')
    expect(historyRequests.filter((item) => /\/api\/v1\/inspections\/[0-9a-f-]+/.test(new URL(item.url).pathname))).toEqual([])
    await page.getByRole('button', { name: 'Next page' }).click()
    await expect(page.getByText('Page 2')).toBeVisible()
    await expect(page.locator('.history-table tbody tr')).toHaveCount(5)

    await page.goto(`${state.frontendUrl}/inspections/new`)
    await expect(page.getByRole('radio')).toHaveCount(2)
    const afterReads = snapshot(state)
    expect(afterReads.database_fingerprint).toBe(beforeReads.database_fingerprint)
    expect(afterReads.row_counts.audit_events).toBe(beforeReads.row_counts.audit_events)

    await page.goto(state.frontendUrl)
    await page.setViewportSize({ width: 768, height: 1024 })
    await expect(page.getByRole('navigation', { name: 'Operator navigation' })).toBeVisible()
    await expect(page.locator('.table-scroll')).toHaveCSS('overflow-x', 'auto')
    await page.getByRole('link', { name: 'New paired inspection' }).click()
    await expect(page.getByLabel(/RGB image/)).toBeVisible()
    await expect(page.getByRole('button', { name: 'Create inspection' })).toBeVisible()
    await assertNoDuplicateIds(page)

    const profiles = queryProfiles(state)
    const last = (method: string, route: string) => profiles.filter((item) => item.method === method && item.path === route).at(-1)
    const history = last('GET', '/api/v1/inspections')
    const recipes = last('GET', '/api/v1/recipes')
    const audit = last('GET', `/api/v1/inspections/${expectedIds.pass}/audit`)
    const report = last('GET', `/api/v1/inspections/${expectedIds.pass}/report`)
    expect(history?.select_queries).toBe(3)
    expect(recipes?.select_queries).toBe(1)
    expect(audit?.select_queries).toBe(2)
    expect(report?.select_queries).toBeGreaterThanOrEqual(4)
    expect(report?.select_queries).toBeLessThanOrEqual(10)
    for (const item of [history, recipes, audit, report]) {
      expect(item?.write_queries).toBe(0)
      expect(item?.elapsed_ms).toBeGreaterThanOrEqual(0)
    }
    boundedQueryProfile = { history, recipes, audit, report }
    await testInfo.attach('bounded-query-profile.json', {
      body: Buffer.from(JSON.stringify(boundedQueryProfile, null, 2)),
      contentType: 'application/json',
    })
  })

  test('covers structured browser error handling, request IDs, accessibility and marks the clean release run complete', async ({ page, networkRecords }, testInfo) => {
    await page.goto(`${state.frontendUrl}/inspections/not-a-uuid`)
    await expect(page.getByRole('alert')).toContainText('Malformed inspection ID')
    const missingId = 'ffffffff-ffff-4fff-8fff-ffffffffffff'
    await page.goto(`${state.frontendUrl}/inspections/${missingId}`)
    await expect(page.getByRole('alert')).toContainText('INSPECTION_NOT_FOUND')
    await expect(page.getByRole('alert')).toContainText(/Request ID/)

    const probe = await createInspection(page, state, 'valid_rgb_png_height_tiff', 'E2E-ERROR-PROBE')
    expect(probe.inspection_id).toBe(expectedIds.errorProbe)
    expect((await validateInspection(page)).validation_outcome).toBe('VALIDATION_PASSED')
    const controlledIds: string[] = []
    const processPattern = '**/api/v1/inspections/*/process'
    await page.route(processPattern, async (route) => {
      const requestId = route.request().headers()['x-request-id']
      expect(requestId).toMatch(/^[0-9a-f-]{36}$/)
      controlledIds.push(requestId)
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        headers: { 'X-Request-ID': 'controlled-503-request' },
        body: JSON.stringify({ code: 'SYNTHETIC_PROCESSING_NOT_CONFIGURED', message: 'Synthetic processing execution is not configured.', request_id: 'controlled-503-request' }),
      })
    })
    await page.getByRole('button', { name: 'Run synthetic processing' }).click()
    await expect(page.getByRole('alert')).toContainText('SYNTHETIC_PROCESSING_NOT_CONFIGURED')
    await expect(page.getByRole('alert')).toContainText('controlled-503-request')
    await page.unroute(processPattern)

    await page.route(processPattern, async (route) => {
      const requestId = route.request().headers()['x-request-id']
      controlledIds.push(requestId)
      await route.fulfill({
        status: 409,
        contentType: 'application/json',
        headers: { 'X-Request-ID': 'controlled-409-request' },
        body: JSON.stringify({ code: 'PROCESSING_LIFECYCLE_CONFLICT', message: 'Processing conflicts with the current lifecycle state.', request_id: 'controlled-409-request' }),
      })
    })
    await page.getByRole('button', { name: 'Run synthetic processing' }).click()
    await expect(page.getByRole('alert')).toContainText('PROCESSING_LIFECYCLE_CONFLICT')
    await expect(page.getByRole('alert')).toContainText('controlled-409-request')
    await page.unroute(processPattern)
    expect(new Set(controlledIds).size).toBe(2)

    await page.goto(`${state.frontendUrl}/inspections/new`)
    const valid = scenarioFiles(state, 'valid_rgb_png_height_tiff')
    await page.getByLabel(/Board ID/).fill('E2E-422')
    await page.getByRole('radio', { name: /synthetic-e2e .* 1\.0/i }).check()
    await page.getByLabel(/RGB image/).setInputFiles(valid.rgb)
    await page.getByLabel(/Height \/ depth map/).setInputFiles(valid.height)
    await page.route('**/api/v1/inspections', async (route) => {
      if (route.request().method() !== 'POST') return route.continue()
      await route.fulfill({
        status: 422,
        contentType: 'application/json',
        headers: { 'X-Request-ID': 'controlled-422-request' },
        body: JSON.stringify({ code: 'INCOMPLETE_OR_INVALID_MULTIPART_REQUEST', message: 'Required multipart fields are invalid.', request_id: 'controlled-422-request' }),
      })
    })
    await page.getByRole('button', { name: 'Create inspection' }).click()
    await expect(page.getByRole('alert')).toContainText('INCOMPLETE_OR_INVALID_MULTIPART_REQUEST')
    await expect(page.getByRole('alert')).toContainText('controlled-422-request')
    await page.unroute('**/api/v1/inspections')

    await page.route('**/api/v1/**', (route) => route.abort('connectionrefused'))
    await page.goto(state.frontendUrl)
    await expect(page.getByText('Backend unavailable')).toBeVisible()
    await expect(page.getByRole('alert')).toContainText('BACKEND_UNAVAILABLE')
    await page.unroute('**/api/v1/**')

    await expect(page.locator('main')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Inspection History' })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Inspection History' })).toBeVisible()
    await expect(page.getByRole('link', { name: 'New Inspection' })).toBeVisible()
    await assertNoDuplicateIds(page)
    const finalSnapshot = snapshot(state)
    expect(finalSnapshot.schema_version).toBe(3)
    expect(finalSnapshot.foreign_key_failures).toEqual([])
    expect(finalSnapshot.fixture_files_verified).toBe(true)
    expect(finalSnapshot.fixture_tree_sha256).toBe(state.fixtureTreeSha256)
    expect(finalSnapshot.report_files).toEqual([])
    expect(finalSnapshot.row_counts.model_versions).toBe(0)
    expect(passSnapshot.schema_version).toBe(3)
    finalDemonstration = {
      ...finalDemonstration,
      request_ids_unique: actionRequestIds.size === 3,
      cleanup_pending_global_teardown: true,
      schema_version: finalSnapshot.schema_version,
      foreign_key_check: 'passed',
      bounded_query_profile: boundedQueryProfile,
    }
    await testInfo.attach('final-release-demonstration.json', {
      body: Buffer.from(JSON.stringify(finalDemonstration, null, 2)),
      contentType: 'application/json',
    })
    process.stdout.write(`\nLIVE_DEMONSTRATION ${JSON.stringify(finalDemonstration)}\n`)
    writeFileSync(successMarker, 'success\n')
    expect(networkRecords.some((item) => item.requestId && item.requestId !== 'controlled-503-request')).toBe(true)
  })
})
