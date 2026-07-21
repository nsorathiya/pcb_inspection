import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getInspectionReport } from '../api/inspections'
import { ApiClientError } from '../api/errors'
import type { InspectionDevelopmentReportResponse } from '../api/types'
import { INSPECTION_ID } from '../test/fixtures'
import { InspectionReportPage } from './InspectionReportPage'

vi.mock('../api/inspections', () => ({ getInspectionReport: vi.fn() }))
const reportMock = vi.mocked(getInspectionReport)

function reportResponse(decision?: 'PASS' | 'FAIL' | 'UNCERTAIN', technicalError = false): InspectionDevelopmentReportResponse {
  const processing = decision || technicalError ? {
    processing_run_id: '33333333-3333-4333-8333-333333333333',
    validation_id: '22222222-2222-4222-8222-222222222222',
    processing_key: 'c'.repeat(64), lifecycle_status: technicalError ? 'ERROR' as const : 'COMPLETED' as const,
    preprocessing_policy: { policy_id: 'synthetic-paired-rgb-height', policy_version: '1.0' },
    preprocessing_implementation: { implementation_id: 'synthetic-copy', implementation_version: '1.0' },
    inference_policy: { policy_id: 'synthetic-deterministic-mock-inference', policy_version: '1.0' },
    engine: { engine_id: 'mock-engine', engine_version: '1.0', engine_type: 'MOCK' },
    started_at: '2026-07-21T08:01:00Z', completed_at: '2026-07-21T08:02:00Z',
    final_decision: technicalError ? null : decision ?? null,
    error: technicalError ? { code: 'INFERENCE_ERROR', message: 'Inspection inference did not complete successfully.' } : null,
    preprocessing: { findings: [] },
    inference: technicalError ? null : { decision: decision ?? null, defect_type: decision === 'FAIL' ? 'missing_part' : null, findings: [] },
    synthetic_input: true, mock_preprocessing: true, mock_inference: !technicalError, production_approved: false as const,
  } : null
  return {
    report: {
      contract_version: 'pcb-aoi-inspection-development-report/1.0', inspection_id: INSPECTION_ID,
      development_only: true, production_approved: false,
      synthetic_evidence_present: processing !== null, mock_inference_present: Boolean(decision),
      inspection: { created_at: '2026-07-21T08:00:00Z', board_id: 'BOARD-A', recipe_id: 'RECIPE-A', recipe_version: '1.0', lot_id: null, operator_id: null, status: technicalError ? 'ERROR' : decision ?? 'RECEIVED', error: technicalError ? { code: 'INFERENCE_ERROR', message: 'Inspection inference did not complete successfully.' } : null },
      artifacts: [], validation: null, processing, audit: [],
      limitations: ['This report is development-only.', 'No real AI model was executed.', 'No confidence was produced.'],
    },
    report_sha256: 'd'.repeat(64), request_id: 'report-request',
  }
}

function renderPage(id = INSPECTION_ID) {
  return render(<MemoryRouter initialEntries={[`/inspections/${id}/report`]}><Routes><Route path="/inspections/:inspectionId/report" element={<InspectionReportPage />} /></Routes></MemoryRouter>)
}

describe('development report page', () => {
  beforeEach(() => {
    reportMock.mockReset()
    reportMock.mockResolvedValue({ data: reportResponse(), requestId: 'report-request' })
  })

  it('renders a RECEIVED partial report with persistent warning, headings, and hash', async () => {
    renderPage()
    expect(await screen.findByRole('heading', { name: 'Development Report' })).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Development-only · nonproduction')
    expect(screen.getByText(/Report SHA-256:/)).toHaveTextContent('d'.repeat(64))
    expect(screen.getByText(/Technical validation evidence is not available/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Inspection summary' })).toBeInTheDocument()
  })

  it.each(['PASS', 'UNCERTAIN'] as const)('renders MOCK %s without a defect type', async (decision) => {
    reportMock.mockResolvedValue({ data: reportResponse(decision), requestId: 'report-request' })
    renderPage()
    expect(await screen.findByText(`MOCK ${decision}`)).toBeInTheDocument()
    expect(screen.queryByText(/Mock defect type:/)).not.toBeInTheDocument()
  })

  it('renders MOCK FAIL with the authoritative defect type', async () => {
    reportMock.mockResolvedValue({ data: reportResponse('FAIL'), requestId: 'report-request' })
    renderPage()
    expect(await screen.findByText('MOCK FAIL')).toBeInTheDocument()
    expect(screen.getByText('Mock defect type: missing_part')).toBeInTheDocument()
  })

  it('renders persisted technical errors without a mock result', async () => {
    reportMock.mockResolvedValue({ data: reportResponse(undefined, true), requestId: 'report-request' })
    renderPage()
    expect(await screen.findAllByText(/TECHNICAL ERROR/)).not.toHaveLength(0)
    expect(screen.queryByText(/MOCK (PASS|FAIL|UNCERTAIN)/)).not.toBeInTheDocument()
  })

  it('downloads the exact report with a safe filename and revokes the object URL', async () => {
    const response = reportResponse('FAIL')
    reportMock.mockResolvedValue({ data: response, requestId: 'report-request' })
    const createUrl = vi.fn(() => 'blob:report')
    const revokeUrl = vi.fn()
    vi.stubGlobal('URL', { ...URL, createObjectURL: createUrl, revokeObjectURL: revokeUrl })
    let downloadName = ''
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { downloadName = this.download })
    renderPage()
    await userEvent.click(await screen.findByRole('button', { name: 'Download JSON' }))
    expect(downloadName).toBe(`inspection-${INSPECTION_ID}-development-report.json`)
    expect(createUrl).toHaveBeenCalledWith(expect.any(Blob))
    expect(revokeUrl).toHaveBeenCalledWith('blob:report')
  })

  it('invokes browser print and refreshes authoritative evidence', async () => {
    const print = vi.spyOn(window, 'print').mockImplementation(() => undefined)
    renderPage()
    await userEvent.click(await screen.findByRole('button', { name: 'Print' }))
    expect(print).toHaveBeenCalledOnce()
    await userEvent.click(screen.getByRole('button', { name: 'Refresh report' }))
    await waitFor(() => expect(reportMock).toHaveBeenCalledTimes(2))
  })

  it.each([
    [404, 'INSPECTION_NOT_FOUND'],
    [500, 'DEVELOPMENT_REPORT_INCONSISTENT'],
  ] as const)('renders structured %s errors', async (status, code) => {
    reportMock.mockRejectedValue(new ApiClientError(status, code, 'Safe report error.', 'report-error-request'))
    renderPage()
    const alert = await screen.findByRole('alert', { name: '' }).catch(() => screen.findByText(code))
    expect(alert).toBeTruthy()
    expect(screen.getByText(code)).toBeInTheDocument()
    expect(screen.getByText('report-error-request')).toBeInTheDocument()
  })

  it('rejects malformed UUIDs before requesting a report', () => {
    renderPage('bad-id')
    expect(screen.getByRole('alert')).toHaveTextContent('Malformed inspection ID')
    expect(reportMock).not.toHaveBeenCalled()
  })
})
