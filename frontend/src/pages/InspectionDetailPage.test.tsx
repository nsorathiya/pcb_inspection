import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  getInspection,
  getProcessing,
  getValidation,
  runProcessing,
  runValidation,
} from '../api/inspections'
import { ApiClientError } from '../api/errors'
import { detailResponse, INSPECTION_ID, processingResponse, validationResponse } from '../test/fixtures'
import { InspectionDetailPage } from './InspectionDetailPage'

vi.mock('../api/inspections', () => ({
  getInspection: vi.fn(),
  getProcessing: vi.fn(),
  getValidation: vi.fn(),
  runProcessing: vi.fn(),
  runValidation: vi.fn(),
}))

const detailMock = vi.mocked(getInspection)
const getValidationMock = vi.mocked(getValidation)
const getProcessingMock = vi.mocked(getProcessing)
const runValidationMock = vi.mocked(runValidation)
const runProcessingMock = vi.mocked(runProcessing)
const missingValidation = new ApiClientError(404, 'INSPECTION_VALIDATION_NOT_FOUND', 'No validation result exists.', 'validation-get')
const missingProcessing = new ApiClientError(404, 'INSPECTION_PROCESSING_NOT_FOUND', 'No processing result exists.', 'processing-get')

function renderPage(id = INSPECTION_ID) {
  return render(
    <MemoryRouter initialEntries={[`/inspections/${id}`]}>
      <Routes><Route path="/inspections/:inspectionId" element={<InspectionDetailPage />} /></Routes>
    </MemoryRouter>,
  )
}

function mockState(status: Parameters<typeof detailResponse>[0], options?: { validation?: boolean; processing?: 'PASS' | 'FAIL' | 'UNCERTAIN' }) {
  detailMock.mockResolvedValue({ data: detailResponse(status), requestId: 'detail-request' })
  getValidationMock.mockImplementation(() => options?.validation
    ? Promise.resolve({ data: validationResponse, requestId: 'validation-request' })
    : Promise.reject(missingValidation))
  getProcessingMock.mockImplementation(() => options?.processing
    ? Promise.resolve({ data: processingResponse(options.processing), requestId: 'processing-request' })
    : Promise.reject(missingProcessing))
}

describe('inspection detail workflow', () => {
  beforeEach(() => mockState('RECEIVED'))

  it('rejects malformed IDs before making API calls', () => {
    renderPage('not-a-uuid')
    expect(screen.getByRole('alert')).toHaveTextContent('Malformed inspection ID')
    expect(detailMock).not.toHaveBeenCalled()
  })

  it('shows only the validation action for RECEIVED and neutral child states', async () => {
    renderPage()
    expect(await screen.findByRole('button', { name: 'Run technical validation' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: 'Run synthetic processing' })).not.toBeInTheDocument()
    expect(screen.getByText('Not validated yet')).toBeInTheDocument()
    expect(screen.getByText('Not processed yet')).toBeInTheDocument()
  })

  it('shows only processing for READY and renders technical validation without PCB PASS wording', async () => {
    mockState('READY', { validation: true })
    renderPage()
    expect(await screen.findByRole('button', { name: 'Run synthetic processing' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: 'Run technical validation' })).not.toBeInTheDocument()
    expect(screen.getByText(/VALIDATION_PASSED means technically ready for preprocessing/)).toBeInTheDocument()
    expect(screen.queryByText(/^PCB PASS$/i)).not.toBeInTheDocument()
    const first = screen.getByText('FIRST_FINDING')
    const second = screen.getByText('SECOND_FINDING')
    expect(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it.each(['VALIDATION_FAILED', 'PROCESSING', 'PASS', 'FAIL', 'UNCERTAIN', 'ERROR'] as const)(
    '%s blocks unsupported processing actions',
    async (status) => {
      mockState(status, { validation: status !== 'VALIDATION_FAILED' })
      renderPage()
      expect(await screen.findByText(status, { selector: '.status-badge' })).toBeInTheDocument()
      expect(screen.queryByRole('button', { name: 'Run synthetic processing' })).not.toBeInTheDocument()
    },
  )

  it.each(['PASS', 'FAIL', 'UNCERTAIN'] as const)('renders MOCK %s with the persistent nonproduction result warning', async (decision) => {
    mockState(decision, { validation: true, processing: decision })
    renderPage()
    expect(await screen.findByRole('heading', { name: `MOCK ${decision}` })).toBeInTheDocument()
    expect(screen.getByText(/not a real AI prediction and not approved for production PCB disposition/)).toBeInTheDocument()
    expect(document.body).not.toHaveTextContent(/confidence/i)
    if (decision === 'FAIL') expect(screen.getByText('missing_part')).toBeInTheDocument()
    else expect(screen.queryByText('missing_part')).not.toBeInTheDocument()
  })

  it('renders a completed validation failure and its ordered findings as technical evidence', async () => {
    const failedValidation = {
      ...validationResponse,
      validation_outcome: 'VALIDATION_FAILED' as const,
      inspection_status: 'VALIDATION_FAILED' as const,
      summary: {
        ...validationResponse.summary,
        error_count: 1,
        blocking_count: 1,
        technically_ready: false,
      },
    }
    detailMock.mockResolvedValue({ data: detailResponse('VALIDATION_FAILED'), requestId: 'detail-request' })
    getValidationMock.mockResolvedValue({ data: failedValidation, requestId: 'validation-request' })
    getProcessingMock.mockRejectedValue(missingProcessing)
    renderPage()
    const validationHeading = await screen.findByRole('heading', { name: 'Technical validation' })
    const validationPanel = validationHeading.closest('section')
    expect(validationPanel).not.toBeNull()
    expect(within(validationPanel as HTMLElement).getByText('VALIDATION_FAILED', { selector: '.status-badge' })).toBeInTheDocument()
    expect(within(validationPanel as HTMLElement).getByText('FIRST_FINDING')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Run synthetic processing' })).not.toBeInTheDocument()
  })

  it('renders persisted processing failure as a technical error without a mock defect label', async () => {
    const failedProcessing = {
      ...processingResponse(),
      processing_status: 'ERROR' as const,
      inspection_status: 'ERROR' as const,
      mock_decision: null,
      defect_type: null,
      inference_execution_outcome: null,
      inference: null,
    }
    detailMock.mockResolvedValue({ data: detailResponse('ERROR'), requestId: 'detail-request' })
    getValidationMock.mockResolvedValue({ data: validationResponse, requestId: 'validation-request' })
    getProcessingMock.mockResolvedValue({ data: failedProcessing, requestId: 'processing-request' })
    renderPage()
    expect(await screen.findByRole('heading', { name: 'TECHNICAL ERROR' })).toBeInTheDocument()
    expect(screen.queryByText('Mock taxonomy label:')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Run synthetic processing' })).not.toBeInTheDocument()
  })

  it('labels an idempotent processing response as persisted replay evidence', async () => {
    const replay = { ...processingResponse('PASS'), lifecycle_idempotent_existing: true }
    detailMock.mockResolvedValue({ data: detailResponse('PASS'), requestId: 'detail-request' })
    getValidationMock.mockResolvedValue({ data: validationResponse, requestId: 'validation-request' })
    getProcessingMock.mockResolvedValue({ data: replay, requestId: 'processing-request' })
    renderPage()
    expect(await screen.findByText('This response is an exact replay of persisted evidence.')).toBeInTheDocument()
  })

  it('runs validation once, disables duplicate action, and refreshes authoritative records', async () => {
    let resolveAction: (() => void) | undefined
    runValidationMock.mockImplementation(() => new Promise((resolve) => {
      resolveAction = () => resolve({ data: validationResponse, requestId: 'validation-action' })
    }))
    renderPage()
    const button = await screen.findByRole('button', { name: 'Run technical validation' })
    await userEvent.click(button)
    expect(screen.getByRole('button', { name: 'Validating…' })).toBeDisabled()
    expect(runValidationMock).toHaveBeenCalledTimes(1)
    resolveAction?.()
    await waitFor(() => expect(detailMock.mock.calls.length).toBeGreaterThan(1))
  })

  it('renders 503 processing configuration errors with request IDs', async () => {
    mockState('READY', { validation: true })
    runProcessingMock.mockRejectedValue(new ApiClientError(503, 'SYNTHETIC_PROCESSING_NOT_CONFIGURED', 'Synthetic processing execution is not configured.', 'process-503-request'))
    renderPage()
    await userEvent.click(await screen.findByRole('button', { name: 'Run synthetic processing' }))
    const alert = await screen.findByRole('alert')
    expect(within(alert).getByText('SYNTHETIC_PROCESSING_NOT_CONFIGURED')).toBeInTheDocument()
    expect(within(alert).getByText('process-503-request')).toBeInTheDocument()
  })

  it('manual refresh retrieves authoritative detail and child state again', async () => {
    renderPage()
    await screen.findByText('BOARD-A')
    await userEvent.click(screen.getByRole('button', { name: 'Refresh' }))
    await waitFor(() => expect(detailMock).toHaveBeenCalledTimes(2))
    expect(getValidationMock).toHaveBeenCalledTimes(2)
    expect(getProcessingMock).toHaveBeenCalledTimes(2)
  })
})
