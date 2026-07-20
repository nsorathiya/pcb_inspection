import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getInspectionHistory } from '../api/inspections'
import { ApiClientError } from '../api/errors'
import { HistoryPage } from './HistoryPage'
import { emptyHistoryResponse, INSPECTION_ID } from '../test/fixtures'
import type { InspectionHistoryResponse } from '../api/types'

vi.mock('../api/inspections', () => ({ getInspectionHistory: vi.fn() }))
const historyMock = vi.mocked(getInspectionHistory)

const populated: InspectionHistoryResponse = {
  ...emptyHistoryResponse,
  items: [{
    inspection_id: INSPECTION_ID,
    status: 'FAIL',
    board_id: 'BOARD-A',
    recipe: { recipe_id: 'RECIPE-A', recipe_version: '1.0' },
    lot_id: 'LOT-7',
    operator_id: 'operator-1',
    created_at: '2026-07-20T10:00:00Z',
    started_at: '2026-07-20T10:01:00Z',
    completed_at: '2026-07-20T10:02:00Z',
    technical_error_code: null,
    validation: { validation_id: 'validation-id', outcome: 'VALIDATION_PASSED', policy_id: 'development-native-rgb-height', policy_version: '1.0', validator_version: '1.0.0', completed_at: '2026-07-20T10:00:30Z', total_findings: 0, blocking_findings: 0, warnings: 0, errors: 0 },
    processing: { processing_run_id: 'run-id', processing_status: 'COMPLETED', preprocessing_id: 'pre-id', preprocessing_outcome: 'PREPROCESSING_SUCCEEDED', inference_id: 'inf-id', inference_execution_outcome: 'INFERENCE_SUCCEEDED', mock_decision: 'FAIL', defect_type: 'missing_part', started_at: '2026-07-20T10:01:00Z', completed_at: '2026-07-20T10:02:00Z', synthetic_input: true, mock_preprocessing: true, mock_inference: true, production_approved: false },
  }],
  page: { limit: 25, has_more: true, next_cursor: 'opaque-next-cursor' },
}

function renderPage() {
  return render(<MemoryRouter><HistoryPage /></MemoryRouter>)
}

describe('inspection history page', () => {
  beforeEach(() => {
    historyMock.mockReset()
    historyMock.mockResolvedValue({ data: emptyHistoryResponse, requestId: 'request-id' })
  })

  it('renders the unfiltered empty database state', async () => {
    renderPage()
    expect(screen.getByText(/Loading inspection history/)).toBeInTheDocument()
    expect(await screen.findByText('No inspections received yet')).toBeInTheDocument()
  })

  it('renders compact rows with explicit mock terminology and no per-row calls', async () => {
    historyMock.mockResolvedValue({ data: populated, requestId: 'request-id' })
    renderPage()
    expect(await screen.findByRole('link', { name: INSPECTION_ID })).toHaveAttribute('href', `/inspections/${INSPECTION_ID}`)
    expect(screen.getByText('MOCK FAIL')).toBeInTheDocument()
    expect(screen.getByText(/Synthetic · missing_part/)).toBeInTheDocument()
    expect(screen.getByText('VALIDATION_PASSED', { selector: '.status-badge' })).toBeInTheDocument()
    expect(historyMock).toHaveBeenCalledTimes(1)
  })

  it('passes the opaque cursor unchanged for the next page', async () => {
    historyMock
      .mockResolvedValueOnce({ data: populated, requestId: 'request-1' })
      .mockResolvedValueOnce({ data: emptyHistoryResponse, requestId: 'request-2' })
    renderPage()
    await userEvent.click(await screen.findByRole('button', { name: 'Next page' }))
    await waitFor(() => expect(historyMock).toHaveBeenLastCalledWith(
      expect.any(Object), 'opaque-next-cursor', 25, expect.any(AbortSignal),
    ))
  })

  it('submits supported exact filters and renders an empty filtered state', async () => {
    renderPage()
    await screen.findByText('No inspections received yet')
    await userEvent.type(screen.getByLabelText('Board ID'), 'BOARD-17')
    await userEvent.selectOptions(screen.getByLabelText('Status'), 'READY')
    await userEvent.click(screen.getByRole('button', { name: 'Apply filters' }))
    await waitFor(() => expect(historyMock).toHaveBeenLastCalledWith(
      expect.objectContaining({ board_id: 'BOARD-17', status: 'READY' }), undefined, 25, expect.any(AbortSignal),
    ))
    expect(await screen.findByText('No inspections match these filters')).toBeInTheDocument()
  })

  it('shows structured errors and request IDs with retry', async () => {
    historyMock.mockRejectedValue(new ApiClientError(500, 'INSPECTION_HISTORY_READ_FAILED', 'History unavailable.', 'history-error-request'))
    renderPage()
    expect(await screen.findByText('History unavailable.')).toBeInTheDocument()
    expect(screen.getByText('INSPECTION_HISTORY_READ_FAILED')).toBeInTheDocument()
    expect(screen.getByText('history-error-request')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeEnabled()
  })
})
