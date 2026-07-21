import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getInspectionAudit } from '../api/inspections'
import { ApiClientError } from '../api/errors'
import { INSPECTION_ID } from '../test/fixtures'
import { AuditTimeline } from './AuditTimeline'

vi.mock('../api/inspections', () => ({ getInspectionAudit: vi.fn() }))
const auditMock = vi.mocked(getInspectionAudit)

describe('audit timeline', () => {
  beforeEach(() => {
    auditMock.mockReset()
    auditMock.mockResolvedValue({
      data: { items: [], page: { limit: 50, has_more: false, next_cursor: null }, request_id: 'audit-request' },
      requestId: 'audit-request',
    })
  })

  it('uses semantic ordered markup and renders the empty state without fabricated events', async () => {
    const { container } = render(<AuditTimeline inspectionId={INSPECTION_ID} />)
    expect(await screen.findByText('No persisted audit events are available for this inspection.')).toBeInTheDocument()
    expect(container.querySelector('ol')).toBeNull()
    expect(screen.queryByText(/inspection received/i)).not.toBeInTheDocument()
  })

  it('renders ordered safe details and a redaction indicator', async () => {
    auditMock.mockResolvedValue({ data: {
      items: [
        { audit_event_id: 'a', inspection_id: INSPECTION_ID, action: 'INSPECTION_RECEIVED', created_at: '2026-07-21T08:00:00Z', actor_id: 'operator-1', request_id: 'historical-1', details: { station_id: 'station-1' }, details_redacted: false, development_only: null, mock_result: null, production_approved: null },
        { audit_event_id: 'b', inspection_id: INSPECTION_ID, action: 'INSPECTION_MOCK_RESULT_FAIL', created_at: '2026-07-21T08:01:00Z', actor_id: null, request_id: null, details: { final_inspection_status: 'FAIL' }, details_redacted: true, development_only: true, mock_result: 'FAIL', production_approved: false },
      ], page: { limit: 50, has_more: false, next_cursor: null }, request_id: 'audit-request',
    }, requestId: 'audit-request' })
    const { container } = render(<AuditTimeline inspectionId={INSPECTION_ID} />)
    expect(await screen.findByText('Inspection received')).toBeInTheDocument()
    const events = container.querySelectorAll('ol.audit-timeline > li')
    expect(events).toHaveLength(2)
    expect(events[0]).toHaveTextContent('station-1')
    expect(events[1]).toHaveTextContent('MOCK FAIL')
    expect(events[1]).toHaveTextContent('redacted by the safety projection')
  })

  it('passes the opaque cursor unchanged when loading more', async () => {
    auditMock.mockResolvedValueOnce({ data: { items: [], page: { limit: 50, has_more: true, next_cursor: 'opaque.cursor/value' }, request_id: 'one' }, requestId: 'one' })
      .mockResolvedValueOnce({ data: { items: [], page: { limit: 50, has_more: false, next_cursor: null }, request_id: 'two' }, requestId: 'two' })
    render(<AuditTimeline inspectionId={INSPECTION_ID} />)
    await userEvent.click(await screen.findByRole('button', { name: 'Load more audit events' }))
    await waitFor(() => expect(auditMock).toHaveBeenCalledTimes(2))
    expect(auditMock.mock.calls[1]?.[1]).toBe('opaque.cursor/value')
  })

  it('shows structured retrieval errors with request IDs', async () => {
    auditMock.mockRejectedValue(new ApiClientError(500, 'AUDIT_RETRIEVAL_FAILED', 'The audit timeline could not be retrieved.', 'audit-error-request'))
    render(<AuditTimeline inspectionId={INSPECTION_ID} />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('AUDIT_RETRIEVAL_FAILED')
    expect(alert).toHaveTextContent('audit-error-request')
  })
})
