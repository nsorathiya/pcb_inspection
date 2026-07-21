import { useCallback, useEffect, useState } from 'react'
import { getInspectionAudit } from '../api/inspections'
import { toApiClientError, type ApiClientError } from '../api/errors'
import type { AuditTimelineItem, JsonValue } from '../api/types'
import { formatTimestamp } from '../utils/format'
import { ErrorPanel } from './ErrorPanel'

const ACTION_NAMES: Record<string, string> = {
  INSPECTION_RECEIVED: 'Inspection received',
  INSPECTION_INTAKE_FAILED: 'Inspection intake failed',
  INSPECTION_VALIDATION_PASSED: 'Technical validation passed',
  INSPECTION_VALIDATION_FAILED: 'Technical validation failed',
  INSPECTION_VALIDATION_ERROR: 'Technical validation error',
  INSPECTION_PROCESSING_STARTED: 'Synthetic processing started',
  INSPECTION_MOCK_RESULT_PASS: 'MOCK PASS recorded',
  INSPECTION_MOCK_RESULT_FAIL: 'MOCK FAIL recorded',
  INSPECTION_MOCK_RESULT_UNCERTAIN: 'MOCK UNCERTAIN recorded',
  INSPECTION_PROCESSING_ERROR: 'Technical processing error',
}

function displayValue(value: JsonValue): string {
  if (value === null) return 'Not available'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function AuditTimeline({ inspectionId }: { inspectionId: string }) {
  const [items, setItems] = useState<AuditTimelineItem[]>([])
  const [cursor, setCursor] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ApiClientError | null>(null)

  const load = useCallback(async (nextCursor?: string, signal?: AbortSignal) => {
    setLoading(true)
    setError(null)
    try {
      const response = await getInspectionAudit(inspectionId, nextCursor, 50, signal)
      setItems((current) => nextCursor ? [...current, ...response.data.items] : response.data.items)
      setCursor(response.data.page.next_cursor)
      setHasMore(response.data.page.has_more)
    } catch (caught) {
      const mapped = toApiClientError(caught)
      if (mapped.code !== 'REQUEST_ABORTED') setError(mapped)
    } finally {
      setLoading(false)
    }
  }, [inspectionId])

  useEffect(() => {
    const controller = new AbortController()
    void load(undefined, controller.signal)
    return () => controller.abort()
  }, [load])

  return (
    <section className="panel audit-panel" id="audit-timeline" aria-labelledby="audit-title">
      <div className="panel-heading">
        <div><p className="step-number">Read-only evidence</p><h3 id="audit-title">Audit Timeline</h3></div>
        <p>Persisted actions only. Viewing this timeline does not create an audit event.</p>
      </div>
      {loading && items.length === 0 && <p className="loading-state">Loading persisted audit events…</p>}
      {error && <ErrorPanel error={error} title="Audit timeline unavailable" onRetry={() => void load()} />}
      {!loading && !error && items.length === 0 && <p className="neutral-state">No persisted audit events are available for this inspection.</p>}
      {items.length > 0 && <ol className="audit-timeline">{items.map((item) => (
        <li key={item.audit_event_id}>
          <div className="audit-marker" aria-hidden="true" />
          <article>
            <div className="subpanel-heading"><div><h4>{ACTION_NAMES[item.action] ?? 'Persisted lifecycle event'}</h4><code>{item.action}</code></div><time dateTime={item.created_at}>{formatTimestamp(item.created_at)}</time></div>
            {(item.actor_id || item.request_id) && <dl className="inline-metadata"><div><dt>Actor ID</dt><dd>{item.actor_id ?? 'Not recorded'}</dd></div><div><dt>Historical request ID</dt><dd className="mono">{item.request_id ?? 'Not recorded'}</dd></div></dl>}
            {Object.keys(item.details).length > 0 && <dl className="audit-details">{Object.entries(item.details).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{displayValue(value)}</dd></div>)}</dl>}
            {item.mock_result && <p className="mock-result-warning"><strong>MOCK {item.mock_result}</strong> — development-only and not approved for production PCB disposition.</p>}
            {item.details_redacted && <p className="redacted-indicator">Some persisted details were redacted by the safety projection.</p>}
          </article>
        </li>
      ))}</ol>}
      {hasMore && cursor && <button className="button secondary" type="button" onClick={() => void load(cursor)} disabled={loading}>{loading ? 'Loading…' : 'Load more audit events'}</button>}
    </section>
  )
}
