import { type FormEvent, useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getInspectionHistory } from '../api/inspections'
import { getDemoWorkspace, loadDemoWorkspace } from '../api/demoWorkspace'
import { toApiClientError, type ApiClientError } from '../api/errors'
import type { DemoWorkspaceResponse, HistoryFilters, InspectionHistoryResponse, InspectionStatus } from '../api/types'
import { ErrorPanel } from '../components/ErrorPanel'
import { StatusBadge } from '../components/StatusBadge'
import { formatTimestamp, toUtcIso } from '../utils/format'

const INSPECTION_STATUSES: InspectionStatus[] = [
  'RECEIVED', 'READY', 'VALIDATION_FAILED', 'PROCESSING', 'PASS', 'FAIL', 'UNCERTAIN', 'ERROR',
]

const EMPTY_FILTERS: HistoryFilters = {
  status: '',
  board_id: '',
  recipe_id: '',
  lot_id: '',
  validation_outcome: '',
  processing_status: '',
  mock_decision: '',
  has_validation: '',
  has_processing: '',
  created_from: '',
  created_to: '',
}

export function HistoryPage() {
  const [draftFilters, setDraftFilters] = useState<HistoryFilters>(EMPTY_FILTERS)
  const [filters, setFilters] = useState<HistoryFilters>(EMPTY_FILTERS)
  const [cursorStack, setCursorStack] = useState<Array<string | undefined>>([undefined])
  const [result, setResult] = useState<InspectionHistoryResponse | null>(null)
  const [error, setError] = useState<ApiClientError | null>(null)
  const [loading, setLoading] = useState(true)
  const [demoWorkspace, setDemoWorkspace] = useState<DemoWorkspaceResponse | null>(null)
  const [demoLoading, setDemoLoading] = useState(false)
  const [demoError, setDemoError] = useState<ApiClientError | null>(null)
  const [demoMessage, setDemoMessage] = useState<string | null>(null)

  const currentCursor = cursorStack.at(-1)
  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true)
    setError(null)
    try {
      const response = await getInspectionHistory(filters, currentCursor, 25, signal)
      setResult(response.data)
    } catch (caught) {
      const mapped = toApiClientError(caught)
      if (mapped.code !== 'REQUEST_ABORTED') setError(mapped)
    } finally {
      setLoading(false)
    }
  }, [currentCursor, filters])

  useEffect(() => {
    const controller = new AbortController()
    void load(controller.signal)
    return () => controller.abort()
  }, [load])

  useEffect(() => {
    const controller = new AbortController()
    void getDemoWorkspace(controller.signal)
      .then((response) => setDemoWorkspace(response.data))
      .catch((caught) => {
        const mapped = toApiClientError(caught)
        if (mapped.code !== 'REQUEST_ABORTED') setDemoError(mapped)
      })
    return () => controller.abort()
  }, [])

  const handleDemoLoad = async () => {
    setDemoLoading(true)
    setDemoError(null)
    setDemoMessage(null)
    try {
      const response = await loadDemoWorkspace()
      setDemoWorkspace(response.data)
      setDemoMessage(response.data.idempotent_existing
        ? 'Demo workspace was already loaded; existing inspections were preserved.'
        : 'Synthetic demo workspace loaded. Inspection history has been refreshed.')
      await load()
    } catch (caught) {
      setDemoError(toApiClientError(caught))
    } finally {
      setDemoLoading(false)
    }
  }

  const update = (key: keyof HistoryFilters, value: string | boolean) => {
    setDraftFilters((current) => ({ ...current, [key]: value }))
  }

  const applyFilters = (event: FormEvent) => {
    event.preventDefault()
    setCursorStack([undefined])
    setFilters({
      ...draftFilters,
      created_from: toUtcIso(draftFilters.created_from ?? ''),
      created_to: toUtcIso(draftFilters.created_to ?? ''),
    })
  }

  const clearFilters = () => {
    setDraftFilters(EMPTY_FILTERS)
    setFilters(EMPTY_FILTERS)
    setCursorStack([undefined])
  }

  const hasFilters = Object.values(filters).some((value) => value !== '' && value !== undefined)

  return (
    <section aria-labelledby="history-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Operator dashboard</p>
          <h2 id="history-title">Inspection History</h2>
          <p>Newest persisted inspections with compact validation and synthetic-processing evidence.</p>
        </div>
        <div className="page-actions">
          {demoWorkspace?.available && (
            <button
              className="button secondary"
              type="button"
              onClick={() => void handleDemoLoad()}
              disabled={demoLoading}
            >
              {demoLoading
                ? 'Loading demo workspaceâ€¦'
                : demoWorkspace.loaded
                  ? 'Verify Demo Workspace'
                  : 'Load Demo Workspace'}
            </button>
          )}
          <Link className="button primary" to="/inspections/new">New paired inspection</Link>
        </div>
      </div>

      <div className="feedback-region demo-feedback" aria-live="polite">
        {demoMessage && <p className="action-feedback">{demoMessage}</p>}
        {demoError && demoWorkspace?.available && (
          <ErrorPanel
            error={demoError}
            title="Demo workspace could not be loaded"
            onRetry={() => void handleDemoLoad()}
          />
        )}
      </div>

      <form className="filter-panel" onSubmit={applyFilters} aria-label="Inspection history filters">
        <div className="filter-grid">
          <label>Status<select value={draftFilters.status} onChange={(e) => update('status', e.target.value)}><option value="">Any status</option>{INSPECTION_STATUSES.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>Board ID<input value={draftFilters.board_id} onChange={(e) => update('board_id', e.target.value)} /></label>
          <label>Recipe ID<input value={draftFilters.recipe_id} onChange={(e) => update('recipe_id', e.target.value)} /></label>
          <label>Lot ID<input value={draftFilters.lot_id} onChange={(e) => update('lot_id', e.target.value)} /></label>
          <label>Validation outcome<select value={draftFilters.validation_outcome} onChange={(e) => update('validation_outcome', e.target.value)}><option value="">Any outcome</option><option>VALIDATION_PASSED</option><option>VALIDATION_FAILED</option><option>VALIDATION_ERROR</option></select></label>
          <label>Processing status<select value={draftFilters.processing_status} onChange={(e) => update('processing_status', e.target.value)}><option value="">Any status</option><option>STARTED</option><option>COMPLETED</option><option>ERROR</option></select></label>
          <label>Mock decision<select value={draftFilters.mock_decision} onChange={(e) => update('mock_decision', e.target.value)}><option value="">Any decision</option><option>PASS</option><option>FAIL</option><option>UNCERTAIN</option></select></label>
          <label>Has validation<select value={String(draftFilters.has_validation)} onChange={(e) => update('has_validation', e.target.value === '' ? '' : e.target.value === 'true')}><option value="">Either</option><option value="true">Yes</option><option value="false">No</option></select></label>
          <label>Has processing<select value={String(draftFilters.has_processing)} onChange={(e) => update('has_processing', e.target.value === '' ? '' : e.target.value === 'true')}><option value="">Either</option><option value="true">Yes</option><option value="false">No</option></select></label>
          <label>Created from<input type="datetime-local" value={draftFilters.created_from} onChange={(e) => update('created_from', e.target.value)} /></label>
          <label>Created to<input type="datetime-local" value={draftFilters.created_to} onChange={(e) => update('created_to', e.target.value)} /></label>
        </div>
        <div className="button-row">
          <button className="button secondary" type="submit" disabled={loading}>Apply filters</button>
          <button className="text-button" type="button" onClick={clearFilters} disabled={loading}>Clear filters</button>
        </div>
      </form>

      <div className="feedback-region" aria-live="polite">
        {loading && <p className="loading-state">Loading inspection history…</p>}
        {error && <ErrorPanel error={error} onRetry={() => void load()} />}
      </div>

      {!loading && !error && result?.items.length === 0 && (
        <div className="empty-state">
          <span className="empty-icon" aria-hidden="true">◇</span>
          <h3>{hasFilters ? 'No inspections match these filters' : 'No inspections received yet'}</h3>
          <p>{hasFilters ? 'Change or clear the exact-match filters.' : 'Create a paired RGB and height-file inspection to begin.'}</p>
        </div>
      )}

      {!loading && !error && result && result.items.length > 0 && (
        <>
          <div className="table-scroll">
            <table className="history-table">
              <caption className="sr-only">Persisted inspection history</caption>
              <thead><tr><th>Created</th><th>Inspection</th><th>Board / recipe</th><th>Lot / operator</th><th>Status</th><th>Validation</th><th>Synthetic processing</th></tr></thead>
              <tbody>{result.items.map((item) => (
                <tr key={item.inspection_id}>
                  <td>{formatTimestamp(item.created_at)}</td>
                  <td><Link className="inspection-link mono" to={`/inspections/${item.inspection_id}`}>{item.inspection_id}</Link></td>
                  <td><strong>{item.board_id}</strong><span>{item.recipe.recipe_id} · {item.recipe.recipe_version}</span></td>
                  <td><span>{item.lot_id ?? '—'}</span><span>{item.operator_id ?? '—'}</span></td>
                  <td><StatusBadge value={item.status} />{item.technical_error_code && <span className="technical-code">{item.technical_error_code}</span>}</td>
                  <td>{item.validation ? <><StatusBadge value={item.validation.outcome} /><span>{item.validation.blocking_findings} blocking · {item.validation.warnings} warnings</span></> : <span className="muted">Not validated</span>}</td>
                  <td>{item.processing ? <><StatusBadge value={item.processing.mock_decision ?? item.processing.processing_status} prefix={item.processing.mock_decision ? 'MOCK' : undefined} /><span>Synthetic · {item.processing.defect_type ?? 'No defect label'}</span></> : <span className="muted">Not processed</span>}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
          <div className="pagination" aria-label="History pagination">
            <button className="button secondary" type="button" disabled={cursorStack.length === 1 || loading} onClick={() => setCursorStack((current) => current.slice(0, -1))}>Previous page</button>
            <span>Page {cursorStack.length}</span>
            <button className="button secondary" type="button" disabled={!result.page.has_more || !result.page.next_cursor || loading} onClick={() => setCursorStack((current) => [...current, result.page.next_cursor ?? undefined])}>{loading ? 'Loading…' : 'Next page'}</button>
          </div>
        </>
      )}
    </section>
  )
}
