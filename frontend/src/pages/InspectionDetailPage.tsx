import { useCallback, useEffect, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import {
  getInspection,
  getProcessing,
  getValidation,
  runProcessing,
  runValidation,
} from '../api/inspections'
import { ApiClientError, toApiClientError } from '../api/errors'
import type {
  InspectionDetailResponse,
  InspectionProcessingResponse,
  InspectionValidationResponse,
} from '../api/types'
import { ErrorPanel } from '../components/ErrorPanel'
import { LifecycleSummary } from '../components/LifecycleSummary'
import { ProcessingPanel } from '../components/ProcessingPanel'
import { StatusBadge } from '../components/StatusBadge'
import { ValidationPanel } from '../components/ValidationPanel'
import { formatBytes, formatTimestamp } from '../utils/format'

const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

interface IntakeLocationState {
  intakeCreated?: boolean
  status?: string
  inspectionId?: string
}

function isNeutralMissing(error: unknown, code: string): boolean {
  return error instanceof ApiClientError && error.status === 404 && error.code === code
}

export function InspectionDetailPage() {
  const { inspectionId = '' } = useParams()
  const location = useLocation()
  const locationState = (location.state ?? {}) as IntakeLocationState
  const validId = CANONICAL_UUID.test(inspectionId)
  const [inspection, setInspection] = useState<InspectionDetailResponse | null>(null)
  const [validation, setValidation] = useState<InspectionValidationResponse | null>(null)
  const [processing, setProcessing] = useState<InspectionProcessingResponse | null>(null)
  const [error, setError] = useState<ApiClientError | null>(null)
  const [loading, setLoading] = useState(validId)
  const [action, setAction] = useState<'validate' | 'process' | null>(null)
  const [feedback, setFeedback] = useState(
    locationState.intakeCreated
      ? `Inspection ${locationState.inspectionId} created with status ${locationState.status}.`
      : '',
  )

  const load = useCallback(async (signal?: AbortSignal) => {
    if (!validId) return
    setLoading(true)
    setError(null)
    try {
      const detail = await getInspection(inspectionId, signal)
      setInspection(detail.data)
      const [validationResult, processingResult] = await Promise.allSettled([
        getValidation(inspectionId, signal),
        getProcessing(inspectionId, signal),
      ])
      if (validationResult.status === 'fulfilled') setValidation(validationResult.value.data)
      else if (isNeutralMissing(validationResult.reason, 'INSPECTION_VALIDATION_NOT_FOUND')) setValidation(null)
      else throw validationResult.reason
      if (processingResult.status === 'fulfilled') setProcessing(processingResult.value.data)
      else if (isNeutralMissing(processingResult.reason, 'INSPECTION_PROCESSING_NOT_FOUND')) setProcessing(null)
      else throw processingResult.reason
    } catch (caught) {
      const mapped = toApiClientError(caught)
      if (mapped.code !== 'REQUEST_ABORTED') setError(mapped)
    } finally {
      setLoading(false)
    }
  }, [inspectionId, validId])

  useEffect(() => {
    const controller = new AbortController()
    void load(controller.signal)
    return () => controller.abort()
  }, [load])

  const validate = async () => {
    if (action) return
    setAction('validate')
    setError(null)
    setFeedback('Running technical validation…')
    try {
      const result = await runValidation(inspectionId)
      setValidation(result.data)
      setFeedback(`Technical validation completed: ${result.data.validation_outcome}.`)
      await load()
    } catch (caught) {
      setError(toApiClientError(caught))
      setFeedback('Technical validation request failed.')
    } finally {
      setAction(null)
    }
  }

  const process = async () => {
    if (action) return
    setAction('process')
    setError(null)
    setFeedback('Running trusted synthetic processing…')
    try {
      const result = await runProcessing(inspectionId)
      setProcessing(result.data)
      setFeedback(`Synthetic processing completed: ${result.data.processing_status}.`)
      await load()
    } catch (caught) {
      setError(toApiClientError(caught))
      setFeedback('Synthetic processing request failed.')
    } finally {
      setAction(null)
    }
  }

  if (!validId) {
    return <section className="not-found" role="alert"><p className="eyebrow">Invalid route parameter</p><h2>Malformed inspection ID</h2><p>The inspection route requires a lowercase, hyphenated UUID.</p><Link className="button secondary" to="/">Return to inspection history</Link></section>
  }

  return (
    <section aria-labelledby="inspection-detail-title">
      <div className="page-heading">
        <div><p className="eyebrow">Persisted workflow evidence</p><h2 id="inspection-detail-title">Inspection Detail</h2><p className="mono page-identifier">{inspectionId}</p></div>
        <button className="button secondary" type="button" onClick={() => void load()} disabled={loading || Boolean(action)}>{loading ? 'Refreshing…' : 'Refresh'}</button>
      </div>
      <div className="feedback-region" aria-live="polite">{feedback && <p className="action-feedback">{feedback}</p>}{loading && !inspection && <p className="loading-state">Loading persisted inspection evidence…</p>}{error && <ErrorPanel error={error} onRetry={() => void load()} />}</div>

      {inspection && (
        <>
          <section className="inspection-overview panel" aria-labelledby="overview-title">
            <div className="panel-heading"><div><p className="step-number">Current state</p><h3 id="overview-title">Inspection overview</h3></div><StatusBadge value={inspection.status} /></div>
            <dl className="metadata-grid overview-grid">
              <div><dt>Inspection ID</dt><dd className="mono">{inspection.inspection_id}</dd></div>
              <div><dt>Created</dt><dd>{formatTimestamp(inspection.created_at)}</dd></div>
              <div><dt>Board</dt><dd>{inspection.board_id}</dd></div>
              <div><dt>Recipe</dt><dd>{inspection.recipe_id} · {inspection.recipe_version}</dd></div>
              <div><dt>Lot</dt><dd>{inspection.lot_id ?? 'Not supplied'}</dd></div>
              <div><dt>Intake request</dt><dd className="mono">{inspection.intake_request_id ?? 'Not available'}</dd></div>
            </dl>
            {inspection.error && <div className="technical-error-evidence"><strong>{inspection.error.code}</strong><p>{inspection.error.message}</p></div>}
            <LifecycleSummary inspection={inspection} validation={validation} processing={processing} />
          </section>

          <section className="panel" aria-labelledby="artifacts-title">
            <div className="panel-heading"><div><p className="step-number">Registered evidence</p><h3 id="artifacts-title">Artifact summaries</h3></div><span>{inspection.artifacts.length} registered</span></div>
            {inspection.artifacts.length === 0 ? <p className="neutral-state">No artifact metadata is registered.</p> : <div className="artifact-grid">{inspection.artifacts.map((artifact) => <article className="artifact-card" key={`${artifact.artifact_type}-${artifact.sha256}`}><div className="subpanel-heading"><h4>{artifact.artifact_type}</h4><span>{formatBytes(artifact.byte_size)}</span></div><dl><div><dt>Media type</dt><dd>{artifact.media_type ?? 'Not reported'}</dd></div><div><dt>SHA-256</dt><dd className="mono hash-value">{artifact.sha256}</dd></div><div><dt>Registered</dt><dd>{formatTimestamp(artifact.created_at)}</dd></div></dl></article>)}</div>}
          </section>

          <ValidationPanel result={validation} inspectionStatus={inspection.status} running={action === 'validate'} onValidate={() => void validate()} />
          <ProcessingPanel result={processing} inspectionStatus={inspection.status} running={action === 'process'} onProcess={() => void process()} />

          {inspection.status === 'PROCESSING' && <div className="manual-refresh-note" role="status"><strong>Processing is recorded as in progress.</strong> Use manual Refresh to retrieve authoritative persisted state. Another processing request is disabled.</div>}
          {inspection.status === 'VALIDATION_FAILED' && <div className="manual-refresh-note"><strong>Processing is unavailable.</strong> Review the completed technical validation findings above.</div>}
          {['PASS', 'FAIL', 'UNCERTAIN'].includes(inspection.status) && <div className="manual-refresh-note"><strong>Final synthetic workflow state is persisted.</strong> Reprocessing is not supported.</div>}
        </>
      )}
    </section>
  )
}
