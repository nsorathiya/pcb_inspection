import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getInspectionReport } from '../api/inspections'
import { toApiClientError, type ApiClientError } from '../api/errors'
import type { InspectionDevelopmentReportResponse, JsonValue } from '../api/types'
import { ErrorPanel } from '../components/ErrorPanel'
import { StatusBadge } from '../components/StatusBadge'
import { formatBytes, formatTimestamp } from '../utils/format'

const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

function valueText(value: JsonValue | undefined): string {
  if (value === undefined || value === null) return 'Not available'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function EvidenceObject({ value }: { value: Record<string, JsonValue> }) {
  return <dl className="report-evidence-grid">{Object.entries(value).map(([key, item]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{valueText(item)}</dd></div>)}</dl>
}

function Findings({ title, items }: { title: string; items: Array<Record<string, JsonValue>> }) {
  const headingId = `${title.replaceAll(' ', '-').toLowerCase()}-title`
  return <section aria-labelledby={headingId}><h4 id={headingId}>{title}</h4>{items.length === 0 ? <p className="neutral-state">No persisted findings.</p> : <ol className="finding-list">{items.map((item, index) => <li key={`${valueText(item.code)}-${index}`}><strong>{valueText(item.code)}</strong><p>{valueText(item.message)}</p><small>{valueText(item.severity)} · {valueText(item.category)}</small></li>)}</ol>}</section>
}

export function InspectionReportPage() {
  const { inspectionId = '' } = useParams()
  const validId = CANONICAL_UUID.test(inspectionId)
  const [response, setResponse] = useState<InspectionDevelopmentReportResponse | null>(null)
  const [loading, setLoading] = useState(validId)
  const [error, setError] = useState<ApiClientError | null>(null)

  const load = useCallback(async (signal?: AbortSignal) => {
    if (!validId) return
    setLoading(true)
    setError(null)
    try {
      const result = await getInspectionReport(inspectionId, signal)
      setResponse(result.data)
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

  const download = () => {
    if (!response) return
    const blob = new Blob([JSON.stringify(response.report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `inspection-${inspectionId}-development-report.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  if (!validId) return <section className="not-found" role="alert"><p className="eyebrow">Invalid route parameter</p><h2>Malformed inspection ID</h2><p>The report route requires a lowercase, hyphenated UUID.</p><Link className="button secondary" to="/">Return to inspection history</Link></section>

  const report = response?.report
  const decision = report?.processing?.final_decision
  const preprocessingFindings = report?.processing?.preprocessing?.findings
  const inferenceFindings = report?.processing?.inference?.findings
  return (
    <article className="development-report" aria-labelledby="development-report-title">
      <div className="report-toolbar no-print"><Link to={`/inspections/${inspectionId}`}>← Back to inspection detail</Link><div><button className="button secondary" type="button" onClick={() => void load()} disabled={loading}>Refresh report</button><button className="button secondary" type="button" onClick={download} disabled={!response}>Download JSON</button><button className="button primary" type="button" onClick={() => window.print()} disabled={!response}>Print</button></div></div>
      <header className="page-heading report-title-block"><div><p className="eyebrow">Persisted development evidence</p><h2 id="development-report-title">Development Report</h2><p className="mono page-identifier">{inspectionId}</p>{response && <p className="mono report-hash">Report SHA-256: {response.report_sha256}</p>}</div></header>
      <div className="report-warning" role="alert"><strong>Development-only · nonproduction</strong><span>This report is not a production inspection certificate or PCB disposition record.</span></div>
      {loading && !response && <p className="loading-state">Loading the authoritative persisted report…</p>}
      {error && <ErrorPanel error={error} title="Development report unavailable" onRetry={() => void load()} />}
      {report && <div className="report-sections">
        <section className="panel" aria-labelledby="report-inspection-title"><div className="panel-heading"><h3 id="report-inspection-title">Inspection summary</h3><StatusBadge value={report.inspection.status} /></div><dl className="metadata-grid overview-grid"><div><dt>Board</dt><dd>{report.inspection.board_id}</dd></div><div><dt>Recipe</dt><dd>{report.inspection.recipe_id} · {report.inspection.recipe_version}</dd></div><div><dt>Created</dt><dd>{formatTimestamp(report.inspection.created_at)}</dd></div><div><dt>Lot</dt><dd>{report.inspection.lot_id ?? 'Not supplied'}</dd></div><div><dt>Operator</dt><dd>{report.inspection.operator_id ?? 'Not recorded'}</dd></div><div><dt>Contract</dt><dd>{report.contract_version}</dd></div></dl>{report.inspection.error && <div className="technical-error-evidence"><strong>TECHNICAL ERROR · {report.inspection.error.code}</strong><p>{report.inspection.error.message}</p></div>}</section>
        <section className="panel" aria-labelledby="report-artifacts-title"><div className="panel-heading"><h3 id="report-artifacts-title">Registered artifact identity</h3><span>{report.artifacts.length} artifact(s)</span></div>{report.artifacts.length === 0 ? <p className="neutral-state">No registered artifact metadata.</p> : <div className="artifact-grid">{report.artifacts.map((item) => <article className="artifact-card" key={`${item.artifact_type}-${item.sha256}`}><h4>{item.artifact_type}</h4><p>{formatBytes(item.byte_size)} · {item.media_type ?? 'Media type not recorded'}</p><p className="mono hash-value">{item.sha256}</p><time dateTime={item.created_at}>{formatTimestamp(item.created_at)}</time></article>)}</div>}</section>
        <section className="panel" aria-labelledby="report-validation-title"><div className="panel-heading"><h3 id="report-validation-title">Technical Validation</h3>{report.validation && <StatusBadge value={report.validation.outcome} />}</div>{!report.validation ? <p className="neutral-state">Technical validation evidence is not available for this partial lifecycle.</p> : <div className="evidence-stack"><dl className="metadata-grid"><div><dt>Validation ID</dt><dd className="mono">{report.validation.validation_id}</dd></div><div><dt>Policy</dt><dd>{report.validation.policy.policy_id} · {report.validation.policy.policy_version}</dd></div><div><dt>Validator</dt><dd>{report.validation.validator_version}</dd></div><div><dt>Completed</dt><dd>{formatTimestamp(report.validation.completed_at)}</dd></div></dl><div className="two-column-evidence"><div><h4>RGB technical summary</h4><EvidenceObject value={report.validation.rgb_technical_summary} /></div><div><h4>Height technical summary</h4><EvidenceObject value={report.validation.height_technical_summary} /></div></div><Findings title="Validation findings" items={report.validation.findings} /></div>}</section>
        <section className="panel" aria-labelledby="report-processing-title"><div className="panel-heading"><h3 id="report-processing-title">Synthetic Preprocessing and Deterministic Mock Inference</h3>{report.processing && <StatusBadge value={report.processing.lifecycle_status} />}</div>{!report.processing ? <p className="neutral-state">Processing evidence is not available for this partial lifecycle.</p> : <div className="evidence-stack"><div className="mock-result-warning"><strong>{decision ? `MOCK ${decision}` : report.processing.error ? 'TECHNICAL ERROR' : 'Processing evidence is partial'}</strong><p>{decision === 'FAIL' && report.processing.inference ? `Mock defect type: ${valueText(report.processing.inference.defect_type)}` : 'Development-only persisted evidence; not approved for production disposition.'}</p></div><dl className="metadata-grid"><div><dt>Processing run</dt><dd className="mono">{report.processing.processing_run_id}</dd></div><div><dt>Started</dt><dd>{formatTimestamp(report.processing.started_at)}</dd></div><div><dt>Preprocessing policy</dt><dd>{report.processing.preprocessing_policy.policy_id} · {report.processing.preprocessing_policy.policy_version}</dd></div><div><dt>Inference engine</dt><dd>{report.processing.engine.engine_id} · {report.processing.engine.engine_version}</dd></div></dl>{report.processing.preprocessing && <><h4>Preprocessing evidence</h4><EvidenceObject value={report.processing.preprocessing} />{Array.isArray(preprocessingFindings) && <Findings title="Preprocessing findings" items={preprocessingFindings as Array<Record<string, JsonValue>>} />}</>}{report.processing.inference && <><h4>Inference evidence</h4><EvidenceObject value={report.processing.inference} />{Array.isArray(inferenceFindings) && <Findings title="Inference findings" items={inferenceFindings as Array<Record<string, JsonValue>>} />}</>}</div>}</section>
        <section className="panel" aria-labelledby="report-audit-title"><div className="panel-heading"><h3 id="report-audit-title">Audit timeline</h3><span>{report.audit.length} event(s)</span></div>{report.audit.length === 0 ? <p className="neutral-state">No persisted audit events.</p> : <ol className="report-audit-list">{report.audit.map((item) => <li key={item.audit_event_id}><time dateTime={item.created_at}>{formatTimestamp(item.created_at)}</time><strong>{item.action}</strong>{item.details_redacted && <span>Safety-projected details were redacted.</span>}</li>)}</ol>}</section>
        <section className="panel report-limitations" aria-labelledby="report-limitations-title"><h3 id="report-limitations-title">Development limitations</h3><ul>{report.limitations.map((item) => <li key={item}>{item}</li>)}</ul></section>
      </div>}
    </article>
  )
}
