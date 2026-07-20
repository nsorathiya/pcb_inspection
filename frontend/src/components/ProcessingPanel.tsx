import type { InspectionProcessingResponse, InspectionStatus, ProcessingFinding } from '../api/types'
import { StatusBadge } from './StatusBadge'
import { formatTimestamp } from '../utils/format'

interface ProcessingPanelProps {
  result: InspectionProcessingResponse | null
  inspectionStatus: InspectionStatus
  running: boolean
  onProcess: () => void
}

function FindingList({ title, findings }: { title: string; findings: ProcessingFinding[] }) {
  return (
    <section aria-label={title}>
      <div className="subpanel-heading"><h4>{title}</h4><span>{findings.length}</span></div>
      {findings.length === 0 ? <p className="neutral-state">No findings were recorded.</p> : <ol className="finding-list">{findings.map((finding, index) => <li key={`${finding.code}-${index}`}><div><StatusBadge value={finding.severity} />{finding.blocking && <span className="blocking-label">Blocking</span>}</div><strong>{finding.code}</strong><p>{finding.message}</p><small>{finding.category}{finding.branch ? ` · ${finding.branch}` : ''}{finding.field ? ` · ${finding.field}` : ''}</small></li>)}</ol>}
    </section>
  )
}

function DecisionResult({ result }: { result: InspectionProcessingResponse }) {
  const isTechnicalError = result.processing_status === 'ERROR' || result.inspection_status === 'ERROR'
  const label = isTechnicalError ? 'TECHNICAL ERROR' : `MOCK ${result.mock_decision ?? 'RESULT UNAVAILABLE'}`
  const icon = isTechnicalError ? '!' : result.mock_decision === 'PASS' ? '✓' : result.mock_decision === 'FAIL' ? '×' : '?'
  return (
    <div className={`decision-result decision-${isTechnicalError ? 'error' : result.mock_decision?.toLowerCase() ?? 'unknown'}`}>
      <span className="decision-icon" aria-hidden="true">{icon}</span>
      <div><p className="eyebrow">Development workflow result</p><h4>{label}</h4>{result.mock_decision === 'FAIL' && result.defect_type && <p>Mock taxonomy label: <strong>{result.defect_type}</strong></p>}</div>
    </div>
  )
}

export function ProcessingPanel({ result, inspectionStatus, running, onProcess }: ProcessingPanelProps) {
  return (
    <section className="panel" aria-labelledby="processing-panel-title">
      <div className="panel-heading">
        <div><p className="step-number">Processing</p><h3 id="processing-panel-title">Trusted synthetic processing</h3></div>
        {inspectionStatus === 'READY' && <button className="button primary" type="button" onClick={onProcess} disabled={running}>{running ? 'Processing…' : 'Run synthetic processing'}</button>}
      </div>
      <div className="policy-callout policy-stack"><span>Preprocessing policy</span><strong>synthetic-paired-rgb-height · 1.0</strong><span>Inference policy</span><strong>synthetic-deterministic-mock-inference · 1.0</strong></div>
      {!result && <div className="neutral-state"><strong>Not processed yet</strong><span>Processing is available only after the inspection becomes READY.</span></div>}
      {result && (
        <div className="evidence-stack">
          <div className="mock-result-warning" role="note"><strong>Deterministic synthetic mock result — not a real AI prediction and not approved for production PCB disposition.</strong></div>
          <DecisionResult result={result} />
          <p className="technical-explanation">The development engine uses deterministic digest bucketing to exercise workflow outcomes; it does not analyze PCB defects.</p>
          <dl className="metadata-grid summary-grid">
            <div><dt>Processing run</dt><dd className="mono">{result.processing_run_id}</dd></div>
            <div><dt>Run status</dt><dd><StatusBadge value={result.processing_status} /></dd></div>
            <div><dt>Preprocessing outcome</dt><dd>{result.preprocessing_outcome}</dd></div>
            <div><dt>Inference execution</dt><dd>{result.inference_execution_outcome ?? 'Not executed'}</dd></div>
            <div><dt>Started</dt><dd>{formatTimestamp(result.started_at)}</dd></div>
            <div><dt>Completed</dt><dd>{formatTimestamp(result.completed_at)}</dd></div>
          </dl>
          <div className="two-column-evidence">
            <section className="technical-summary"><h4>Preprocessing evidence</h4><dl className="metadata-grid"><div><dt>Identity</dt><dd className="mono">{result.preprocessing.preprocessing_id}</dd></div><div><dt>Policy</dt><dd>{result.preprocessing.policy_id} · {result.preprocessing.policy_version}</dd></div><div><dt>Implementation</dt><dd>{result.preprocessing.implementation_id} · {result.preprocessing.implementation_version}</dd></div><div><dt>Findings</dt><dd>{result.preprocessing.summary.total_findings}</dd></div></dl></section>
            <section className="technical-summary"><h4>Mock inference evidence</h4>{result.inference ? <dl className="metadata-grid"><div><dt>Identity</dt><dd className="mono">{result.inference.inference_id}</dd></div><div><dt>Policy</dt><dd>{result.inference.policy_id} · {result.inference.policy_version}</dd></div><div><dt>Engine</dt><dd>{result.inference.engine_id} · {result.inference.engine_version} · {result.inference.engine_type}</dd></div><div><dt>Findings</dt><dd>{result.inference.summary.total_findings}</dd></div></dl> : <p>Inference was not executed because preprocessing did not succeed.</p>}</section>
          </div>
          <FindingList title="Ordered preprocessing findings" findings={result.preprocessing.findings} />
          {result.inference && <FindingList title="Ordered mock-inference findings" findings={result.inference.findings} />}
          {result.lifecycle_idempotent_existing && <p className="replay-label">This response is an exact replay of persisted evidence.</p>}
        </div>
      )}
    </section>
  )
}
