import type { InspectionStatus, InspectionValidationResponse, ValidationArtifact } from '../api/types'
import { StatusBadge } from './StatusBadge'
import { formatBytes, formatTimestamp } from '../utils/format'

interface ValidationPanelProps {
  result: InspectionValidationResponse | null
  inspectionStatus: InspectionStatus
  running: boolean
  onValidate: () => void
}

function ArtifactTechnicalSummary({ label, artifact }: { label: string; artifact: ValidationArtifact }) {
  return (
    <section className="technical-summary" aria-label={`${label} technical summary`}>
      <div className="subpanel-heading"><h4>{label}</h4><StatusBadge value={artifact.readability_status} /></div>
      <dl className="metadata-grid">
        <div><dt>Detected format</dt><dd>{artifact.detected_format ?? 'Not detected'}</dd></div>
        <div><dt>Dimensions</dt><dd>{artifact.width && artifact.height ? `${artifact.width} × ${artifact.height}` : 'Not available'}</dd></div>
        <div><dt>Channels</dt><dd>{artifact.channels ?? 'Not available'}</dd></div>
        <div><dt>Bit depth</dt><dd>{artifact.bit_depth ?? 'Not available'}</dd></div>
        <div><dt>Storage type</dt><dd>{artifact.storage_data_type ?? 'Not reported'}</dd></div>
        <div><dt>Byte size</dt><dd>{artifact.byte_size === null ? 'Not available' : formatBytes(artifact.byte_size)}</dd></div>
      </dl>
    </section>
  )
}

export function ValidationPanel({ result, inspectionStatus, running, onValidate }: ValidationPanelProps) {
  return (
    <section className="panel" aria-labelledby="validation-panel-title">
      <div className="panel-heading">
        <div><p className="step-number">Validation</p><h3 id="validation-panel-title">Technical validation</h3></div>
        {inspectionStatus === 'RECEIVED' && <button className="button primary" type="button" onClick={onValidate} disabled={running}>{running ? 'Validating…' : 'Run technical validation'}</button>}
      </div>
      <div className="policy-callout"><span>Selected policy</span><strong>development-native-rgb-height · 1.0</strong><small>Development policy; not production approval.</small></div>
      {!result && <div className="neutral-state"><strong>Not validated yet</strong><span>A persisted technical validation result is not available.</span></div>}
      {result && (
        <div className="evidence-stack">
          <div className="result-heading"><StatusBadge value={result.validation_outcome} /><span>Completed {formatTimestamp(result.completed_at)}</span>{result.idempotent_existing && <span className="replay-label">Exact persisted replay</span>}</div>
          <p className="technical-explanation"><strong>VALIDATION_PASSED means technically ready for preprocessing.</strong> It is not a PCB quality decision.</p>
          <dl className="metadata-grid summary-grid">
            <div><dt>Validation ID</dt><dd className="mono">{result.validation_id}</dd></div>
            <div><dt>Policy</dt><dd>{result.policy.policy_id} · {result.policy.policy_version}</dd></div>
            <div><dt>Validator</dt><dd>{result.validator_version}</dd></div>
            <div><dt>Total findings</dt><dd>{result.summary.finding_count}</dd></div>
            <div><dt>Blocking</dt><dd>{result.summary.blocking_count}</dd></div>
            <div><dt>Warnings / errors</dt><dd>{result.summary.warning_count} / {result.summary.error_count}</dd></div>
          </dl>
          <div className="two-column-evidence"><ArtifactTechnicalSummary label="RGB source" artifact={result.artifacts.rgb} /><ArtifactTechnicalSummary label="Height source" artifact={result.artifacts.height} /></div>
          <section aria-labelledby="validation-findings-heading">
            <div className="subpanel-heading"><h4 id="validation-findings-heading">Ordered validation findings</h4><span>{result.findings.length}</span></div>
            {result.findings.length === 0 ? <p className="neutral-state">No technical findings were recorded.</p> : <ol className="finding-list">{result.findings.map((finding, index) => <li key={`${finding.code}-${index}`}><div><StatusBadge value={finding.severity} />{finding.blocking && <span className="blocking-label">Blocking</span>}</div><strong>{finding.code}</strong><p>{finding.message}</p><small>{finding.category}{finding.artifact_type ? ` · ${finding.artifact_type}` : ''}{finding.field ? ` · ${finding.field}` : ''}</small></li>)}</ol>}
          </section>
        </div>
      )}
    </section>
  )
}
