import type { InspectionDetailResponse, InspectionProcessingResponse, InspectionValidationResponse } from '../api/types'

interface LifecycleSummaryProps {
  inspection: InspectionDetailResponse
  validation: InspectionValidationResponse | null
  processing: InspectionProcessingResponse | null
}

export function LifecycleSummary({ inspection, validation, processing }: LifecycleSummaryProps) {
  const final = ['PASS', 'FAIL', 'UNCERTAIN', 'ERROR'].includes(inspection.status)
  const steps = [
    { label: 'Intake', state: 'complete', detail: 'Paired files registered' },
    { label: 'Validation', state: validation ? 'complete' : inspection.status === 'RECEIVED' ? 'current' : 'pending', detail: validation?.validation_outcome ?? 'Not validated' },
    { label: 'Processing', state: processing ? 'complete' : inspection.status === 'READY' ? 'current' : inspection.status === 'PROCESSING' ? 'current' : 'pending', detail: processing?.processing_status ?? (inspection.status === 'PROCESSING' ? 'PROCESSING' : 'Not processed') },
    { label: 'Result', state: final ? 'complete' : 'pending', detail: final ? inspection.status : 'No final result' },
  ]
  return <ol className="lifecycle-summary" aria-label="Technical workflow progression">{steps.map((step) => <li className={`lifecycle-${step.state}`} key={step.label}><span className="lifecycle-marker" aria-hidden="true" /><div><strong>{step.label}</strong><span>{step.detail}</span></div></li>)}</ol>
}
