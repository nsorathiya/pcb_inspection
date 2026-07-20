import type { ApiClientError } from '../api/errors'

interface ErrorPanelProps {
  error: ApiClientError
  onRetry?: () => void
  title?: string
}

export function ErrorPanel({ error, onRetry, title = 'Request could not be completed' }: ErrorPanelProps) {
  return (
    <section className="error-panel" role="alert" aria-labelledby="request-error-title">
      <div>
        <h2 id="request-error-title">{title}</h2>
        <p>{error.message}</p>
        <dl className="inline-metadata">
          <div><dt>Error code</dt><dd>{error.code}</dd></div>
          <div><dt>Request ID</dt><dd className="mono">{error.requestId}</dd></div>
        </dl>
      </div>
      {onRetry && <button type="button" className="button secondary" onClick={onRetry}>Try again</button>}
    </section>
  )
}
