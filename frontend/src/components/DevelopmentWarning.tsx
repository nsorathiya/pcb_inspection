export function DevelopmentWarning() {
  return (
    <div className="development-warning" role="note">
      <span className="warning-mark" aria-hidden="true">!</span>
      <strong>Development mode:</strong> processing uses deterministic synthetic mock
      inference. Results are not production PCB decisions.
    </div>
  )
}
