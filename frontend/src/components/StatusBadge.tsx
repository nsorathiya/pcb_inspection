interface StatusBadgeProps {
  value: string
  prefix?: string
}

export function StatusBadge({ value, prefix }: StatusBadgeProps) {
  const normalized = value.toLowerCase().replaceAll('_', '-')
  return (
    <span className={`status-badge status-${normalized}`}>
      <span className="status-dot" aria-hidden="true" />
      {prefix ? `${prefix} ${value}` : value}
    </span>
  )
}
