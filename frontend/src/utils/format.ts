export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return 'Not available'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString()
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`
}

export function fileExtension(filename: string): string {
  const index = filename.lastIndexOf('.')
  return index >= 0 ? filename.slice(index).toLowerCase() : 'No extension'
}

export function toUtcIso(localValue: string): string | undefined {
  if (!localValue) return undefined
  const parsed = new Date(localValue)
  return Number.isNaN(parsed.getTime()) ? localValue : parsed.toISOString()
}
