import { useRef } from 'react'
import { fileExtension, formatBytes } from '../utils/format'

interface FileSelectionProps {
  id: string
  label: string
  hint: string
  accept: string
  file: File | null
  onChange: (file: File | null) => void
  error?: string
}

export function FileSelection({ id, label, hint, accept, file, onChange, error }: FileSelectionProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const clear = () => {
    if (inputRef.current) inputRef.current.value = ''
    onChange(null)
  }

  return (
    <div className={`file-field ${error ? 'field-invalid' : ''}`}>
      <label htmlFor={id}>{label} <span aria-hidden="true">*</span></label>
      <p className="field-hint" id={`${id}-hint`}>{hint}</p>
      <input
        ref={inputRef}
        id={id}
        name={id}
        type="file"
        accept={accept}
        aria-describedby={`${id}-hint${error ? ` ${id}-error` : ''}`}
        aria-invalid={Boolean(error)}
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />
      {file && (
        <div className="selected-file">
          <div>
            <strong>{file.name}</strong>
            <span>{formatBytes(file.size)} · {file.type || 'Type not reported'} · {fileExtension(file.name)}</span>
          </div>
          <button type="button" className="text-button" onClick={clear}>Clear</button>
        </div>
      )}
      {error && <p className="field-error" id={`${id}-error`}>{error}</p>}
    </div>
  )
}
