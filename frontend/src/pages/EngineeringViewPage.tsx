import { useCallback, useEffect, useRef, useState, type FormEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'
import { engineeringPreviewUrl, getEngineeringSample, getEngineeringView } from '../api/engineeringViewer'
import { toApiClientError, type ApiClientError } from '../api/errors'
import type { EngineeringRasterMetadata, EngineeringSampleResponse, EngineeringViewResponse } from '../api/types'
import { ErrorPanel } from '../components/ErrorPanel'
import { StatusBadge } from '../components/StatusBadge'
import { formatBytes } from '../utils/format'

const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const VIEW_MODES = ['RGB', 'Height', 'Side-by-side', 'Alpha overlay', 'Split comparison'] as const
type ViewMode = (typeof VIEW_MODES)[number]
type CanvasTool = 'pan' | 'crosshair'

interface CoordinateState {
  rgbX: number
  rgbY: number
  heightX: number
  heightY: number
}

function MetadataBlock({ title, metadata }: { title: string; metadata: EngineeringRasterMetadata }) {
  return (
    <section className="engineering-metadata-block" aria-label={`${title} metadata`}>
      <h4>{title}</h4>
      <dl className="engineering-definition-list">
        <div><dt>Dimensions</dt><dd>{metadata.width} x {metadata.height}</dd></div>
        <div><dt>Format</dt><dd>{metadata.detected_format}</dd></div>
        <div><dt>Native type</dt><dd>{metadata.storage_data_type ?? 'Not reported'}</dd></div>
        <div><dt>Channels / depth</dt><dd>{metadata.channels} / {metadata.bit_depth}-bit</dd></div>
        <div><dt>Color mode</dt><dd>{metadata.color_mode}</dd></div>
        <div><dt>Artifact size</dt><dd>{formatBytes(metadata.byte_size)}</dd></div>
        <div className="engineering-definition-wide"><dt>SHA-256</dt><dd className="mono">{metadata.sha256}</dd></div>
      </dl>
    </section>
  )
}

function Crosshair({ x, y }: { x: number; y: number }) {
  return (
    <span className="engineering-crosshair" style={{ left: `${x}%`, top: `${y}%` }} aria-hidden="true">
      <span className="engineering-crosshair-horizontal" />
      <span className="engineering-crosshair-vertical" />
      <span className="engineering-crosshair-centre" />
    </span>
  )
}

function EvidenceImage({
  kind,
  src,
  metadata,
  crosshair,
  showCrosshair,
}: {
  kind: 'RGB' | 'Height'
  src: string
  metadata: EngineeringRasterMetadata
  crosshair: { x: number; y: number }
  showCrosshair: boolean
}) {
  return (
    <div className={`engineering-image-frame engineering-image-${kind.toLowerCase()}`} data-evidence-kind={kind}>
      <span className="engineering-image-label">{kind} · {metadata.width} x {metadata.height}</span>
      <img src={src} alt={`${kind} evidence preview`} draggable={false} />
      {showCrosshair && <Crosshair x={crosshair.x} y={crosshair.y} />}
    </div>
  )
}

function HeightLegend({ minimum, maximum }: { minimum: number; maximum: number }) {
  return (
    <div className="height-palette-legend" aria-label="Derived height preview palette legend">
      <span>{minimum}</span>
      <span className="height-palette-ramp" aria-hidden="true" />
      <span>{maximum}</span>
      <small>Derived display intensity · native values remain unchanged</small>
    </div>
  )
}

function Histogram({ view }: { view: EngineeringViewResponse }) {
  const histogram = view.height_statistics.histogram
  const maximum = Math.max(...histogram.counts, 1)
  return (
    <section className="engineering-histogram" aria-labelledby="height-histogram-title">
      <div className="subpanel-heading">
        <h4 id="height-histogram-title">Native height distribution</h4>
        <span className="mono">64 bins</span>
      </div>
      <svg viewBox="0 0 640 130" role="img" aria-label="64-bin native height histogram" preserveAspectRatio="none">
        <title>64-bin native height histogram</title>
        {histogram.counts.map((count, index) => {
          const height = (count / maximum) * 110
          return <rect key={index} data-histogram-bin={index} x={index * 10 + 1} y={120 - height} width="8" height={height} />
        })}
        <line x1="0" y1="120" x2="640" y2="120" />
      </svg>
      <div className="histogram-axis"><span>{histogram.native_min}</span><span>Native sample value</span><span>{histogram.native_max}</span></div>
      <dl className="engineering-definition-list compact">
        <div><dt>Valid</dt><dd>{view.height_statistics.valid_count.toLocaleString()}</dd></div>
        <div><dt>Invalid</dt><dd>{view.height_statistics.invalid_count.toLocaleString()}</dd></div>
      </dl>
    </section>
  )
}

function PipelineCard({ id, title, available, children }: { id?: string; title: string; available: boolean; children: ReactNode }) {
  return (
    <article id={id} className="pipeline-card">
      <div className="subpanel-heading"><h4>{title}</h4><span className={available ? 'pipeline-available' : 'pipeline-unavailable'}>{available ? 'Persisted' : 'Not available'}</span></div>
      {children}
    </article>
  )
}

export function EngineeringViewPage() {
  const { inspectionId = '' } = useParams()
  const validId = CANONICAL_UUID.test(inspectionId)
  const [view, setView] = useState<EngineeringViewResponse | null>(null)
  const [error, setError] = useState<ApiClientError | null>(null)
  const [loading, setLoading] = useState(validId)
  const [mode, setMode] = useState<ViewMode>('Side-by-side')
  const [tool, setTool] = useState<CanvasTool>('pan')
  const [zoom, setZoom] = useState(1)
  const [scaleMode, setScaleMode] = useState<'fit' | 'actual'>('fit')
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [overlayOpacity, setOverlayOpacity] = useState(50)
  const [splitPosition, setSplitPosition] = useState(50)
  const [coordinates, setCoordinates] = useState<CoordinateState>({ rgbX: 0, rgbY: 0, heightX: 0, heightY: 0 })
  const [sample, setSample] = useState<EngineeringSampleResponse | null>(null)
  const [sampleError, setSampleError] = useState<ApiClientError | null>(null)
  const [sampling, setSampling] = useState(false)
  const drag = useRef<{ clientX: number; clientY: number; panX: number; panY: number } | null>(null)

  const load = useCallback(async (signal?: AbortSignal) => {
    if (!validId) return
    setLoading(true)
    setError(null)
    try {
      const response = await getEngineeringView(inspectionId, signal)
      setView(response.data)
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

  if (!validId) {
    return <section className="not-found" role="alert"><p className="eyebrow">Invalid route parameter</p><h2>Malformed inspection ID</h2><p>The engineering-view route requires a lowercase, hyphenated UUID.</p><Link className="button secondary" to="/">Return to inspection history</Link></section>
  }

  const resetCanvas = () => {
    setMode('Side-by-side')
    setTool('pan')
    setZoom(1)
    setScaleMode('fit')
    setPan({ x: 0, y: 0 })
    setOverlayOpacity(50)
    setSplitPosition(50)
  }

  const changeZoom = (delta: number) => {
    setScaleMode('fit')
    setZoom((current) => Math.min(4, Math.max(0.25, Number((current + delta).toFixed(2)))))
  }

  const movePan = (x: number, y: number) => setPan((current) => ({
    x: Math.min(1, Math.max(-1, current.x + x)),
    y: Math.min(1, Math.max(-1, current.y + y)),
  }))

  const pointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (tool !== 'pan') return
    event.currentTarget.setPointerCapture?.(event.pointerId)
    drag.current = { clientX: event.clientX, clientY: event.clientY, panX: pan.x, panY: pan.y }
  }

  const pointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current || tool !== 'pan') return
    const bounds = event.currentTarget.getBoundingClientRect()
    const width = Math.max(bounds.width, 1)
    const height = Math.max(bounds.height, 1)
    setPan({
      x: Math.min(1, Math.max(-1, drag.current.panX + (event.clientX - drag.current.clientX) / width)),
      y: Math.min(1, Math.max(-1, drag.current.panY + (event.clientY - drag.current.clientY) / height)),
    })
  }

  const sampleCoordinates = async (event: FormEvent) => {
    event.preventDefault()
    if (!view || sampling) return
    setSampling(true)
    setSampleError(null)
    try {
      const response = await getEngineeringSample(inspectionId, coordinates)
      setSample(response.data)
    } catch (caught) {
      setSampleError(toApiClientError(caught))
    } finally {
      setSampling(false)
    }
  }

  const coordinateValid = Boolean(view) && coordinates.rgbX >= 0 && coordinates.rgbX < view!.rgb.width && coordinates.rgbY >= 0 && coordinates.rgbY < view!.rgb.height && coordinates.heightX >= 0 && coordinates.heightX < view!.height.width && coordinates.heightY >= 0 && coordinates.heightY < view!.height.height

  if (loading && !view) return <p className="loading-state">Loading integrity-verified engineering evidence…</p>
  if (error && !view) return <ErrorPanel error={error} onRetry={() => void load()} title="Engineering workspace unavailable" />
  if (!view) return null

  const rgbUrl = engineeringPreviewUrl(inspectionId, 'rgb')
  const heightUrl = engineeringPreviewUrl(inspectionId, 'height')
  const dimensionMismatch = view.rgb.width !== view.height.width || view.rgb.height !== view.height.height
  const rgbCrosshair = { x: ((coordinates.rgbX + 0.5) / view.rgb.width) * 100, y: ((coordinates.rgbY + 0.5) / view.rgb.height) * 100 }
  const heightCrosshair = { x: ((coordinates.heightX + 0.5) / view.height.width) * 100, y: ((coordinates.heightY + 0.5) / view.height.height) * 100 }
  const showCrosshair = tool === 'crosshair'
  const transform = `translate(${pan.x * 100}%, ${pan.y * 100}%) scale(${zoom})`

  const rgbImage = <EvidenceImage kind="RGB" src={rgbUrl} metadata={view.rgb} crosshair={rgbCrosshair} showCrosshair={showCrosshair} />
  const heightImage = <EvidenceImage kind="Height" src={heightUrl} metadata={view.height} crosshair={heightCrosshair} showCrosshair={showCrosshair} />

  return (
    <section className="engineering-page" aria-labelledby="engineering-workspace-title">
      <div className="engineering-page-header">
        <div><p className="eyebrow">Read-only synthetic evidence viewer</p><h2 id="engineering-workspace-title">PCB 2D/3D Vision Engineering Workspace</h2><p className="mono page-identifier">{inspectionId}</p></div>
        <div className="page-actions"><StatusBadge value={view.inspection_status} /><Link className="button secondary" to={`/inspections/${inspectionId}`}>Inspection detail</Link></div>
      </div>

      <div className="engineering-safety-banner" role="note">
        <strong>Synthetic engineering data</strong>
        <span>No production measurement · No physical calibration · No real registration · No production inspection decision</span>
      </div>
      {dimensionMismatch && <div className="dimension-warning" role="alert"><strong>Dimension mismatch:</strong> RGB is {view.rgb.width} x {view.rgb.height}; height is {view.height.width} x {view.height.height}. Coordinates remain independent and overlay alignment is illustrative only.</div>}

      <div className="engineering-workspace-grid">
        <nav className="evidence-navigator" aria-label="Engineering evidence navigator">
          <p className="step-number">Evidence navigator</p>
          <button type="button" onClick={() => setMode('RGB')}>RGB evidence</button>
          <button type="button" onClick={() => setMode('Height')}>Height evidence</button>
          <a href="#technical-validation">Technical validation</a>
          <a href="#synthetic-preprocessing">Synthetic preprocessing</a>
          <a href="#mock-inference">Deterministic mock inference</a>
          <a href="#persisted-result">Persisted result</a>
          <Link to={`/inspections/${inspectionId}#audit-timeline`}>Audit</Link>
          <Link to={`/inspections/${inspectionId}/report`}>Development report</Link>
          <p className="evidence-readonly-note">GET-only workspace<br />Persisted evidence only</p>
        </nav>

        <section className="vision-workbench" aria-label="Vision canvas workspace">
          <div className="view-mode-toolbar" role="group" aria-label="Vision view modes">
            {VIEW_MODES.map((item) => <button key={item} type="button" aria-pressed={mode === item} onClick={() => setMode(item)}>{item}</button>)}
          </div>
          <div className="canvas-toolbar" role="toolbar" aria-label="Vision canvas controls">
            <button type="button" onClick={() => changeZoom(-0.25)} aria-label="Zoom out">−</button>
            <output aria-label="Zoom level">{Math.round(zoom * 100)}%</output>
            <button type="button" onClick={() => changeZoom(0.25)} aria-label="Zoom in">+</button>
            <button type="button" aria-pressed={scaleMode === 'fit'} onClick={() => { setScaleMode('fit'); setZoom(1); setPan({ x: 0, y: 0 }) }}>Fit</button>
            <button type="button" aria-pressed={scaleMode === 'actual'} onClick={() => { setScaleMode('actual'); setZoom(1); setPan({ x: 0, y: 0 }) }}>Actual pixels</button>
            <button type="button" onClick={resetCanvas}>Reset</button>
            <span className="toolbar-separator" aria-hidden="true" />
            <button type="button" aria-pressed={tool === 'pan'} onClick={() => setTool('pan')}>Pan</button>
            <button type="button" aria-pressed={tool === 'crosshair'} onClick={() => setTool('crosshair')}>Shared crosshair</button>
            <div className="pan-nudges" aria-label="Pan position controls">
              <button type="button" onClick={() => movePan(-0.05, 0)} aria-label="Pan left">←</button>
              <button type="button" onClick={() => movePan(0, -0.05)} aria-label="Pan up">↑</button>
              <button type="button" onClick={() => movePan(0, 0.05)} aria-label="Pan down">↓</button>
              <button type="button" onClick={() => movePan(0.05, 0)} aria-label="Pan right">→</button>
            </div>
          </div>
          {(mode === 'Alpha overlay' || mode === 'Split comparison') && (
            <div className="comparison-controls">
              {mode === 'Alpha overlay' && <label>Overlay opacity <input aria-label="Overlay opacity" type="range" min="0" max="100" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} /><output>{overlayOpacity}%</output></label>}
              {mode === 'Split comparison' && <label>Split position <input aria-label="Split position" type="range" min="0" max="100" value={splitPosition} onChange={(event) => setSplitPosition(Number(event.target.value))} /><output>{splitPosition}%</output></label>}
            </div>
          )}
          <div
            className={`vision-canvas tool-${tool} scale-${scaleMode}`}
            data-view-mode={mode}
            data-pan={`${pan.x.toFixed(2)},${pan.y.toFixed(2)}`}
            onPointerDown={pointerDown}
            onPointerMove={pointerMove}
            onPointerUp={() => { drag.current = null }}
            onPointerCancel={() => { drag.current = null }}
          >
            <div className="vision-canvas-grid" aria-hidden="true" />
            <div className="vision-transform-surface" style={{ transform }}>
              {mode === 'RGB' && <div id="rgb-evidence" className="single-evidence">{rgbImage}</div>}
              {mode === 'Height' && <div id="height-evidence" className="single-evidence">{heightImage}</div>}
              {mode === 'Side-by-side' && <div className="side-by-side-evidence"><div id="rgb-evidence">{rgbImage}</div><div id="height-evidence">{heightImage}</div></div>}
              {mode === 'Alpha overlay' && <div className="stacked-evidence"><div>{rgbImage}</div><div className="overlay-layer" style={{ opacity: overlayOpacity / 100 }}>{heightImage}</div></div>}
              {mode === 'Split comparison' && <div className="stacked-evidence"><div>{rgbImage}</div><div className="split-layer" style={{ clipPath: `inset(0 0 0 ${splitPosition}%)` }}>{heightImage}</div><span className="split-divider" style={{ left: `${splitPosition}%` }} aria-hidden="true" /></div>}
            </div>
          </div>
          {(mode !== 'RGB') && <HeightLegend minimum={view.height_statistics.native_min} maximum={view.height_statistics.native_max} />}
          <p className="canvas-status mono" aria-live="polite">Zoom {Math.round(zoom * 100)}% · normalized pan x {pan.x.toFixed(2)}, y {pan.y.toFixed(2)} · {tool} tool</p>
        </section>

        <aside className="engineering-inspector" aria-label="Metadata and pixel inspector">
          <p className="step-number">Metadata inspector</p>
          <section className="engineering-metadata-block" aria-label="Pair status metadata">
            <h4>Pair status</h4>
            <dl className="engineering-definition-list">
              <div><dt>Calibration</dt><dd>{view.calibration_status}</dd></div>
              <div><dt>Registration</dt><dd>{view.registration_status}</dd></div>
              <div><dt>Physical height unit</dt><dd>Unavailable</dd></div>
              <div><dt>Production approved</dt><dd>No</dd></div>
            </dl>
          </section>
          <MetadataBlock title="RGB evidence" metadata={view.rgb} />
          <MetadataBlock title="Height evidence" metadata={view.height} />
          <Histogram view={view} />

          <form className="pixel-inspector" onSubmit={(event) => void sampleCoordinates(event)}>
            <div className="subpanel-heading"><h4>Native pixel inspector</h4><span>Separate spaces</span></div>
            <fieldset><legend>RGB coordinate</legend><label>X<input aria-label="RGB X coordinate" type="number" min="0" max={view.rgb.width - 1} value={coordinates.rgbX} onChange={(event) => setCoordinates((current) => ({ ...current, rgbX: Number(event.target.value) }))} /></label><label>Y<input aria-label="RGB Y coordinate" type="number" min="0" max={view.rgb.height - 1} value={coordinates.rgbY} onChange={(event) => setCoordinates((current) => ({ ...current, rgbY: Number(event.target.value) }))} /></label></fieldset>
            <fieldset><legend>Height coordinate</legend><label>X<input aria-label="Height X coordinate" type="number" min="0" max={view.height.width - 1} value={coordinates.heightX} onChange={(event) => setCoordinates((current) => ({ ...current, heightX: Number(event.target.value) }))} /></label><label>Y<input aria-label="Height Y coordinate" type="number" min="0" max={view.height.height - 1} value={coordinates.heightY} onChange={(event) => setCoordinates((current) => ({ ...current, heightY: Number(event.target.value) }))} /></label></fieldset>
            {!coordinateValid && <p className="field-error" role="alert">Coordinates must remain inside their respective raster bounds.</p>}
            <button className="button primary" type="submit" disabled={!coordinateValid || sampling}>{sampling ? 'Sampling…' : 'Sample native values'}</button>
            {sampleError && <div className="sample-error" role="alert"><strong>{sampleError.code}</strong><span>{sampleError.message}</span></div>}
            {sample && <div className="sample-result" aria-live="polite"><dl className="engineering-definition-list"><div><dt>RGB [{sample.rgb.x}, {sample.rgb.y}]</dt><dd className="mono">[{sample.rgb.values.join(', ')}]</dd></div><div><dt>RGB type</dt><dd>{sample.rgb.storage_data_type ?? 'Not reported'}</dd></div><div><dt>Height [{sample.height.x}, {sample.height.y}]</dt><dd className="mono">{sample.height.value === null ? 'Invalid / NaN' : sample.height.value}</dd></div><div><dt>Height type</dt><dd>{sample.height.storage_data_type ?? 'Not reported'}</dd></div><div><dt>Physical unit</dt><dd>Unavailable</dd></div><div><dt>Validity</dt><dd>{sample.height.valid ? 'Valid' : 'Invalid'}</dd></div></dl></div>}
          </form>
        </aside>
      </div>

      <section className="pipeline-evidence-panel" aria-labelledby="pipeline-evidence-title">
        <div className="panel-heading"><div><p className="step-number">Pipeline evidence and limitations</p><h3 id="pipeline-evidence-title">Persisted workflow record</h3></div><p>No validation or processing is executed by this workspace.</p></div>
        <div className="pipeline-evidence-grid">
          <PipelineCard id="technical-validation" title="Technical validation" available={view.validation.available}><dl><div><dt>Outcome</dt><dd>{view.validation.outcome ?? 'Not recorded'}</dd></div><div><dt>Policy</dt><dd>{view.validation.policy_id ? `${view.validation.policy_id} ${view.validation.policy_version}` : 'Not recorded'}</dd></div><div><dt>Findings</dt><dd>{view.validation.finding_codes.join(', ') || 'None recorded'}</dd></div></dl></PipelineCard>
          <PipelineCard id="synthetic-preprocessing" title="Synthetic preprocessing" available={view.processing.available}><dl><div><dt>Outcome</dt><dd>{view.processing.preprocessing_outcome ?? 'Not recorded'}</dd></div><div><dt>Synthetic input verified</dt><dd>{view.processing.synthetic_input_verified === null ? 'Not recorded' : view.processing.synthetic_input_verified ? 'Yes' : 'No'}</dd></div></dl></PipelineCard>
          <PipelineCard id="mock-inference" title="Deterministic mock inference" available={view.processing.available}><dl><div><dt>Decision</dt><dd>{view.processing.mock_decision ?? 'Not recorded'}</dd></div><div><dt>Processing status</dt><dd>{view.processing.processing_status ?? 'Not recorded'}</dd></div><div><dt>Production approved</dt><dd>No</dd></div></dl></PipelineCard>
          <PipelineCard id="persisted-result" title="Persisted result" available={true}><dl><div><dt>Inspection status</dt><dd>{view.inspection_status}</dd></div><div><dt>Calibration</dt><dd>{view.calibration_status}</dd></div><div><dt>Registration</dt><dd>{view.registration_status}</dd></div></dl></PipelineCard>
        </div>
        <div className="engineering-limitations">
          <h4>Engineering limitations</h4>
          <ul><li>Synthetic engineering data only.</li><li>No production measurement or fabricated physical units.</li><li>No physical calibration.</li><li>No real registration.</li><li>No production inspection decision.</li>{view.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
        </div>
      </section>
    </section>
  )
}
