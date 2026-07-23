import { useCallback, useEffect, useRef, useState, type CSSProperties, type FormEvent, type PointerEvent as ReactPointerEvent, type ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'
import { engineeringPreviewUrl, getEngineeringHeightRoi, getEngineeringSample, getEngineeringView } from '../api/engineeringViewer'
import { toApiClientError, type ApiClientError } from '../api/errors'
import type { EngineeringRasterMetadata, EngineeringSampleResponse, EngineeringViewResponse } from '../api/types'
import { ErrorPanel } from '../components/ErrorPanel'
import { StatusBadge } from '../components/StatusBadge'
import { formatBytes } from '../utils/format'
import {
  DEFAULT_ALIGNMENT,
  DEFAULT_ENGINEERING_SESSION,
  SESSION_HISTORY_LIMIT,
  affineMatrix,
  alignmentExportFilename,
  buildAlignmentExport,
  clampImagePoint,
  commitSessionHistory,
  correspondenceResidual,
  createSessionHistory,
  cssAffineMatrix,
  displayPointToImagePoint,
  redoSessionHistory,
  residualSummary,
  suggestedTranslation,
  transformHeightPointToRgb,
  undoSessionHistory,
  type AlignmentViewState,
  type CorrespondencePoint,
  type EngineeringCoordinateSpace,
  type EngineeringRoi,
  type EngineeringSessionSnapshot,
  type EngineeringTool,
  type EngineeringViewMode,
  type HeightPreviewPalette,
  type PixelPoint,
  type SessionHistory,
} from '../utils/engineeringSession'

const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
const VIEW_MODES: EngineeringViewMode[] = ['RGB', 'Height', 'Side-by-side', 'Alpha overlay', 'Split comparison']
const TRANSLATION_LIMIT = 10_000
const ROTATION_LIMIT = 180
const SCALE_MINIMUM = 0.01
const SCALE_MAXIMUM = 100
const TOOL_DEFINITIONS: Array<{ tool: EngineeringTool; label: string; shortcut: string; help: string }> = [
  { tool: 'pointer', label: 'Pointer', shortcut: 'V', help: 'click the RGB or height view to select a native pixel.' },
  { tool: 'pan', label: 'Pan', shortcut: 'H', help: 'drag the canvas or use the pan buttons.' },
  { tool: 'sample', label: 'Sample', shortcut: 'S', help: 'click the RGB or height view to select and sample native values.' },
  { tool: 'correspondence', label: 'Correspondence', shortcut: 'C', help: 'select RGB, then height, then confirm with Add Pair.' },
  { tool: 'rectangle', label: 'Rectangle', shortcut: 'R', help: 'drag a pixel rectangle in the active coordinate space.' },
  { tool: 'line', label: 'Line', shortcut: 'L', help: 'drag a pixel line in the active coordinate space.' },
]

type RasterKind = 'RGB' | 'Height'

function Crosshair({ kind, point, metadata }: { kind: RasterKind; point: PixelPoint; metadata: EngineeringRasterMetadata }) {
  const left = ((point.x + 0.5) / metadata.width) * 100
  const top = ((point.y + 0.5) / metadata.height) * 100
  return (
    <span
      className={`engineering-crosshair engineering-crosshair-${kind.toLowerCase()}`}
      style={{ left: `${left}%`, top: `${top}%` }}
      role="img"
      aria-label={`${kind} selected coordinate X ${point.x}, Y ${point.y}`}
      data-testid={`${kind.toLowerCase()}-crosshair`}
    >
      <span className="engineering-crosshair-horizontal" aria-hidden="true" />
      <span className="engineering-crosshair-vertical" aria-hidden="true" />
      <span className="engineering-crosshair-centre" aria-hidden="true" />
      <span className="engineering-crosshair-label">{kind} X {point.x} Y {point.y}</span>
    </span>
  )
}

function EvidenceImage({
  kind,
  src,
  metadata,
  selectedPoint,
  rois,
  correspondences,
  selectedCorrespondenceId,
  onSelectCorrespondence,
  residualAlignment,
  residualDimensions,
  showResiduals,
  highestResidualId,
  alignmentTransform,
}: {
  kind: RasterKind
  src: string
  metadata: EngineeringRasterMetadata
  selectedPoint: PixelPoint | null
  rois: EngineeringRoi[]
  correspondences: CorrespondencePoint[]
  selectedCorrespondenceId: string | null
  onSelectCorrespondence: (id: string) => void
  residualAlignment: EngineeringSessionSnapshot['alignment']
  residualDimensions: { rgb: { width: number; height: number }; height: { width: number; height: number } }
  showResiduals: boolean
  highestResidualId: string | null
  alignmentTransform?: string
}) {
  const coordinateSpace = kind === 'RGB' ? 'RGB' : 'HEIGHT'
  const strokeWidth = Math.max(metadata.width, metadata.height) / 200
  return (
    <div className={`engineering-image-frame engineering-image-${kind.toLowerCase()}`} data-evidence-kind={kind} data-testid={`${kind.toLowerCase()}-evidence-frame`}>
      <span className="engineering-image-label">{kind} - {metadata.width} x {metadata.height}</span>
      <div
        className="engineering-raster-anchor"
        style={{
          aspectRatio: `${metadata.width} / ${metadata.height}`,
          '--raster-width': `${metadata.width}px`,
          '--raster-height': `${metadata.height}px`,
        } as CSSProperties}
        data-raster-space={coordinateSpace}
        data-testid={`${kind.toLowerCase()}-raster`}
      >
        <div className="engineering-raster-layer" style={alignmentTransform ? { transform: alignmentTransform, transformOrigin: 'center' } : undefined}>
          <img src={src} alt={`${kind} evidence preview`} draggable={false} />
          <svg className="engineering-roi-overlay" viewBox={`0 0 ${metadata.width} ${metadata.height}`} aria-label={`${kind} measurements`}>
            <title>{kind} session measurements</title>
            {rois.filter((roi) => roi.coordinateSpace === coordinateSpace).map((roi) => {
              if (roi.kind === 'POINT') return <circle key={roi.id} cx={roi.x + 0.5} cy={roi.y + 0.5} r={strokeWidth * 1.8} data-roi-id={roi.id} />
              if (roi.kind === 'RECTANGLE') return <rect key={roi.id} x={roi.x} y={roi.y} width={roi.width} height={roi.height} data-roi-id={roi.id} />
              return <line key={roi.id} x1={roi.x1 + 0.5} y1={roi.y1 + 0.5} x2={roi.x2 + 0.5} y2={roi.y2 + 0.5} data-roi-id={roi.id} />
            })}
          </svg>
          <svg className="engineering-landmark-overlay" viewBox={`0 0 ${metadata.width} ${metadata.height}`} aria-label={`${kind} correspondence landmarks`}>
            <title>{kind} numbered correspondence landmarks</title>
            {kind === 'RGB' && showResiduals && correspondences.map((point) => {
              const transformed = transformHeightPointToRgb(point.height, residualAlignment, residualDimensions)
              const residual = correspondenceResidual(point, residualAlignment, residualDimensions)
              const labelX = (point.rgb.x + transformed.x) / 2
              const labelY = (point.rgb.y + transformed.y) / 2
              return (
                <g key={`residual-${point.id}`} className={`engineering-residual ${highestResidualId === point.id ? 'highest' : ''}`} data-testid={`residual-${point.id}`}>
                  <line x1={transformed.x + 0.5} y1={transformed.y + 0.5} x2={point.rgb.x + 0.5} y2={point.rgb.y + 0.5} />
                  <circle className="transformed-height-point" cx={transformed.x + 0.5} cy={transformed.y + 0.5} r={strokeWidth * 1.5} />
                  <text x={labelX + 0.5} y={labelY + 0.5}>{point.id} {residual}px</text>
                </g>
              )
            })}
            {correspondences.map((point) => {
              const coordinate = kind === 'RGB' ? point.rgb : point.height
              const selected = selectedCorrespondenceId === point.id
              const half = strokeWidth * 2.6
              return (
                <g
                  key={`${kind}-${point.id}`}
                  className={`engineering-landmark ${kind.toLowerCase()} ${selected ? 'selected' : ''}`}
                  data-testid={`${kind.toLowerCase()}-landmark-${point.id}`}
                  role="button"
                  tabIndex={0}
                  aria-label={`Select ${point.id} ${kind} landmark`}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={(event) => { event.stopPropagation(); onSelectCorrespondence(point.id) }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      event.stopPropagation()
                      onSelectCorrespondence(point.id)
                    }
                  }}
                >
                  {kind === 'RGB'
                    ? <circle cx={coordinate.x + 0.5} cy={coordinate.y + 0.5} r={half} />
                    : <polygon points={`${coordinate.x + 0.5},${coordinate.y + 0.5 - half} ${coordinate.x + 0.5 + half},${coordinate.y + 0.5} ${coordinate.x + 0.5},${coordinate.y + 0.5 + half} ${coordinate.x + 0.5 - half},${coordinate.y + 0.5}`} />}
                  <text x={coordinate.x + 0.5 + half * 1.35} y={coordinate.y + 0.5 - half * 1.35}>{point.id}</text>
                </g>
              )
            })}
          </svg>
          {selectedPoint && <Crosshair kind={kind} point={selectedPoint} metadata={metadata} />}
        </div>
      </div>
    </div>
  )
}

function HeightLegend({
  palette,
  minimum,
  maximum,
  showInvalid,
}: {
  palette: HeightPreviewPalette
  minimum: number
  maximum: number
  showInvalid: boolean
}) {
  return (
    <div className="height-palette-legend" aria-label="Derived height preview palette legend">
      <span>{minimum}</span><span className={`height-palette-ramp palette-${palette}`} aria-hidden="true" /><span>{maximum}</span>
      {showInvalid && <span className="invalid-height-key"><i aria-hidden="true" />Invalid</span>}
      <small>{palette} derived colour view</small>
    </div>
  )
}

function Histogram({
  view,
  displayMin,
  displayMax,
  sampledHeight,
  selectedRoi,
  onUseRange,
  onResetRange,
}: {
  view: EngineeringViewResponse
  displayMin: number
  displayMax: number
  sampledHeight: number | null
  selectedRoi: Extract<EngineeringRoi, { kind: 'RECTANGLE' }> | null
  onUseRange: (minimum: number, maximum: number) => void
  onResetRange: () => void
}) {
  const histogram = view.height_statistics.histogram
  const maximum = Math.max(...histogram.counts, 1)
  const [activeBin, setActiveBin] = useState<number | null>(null)
  const [selectedBin, setSelectedBin] = useState<number | null>(null)
  const span = histogram.native_max - histogram.native_min
  const binBounds = (index: number) => {
    if (span === 0) return [histogram.native_min, histogram.native_max] as const
    const width = span / 64
    return [
      histogram.native_min + width * index,
      index === 63 ? histogram.native_max : histogram.native_min + width * (index + 1),
    ] as const
  }
  const xForValue = (value: number) => span === 0 ? 0 : Math.max(0, Math.min(640, ((value - histogram.native_min) / span) * 640))
  const activeBounds = activeBin === null ? null : binBounds(activeBin)
  const selectedStatistics = selectedRoi?.nativeHeightStatistics
  return (
    <section className="engineering-histogram" aria-labelledby="height-histogram-title">
      <div className="subpanel-heading"><h4 id="height-histogram-title">Native height distribution</h4><span className="mono">64 bins</span></div>
      <svg viewBox="0 0 640 130" role="group" aria-label="64-bin native height histogram" preserveAspectRatio="none">
        <title>64-bin native height histogram</title>
        {histogram.counts.map((count, index) => {
          const height = (count / maximum) * 110
          const [minimum, maximumValue] = binBounds(index)
          return (
            <g
              key={index}
              role="button"
              tabIndex={0}
              aria-label={`Bin ${index + 1}, native range ${minimum} to ${maximumValue}, count ${count}`}
              data-histogram-bin={index}
              onMouseEnter={() => setActiveBin(index)}
              onMouseLeave={() => setActiveBin(null)}
              onFocus={() => setActiveBin(index)}
              onBlur={() => setActiveBin(null)}
              onClick={() => setSelectedBin(index)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  setSelectedBin(index)
                }
              }}
            >
              <rect x={index * 10 + 1} y={120 - height} width="8" height={height} />
            </g>
          )
        })}
        {selectedStatistics && <rect className="roi-range-marker" x={xForValue(selectedStatistics.nativeMin)} y="2" width={Math.max(2, xForValue(selectedStatistics.nativeMax) - xForValue(selectedStatistics.nativeMin))} height="116" />}
        <line className="display-range-marker" x1={xForValue(displayMin)} y1="0" x2={xForValue(displayMin)} y2="120" />
        <line className="display-range-marker" x1={xForValue(displayMax)} y1="0" x2={xForValue(displayMax)} y2="120" />
        {sampledHeight !== null && <line className="sample-value-marker" x1={xForValue(sampledHeight)} y1="0" x2={xForValue(sampledHeight)} y2="120" />}
        <line x1="0" y1="120" x2="640" y2="120" />
      </svg>
      <div className="histogram-axis"><span>{histogram.native_min}</span><span>Native sample value</span><span>{histogram.native_max}</span></div>
      <p className="histogram-bin-detail" aria-live="polite">{activeBounds ? `Bin ${activeBin! + 1}: ${activeBounds[0]} to ${activeBounds[1]}; count ${histogram.counts[activeBin!]}` : 'Focus or hover a bin for its native range and count.'}</p>
      <div className="histogram-actions">
        <button type="button" disabled={selectedBin === null} onClick={() => {
          if (selectedBin !== null) {
            const [minimum, maximumValue] = binBounds(selectedBin)
            onUseRange(minimum, maximumValue)
          }
        }}>Use selected bin as display range</button>
        <button type="button" onClick={onResetRange}>Reset display range</button>
      </div>
      <dl className="engineering-definition-list compact">
        <div><dt>Valid total</dt><dd>{view.height_statistics.valid_count.toLocaleString()}</dd></div>
        <div><dt>Excluded invalid</dt><dd>{view.height_statistics.invalid_count.toLocaleString()}</dd></div>
        <div><dt>Display range</dt><dd>{displayMin} to {displayMax}</dd></div>
        <div><dt>Selected ROI range</dt><dd>{selectedStatistics ? `${selectedStatistics.nativeMin} to ${selectedStatistics.nativeMax}` : 'None'}</dd></div>
      </dl>
    </section>
  )
}

function PipelineCard({ id, title, available, children }: { id?: string; title: string; available: boolean; children: ReactNode }) {
  return <article id={id} className="pipeline-card"><div className="subpanel-heading"><h4>{title}</h4><span className={available ? 'pipeline-available' : 'pipeline-unavailable'}>{available ? 'Persisted' : 'Not available'}</span></div>{children}</article>
}

function sampledRgbValues(sample: EngineeringSampleResponse | null): [string, string, string] {
  return [String(sample?.rgb.values[0] ?? 'Not sampled'), String(sample?.rgb.values[1] ?? 'Not sampled'), String(sample?.rgb.values[2] ?? 'Not sampled')]
}

export function EngineeringViewPage() {
  const { inspectionId = '' } = useParams()
  const validId = CANONICAL_UUID.test(inspectionId)
  const [view, setView] = useState<EngineeringViewResponse | null>(null)
  const [error, setError] = useState<ApiClientError | null>(null)
  const [loading, setLoading] = useState(validId)
  const [history, setHistory] = useState<SessionHistory<EngineeringSessionSnapshot>>(() => createSessionHistory(DEFAULT_ENGINEERING_SESSION))
  const [tool, setTool] = useState<EngineeringTool>('pointer')
  const [sample, setSample] = useState<EngineeringSampleResponse | null>(null)
  const [sampleError, setSampleError] = useState<ApiClientError | null>(null)
  const [sampling, setSampling] = useState(false)
  const [roiError, setRoiError] = useState<ApiClientError | null>(null)
  const [roiLoading, setRoiLoading] = useState(false)
  const [displayMinDraft, setDisplayMinDraft] = useState('')
  const [displayMaxDraft, setDisplayMaxDraft] = useState('')
  const [displayRangeError, setDisplayRangeError] = useState<string | null>(null)
  const [guideOpen, setGuideOpen] = useState(true)
  const [keyboardHelpOpen, setKeyboardHelpOpen] = useState(false)
  const [flickerRunning, setFlickerRunning] = useState(false)
  const [flickerPhase, setFlickerPhase] = useState<AlignmentViewState>('DEVELOPMENT')
  const [reducedMotion, setReducedMotion] = useState(() => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false)
  const [flickerNotice, setFlickerNotice] = useState<string | null>(null)
  const drag = useRef<{ clientX: number; clientY: number; panX: number; panY: number; before: EngineeringSessionSnapshot; moved: boolean } | null>(null)
  const roiStart = useRef<{ point: PixelPoint; space: EngineeringCoordinateSpace } | null>(null)
  const sampleAbort = useRef<AbortController | null>(null)
  const sampleSequence = useRef(0)
  const roiAbort = useRef<AbortController | null>(null)
  const roiSequence = useRef(0)
  const session = history.present

  const commit = useCallback((update: (current: EngineeringSessionSnapshot) => EngineeringSessionSnapshot) => {
    setHistory((current) => commitSessionHistory(current, update(current.present)))
  }, [])

  const undo = useCallback(() => {
    drag.current = null
    roiStart.current = null
    setHistory((current) => undoSessionHistory(current))
  }, [])

  const redo = useCallback(() => {
    drag.current = null
    roiStart.current = null
    setHistory((current) => redoSessionHistory(current))
  }, [])

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

  useEffect(() => {
    setHistory(createSessionHistory(DEFAULT_ENGINEERING_SESSION))
    setTool('pointer')
    setSample(null)
    setSampleError(null)
    setGuideOpen(true)
    setKeyboardHelpOpen(false)
    setFlickerRunning(false)
    setFlickerPhase('DEVELOPMENT')
    setFlickerNotice(null)
    drag.current = null
    roiStart.current = null
    sampleAbort.current?.abort()
    roiAbort.current?.abort()
    roiSequence.current += 1
    setRoiError(null)
    setRoiLoading(false)
    setDisplayMinDraft('')
    setDisplayMaxDraft('')
    setDisplayRangeError(null)
  }, [inspectionId])

  useEffect(() => () => {
    sampleAbort.current?.abort()
    roiAbort.current?.abort()
  }, [])

  useEffect(() => {
    if (!view) return
    setDisplayMinDraft(String(session.heightDisplayMin ?? view.height_statistics.native_min))
    setDisplayMaxDraft(String(session.heightDisplayMax ?? view.height_statistics.native_max))
    setDisplayRangeError(null)
  }, [session.heightDisplayMax, session.heightDisplayMin, view])

  useEffect(() => {
    const query = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!query) return
    const update = () => setReducedMotion(query.matches)
    update()
    query.addEventListener?.('change', update)
    return () => query.removeEventListener?.('change', update)
  }, [])

  useEffect(() => {
    if (!flickerRunning) return
    const stopWhenHidden = () => {
      if (document.hidden) {
        setFlickerRunning(false)
        setFlickerPhase('DEVELOPMENT')
        setFlickerNotice('Flicker stopped because this browser tab is hidden.')
      }
    }
    const interval = window.setInterval(() => {
      if (!document.hidden) setFlickerPhase((current) => current === 'ORIGINAL' ? 'DEVELOPMENT' : 'ORIGINAL')
    }, 400)
    document.addEventListener('visibilitychange', stopWhenHidden)
    return () => {
      window.clearInterval(interval)
      document.removeEventListener('visibilitychange', stopWhenHidden)
    }
  }, [flickerRunning])

  useEffect(() => {
    if (session.mode !== 'Alpha overlay' && flickerRunning) {
      setFlickerRunning(false)
      setFlickerPhase('DEVELOPMENT')
      setFlickerNotice('Flicker stopped after leaving Alpha overlay mode.')
    }
  }, [flickerRunning, session.mode])

  useEffect(() => {
    if (reducedMotion && flickerRunning) {
      setFlickerRunning(false)
      setFlickerPhase('DEVELOPMENT')
      setFlickerNotice('Flicker stopped because reduced motion is enabled in the browser or operating system.')
    }
  }, [flickerRunning, reducedMotion])

  const requestSample = useCallback(async (coordinates: EngineeringSessionSnapshot['coordinates']) => {
    if (!view) return
    sampleAbort.current?.abort()
    const controller = new AbortController()
    sampleAbort.current = controller
    const sequence = ++sampleSequence.current
    setSampling(true)
    setSampleError(null)
    try {
      const response = await getEngineeringSample(inspectionId, coordinates, controller.signal)
      if (sequence === sampleSequence.current) setSample(response.data)
    } catch (caught) {
      const mapped = toApiClientError(caught)
      if (sequence === sampleSequence.current && mapped.code !== 'REQUEST_ABORTED') setSampleError(mapped)
    } finally {
      if (sequence === sampleSequence.current) setSampling(false)
    }
  }, [inspectionId, view])

  const selectTool = useCallback((next: EngineeringTool) => {
    drag.current = null
    roiStart.current = null
    if (tool === 'correspondence' && next !== 'correspondence' && (session.pendingRgbPoint || session.pendingHeightPoint)) {
      commit((current) => ({ ...current, pendingRgbPoint: null, pendingHeightPoint: null }))
    }
    setTool(next)
  }, [commit, session.pendingHeightPoint, session.pendingRgbPoint, tool])

  const cancelCurrentAction = useCallback(() => {
    drag.current = null
    roiStart.current = null
    sampleAbort.current?.abort()
    roiAbort.current?.abort()
    roiSequence.current += 1
    setRoiLoading(false)
    if (session.pendingRgbPoint || session.pendingHeightPoint) {
      commit((current) => ({ ...current, pendingRgbPoint: null, pendingHeightPoint: null }))
    }
  }, [commit, session.pendingHeightPoint, session.pendingRgbPoint])

  const changeZoom = useCallback((delta: number) => {
    commit((current) => ({ ...current, scaleMode: 'fit', zoom: Math.min(4, Math.max(0.25, Number((current.zoom + delta).toFixed(2)))) }))
  }, [commit])

  const fitToView = useCallback(() => commit((current) => ({ ...current, scaleMode: 'fit', zoom: 1, pan: { x: 0, y: 0 } })), [commit])
  const actualPixels = useCallback(() => commit((current) => ({ ...current, scaleMode: 'actual', zoom: 1, pan: { x: 0, y: 0 } })), [commit])
  const movePan = useCallback((x: number, y: number) => commit((current) => ({ ...current, pan: { x: Math.min(1, Math.max(-1, current.pan.x + x)), y: Math.min(1, Math.max(-1, current.pan.y + y)) } })), [commit])

  const setCoordinate = useCallback((space: EngineeringCoordinateSpace, point: PixelPoint, shouldSample = false) => {
    const nextCoordinates: EngineeringSessionSnapshot['coordinates'] = space === 'RGB'
      ? { ...session.coordinates, rgbX: point.x, rgbY: point.y, rgbSelected: true, activeSpace: 'RGB' }
      : { ...session.coordinates, heightX: point.x, heightY: point.y, heightSelected: true, activeSpace: 'HEIGHT' }
    commit((current) => ({ ...current, coordinates: nextCoordinates }))
    if (shouldSample) void requestSample(nextCoordinates)
  }, [commit, requestSample, session.coordinates])

  const adjustActiveCoordinate = useCallback((dx: number, dy: number) => {
    if (!view) return
    const space = session.coordinates.activeSpace
    const selected = space === 'RGB' ? session.coordinates.rgbSelected : session.coordinates.heightSelected
    if (!selected) return
    const metadata = space === 'RGB' ? view.rgb : view.height
    const point = space === 'RGB'
      ? { x: session.coordinates.rgbX + dx, y: session.coordinates.rgbY + dy }
      : { x: session.coordinates.heightX + dx, y: session.coordinates.heightY + dy }
    setCoordinate(space, clampImagePoint(point, metadata), false)
  }, [session.coordinates, setCoordinate, view])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target
      if (target instanceof HTMLElement && (target.matches('input, textarea, select') || target.isContentEditable)) return
      const key = event.key.toLowerCase()
      if ((event.ctrlKey || event.metaKey) && key === 'z') {
        event.preventDefault()
        if (event.shiftKey) redo(); else undo()
        return
      }
      if ((event.ctrlKey || event.metaKey) && key === 'y') {
        event.preventDefault(); redo(); return
      }
      if (key === 'escape') {
        event.preventDefault()
        if (keyboardHelpOpen) setKeyboardHelpOpen(false); else cancelCurrentAction()
        return
      }
      const toolShortcut = TOOL_DEFINITIONS.find((item) => item.shortcut.toLowerCase() === key)
      if (toolShortcut) { event.preventDefault(); selectTool(toolShortcut.tool); return }
      if (key === '+' || key === '=') { event.preventDefault(); changeZoom(0.25); return }
      if (key === '-') { event.preventDefault(); changeZoom(-0.25); return }
      if (key === 'f') { event.preventDefault(); fitToView(); return }
      if (key === '0') { event.preventDefault(); actualPixels(); return }
      const amount = event.shiftKey ? 10 : 1
      if (key === 'arrowleft') { event.preventDefault(); adjustActiveCoordinate(-amount, 0) }
      if (key === 'arrowright') { event.preventDefault(); adjustActiveCoordinate(amount, 0) }
      if (key === 'arrowup') { event.preventDefault(); adjustActiveCoordinate(0, -amount) }
      if (key === 'arrowdown') { event.preventDefault(); adjustActiveCoordinate(0, amount) }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [actualPixels, adjustActiveCoordinate, cancelCurrentAction, changeZoom, fitToView, keyboardHelpOpen, redo, selectTool, undo])

  if (!validId) return <section className="not-found" role="alert"><p className="eyebrow">Invalid route parameter</p><h2>Malformed inspection ID</h2><p>The engineering-view route requires a lowercase, hyphenated UUID.</p><Link className="button secondary" to="/">Return to inspection history</Link></section>
  if (loading && !view) return <p className="loading-state">Loading integrity-verified engineering evidence...</p>
  if (error && !view) return <ErrorPanel error={error} onRetry={() => void load()} title="Engineering workspace unavailable" />
  if (!view) return null

  const renderedAlignmentView = flickerRunning ? flickerPhase : session.alignmentView
  const renderedAlignment = renderedAlignmentView === 'DEVELOPMENT' ? session.alignment : DEFAULT_ALIGNMENT
  const comparisonDimensions = { rgb: { width: view.rgb.width, height: view.rgb.height }, height: { width: view.height.width, height: view.height.height } }

  const pointForSpace = (event: ReactPointerEvent<HTMLDivElement>, space: EngineeringCoordinateSpace): PixelPoint | null => {
    const anchor = event.currentTarget.querySelector<HTMLElement>(`[data-raster-space="${space}"]`)
    if (!anchor) return null
    const bounds = anchor.getBoundingClientRect()
    const metadata = space === 'RGB' ? view.rgb : view.height
    const outerScale = {
      x: anchor.offsetWidth > 0 ? bounds.width / anchor.offsetWidth : session.zoom,
      y: anchor.offsetHeight > 0 ? bounds.height / anchor.offsetHeight : session.zoom,
    }
    return displayPointToImagePoint(
      { x: event.clientX, y: event.clientY },
      bounds,
      metadata,
      space === 'HEIGHT' ? renderedAlignment : DEFAULT_ALIGNMENT,
      outerScale,
    )
  }

  const resolveInteraction = (event: ReactPointerEvent<HTMLDivElement>, forcedSpace?: EngineeringCoordinateSpace) => {
    const spaces: EngineeringCoordinateSpace[] = forcedSpace
      ? [forcedSpace]
      : session.mode === 'RGB'
        ? ['RGB']
        : session.mode === 'Height'
          ? ['HEIGHT']
          : session.mode === 'Side-by-side'
            ? ['RGB', 'HEIGHT']
            : [session.coordinates.activeSpace]
    for (const space of spaces) {
      const point = pointForSpace(event, space)
      if (point) return { space, point }
    }
    return null
  }

  const addCorrespondenceSelection = (space: EngineeringCoordinateSpace, point: PixelPoint) => {
    commit((current) => {
      const coordinates = space === 'RGB'
        ? { ...current.coordinates, rgbX: point.x, rgbY: point.y, rgbSelected: true, activeSpace: 'RGB' as const }
        : { ...current.coordinates, heightX: point.x, heightY: point.y, heightSelected: true, activeSpace: 'HEIGHT' as const }
      return {
        ...current,
        coordinates,
        pendingRgbPoint: space === 'RGB' ? point : current.pendingRgbPoint,
        pendingHeightPoint: space === 'HEIGHT' && current.pendingRgbPoint ? point : current.pendingHeightPoint,
      }
    })
  }

  const addPendingCorrespondence = () => {
    if (!session.pendingRgbPoint || !session.pendingHeightPoint) return
    commit((current) => {
      if (!current.pendingRgbPoint || !current.pendingHeightPoint) return current
      const id = `P${current.nextCorrespondenceNumber}`
      return {
        ...current,
        correspondences: [...current.correspondences, { id, rgb: current.pendingRgbPoint, height: current.pendingHeightPoint }],
        selectedCorrespondenceId: id,
        nextCorrespondenceNumber: current.nextCorrespondenceNumber + 1,
        pendingRgbPoint: null,
        pendingHeightPoint: null,
      }
    })
  }

  const pointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (tool === 'pan') {
      event.currentTarget.setPointerCapture?.(event.pointerId)
      drag.current = { clientX: event.clientX, clientY: event.clientY, panX: session.pan.x, panY: session.pan.y, before: session, moved: false }
      return
    }
    const interaction = resolveInteraction(event)
    if (!interaction) return
    if (tool === 'pointer') { setCoordinate(interaction.space, interaction.point); return }
    if (tool === 'sample') { setCoordinate(interaction.space, interaction.point, true); return }
    if (tool === 'correspondence') { addCorrespondenceSelection(interaction.space, interaction.point); return }
    roiStart.current = interaction
    event.currentTarget.setPointerCapture?.(event.pointerId)
  }

  const pointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current || tool !== 'pan') return
    const bounds = event.currentTarget.getBoundingClientRect()
    const nextPan = {
      x: Math.min(1, Math.max(-1, drag.current.panX + (event.clientX - drag.current.clientX) / Math.max(bounds.width, 1))),
      y: Math.min(1, Math.max(-1, drag.current.panY + (event.clientY - drag.current.clientY) / Math.max(bounds.height, 1))),
    }
    drag.current.moved = drag.current.moved || nextPan.x !== drag.current.panX || nextPan.y !== drag.current.panY
    setHistory((current) => ({ ...current, present: { ...current.present, pan: nextPan } }))
  }

  const pointerUp = async (event: ReactPointerEvent<HTMLDivElement>) => {
    const completedDrag = drag.current
    drag.current = null
    if (completedDrag) {
      if (completedDrag.moved) setHistory((current) => ({ past: [...current.past, completedDrag.before].slice(-SESSION_HISTORY_LIMIT), present: current.present, future: [] }))
      return
    }
    const start = roiStart.current
    roiStart.current = null
    if (!start || (tool !== 'rectangle' && tool !== 'line')) return
    const end = resolveInteraction(event, start.space)
    if (!end) return
    setRoiError(null)
    if (tool === 'line') {
      const deltaX = end.point.x - start.point.x
      const deltaY = end.point.y - start.point.y
      commit((current) => {
        const id = `M${current.rois.length + 1}`
        return {
          ...current,
          coordinates: { ...current.coordinates, activeSpace: start.space },
          selectedRoiId: id,
          rois: [...current.rois, {
            id,
            kind: 'LINE',
            coordinateSpace: start.space,
            x1: start.point.x,
            y1: start.point.y,
            x2: end.point.x,
            y2: end.point.y,
            deltaXPixels: deltaX,
            deltaYPixels: deltaY,
            distancePixels: Number(Math.hypot(deltaX, deltaY).toFixed(4)),
            directionDegrees: Number((Math.atan2(deltaY, deltaX) * 180 / Math.PI).toFixed(4)),
          }],
        }
      })
      return
    }
    const rectangle = { x: Math.min(start.point.x, end.point.x), y: Math.min(start.point.y, end.point.y), width: Math.abs(end.point.x - start.point.x) + 1, height: Math.abs(end.point.y - start.point.y) + 1 }
    let nativeHeightStatistics: Extract<EngineeringRoi, { kind: 'RECTANGLE' }>['nativeHeightStatistics']
    if (start.space === 'HEIGHT') {
      roiAbort.current?.abort()
      const controller = new AbortController()
      roiAbort.current = controller
      const sequence = ++roiSequence.current
      setRoiLoading(true)
      try {
        const response = await getEngineeringHeightRoi(inspectionId, rectangle, controller.signal)
        if (sequence !== roiSequence.current) return
        nativeHeightStatistics = { nativeMin: response.data.native_min, nativeMax: response.data.native_max, nativeMean: response.data.native_mean, validCount: response.data.valid_count, invalidCount: response.data.invalid_count, storageDataType: response.data.storage_data_type }
      } catch (caught) {
        const mapped = toApiClientError(caught)
        if (sequence === roiSequence.current && mapped.code !== 'REQUEST_ABORTED') setRoiError(mapped)
        if (sequence === roiSequence.current) setRoiLoading(false)
        return
      }
      if (sequence === roiSequence.current) setRoiLoading(false)
    }
    commit((current) => {
      const id = `M${current.rois.length + 1}`
      return { ...current, coordinates: { ...current.coordinates, activeSpace: start.space }, selectedRoiId: id, rois: [...current.rois, { id, kind: 'RECTANGLE', coordinateSpace: start.space, ...rectangle, nativeHeightStatistics }] }
    })
  }

  const sampleCoordinates = (event: FormEvent) => { event.preventDefault(); void requestSample(session.coordinates) }
  const coordinateValid = session.coordinates.rgbX >= 0 && session.coordinates.rgbX < view.rgb.width && session.coordinates.rgbY >= 0 && session.coordinates.rgbY < view.rgb.height && session.coordinates.heightX >= 0 && session.coordinates.heightX < view.height.width && session.coordinates.heightY >= 0 && session.coordinates.heightY < view.height.height
  const resetCanvas = () => commit((current) => ({ ...current, mode: 'Side-by-side', zoom: 1, scaleMode: 'fit', pan: { x: 0, y: 0 }, overlayOpacity: 50, splitPosition: 50 }))
  const nativeHeightMin = view.height_statistics.native_min
  const nativeHeightMax = view.height_statistics.native_max
  const activeDisplayMin = session.heightDisplayMin ?? nativeHeightMin
  const activeDisplayMax = session.heightDisplayMax ?? nativeHeightMax
  const applyDisplayRange = (minimum: number, maximum: number) => {
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) {
      setDisplayRangeError('Display minimum and maximum must be finite numbers.')
      return
    }
    if (minimum >= maximum) {
      setDisplayRangeError('Display minimum must be less than display maximum.')
      return
    }
    if (minimum < nativeHeightMin || maximum > nativeHeightMax) {
      setDisplayRangeError(`Display range must remain within native limits ${nativeHeightMin} to ${nativeHeightMax}.`)
      return
    }
    setDisplayRangeError(null)
    setDisplayMinDraft(String(minimum))
    setDisplayMaxDraft(String(maximum))
    commit((current) => ({ ...current, heightDisplayMin: minimum, heightDisplayMax: maximum }))
  }
  const resetDisplayRange = () => {
    setDisplayRangeError(null)
    setDisplayMinDraft(String(nativeHeightMin))
    setDisplayMaxDraft(String(nativeHeightMax))
    commit((current) => ({ ...current, heightDisplayMin: null, heightDisplayMax: null }))
  }
  const resetEngineeringSession = () => {
    if (!window.confirm('Reset all engineering session controls and measurements? Persisted evidence will not be reloaded or changed.')) return
    sampleAbort.current?.abort()
    roiAbort.current?.abort()
    sampleSequence.current += 1
    roiSequence.current += 1
    drag.current = null
    roiStart.current = null
    setHistory(createSessionHistory(DEFAULT_ENGINEERING_SESSION))
    setTool('pointer')
    setSample(null)
    setSampleError(null)
    setSampling(false)
    setRoiError(null)
    setRoiLoading(false)
    setDisplayRangeError(null)
    setDisplayMinDraft(String(nativeHeightMin))
    setDisplayMaxDraft(String(nativeHeightMax))
    setFlickerRunning(false)
    setFlickerPhase('DEVELOPMENT')
    setFlickerNotice(null)
  }
  const clearSelections = () => commit((current) => ({ ...current, coordinates: { ...current.coordinates, rgbSelected: false, heightSelected: false } }))
  const rgbSelectedPoint = session.coordinates.rgbSelected ? { x: session.coordinates.rgbX, y: session.coordinates.rgbY } : null
  const heightSelectedPoint = session.coordinates.heightSelected ? { x: session.coordinates.heightX, y: session.coordinates.heightY } : null
  const matrix = affineMatrix(session.alignment)
  const residuals = residualSummary(session.correspondences, session.alignment, comparisonDimensions)
  const translationSuggestion = suggestedTranslation(session.correspondences, session.alignment, comparisonDimensions)
  const correspondenceStep = !session.pendingRgbPoint ? 1 : !session.pendingHeightPoint ? 2 : 3
  const rgbValues = sampledRgbValues(sample)
  const toolDefinition = TOOL_DEFINITIONS.find((item) => item.tool === tool) ?? TOOL_DEFINITIONS[0]!
  const transform = `translate(${session.pan.x * 100}%, ${session.pan.y * 100}%) scale(${session.zoom})`
  const dimensionMismatch = view.rgb.width !== view.height.width || view.rgb.height !== view.height.height
  const selectedHeightRoi = session.rois.find((roi): roi is Extract<EngineeringRoi, { kind: 'RECTANGLE' }> => roi.id === session.selectedRoiId && roi.kind === 'RECTANGLE' && roi.coordinateSpace === 'HEIGHT') ?? null
  const selectCorrespondence = (id: string) => commit((current) => ({ ...current, selectedCorrespondenceId: id }))
  const sharedEvidenceProps = {
    correspondences: session.correspondences,
    selectedCorrespondenceId: session.selectedCorrespondenceId,
    onSelectCorrespondence: selectCorrespondence,
    residualAlignment: session.alignment,
    residualDimensions: comparisonDimensions,
    showResiduals: session.showResiduals,
    highestResidualId: residuals.highestPairId,
  }
  const rgbImage = <EvidenceImage kind="RGB" src={engineeringPreviewUrl(inspectionId, 'rgb')} metadata={view.rgb} selectedPoint={rgbSelectedPoint} rois={session.rois} {...sharedEvidenceProps} />
  const heightImage = <EvidenceImage kind="Height" src={engineeringPreviewUrl(inspectionId, 'height', { palette: session.heightPalette, displayMin: session.heightDisplayMin, displayMax: session.heightDisplayMax, showInvalid: session.showInvalidHeight })} metadata={view.height} selectedPoint={heightSelectedPoint} rois={session.rois} alignmentTransform={renderedAlignmentView === 'DEVELOPMENT' ? cssAffineMatrix(session.alignment) : undefined} {...sharedEvidenceProps} />

  const addManualCorrespondence = (space: EngineeringCoordinateSpace) => {
    const point = space === 'RGB' ? { x: session.coordinates.rgbX, y: session.coordinates.rgbY } : { x: session.coordinates.heightX, y: session.coordinates.heightY }
    addCorrespondenceSelection(space, point)
  }
  const setAlignmentView = (next: AlignmentViewState) => {
    setFlickerRunning(false)
    setFlickerPhase(next)
    setFlickerNotice(null)
    commit((current) => ({ ...current, alignmentView: next }))
  }
  const startFlicker = () => {
    if (reducedMotion) {
      setFlickerNotice('Flicker is unavailable because reduced motion is enabled in the browser or operating system.')
      return
    }
    commit((current) => ({ ...current, mode: 'Alpha overlay', alignmentView: 'DEVELOPMENT' }))
    setFlickerPhase('ORIGINAL')
    setFlickerNotice('Manual flicker is running at 2.5 changes per second. This is a visual comparison, not proof of registration quality.')
    setFlickerRunning(true)
  }
  const stopFlicker = () => {
    setFlickerRunning(false)
    setFlickerPhase('DEVELOPMENT')
    setFlickerNotice('Flicker stopped. Development-aligned view is shown.')
  }
  const bounded = (value: number, minimum: number, maximum: number) => Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, value)) : 0
  const exportAlignment = () => {
    const payload = buildAlignmentExport(inspectionId, session.alignment, session.correspondences, session.rois, session.overlayOpacity, { activeView: session.alignmentView, dimensions: comparisonDimensions })
    const url = URL.createObjectURL(new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: 'application/json' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = alignmentExportFilename(inspectionId); anchor.click(); URL.revokeObjectURL(url)
  }

  return (
    <section className="engineering-page" aria-labelledby="engineering-workspace-title">
      <div className="engineering-page-header">
        <div><p className="eyebrow">Read-only synthetic evidence viewer</p><h2 id="engineering-workspace-title">PCB 2D/3D Vision Engineering Workspace</h2><p className="mono page-identifier">{inspectionId}</p></div>
        <div className="page-actions"><button type="button" className="button secondary" onClick={() => setGuideOpen(true)}>Show guide</button><StatusBadge value={view.inspection_status} /><Link className="button secondary" to={`/inspections/${inspectionId}`}>Inspection detail</Link></div>
      </div>

      <div className="engineering-safety-banner" role="note"><strong>Synthetic engineering data</strong><span>No production measurement - No physical calibration - No real registration - No production inspection decision</span></div>
      {guideOpen && <section className="engineering-onboarding" aria-labelledby="engineering-guide-title"><div><p className="step-number">Quick start</p><h3 id="engineering-guide-title">Engineering workspace guide</h3></div><ol><li>Choose a view mode.</li><li>Inspect RGB and height evidence.</li><li>Use Sample and click each image.</li><li>Review native values.</li><li>Review alignment and persisted pipeline evidence.</li></ol><button type="button" className="button secondary" onClick={() => setGuideOpen(false)}>Dismiss guide</button></section>}
      {dimensionMismatch && <div className="dimension-warning" role="alert"><strong>Dimension mismatch:</strong> RGB is {view.rgb.width} x {view.rgb.height}; height is {view.height.width} x {view.height.height}. Coordinates remain independent and overlay alignment is illustrative only.</div>}

      <div className="engineering-workspace-grid">
        <nav className="evidence-navigator" aria-label="Engineering evidence navigator">
          <p className="step-number">Evidence navigator</p><button type="button" onClick={() => commit((current) => ({ ...current, mode: 'RGB' }))}>RGB evidence</button><button type="button" onClick={() => commit((current) => ({ ...current, mode: 'Height' }))}>Height evidence</button><a href="#technical-validation">Technical validation</a><a href="#synthetic-preprocessing">Synthetic preprocessing</a><a href="#mock-inference">Deterministic mock inference</a><a href="#persisted-result">Persisted result</a><Link to={`/inspections/${inspectionId}#audit-timeline`}>Audit</Link><Link to={`/inspections/${inspectionId}/report`}>Development report</Link><p className="evidence-readonly-note">GET-only workspace<br />Persisted evidence only</p>
        </nav>

        <section className="vision-workbench" aria-label="Vision canvas workspace" aria-describedby="active-tool-help">
          <div className="view-mode-toolbar" role="group" aria-label="Vision view modes">{VIEW_MODES.map((item) => <button key={item} type="button" aria-pressed={session.mode === item} onClick={() => commit((current) => ({ ...current, mode: item }))}>{item}</button>)}</div>
          <div className="canvas-toolbar" role="toolbar" aria-label="Vision canvas controls">
            <button type="button" onClick={() => changeZoom(-0.25)} aria-label="Zoom out" aria-keyshortcuts="-">-</button><output aria-label="Zoom level">{Math.round(session.zoom * 100)}%</output><button type="button" onClick={() => changeZoom(0.25)} aria-label="Zoom in" aria-keyshortcuts="+ =">+</button>
            <button type="button" aria-pressed={session.scaleMode === 'fit'} onClick={fitToView} aria-keyshortcuts="F">Fit</button><button type="button" aria-pressed={session.scaleMode === 'actual'} onClick={actualPixels} aria-keyshortcuts="0">Actual pixels</button><button type="button" onClick={resetCanvas}>Reset view</button>
            <span className="toolbar-separator" aria-hidden="true" />
            <button type="button" onClick={undo} disabled={!history.past.length} aria-keyshortcuts="Control+Z Meta+Z">Undo</button><button type="button" onClick={redo} disabled={!history.future.length} aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z Control+Y Meta+Y">Redo</button><button type="button" onClick={() => setKeyboardHelpOpen((current) => !current)} aria-expanded={keyboardHelpOpen} aria-controls="engineering-keyboard-help">Keyboard help</button><button type="button" className="danger-outline" onClick={resetEngineeringSession}>Reset Engineering Session</button>
          </div>
          <div className="engineering-tool-strip" role="toolbar" aria-label="Engineering tools">{TOOL_DEFINITIONS.map((item) => <button key={item.tool} type="button" aria-pressed={tool === item.tool} aria-keyshortcuts={item.shortcut} onClick={() => selectTool(item.tool)}>{item.label}<kbd>{item.shortcut}</kbd></button>)}</div>
          <div className="active-tool-status" id="active-tool-help" role="status" aria-live="polite"><strong>Active tool: {toolDefinition.label}</strong><span>{toolDefinition.help}</span><span>Active coordinate space: {session.coordinates.activeSpace === 'RGB' ? 'RGB' : 'Height'}</span></div>
          {keyboardHelpOpen && <section className="keyboard-help-panel" id="engineering-keyboard-help" role="dialog" aria-modal="false" aria-labelledby="keyboard-help-title"><div className="subpanel-heading"><h3 id="keyboard-help-title">Keyboard shortcuts</h3><button type="button" onClick={() => setKeyboardHelpOpen(false)}>Close</button></div><dl><div><dt>Tools</dt><dd>V Pointer, H Pan, S Sample, C Correspondence, R Rectangle, L Line</dd></div><div><dt>View</dt><dd>+/- Zoom, F Fit, 0 Actual pixels</dd></div><div><dt>Selection</dt><dd>Arrow keys move 1 pixel; Shift+Arrow moves 10 pixels</dd></div><div><dt>History</dt><dd>Ctrl/Cmd+Z Undo; Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y Redo</dd></div><div><dt>Cancel</dt><dd>Escape cancels an incomplete action</dd></div></dl></section>}
          <div className="coordinate-space-controls" role="group" aria-label="Active coordinate space"><span>Coordinate space</span><button type="button" aria-pressed={session.coordinates.activeSpace === 'RGB'} onClick={() => commit((current) => ({ ...current, coordinates: { ...current.coordinates, activeSpace: 'RGB' } }))}>RGB</button><button type="button" aria-pressed={session.coordinates.activeSpace === 'HEIGHT'} onClick={() => commit((current) => ({ ...current, coordinates: { ...current.coordinates, activeSpace: 'HEIGHT' } }))}>Height</button><button type="button" disabled={!session.coordinates.rgbSelected && !session.coordinates.heightSelected} onClick={clearSelections}>Clear selected coordinates</button></div>
          {(session.mode === 'Alpha overlay' || session.mode === 'Split comparison') && <div className="comparison-controls"><strong className="overlay-space-label">Interaction targets {session.coordinates.activeSpace === 'RGB' ? 'RGB' : 'Height'} coordinates; no registration is implied.</strong>{session.mode === 'Alpha overlay' && <label>Overlay opacity <input aria-label="Overlay opacity" type="range" min="0" max="100" value={session.overlayOpacity} onChange={(event) => commit((current) => ({ ...current, overlayOpacity: Number(event.target.value) }))} /><output>{session.overlayOpacity}%</output></label>}{session.mode === 'Split comparison' && <label>Split position <input aria-label="Split position" type="range" min="0" max="100" value={session.splitPosition} onChange={(event) => commit((current) => ({ ...current, splitPosition: Number(event.target.value) }))} /><output>{session.splitPosition}%</output></label>}</div>}
          <div className={`vision-canvas tool-${tool} scale-${session.scaleMode}`} data-testid="vision-canvas" data-view-mode={session.mode} data-alignment-view={renderedAlignmentView} data-flicker-running={flickerRunning ? 'true' : 'false'} data-pan={`${session.pan.x.toFixed(2)},${session.pan.y.toFixed(2)}`} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={(event) => void pointerUp(event)} onPointerCancel={() => { drag.current = null; roiStart.current = null }}>
            <div className="vision-canvas-grid" aria-hidden="true" /><div className="vision-transform-surface" style={{ transform }}>
              {session.mode === 'RGB' && <div id="rgb-evidence" className="single-evidence">{rgbImage}</div>}{session.mode === 'Height' && <div id="height-evidence" className="single-evidence">{heightImage}</div>}{session.mode === 'Side-by-side' && <div className="side-by-side-evidence"><div id="rgb-evidence">{rgbImage}</div><div id="height-evidence">{heightImage}</div></div>}{session.mode === 'Alpha overlay' && <div className="stacked-evidence"><div>{rgbImage}</div><div className="overlay-layer" style={{ opacity: session.overlayOpacity / 100 }}>{heightImage}</div></div>}{session.mode === 'Split comparison' && <div className="stacked-evidence"><div>{rgbImage}</div><div className="split-layer" style={{ clipPath: `inset(0 0 0 ${session.splitPosition}%)` }}>{heightImage}</div><span className="split-divider" style={{ left: `${session.splitPosition}%` }} aria-hidden="true" /></div>}
            </div>
          </div>
          {session.mode !== 'RGB' && <section className="height-display-controls" aria-labelledby="height-display-controls-title">
            <div className="subpanel-heading"><h3 id="height-display-controls-title">Derived height display</h3><span>Preview only</span></div>
            <div className="height-display-control-grid">
              <label>Palette<select aria-label="Height palette" value={session.heightPalette} onChange={(event) => commit((current) => ({ ...current, heightPalette: event.target.value as HeightPreviewPalette }))}><option value="grayscale">Grayscale</option><option value="blue-yellow">Blue-yellow</option><option value="viridis-like">Viridis-like</option><option value="high-contrast">High-contrast</option></select></label>
              <label>Display minimum<input aria-label="Height display minimum" type="number" min={nativeHeightMin} max={nativeHeightMax} value={displayMinDraft} onChange={(event) => setDisplayMinDraft(event.target.value)} /></label>
              <label>Display maximum<input aria-label="Height display maximum" type="number" min={nativeHeightMin} max={nativeHeightMax} value={displayMaxDraft} onChange={(event) => setDisplayMaxDraft(event.target.value)} /></label>
              <button type="button" onClick={() => applyDisplayRange(Number(displayMinDraft), Number(displayMaxDraft))}>Apply display range</button>
              <button type="button" onClick={resetDisplayRange}>Reset to native range</button>
              <label className="invalid-height-toggle"><input type="checkbox" checked={session.showInvalidHeight} onChange={(event) => commit((current) => ({ ...current, showInvalidHeight: event.target.checked }))} />Show invalid pixels <span className="invalid-height-swatch" aria-hidden="true" /></label>
            </div>
            {displayRangeError && <p className="field-error" role="alert">{displayRangeError}</p>}
            <p className="height-display-status">Palette {session.heightPalette}; native {nativeHeightMin} to {nativeHeightMax}; display {activeDisplayMin} to {activeDisplayMax}; invalid pixels {session.showInvalidHeight ? 'shown in magenta' : 'hidden in the low-end display colour'}.</p>
            <p className="height-display-warning">Display range changes only the derived colour view. Native height values remain unchanged.</p>
            <p className="engineering-panel-note">Invalid according to the current synthetic decoder. Count: {view.height_statistics.invalid_count.toLocaleString()}.</p>
            <HeightLegend palette={session.heightPalette} minimum={activeDisplayMin} maximum={activeDisplayMax} showInvalid={session.showInvalidHeight} />
          </section>}
          <div className="pan-nudges" aria-label="Pan position controls"><button type="button" onClick={() => movePan(-0.05, 0)} aria-label="Pan left">Left</button><button type="button" onClick={() => movePan(0, -0.05)} aria-label="Pan up">Up</button><button type="button" onClick={() => movePan(0, 0.05)} aria-label="Pan down">Down</button><button type="button" onClick={() => movePan(0.05, 0)} aria-label="Pan right">Right</button></div>
        </section>

        <aside className="engineering-inspector" aria-label="Metadata and pixel inspector">
          <p className="step-number">Pixel and evidence inspector</p>
          <form className="pixel-inspector" onSubmit={sampleCoordinates}>
            <section className="coordinate-inspector" aria-labelledby="rgb-inspector-title"><div className="subpanel-heading"><h4 id="rgb-inspector-title">RGB evidence</h4><span>RGB coordinate space</span></div><fieldset><legend>Selected coordinate</legend><label>X<input aria-label="RGB X coordinate" type="number" min="0" max={view.rgb.width - 1} value={session.coordinates.rgbX} onChange={(event) => setCoordinate('RGB', clampImagePoint({ x: Number(event.target.value), y: session.coordinates.rgbY }, view.rgb))} /></label><label>Y<input aria-label="RGB Y coordinate" type="number" min="0" max={view.rgb.height - 1} value={session.coordinates.rgbY} onChange={(event) => setCoordinate('RGB', clampImagePoint({ x: session.coordinates.rgbX, y: Number(event.target.value) }, view.rgb))} /></label></fieldset><dl className="engineering-definition-list"><div><dt>Selection</dt><dd>{session.coordinates.rgbSelected ? `X ${session.coordinates.rgbX}, Y ${session.coordinates.rgbY}` : 'No RGB pixel selected'}</dd></div><div><dt>Dimensions</dt><dd>{view.rgb.width} x {view.rgb.height}</dd></div><div><dt>R</dt><dd>{rgbValues[0]}</dd></div><div><dt>G</dt><dd>{rgbValues[1]}</dd></div><div><dt>B</dt><dd>{rgbValues[2]}</dd></div><div><dt>Integrity</dt><dd>Verified before decode</dd></div></dl><details><summary>RGB integrity metadata</summary><p>{view.rgb.detected_format} - {view.rgb.storage_data_type ?? 'Type unavailable'} - {formatBytes(view.rgb.byte_size)}</p><p className="mono hash-value">{view.rgb.sha256}</p></details></section>
            <section className="coordinate-inspector" aria-labelledby="height-inspector-title"><div className="subpanel-heading"><h4 id="height-inspector-title">Height evidence</h4><span>Height coordinate space</span></div><fieldset><legend>Selected coordinate</legend><label>X<input aria-label="Height X coordinate" type="number" min="0" max={view.height.width - 1} value={session.coordinates.heightX} onChange={(event) => setCoordinate('HEIGHT', clampImagePoint({ x: Number(event.target.value), y: session.coordinates.heightY }, view.height))} /></label><label>Y<input aria-label="Height Y coordinate" type="number" min="0" max={view.height.height - 1} value={session.coordinates.heightY} onChange={(event) => setCoordinate('HEIGHT', clampImagePoint({ x: session.coordinates.heightX, y: Number(event.target.value) }, view.height))} /></label></fieldset><dl className="engineering-definition-list"><div><dt>Selection</dt><dd>{session.coordinates.heightSelected ? `X ${session.coordinates.heightX}, Y ${session.coordinates.heightY}` : 'No height pixel selected'}</dd></div><div><dt>Dimensions</dt><dd>{view.height.width} x {view.height.height}</dd></div><div><dt>Native height</dt><dd>{sample ? sample.height.value === null ? 'Invalid / NaN' : sample.height.value : 'Not sampled'}</dd></div><div><dt>Validity</dt><dd>{sample ? sample.height.valid ? 'Valid' : 'Invalid' : 'Not sampled'}</dd></div><div><dt>Physical value</dt><dd>Unavailable</dd></div><div><dt>Physical unit</dt><dd>Unavailable</dd></div><div><dt>Calibration</dt><dd>{view.calibration_status}</dd></div><div><dt>Registration</dt><dd>{view.registration_status}</dd></div></dl><details><summary>Height integrity metadata</summary><p>{view.height.detected_format} - {view.height.storage_data_type ?? 'Type unavailable'} - {formatBytes(view.height.byte_size)}</p><p className="mono hash-value">{view.height.sha256}</p></details></section>
            {!coordinateValid && <p className="field-error" role="alert">Coordinates must remain inside their respective raster bounds.</p>}<button className="button primary" type="submit" disabled={!coordinateValid || sampling}>{sampling ? 'Sampling...' : 'Sample values'}</button>
            {sampleError && <div className="sample-error" role="alert"><strong>{sampleError.code}</strong><span>{sampleError.message}</span><span>Request ID: {sampleError.requestId}</span></div>}{sample && <p className="sample-evidence-note" aria-live="polite">Last authoritative sample: RGB X {sample.rgb.x}, Y {sample.rgb.y}; height X {sample.height.x}, Y {sample.height.y}. Coordinate spaces remain independent.</p>}
          </form>
          <Histogram view={view} displayMin={activeDisplayMin} displayMax={activeDisplayMax} sampledHeight={sample?.height.valid && sample.height.value !== null ? sample.height.value : null} selectedRoi={selectedHeightRoi} onUseRange={applyDisplayRange} onResetRange={resetDisplayRange} />

          <section className="engineering-session-panel" aria-labelledby="session-alignment-title">
            <div className="subpanel-heading"><h4 id="session-alignment-title">Session alignment</h4><span className="development-badge">Development · not persisted</span></div>
            <p className="engineering-panel-note">Optional visual development workflow only. Reload clears every point and transform; no automatic or production registration is performed.</p>

            <section className="alignment-group" aria-labelledby="alignment-view-group-title">
              <div className="subpanel-heading"><h5 id="alignment-view-group-title">View</h5><strong className={`alignment-view-state ${renderedAlignmentView.toLowerCase()}`}>{renderedAlignmentView === 'ORIGINAL' ? 'Original' : 'Development-aligned'} view</strong></div>
              <p>Choose what the browser renders. The original artifact is never modified.</p>
              <div className="segmented-controls" role="group" aria-label="Original and development alignment view">
                <button type="button" aria-pressed={session.alignmentView === 'ORIGINAL' && !flickerRunning} onClick={() => setAlignmentView('ORIGINAL')}>Return to original</button>
                <button type="button" aria-pressed={session.alignmentView === 'DEVELOPMENT' && !flickerRunning} onClick={() => setAlignmentView('DEVELOPMENT')}>Apply transform to view only</button>
              </div>
              <label>Overlay opacity<input type="range" min="0" max="100" value={session.overlayOpacity} onChange={(event) => commit((current) => ({ ...current, overlayOpacity: Number(event.target.value) }))} /><output>{session.overlayOpacity}%</output></label>
              <div className="flicker-controls"><button type="button" onClick={startFlicker} disabled={flickerRunning || reducedMotion}>Start flicker comparison</button><button type="button" onClick={stopFlicker} disabled={!flickerRunning}>Stop flicker</button></div>
              <p className="engineering-panel-note" role="status" aria-live="polite">{flickerNotice ?? 'Manual flicker alternates original and development-aligned rendering at 2.5 changes per second; it is not proof of registration.'}</p>
            </section>

            <section className="alignment-group" aria-labelledby="alignment-transform-group-title">
              <div className="subpanel-heading"><h5 id="alignment-transform-group-title">Transform</h5><span>px / degrees / unitless scale</span></div>
              <div className="alignment-control-grid">
                <label>Translation X (pixels)<input type="number" min={-TRANSLATION_LIMIT} max={TRANSLATION_LIMIT} value={session.alignment.translationX} onChange={(event) => commit((current) => ({ ...current, alignment: { ...current.alignment, translationX: bounded(Number(event.target.value), -TRANSLATION_LIMIT, TRANSLATION_LIMIT) } }))} /></label>
                <label>Translation Y (pixels)<input type="number" min={-TRANSLATION_LIMIT} max={TRANSLATION_LIMIT} value={session.alignment.translationY} onChange={(event) => commit((current) => ({ ...current, alignment: { ...current.alignment, translationY: bounded(Number(event.target.value), -TRANSLATION_LIMIT, TRANSLATION_LIMIT) } }))} /></label>
                <label>Rotation (degrees)<input type="number" min={-ROTATION_LIMIT} max={ROTATION_LIMIT} step="0.1" value={session.alignment.rotationDegrees} onChange={(event) => commit((current) => ({ ...current, alignment: { ...current.alignment, rotationDegrees: bounded(Number(event.target.value), -ROTATION_LIMIT, ROTATION_LIMIT) } }))} /></label>
                <label>Scale X (unitless)<input type="number" min={SCALE_MINIMUM} max={SCALE_MAXIMUM} step="0.01" value={session.alignment.scaleX} onChange={(event) => commit((current) => ({ ...current, alignment: { ...current.alignment, scaleX: bounded(Number(event.target.value), SCALE_MINIMUM, SCALE_MAXIMUM) } }))} /></label>
                <label>Scale Y (unitless)<input type="number" min={SCALE_MINIMUM} max={SCALE_MAXIMUM} step="0.01" value={session.alignment.scaleY} onChange={(event) => commit((current) => ({ ...current, alignment: { ...current.alignment, scaleY: bounded(Number(event.target.value), SCALE_MINIMUM, SCALE_MAXIMUM) } }))} /></label>
              </div>
              <button type="button" className="button secondary" onClick={() => commit((current) => ({ ...current, alignment: DEFAULT_ALIGNMENT, alignmentView: 'DEVELOPMENT' }))}>Reset transform to identity</button>
            </section>

            <section className="alignment-group correspondence-workflow" aria-labelledby="correspondence-points-title">
              <div className="subpanel-heading"><h5 id="correspondence-points-title">Correspondence</h5><span>Optional · browser session only</span></div>
              <ol className="correspondence-steps" aria-label="Guided correspondence steps">
                <li className={correspondenceStep === 1 ? 'current' : session.pendingRgbPoint ? 'complete' : ''}><strong>1</strong><span>Select RGB point</span></li>
                <li className={correspondenceStep === 2 ? 'current' : session.pendingHeightPoint ? 'complete' : ''}><strong>2</strong><span>Select height point</span></li>
                <li className={correspondenceStep === 3 ? 'current' : ''}><strong>3</strong><span>Add Pair</span></li>
                <li className={session.correspondences.length ? 'available' : ''}><strong>4</strong><span>Repeat, review development residuals, then adjust or apply the display transform</span></li>
              </ol>
              <p className="correspondence-current-step" role="status" aria-live="polite"><strong>Current step {correspondenceStep}:</strong> {correspondenceStep === 1 ? `Select the RGB point for P${session.nextCorrespondenceNumber}.` : correspondenceStep === 2 ? `Select the matching height point for P${session.nextCorrespondenceNumber}.` : `Review both coordinates and add P${session.nextCorrespondenceNumber}.`}</p>
              <div className="correspondence-actions">
                <button type="button" onClick={() => addManualCorrespondence('RGB')}>Use selected RGB point</button>
                <button type="button" disabled={!session.pendingRgbPoint} onClick={() => addManualCorrespondence('HEIGHT')}>Use selected height point</button>
                <button type="button" className="add-pair" disabled={!session.pendingRgbPoint || !session.pendingHeightPoint} onClick={addPendingCorrespondence}>Add Pair</button>
                <button type="button" disabled={!session.pendingRgbPoint && !session.pendingHeightPoint} onClick={cancelCurrentAction}>Cancel current pair</button>
                <button type="button" disabled={!session.correspondences.length && !session.pendingRgbPoint && !session.pendingHeightPoint} onClick={() => {
                  if (window.confirm('Clear all correspondence pairs from this browser session?')) commit((current) => ({ ...current, correspondences: [], selectedCorrespondenceId: null, pendingRgbPoint: null, pendingHeightPoint: null }))
                }}>Clear all correspondence points</button>
              </div>
              {(session.pendingRgbPoint || session.pendingHeightPoint) && <p className="engineering-panel-note pending-pair" aria-live="polite">Pending P{session.nextCorrespondenceNumber}: RGB {session.pendingRgbPoint ? `(${session.pendingRgbPoint.x}, ${session.pendingRgbPoint.y})` : 'not selected'}; height {session.pendingHeightPoint ? `(${session.pendingHeightPoint.x}, ${session.pendingHeightPoint.y})` : 'not selected'}.</p>}
              {session.correspondences.length ? <ul className="correspondence-list" aria-label="Correspondence pair list">{session.correspondences.map((point) => <li key={point.id} className={`${session.selectedCorrespondenceId === point.id ? 'selected' : ''} ${residuals.highestPairId === point.id ? 'highest-residual' : ''}`}><button type="button" className="correspondence-select" aria-pressed={session.selectedCorrespondenceId === point.id} onClick={() => selectCorrespondence(point.id)}><strong>{point.id}</strong> RGB ({point.rgb.x}, {point.rgb.y}) / height ({point.height.x}, {point.height.y})</button><span>Development residual: <output>{correspondenceResidual(point, session.alignment, comparisonDimensions)}</output> px {residuals.highestPairId === point.id ? '· highest' : ''}</span><button type="button" aria-label={`Remove ${point.id}`} onClick={() => commit((current) => ({ ...current, correspondences: current.correspondences.filter((item) => item.id !== point.id), selectedCorrespondenceId: current.selectedCorrespondenceId === point.id ? null : current.selectedCorrespondenceId }))}>Remove</button></li>)}</ul> : <p className="engineering-empty-inline">No correspondence pairs in this browser session. Development residuals are unavailable.</p>}
            </section>

            <section className="alignment-group" aria-labelledby="alignment-results-group-title">
              <div className="subpanel-heading"><h5 id="alignment-results-group-title">Results</h5><span className="development-badge">Development view only</span></div>
              <label className="residual-toggle"><input type="checkbox" checked={session.showResiduals} onChange={(event) => commit((current) => ({ ...current, showResiduals: event.target.checked }))} />Show development residual lines on RGB view</label>
              <p className="engineering-panel-note">Residual coordinate space: transformed height display coordinates normalized into RGB display pixels. These values are visual development residuals, not a quality or registration claim.</p>
              <dl className="engineering-definition-list compact residual-summary">
                <div><dt>Pair count</dt><dd>{session.correspondences.length}</dd></div><div><dt>Mean development residual</dt><dd>{residuals.meanPixels === null ? 'Unavailable' : `${residuals.meanPixels} px`}</dd></div>
                <div><dt>Maximum development residual</dt><dd>{residuals.maximumPixels === null ? 'Unavailable' : `${residuals.maximumPixels} px`}</dd></div><div><dt>Minimum development residual</dt><dd>{residuals.minimumPixels === null ? 'Unavailable' : `${residuals.minimumPixels} px`}</dd></div>
                <div><dt>Median development residual</dt><dd>{residuals.medianPixels === null ? 'Unavailable' : `${residuals.medianPixels} px`}</dd></div><div><dt>Highest residual pair</dt><dd>{residuals.highestPairId ?? 'Unavailable'}</dd></div>
              </dl>
              {translationSuggestion && <div className="translation-suggestion"><span>Optional translation suggestion: X {translationSuggestion.x}px, Y {translationSuggestion.y}px</span><button type="button" onClick={() => commit((current) => ({ ...current, alignment: { ...current.alignment, translationX: translationSuggestion.x, translationY: translationSuggestion.y } }))}>Apply translation suggestion</button></div>}
              <div className="matrix-context"><strong>3 × 3 affine display matrix</strong><span>Source: height display coordinates</span><span>Target: RGB display coordinates</span><span>Origin: RGB display centre</span><span>Translation: pixels · rotation: degrees · scale: unitless</span><span>Application: browser view only</span></div>
              <div className="affine-matrix" role="table" aria-label="3 by 3 development affine display matrix">{matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => <output key={`${rowIndex}-${columnIndex}`} role="cell" data-testid={`matrix-${rowIndex}-${columnIndex}`}>{value}</output>))}</div>
              <button type="button" className="button secondary export-alignment" onClick={exportAlignment}>Export alignment JSON</button>
            </section>
          </section>

          <section className="engineering-session-panel" aria-labelledby="measurement-tools-title">
            <div className="subpanel-heading"><h4 id="measurement-tools-title">Pixel measurements</h4><span>{session.coordinates.activeSpace === 'RGB' ? 'RGB' : 'Height'} space</span></div>
            <p className="engineering-panel-note">Use Rectangle or Line on the canvas. Height rectangles request bounded native statistics. Native values; physical units unavailable.</p>
            {roiLoading && <p role="status">Calculating native height statistics...</p>}
            {roiError && <div className="sample-error" role="alert"><strong>{roiError.code}</strong><span>{roiError.message}</span><span>Request ID: {roiError.requestId}</span></div>}
            <div className="measurement-heading">
              <strong>Measurements</strong>
              <button type="button" disabled={!selectedHeightRoi} onClick={() => {
                if (!selectedHeightRoi) return
                commit((current) => ({ ...current, selectedRoiId: null, rois: current.rois.filter((item) => item.id !== selectedHeightRoi.id) }))
              }}>Clear ROI</button>
              <button type="button" disabled={!session.rois.length} onClick={() => commit((current) => ({ ...current, selectedRoiId: null, rois: [] }))}>Clear measurements</button>
            </div>
            {session.rois.length ? <ul className="measurement-list">{session.rois.map((roi) => (
              <li key={roi.id} className={session.selectedRoiId === roi.id ? 'selected' : ''} data-measurement-kind={roi.kind}>
                <button type="button" className="measurement-select" aria-pressed={session.selectedRoiId === roi.id} onClick={() => commit((current) => ({ ...current, selectedRoiId: roi.id }))}>
                  <strong>{roi.id} - {roi.coordinateSpace} {roi.kind.toLowerCase()}</strong>
                  {roi.kind === 'POINT' && <span>X {roi.x}, Y {roi.y} pixels</span>}
                  {roi.kind === 'RECTANGLE' && <span>X {roi.x}, Y {roi.y}; W {roi.width}, H {roi.height} pixels; area {roi.width * roi.height} pixels squared</span>}
                  {roi.kind === 'LINE' && <span>Start X {roi.x1}, Y {roi.y1}; end X {roi.x2}, Y {roi.y2}; dx {roi.deltaXPixels}, dy {roi.deltaYPixels} pixels; distance {roi.distancePixels} pixels; direction {roi.directionDegrees} degrees</span>}
                  {roi.kind === 'RECTANGLE' && roi.nativeHeightStatistics && <span>Native min {roi.nativeHeightStatistics.nativeMin}, max {roi.nativeHeightStatistics.nativeMax}, mean {roi.nativeHeightStatistics.nativeMean}, range {Number((roi.nativeHeightStatistics.nativeMax - roi.nativeHeightStatistics.nativeMin).toFixed(6))}; {roi.nativeHeightStatistics.validCount} valid / {roi.nativeHeightStatistics.invalidCount} invalid</span>}
                </button>
                <button type="button" aria-label={`Remove ${roi.id}`} onClick={() => commit((current) => ({ ...current, selectedRoiId: current.selectedRoiId === roi.id ? null : current.selectedRoiId, rois: current.rois.filter((item) => item.id !== roi.id) }))}>Remove</button>
              </li>
            ))}</ul> : <p className="engineering-empty-inline">No pixel measurements in this browser session.</p>}
          </section>
        </aside>
      </div>

      <div className="engineering-status-bar" role="status" aria-live="polite" aria-label="Engineering workspace status"><span>Inspection <strong className="mono">{inspectionId}</strong></span><span>RGB {view.rgb.width}x{view.rgb.height}</span><span>Height {view.height.width}x{view.height.height}</span><span>Zoom {Math.round(session.zoom * 100)}%</span><span>Tool {toolDefinition.label}</span><span>RGB {session.coordinates.rgbSelected ? `${session.coordinates.rgbX},${session.coordinates.rgbY}` : 'none'}</span><span>Height {session.coordinates.heightSelected ? `${session.coordinates.heightX},${session.coordinates.heightY}` : 'none'}</span><span>Palette {session.heightPalette}</span><span>Display {activeDisplayMin}..{activeDisplayMax}</span><span>Invalid {session.showInvalidHeight ? 'shown' : 'hidden'}</span><span>Pairs {session.correspondences.length}</span><span>Registration {view.registration_status}</span><span>Units unavailable</span></div>

      <section className="pipeline-evidence-panel" aria-labelledby="pipeline-evidence-title"><div className="panel-heading"><div><p className="step-number">Pipeline evidence and limitations</p><h3 id="pipeline-evidence-title">Persisted workflow record</h3></div><p>No validation or processing is executed by this workspace.</p></div><div className="pipeline-evidence-grid"><PipelineCard id="technical-validation" title="Technical validation" available={view.validation.available}><dl><div><dt>Outcome</dt><dd>{view.validation.outcome ?? 'Not recorded'}</dd></div><div><dt>Policy</dt><dd>{view.validation.policy_id ? `${view.validation.policy_id} ${view.validation.policy_version}` : 'Not recorded'}</dd></div><div><dt>Findings</dt><dd>{view.validation.finding_codes.join(', ') || 'None recorded'}</dd></div></dl></PipelineCard><PipelineCard id="synthetic-preprocessing" title="Synthetic preprocessing" available={view.processing.available}><dl><div><dt>Outcome</dt><dd>{view.processing.preprocessing_outcome ?? 'Not recorded'}</dd></div><div><dt>Synthetic input verified</dt><dd>{view.processing.synthetic_input_verified === null ? 'Not recorded' : view.processing.synthetic_input_verified ? 'Yes' : 'No'}</dd></div></dl></PipelineCard><PipelineCard id="mock-inference" title="Deterministic mock inference" available={view.processing.available}><dl><div><dt>Decision</dt><dd>{view.processing.mock_decision ?? 'Not recorded'}</dd></div><div><dt>Processing status</dt><dd>{view.processing.processing_status ?? 'Not recorded'}</dd></div></dl></PipelineCard><PipelineCard id="persisted-result" title="Persisted result" available={true}><dl><div><dt>Inspection status</dt><dd>{view.inspection_status}</dd></div><div><dt>Calibration</dt><dd>{view.calibration_status}</dd></div><div><dt>Registration</dt><dd>{view.registration_status}</dd></div></dl></PipelineCard></div><div className="engineering-limitations"><h4>Engineering limitations</h4><ul><li>Synthetic engineering data only.</li><li>No production measurement or fabricated physical units.</li><li>No physical calibration.</li><li>No real registration.</li><li>No production inspection decision.</li>{view.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div></section>
    </section>
  )
}
