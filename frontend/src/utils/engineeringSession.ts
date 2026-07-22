export interface SessionAlignment {
  translationX: number
  translationY: number
  rotationDegrees: number
  scaleX: number
  scaleY: number
}

export interface PixelPoint {
  x: number
  y: number
}

export type EngineeringCoordinateSpace = 'RGB' | 'HEIGHT'
export type EngineeringTool = 'pointer' | 'pan' | 'sample' | 'correspondence' | 'rectangle' | 'line'
export type EngineeringViewMode = 'RGB' | 'Height' | 'Side-by-side' | 'Alpha overlay' | 'Split comparison'
export type AlignmentViewState = 'ORIGINAL' | 'DEVELOPMENT'

export interface RasterComparisonDimensions {
  rgb: { width: number; height: number }
  height: { width: number; height: number }
}

export interface EngineeringCoordinates {
  rgbX: number
  rgbY: number
  heightX: number
  heightY: number
  rgbSelected: boolean
  heightSelected: boolean
  activeSpace: EngineeringCoordinateSpace
}

export interface CorrespondencePoint {
  id: string
  rgb: PixelPoint
  height: PixelPoint
}

export interface NativeHeightRoiStatistics {
  nativeMin: number
  nativeMax: number
  nativeMean: number
  validCount: number
  invalidCount: number
  storageDataType: string | null
}

export type EngineeringRoi =
  | { id: string; kind: 'POINT'; coordinateSpace: 'RGB' | 'HEIGHT'; x: number; y: number }
  | { id: string; kind: 'RECTANGLE'; coordinateSpace: 'RGB' | 'HEIGHT'; x: number; y: number; width: number; height: number; nativeHeightStatistics?: NativeHeightRoiStatistics }
  | { id: string; kind: 'LINE'; coordinateSpace: 'RGB' | 'HEIGHT'; x1: number; y1: number; x2: number; y2: number; distancePixels: number }

export interface EngineeringSessionSnapshot {
  mode: EngineeringViewMode
  zoom: number
  scaleMode: 'fit' | 'actual'
  pan: PixelPoint
  overlayOpacity: number
  splitPosition: number
  coordinates: EngineeringCoordinates
  alignment: SessionAlignment
  alignmentView: AlignmentViewState
  showResiduals: boolean
  selectedCorrespondenceId: string | null
  nextCorrespondenceNumber: number
  correspondences: CorrespondencePoint[]
  pendingRgbPoint: PixelPoint | null
  pendingHeightPoint: PixelPoint | null
  rois: EngineeringRoi[]
}

export interface SessionHistory<T> {
  past: T[]
  present: T
  future: T[]
}

export const SESSION_HISTORY_LIMIT = 50

export const DEFAULT_ALIGNMENT: SessionAlignment = {
  translationX: 0,
  translationY: 0,
  rotationDegrees: 0,
  scaleX: 1,
  scaleY: 1,
}

export const DEFAULT_ENGINEERING_SESSION: EngineeringSessionSnapshot = {
  mode: 'Side-by-side',
  zoom: 1,
  scaleMode: 'fit',
  pan: { x: 0, y: 0 },
  overlayOpacity: 50,
  splitPosition: 50,
  coordinates: {
    rgbX: 0,
    rgbY: 0,
    heightX: 0,
    heightY: 0,
    rgbSelected: false,
    heightSelected: false,
    activeSpace: 'RGB',
  },
  alignment: DEFAULT_ALIGNMENT,
  alignmentView: 'ORIGINAL',
  showResiduals: true,
  selectedCorrespondenceId: null,
  nextCorrespondenceNumber: 1,
  correspondences: [],
  pendingRgbPoint: null,
  pendingHeightPoint: null,
  rois: [],
}

export function createSessionHistory<T>(initial: T): SessionHistory<T> {
  return { past: [], present: initial, future: [] }
}

export function commitSessionHistory<T>(
  history: SessionHistory<T>,
  next: T,
  limit = SESSION_HISTORY_LIMIT,
): SessionHistory<T> {
  if (Object.is(history.present, next)) return history
  return {
    past: [...history.past, history.present].slice(-limit),
    present: next,
    future: [],
  }
}

export function undoSessionHistory<T>(history: SessionHistory<T>): SessionHistory<T> {
  const previous = history.past.at(-1)
  if (previous === undefined) return history
  return {
    past: history.past.slice(0, -1),
    present: previous,
    future: [history.present, ...history.future],
  }
}

export function redoSessionHistory<T>(history: SessionHistory<T>): SessionHistory<T> {
  const next = history.future[0]
  if (next === undefined) return history
  return {
    past: [...history.past, history.present].slice(-SESSION_HISTORY_LIMIT),
    present: next,
    future: history.future.slice(1),
  }
}

export interface DisplayRectangle {
  left: number
  top: number
  width: number
  height: number
}

export function displayPointToImagePoint(
  clientPoint: PixelPoint,
  display: DisplayRectangle,
  image: { width: number; height: number },
  alignment: SessionAlignment = DEFAULT_ALIGNMENT,
  outerScale: PixelPoint = { x: 1, y: 1 },
): PixelPoint | null {
  if (
    display.width <= 0
    || display.height <= 0
    || image.width <= 0
    || image.height <= 0
    || alignment.scaleX <= 0
    || alignment.scaleY <= 0
    || outerScale.x <= 0
    || outerScale.y <= 0
  ) return null

  const centreX = display.left + display.width / 2
  const centreY = display.top + display.height / 2
  const translatedX = clientPoint.x - centreX - alignment.translationX * outerScale.x
  const translatedY = clientPoint.y - centreY - alignment.translationY * outerScale.y
  const radians = alignment.rotationDegrees * Math.PI / 180
  const cosine = Math.cos(radians)
  const sine = Math.sin(radians)
  const unrotatedX = cosine * translatedX + sine * translatedY
  const unrotatedY = -sine * translatedX + cosine * translatedY
  const localX = unrotatedX / alignment.scaleX + display.width / 2
  const localY = unrotatedY / alignment.scaleY + display.height / 2
  if (localX < 0 || localY < 0 || localX >= display.width || localY >= display.height) return null
  return {
    x: Math.min(image.width - 1, Math.floor(localX / display.width * image.width)),
    y: Math.min(image.height - 1, Math.floor(localY / display.height * image.height)),
  }
}

export function clampImagePoint(point: PixelPoint, image: { width: number; height: number }): PixelPoint {
  return {
    x: Math.min(image.width - 1, Math.max(0, point.x)),
    y: Math.min(image.height - 1, Math.max(0, point.y)),
  }
}

function clean(value: number): number {
  const normalized = Math.abs(value) < 1e-12 ? 0 : value
  return Number(normalized.toFixed(8))
}

export type AffineMatrix3x3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
]

export function affineMatrix(alignment: SessionAlignment): AffineMatrix3x3 {
  const radians = alignment.rotationDegrees * Math.PI / 180
  const cosine = Math.cos(radians)
  const sine = Math.sin(radians)
  return [
    [clean(cosine * alignment.scaleX), clean(-sine * alignment.scaleY), clean(alignment.translationX)],
    [clean(sine * alignment.scaleX), clean(cosine * alignment.scaleY), clean(alignment.translationY)],
    [0, 0, 1],
  ]
}

export function cssAffineMatrix(alignment: SessionAlignment): string {
  const matrix = affineMatrix(alignment)
  return `matrix(${matrix[0][0]}, ${matrix[1][0]}, ${matrix[0][1]}, ${matrix[1][1]}, ${matrix[0][2]}, ${matrix[1][2]})`
}

export function transformHeightPoint(point: PixelPoint, alignment: SessionAlignment): PixelPoint {
  const matrix = affineMatrix(alignment)
  return {
    x: clean(matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2]),
    y: clean(matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2]),
  }
}

export function transformHeightPointToRgb(
  point: PixelPoint,
  alignment: SessionAlignment,
  dimensions?: RasterComparisonDimensions,
): PixelPoint {
  if (!dimensions) return transformHeightPoint(point, alignment)
  const mapped = {
    x: (point.x + 0.5) * dimensions.rgb.width / dimensions.height.width - 0.5,
    y: (point.y + 0.5) * dimensions.rgb.height / dimensions.height.height - 0.5,
  }
  const centre = {
    x: (dimensions.rgb.width - 1) / 2,
    y: (dimensions.rgb.height - 1) / 2,
  }
  const transformed = transformHeightPoint(
    { x: mapped.x - centre.x, y: mapped.y - centre.y },
    alignment,
  )
  return {
    x: clean(transformed.x + centre.x),
    y: clean(transformed.y + centre.y),
  }
}

export function correspondenceResidual(
  point: CorrespondencePoint,
  alignment: SessionAlignment,
  dimensions?: RasterComparisonDimensions,
): number {
  const transformed = transformHeightPointToRgb(point.height, alignment, dimensions)
  return clean(Math.hypot(point.rgb.x - transformed.x, point.rgb.y - transformed.y))
}

export function residualSummary(
  points: CorrespondencePoint[],
  alignment: SessionAlignment,
  dimensions?: RasterComparisonDimensions,
) {
  const residuals = points.map((point) => correspondenceResidual(point, alignment, dimensions))
  const sorted = [...residuals].sort((left, right) => left - right)
  const midpoint = Math.floor(sorted.length / 2)
  const median = sorted.length
    ? sorted.length % 2
      ? sorted[midpoint]!
      : clean((sorted[midpoint - 1]! + sorted[midpoint]!) / 2)
    : null
  const maximum = residuals.length ? clean(Math.max(...residuals)) : null
  const highestIndex = maximum === null ? -1 : residuals.indexOf(maximum)
  return {
    residuals,
    meanPixels: residuals.length ? clean(residuals.reduce((sum, value) => sum + value, 0) / residuals.length) : null,
    maximumPixels: maximum,
    minimumPixels: residuals.length ? clean(Math.min(...residuals)) : null,
    medianPixels: median,
    highestPairId: highestIndex >= 0 ? points[highestIndex]!.id : null,
  }
}

export function suggestedTranslation(
  points: CorrespondencePoint[],
  alignment: SessionAlignment,
  dimensions?: RasterComparisonDimensions,
): PixelPoint | null {
  if (!points.length) return null
  const withoutTranslation = { ...alignment, translationX: 0, translationY: 0 }
  const deltas = points.map((point) => {
    const transformed = transformHeightPointToRgb(point.height, withoutTranslation, dimensions)
    return { x: point.rgb.x - transformed.x, y: point.rgb.y - transformed.y }
  })
  return {
    x: clean(deltas.reduce((sum, point) => sum + point.x, 0) / deltas.length),
    y: clean(deltas.reduce((sum, point) => sum + point.y, 0) / deltas.length),
  }
}

export function buildAlignmentExport(
  inspectionId: string,
  alignment: SessionAlignment,
  points: CorrespondencePoint[],
  rois: EngineeringRoi[],
  overlayOpacityPercent: number,
  options: {
    activeView?: AlignmentViewState
    dimensions?: RasterComparisonDimensions
  } = {},
) {
  const residuals = residualSummary(points, alignment, options.dimensions)
  return {
    contract_version: 'pcb-aoi-development-alignment/1.0',
    inspection_id: inspectionId,
    development_only: true,
    production_approved: false,
    units: 'pixels',
    active_view: options.activeView ?? 'ORIGINAL',
    comparison_coordinate_space: 'RGB_DISPLAY_PIXELS',
    alignment: {
      translation: { x: alignment.translationX, y: alignment.translationY },
      rotation_degrees: alignment.rotationDegrees,
      scale: { x: alignment.scaleX, y: alignment.scaleY },
      overlay_opacity_percent: overlayOpacityPercent,
      affine_matrix_3x3: affineMatrix(alignment),
      source_coordinate_space: 'HEIGHT_DISPLAY_PIXELS',
      target_coordinate_space: 'RGB_DISPLAY_PIXELS',
      transform_origin: 'RGB_DISPLAY_CENTRE',
      scale_units: 'UNITLESS',
      application: 'BROWSER_VIEW_ONLY',
    },
    correspondences: points.map((point, index) => ({
      id: point.id,
      pair_number: Number(point.id.replace(/^P/, '')),
      rgb: point.rgb,
      height: point.height,
      residual_pixels: residuals.residuals[index],
    })),
    residual_summary: {
      mean_pixels: residuals.meanPixels,
      maximum_pixels: residuals.maximumPixels,
      minimum_pixels: residuals.minimumPixels,
      median_pixels: residuals.medianPixels,
      highest_pair_id: residuals.highestPairId,
    },
    measurements: rois,
    limitations: [
      'SESSION_ONLY_ALIGNMENT',
      'NO_AUTOMATIC_PRODUCTION_REGISTRATION',
      'NO_PHYSICAL_UNIT_CONVERSION',
      'NO_PRODUCTION_INSPECTION_DECISION',
      'RESIDUALS_ARE_DEVELOPMENT_VISUALIZATION_NOT_A_QUALITY_CLAIM',
      'ALIGNMENT_APPLIES_TO_BROWSER_RENDERING_ONLY',
    ],
  }
}

export function alignmentExportFilename(inspectionId: string): string {
  return `inspection-${inspectionId}-development-alignment.json`
}
