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

export const DEFAULT_ALIGNMENT: SessionAlignment = {
  translationX: 0,
  translationY: 0,
  rotationDegrees: 0,
  scaleX: 1,
  scaleY: 1,
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

export function correspondenceResidual(point: CorrespondencePoint, alignment: SessionAlignment): number {
  const transformed = transformHeightPoint(point.height, alignment)
  return clean(Math.hypot(point.rgb.x - transformed.x, point.rgb.y - transformed.y))
}

export function residualSummary(points: CorrespondencePoint[], alignment: SessionAlignment) {
  const residuals = points.map((point) => correspondenceResidual(point, alignment))
  return {
    residuals,
    meanPixels: residuals.length ? clean(residuals.reduce((sum, value) => sum + value, 0) / residuals.length) : null,
    maximumPixels: residuals.length ? clean(Math.max(...residuals)) : null,
  }
}

export function suggestedTranslation(points: CorrespondencePoint[], alignment: SessionAlignment): PixelPoint | null {
  if (!points.length) return null
  const withoutTranslation = { ...alignment, translationX: 0, translationY: 0 }
  const deltas = points.map((point) => {
    const transformed = transformHeightPoint(point.height, withoutTranslation)
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
) {
  const residuals = residualSummary(points, alignment)
  return {
    contract_version: 'pcb-aoi-development-alignment/1.0',
    inspection_id: inspectionId,
    development_only: true,
    production_approved: false,
    units: 'pixels',
    alignment: {
      translation: { x: alignment.translationX, y: alignment.translationY },
      rotation_degrees: alignment.rotationDegrees,
      scale: { x: alignment.scaleX, y: alignment.scaleY },
      overlay_opacity_percent: overlayOpacityPercent,
      affine_matrix_3x3: affineMatrix(alignment),
    },
    correspondences: points.map((point, index) => ({
      id: point.id,
      rgb: point.rgb,
      height: point.height,
      residual_pixels: residuals.residuals[index],
    })),
    residual_summary: {
      mean_pixels: residuals.meanPixels,
      maximum_pixels: residuals.maximumPixels,
    },
    measurements: rois,
    limitations: [
      'SESSION_ONLY_ALIGNMENT',
      'NO_AUTOMATIC_PRODUCTION_REGISTRATION',
      'NO_PHYSICAL_UNIT_CONVERSION',
      'NO_PRODUCTION_INSPECTION_DECISION',
    ],
  }
}

export function alignmentExportFilename(inspectionId: string): string {
  return `inspection-${inspectionId}-development-alignment.json`
}
