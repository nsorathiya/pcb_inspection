import { describe, expect, it } from 'vitest'
import {
  DEFAULT_ALIGNMENT,
  affineMatrix,
  alignmentExportFilename,
  buildAlignmentExport,
  correspondenceResidual,
  residualSummary,
  suggestedTranslation,
  transformHeightPoint,
  type CorrespondencePoint,
  type EngineeringRoi,
} from './engineeringSession'

const points: CorrespondencePoint[] = [
  { id: 'P1', rgb: { x: 12, y: 8 }, height: { x: 10, y: 5 } },
  { id: 'P2', rgb: { x: 22, y: 18 }, height: { x: 20, y: 15 } },
]

describe('session-only engineering calculations', () => {
  it('builds a deterministic identity matrix and transforms native pixel coordinates', () => {
    expect(affineMatrix(DEFAULT_ALIGNMENT)).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expect(transformHeightPoint({ x: 4, y: 6 }, { translationX: 3, translationY: -2, rotationDegrees: 0, scaleX: 2, scaleY: 0.5 })).toEqual({ x: 11, y: 1 })
  })

  it('calculates per-point, mean, maximum, and suggested translation residuals in pixels', () => {
    expect(correspondenceResidual(points[0]!, DEFAULT_ALIGNMENT)).toBeCloseTo(Math.sqrt(13))
    const summary = residualSummary(points, DEFAULT_ALIGNMENT)
    expect(summary.meanPixels).toBeCloseTo(Math.sqrt(13))
    expect(summary.maximumPixels).toBeCloseTo(Math.sqrt(13))
    expect(suggestedTranslation(points, DEFAULT_ALIGNMENT)).toEqual({ x: 2, y: 3 })
    expect(residualSummary(points, { ...DEFAULT_ALIGNMENT, translationX: 2, translationY: 3 }).maximumPixels).toBe(0)
  })

  it('exports a versioned, deterministic development-only contract without operational metadata', () => {
    const rois: EngineeringRoi[] = [{ id: 'M1', kind: 'LINE', coordinateSpace: 'RGB', x1: 0, y1: 0, x2: 3, y2: 4, distancePixels: 5 }]
    const first = buildAlignmentExport('inspection-id', DEFAULT_ALIGNMENT, points, rois, 45)
    const second = buildAlignmentExport('inspection-id', DEFAULT_ALIGNMENT, points, rois, 45)
    expect(first).toEqual(second)
    expect(first).toMatchObject({
      contract_version: 'pcb-aoi-development-alignment/1.0',
      development_only: true,
      production_approved: false,
      units: 'pixels',
      alignment: { overlay_opacity_percent: 45 },
    })
    expect(alignmentExportFilename('inspection-id')).toBe('inspection-inspection-id-development-alignment.json')
    const serialized = JSON.stringify(first).toLowerCase()
    expect(serialized).not.toContain('path')
    expect(serialized).not.toContain('request_id')
    expect(serialized).not.toContain('confidence')
    expect(serialized).not.toContain('millimet')
    expect(serialized).not.toContain('microm')
  })
})
