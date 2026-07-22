import { describe, expect, it } from 'vitest'
import {
  DEFAULT_ALIGNMENT,
  affineMatrix,
  alignmentExportFilename,
  buildAlignmentExport,
  clampImagePoint,
  commitSessionHistory,
  correspondenceResidual,
  createSessionHistory,
  displayPointToImagePoint,
  redoSessionHistory,
  residualSummary,
  suggestedTranslation,
  transformHeightPoint,
  transformHeightPointToRgb,
  undoSessionHistory,
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

  it('calculates complete development residual statistics and a translation suggestion in pixels', () => {
    expect(correspondenceResidual(points[0]!, DEFAULT_ALIGNMENT)).toBeCloseTo(Math.sqrt(13))
    const summary = residualSummary(points, DEFAULT_ALIGNMENT)
    expect(summary.meanPixels).toBeCloseTo(Math.sqrt(13))
    expect(summary.maximumPixels).toBeCloseTo(Math.sqrt(13))
    expect(summary.minimumPixels).toBeCloseTo(Math.sqrt(13))
    expect(summary.medianPixels).toBeCloseTo(Math.sqrt(13))
    expect(summary.highestPairId).toBe('P1')
    expect(suggestedTranslation(points, DEFAULT_ALIGNMENT)).toEqual({ x: 2, y: 3 })
    expect(residualSummary(points, { ...DEFAULT_ALIGNMENT, translationX: 2, translationY: 3 }).maximumPixels).toBe(0)
  })

  it('normalizes mismatched height dimensions into RGB display coordinates around the display centre', () => {
    const dimensions = { rgb: { width: 640, height: 480 }, height: { width: 320, height: 240 } }
    const mapped = transformHeightPointToRgb({ x: 159.5, y: 119.5 }, DEFAULT_ALIGNMENT, dimensions)
    expect(mapped).toEqual({ x: 319.5, y: 239.5 })
    const normalized: CorrespondencePoint = { id: 'P3', rgb: mapped, height: { x: 159.5, y: 119.5 } }
    expect(correspondenceResidual(normalized, DEFAULT_ALIGNMENT, dimensions)).toBe(0)
  })

  it('returns an explicit empty development residual state', () => {
    expect(residualSummary([], DEFAULT_ALIGNMENT)).toEqual({
      residuals: [], meanPixels: null, maximumPixels: null, minimumPixels: null,
      medianPixels: null, highestPairId: null,
    })
  })

  it('exports a versioned, deterministic development-only contract without operational metadata', () => {
    const rois: EngineeringRoi[] = [{ id: 'M1', kind: 'LINE', coordinateSpace: 'RGB', x1: 0, y1: 0, x2: 3, y2: 4, distancePixels: 5 }]
    const first = buildAlignmentExport('inspection-id', DEFAULT_ALIGNMENT, points, rois, 45, { activeView: 'DEVELOPMENT' })
    const second = buildAlignmentExport('inspection-id', DEFAULT_ALIGNMENT, points, rois, 45, { activeView: 'DEVELOPMENT' })
    expect(first).toEqual(second)
    expect(first).toMatchObject({
      contract_version: 'pcb-aoi-development-alignment/1.0',
      development_only: true,
      production_approved: false,
      units: 'pixels',
      active_view: 'DEVELOPMENT',
      comparison_coordinate_space: 'RGB_DISPLAY_PIXELS',
      alignment: {
        overlay_opacity_percent: 45,
        source_coordinate_space: 'HEIGHT_DISPLAY_PIXELS',
        target_coordinate_space: 'RGB_DISPLAY_PIXELS',
        transform_origin: 'RGB_DISPLAY_CENTRE',
        application: 'BROWSER_VIEW_ONLY',
      },
      residual_summary: {
        minimum_pixels: expect.any(Number),
        median_pixels: expect.any(Number),
        highest_pair_id: 'P1',
      },
    })
    expect(first.correspondences.map((point) => point.pair_number)).toEqual([1, 2])
    expect(alignmentExportFilename('inspection-id')).toBe('inspection-inspection-id-development-alignment.json')
    const serialized = JSON.stringify(first).toLowerCase()
    expect(serialized).not.toContain('path')
    expect(serialized).not.toContain('request_id')
    expect(serialized).not.toContain('confidence')
    expect(serialized).not.toContain('millimet')
    expect(serialized).not.toContain('microm')
  })

  it('converts post-zoom and pan display coordinates and safely rejects outside clicks', () => {
    const display = { left: 120, top: 80, width: 640, height: 480 }
    expect(displayPointToImagePoint({ x: 280, y: 200 }, display, { width: 640, height: 480 })).toEqual({ x: 160, y: 120 })
    expect(displayPointToImagePoint({ x: 119, y: 200 }, display, { width: 640, height: 480 })).toBeNull()
    expect(displayPointToImagePoint({ x: 760, y: 200 }, display, { width: 640, height: 480 })).toBeNull()
  })

  it('inverts session translation, rotation, and scaling during height coordinate conversion', () => {
    const display = { left: 100, top: 100, width: 200, height: 100 }
    const aligned = { translationX: 10, translationY: -5, rotationDegrees: 90, scaleX: 2, scaleY: 1 }
    const centre = displayPointToImagePoint(
      { x: 220, y: 140 },
      display,
      { width: 20, height: 10 },
      aligned,
      { x: 2, y: 2 },
    )
    expect(centre).toEqual({ x: 10, y: 5 })
    expect(clampImagePoint({ x: -3, y: 99 }, { width: 20, height: 10 })).toEqual({ x: 0, y: 9 })
  })

  it('keeps bounded immutable undo/redo history and clears redo after a new action', () => {
    let history = createSessionHistory(0)
    for (let value = 1; value <= 75; value += 1) history = commitSessionHistory(history, value)
    expect(history.past).toHaveLength(50)
    expect(history.present).toBe(75)
    history = undoSessionHistory(history)
    expect(history.present).toBe(74)
    expect(history.future).toEqual([75])
    history = redoSessionHistory(history)
    expect(history.present).toBe(75)
    history = undoSessionHistory(history)
    history = commitSessionHistory(history, 100)
    expect(history.present).toBe(100)
    expect(history.future).toEqual([])
  })
})
