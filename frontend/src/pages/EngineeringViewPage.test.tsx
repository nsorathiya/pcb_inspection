import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getEngineeringSample, getEngineeringView } from '../api/engineeringViewer'
import { ApiClientError } from '../api/errors'
import type { EngineeringSampleResponse, EngineeringViewResponse } from '../api/types'
import { INSPECTION_ID } from '../test/fixtures'
import { EngineeringViewPage } from './EngineeringViewPage'

vi.mock('../api/engineeringViewer', () => ({
  getEngineeringView: vi.fn(),
  getEngineeringSample: vi.fn(),
  getEngineeringHeightRoi: vi.fn(),
  engineeringPreviewUrl: vi.fn((_id: string, kind: string) => `/preview/${kind}.png`),
}))

const viewMock = vi.mocked(getEngineeringView)
const sampleMock = vi.mocked(getEngineeringSample)

function engineeringResponse(mismatched = false): EngineeringViewResponse {
  const counts = Array.from({ length: 64 }, (_, index) => index + 1)
  return {
    inspection_id: INSPECTION_ID,
    inspection_status: 'FAIL',
    rgb: {
      artifact_type: 'RGB_RAW', detected_format: 'PNG', width: 640, height: 480,
      channels: 3, bit_depth: 8, color_mode: 'RGB', storage_data_type: 'uint8',
      sha256: 'a'.repeat(64), byte_size: 12345,
    },
    height: {
      artifact_type: 'HEIGHT_RAW', detected_format: 'TIFF', width: mismatched ? 320 : 640, height: 480,
      channels: 1, bit_depth: 16, color_mode: 'HEIGHT', storage_data_type: 'uint16',
      sha256: 'b'.repeat(64), byte_size: 54321,
    },
    height_statistics: {
      native_min: 100, native_max: 4000, valid_count: 153600, invalid_count: 0,
      histogram: { bin_count: 64, native_min: 100, native_max: 4000, counts },
    },
    calibration_status: 'NOT_CALIBRATED',
    registration_status: 'NOT_REGISTERED',
    physical_height_unit: null,
    validation: {
      available: true, validation_id: 'validation-1', outcome: 'VALIDATION_PASSED',
      policy_id: 'development-native-rgb-height', policy_version: '1.0', technically_ready: true,
      finding_codes: ['SYNTHETIC_INPUT'],
    },
    processing: {
      available: true, processing_run_id: 'processing-1', processing_status: 'COMPLETED',
      preprocessing_outcome: 'PREPROCESSING_PASSED', mock_decision: 'FAIL',
      production_approved: false, synthetic_input_verified: true, finding_codes: [],
    },
    warnings: ['Height preview is derived for browser display.'],
    synthetic_input_verified: true,
    production_approved: false,
    request_id: 'engineering-request',
  }
}

const sampleResponse: EngineeringSampleResponse = {
  inspection_id: INSPECTION_ID,
  rgb: { x: 12, y: 34, storage_data_type: 'uint8', values: [10, 20, 30] },
  height: { x: 56, y: 78, storage_data_type: 'uint16', value: 2048, valid: true, physical_unit: null },
  warnings: [],
  request_id: 'sample-request',
}

function renderPage(id = INSPECTION_ID) {
  return render(
    <MemoryRouter initialEntries={[`/inspections/${id}/engineering-view`]}>
      <Routes><Route path="/inspections/:inspectionId/engineering-view" element={<EngineeringViewPage />} /></Routes>
    </MemoryRouter>,
  )
}

function setRasterGeometry(element: HTMLElement, rect: { left: number; top: number; width: number; height: number }) {
  Object.defineProperty(element, 'offsetWidth', { configurable: true, value: rect.width })
  Object.defineProperty(element, 'offsetHeight', { configurable: true, value: rect.height })
  element.getBoundingClientRect = () => ({ ...rect, right: rect.left + rect.width, bottom: rect.top + rect.height, x: rect.left, y: rect.top, toJSON: () => ({}) })
}

function addManualPair(
  rgb: { x: number; y: number },
  height: { x: number; y: number },
) {
  fireEvent.change(screen.getByLabelText('RGB X coordinate'), { target: { value: String(rgb.x) } })
  fireEvent.change(screen.getByLabelText('RGB Y coordinate'), { target: { value: String(rgb.y) } })
  fireEvent.change(screen.getByLabelText('Height X coordinate'), { target: { value: String(height.x) } })
  fireEvent.change(screen.getByLabelText('Height Y coordinate'), { target: { value: String(height.y) } })
  fireEvent.click(screen.getByRole('button', { name: 'Use selected RGB point' }))
  fireEvent.click(screen.getByRole('button', { name: 'Use selected height point' }))
  fireEvent.click(screen.getByRole('button', { name: 'Add Pair' }))
}

describe('vision engineering workspace', () => {
  beforeEach(() => {
    vi.stubGlobal('PointerEvent', MouseEvent)
    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: vi.fn().mockImplementation(() => ({
        matches: false,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      })),
    })
    viewMock.mockReset()
    sampleMock.mockReset()
    viewMock.mockResolvedValue({ data: engineeringResponse(), requestId: 'engineering-request' })
    sampleMock.mockResolvedValue({ data: sampleResponse, requestId: 'sample-request' })
  })

  it('switches among all five view modes and exposes comparison controls', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })

    const modes = screen.getByRole('group', { name: 'Vision view modes' })
    for (const mode of ['RGB', 'Height', 'Side-by-side', 'Alpha overlay', 'Split comparison']) {
      await user.click(within(modes).getByRole('button', { name: mode }))
      expect(document.querySelector('.vision-canvas')).toHaveAttribute('data-view-mode', mode)
    }
    expect(screen.getByRole('slider', { name: 'Split position' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Alpha overlay' }))
    expect(screen.getByRole('slider', { name: 'Overlay opacity' })).toBeInTheDocument()
  }, 10_000)

  it('applies synchronized zoom and normalized pan controls', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    await user.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%')
    await user.click(screen.getByRole('button', { name: 'Pan right' }))
    expect(document.querySelector('.vision-canvas')).toHaveAttribute('data-pan', '0.05,0.00')
    expect(screen.getByRole('status', { name: 'Engineering workspace status' })).toHaveTextContent('Zoom 125%')
    await user.click(screen.getByRole('button', { name: 'Reset view' }))
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%')
  })

  it('samples independent RGB and height coordinates only after explicit action', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    await user.clear(screen.getByLabelText('RGB X coordinate'))
    await user.type(screen.getByLabelText('RGB X coordinate'), '12')
    await user.clear(screen.getByLabelText('RGB Y coordinate'))
    await user.type(screen.getByLabelText('RGB Y coordinate'), '34')
    await user.clear(screen.getByLabelText('Height X coordinate'))
    await user.type(screen.getByLabelText('Height X coordinate'), '56')
    await user.clear(screen.getByLabelText('Height Y coordinate'))
    await user.type(screen.getByLabelText('Height Y coordinate'), '78')
    expect(sampleMock).not.toHaveBeenCalled()
    await user.click(screen.getByRole('button', { name: 'Sample values' }))
    await waitFor(() => expect(sampleMock).toHaveBeenCalledWith(INSPECTION_ID, { rgbX: 12, rgbY: 34, heightX: 56, heightY: 78, rgbSelected: true, heightSelected: true, activeSpace: 'HEIGHT' }, expect.any(AbortSignal)))
    const rgbInspector = screen.getByRole('region', { name: 'RGB evidence' })
    expect(within(rgbInspector).getByText('10')).toBeInTheDocument()
    expect(screen.getByText('2048')).toBeInTheDocument()
    expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0)
  }, 10_000)

  it('renders the SVG 64-bin histogram with native bounds and counts', async () => {
    const { container } = renderPage()
    expect(await screen.findByRole('img', { name: '64-bin native height histogram' })).toBeInTheDocument()
    expect(container.querySelectorAll('[data-histogram-bin]')).toHaveLength(64)
    const validCount = screen.getByText('Valid').parentElement?.textContent ?? ''
    expect(validCount.replace(/\D/g, '')).toBe('153600')
    expect(screen.getAllByText('4000').length).toBeGreaterThan(0)
  })

  it('shows integrity metadata and persisted pipeline evidence without execution actions', async () => {
    const user = userEvent.setup()
    renderPage()
    const inspector = await screen.findByRole('complementary', { name: 'Metadata and pixel inspector' })
    await user.click(within(inspector).getByText('RGB integrity metadata'))
    await user.click(within(inspector).getByText('Height integrity metadata'))
    expect(within(inspector).getByText(/PNG - uint8/)).toBeInTheDocument()
    expect(within(inspector).getByText(/TIFF - uint16/)).toBeInTheDocument()
    expect(within(inspector).getByText('NOT_CALIBRATED')).toBeInTheDocument()
    expect(within(inspector).getByText('NOT_REGISTERED')).toBeInTheDocument()
    expect(screen.getByText('VALIDATION_PASSED')).toBeInTheDocument()
    expect(screen.getByText('PREPROCESSING_PASSED')).toBeInTheDocument()
    expect(screen.getByText('No validation or processing is executed by this workspace.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /run|process|validate/i })).not.toBeInTheDocument()
  })

  it('warns when RGB and height dimensions differ and keeps coordinates separate', async () => {
    viewMock.mockResolvedValue({ data: engineeringResponse(true), requestId: 'engineering-request' })
    renderPage()
    const warning = await screen.findByRole('alert')
    expect(warning).toHaveTextContent('Dimension mismatch')
    expect(warning).toHaveTextContent('Coordinates remain independent')
    expect(screen.getByLabelText('Height X coordinate')).toHaveAttribute('max', '319')
    expect(screen.getByLabelText('RGB X coordinate')).toHaveAttribute('max', '639')
  })

  it('provides named landmarks, tools, warnings, guide, and persistent status', async () => {
    const user = userEvent.setup()
    renderPage()
    expect(await screen.findByRole('navigation', { name: 'Engineering evidence navigator' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Vision canvas workspace' })).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: 'Metadata and pixel inspector' })).toBeInTheDocument()
    expect(screen.getByRole('toolbar', { name: 'Vision canvas controls' })).toBeInTheDocument()
    expect(screen.getAllByText(/No physical calibration/).length).toBeGreaterThan(0)
    const tools = screen.getByRole('toolbar', { name: 'Engineering tools' })
    expect(within(tools).getByRole('button', { name: /Pointer/ })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('region', { name: 'Engineering workspace guide' })).toBeInTheDocument()
    expect(screen.getByRole('status', { name: 'Engineering workspace status' })).toHaveTextContent('Units unavailable')
    await user.click(screen.getByRole('button', { name: 'Dismiss guide' }))
    expect(screen.queryByRole('region', { name: 'Engineering workspace guide' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Show guide' }))
    expect(screen.getByRole('region', { name: 'Engineering workspace guide' })).toBeInTheDocument()
  })

  it('keeps affine alignment in session state and resets it without a backend write', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'Session alignment' })
    const translationX = screen.getByLabelText('Translation X (pixels)')
    const rotation = screen.getByLabelText('Rotation (degrees)')
    await user.clear(translationX)
    await user.type(translationX, '7')
    await user.clear(rotation)
    await user.type(rotation, '10')
    expect(screen.getByTestId('matrix-0-2')).toHaveTextContent('7')
    await user.click(screen.getByRole('button', { name: 'Apply transform to view only' }))
    expect(screen.getByTestId('height-raster').firstElementChild).toHaveStyle({ transformOrigin: 'center' })
    expect(viewMock).toHaveBeenCalledTimes(1)
    await user.click(screen.getByRole('button', { name: 'Reset transform to identity' }))
    expect(translationX).toHaveValue(0)
    expect(rotation).toHaveValue(0)
    expect(screen.getByTestId('matrix-0-0')).toHaveTextContent('1')
  })

  it('pairs explicit RGB and height points, reports residuals, and applies only an optional suggestion', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'Correspondence' })
    const setCoordinate = async (label: string, value: string) => {
      const input = screen.getByLabelText(label)
      await user.clear(input)
      await user.type(input, value)
    }
    await setCoordinate('RGB X coordinate', '12')
    await setCoordinate('RGB Y coordinate', '8')
    await setCoordinate('Height X coordinate', '10')
    await setCoordinate('Height Y coordinate', '5')
    expect(screen.getByRole('button', { name: 'Add Pair' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Use selected RGB point' }))
    expect(screen.getByText(/Pending P1: RGB \(12, 8\)/)).toBeInTheDocument()
    expect(screen.getByText(/Current step 2:/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Use selected height point' }))
    expect(screen.getByText(/Current step 3:/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Add Pair' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Add Pair' }))
    expect(screen.getByText(/Development residual:/)).toHaveTextContent(/3\.6055/)
    expect(screen.getByText(/Optional translation suggestion:/)).toHaveTextContent('X 2px, Y 3px')
    expect(screen.getByLabelText('Translation X (pixels)')).toHaveValue(0)
    await user.click(screen.getByRole('button', { name: 'Apply translation suggestion' }))
    expect(screen.getByLabelText('Translation X (pixels)')).toHaveValue(2)
    expect(screen.getByLabelText('Translation Y (pixels)')).toHaveValue(3)
    expect(screen.getByText(/Development residual:/)).toHaveTextContent('0 px')
    await user.click(screen.getByRole('button', { name: 'Remove P1' }))
    expect(screen.getByText(/No correspondence pairs in this browser session/)).toBeInTheDocument()
  }, 10_000)

  it('numbers visible landmarks, selects a pair from canvas or list, and confirms clear all', async () => {
    renderPage()
    await screen.findByRole('heading', { name: 'Correspondence' })
    addManualPair({ x: 12, y: 8 }, { x: 10, y: 5 })
    addManualPair({ x: 30, y: 30 }, { x: 30, y: 30 })

    expect(screen.getByTestId('rgb-landmark-P1')).toBeInTheDocument()
    expect(screen.getByTestId('height-landmark-P1')).toBeInTheDocument()
    expect(screen.getByTestId('rgb-landmark-P2')).toBeInTheDocument()
    expect(screen.getByTestId('residual-P1')).toHaveClass('highest')
    expect(screen.getByText('Highest residual pair').parentElement).toHaveTextContent('P1')

    fireEvent.click(screen.getByRole('button', { name: 'Select P1 RGB landmark' }))
    const pairList = screen.getByRole('list', { name: 'Correspondence pair list' })
    expect(within(pairList).getByRole('button', { name: /P1 RGB/ })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Select P2 Height landmark' }))
    expect(screen.getByTestId('height-landmark-P2')).toHaveClass('selected')
    fireEvent.click(within(pairList).getByRole('button', { name: /P1 RGB/ }))
    expect(screen.getByTestId('rgb-landmark-P1')).toHaveClass('selected')

    const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
    fireEvent.click(screen.getByRole('button', { name: 'Clear all correspondence points' }))
    expect(screen.getByTestId('rgb-landmark-P1')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Clear all correspondence points' }))
    expect(screen.queryByTestId('rgb-landmark-P1')).not.toBeInTheDocument()
    expect(confirm).toHaveBeenCalledTimes(2)
    confirm.mockRestore()
  }, 10_000)

  it('shows complete development residual statistics and toggles residual visualization', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'Results' })
    addManualPair({ x: 3, y: 4 }, { x: 0, y: 0 })
    addManualPair({ x: 10, y: 0 }, { x: 0, y: 0 })
    expect(screen.getByText('Pair count').parentElement).toHaveTextContent('2')
    expect(screen.getByText('Mean development residual').parentElement).toHaveTextContent('7.5 px')
    expect(screen.getByText('Maximum development residual').parentElement).toHaveTextContent('10 px')
    expect(screen.getByText('Minimum development residual').parentElement).toHaveTextContent('5 px')
    expect(screen.getByText('Median development residual').parentElement).toHaveTextContent('7.5 px')
    expect(screen.getAllByTestId(/^residual-P/)).toHaveLength(2)
    await user.click(screen.getByRole('checkbox', { name: 'Show development residual lines on RGB view' }))
    expect(screen.queryByTestId('residual-P1')).not.toBeInTheDocument()
  })

  it('switches original and development rendering, resets identity, and undo/redoes transforms', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'Transform' })
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-alignment-view', 'ORIGINAL')
    fireEvent.change(screen.getByLabelText('Translation X (pixels)'), { target: { value: '9' } })
    await user.click(screen.getByRole('button', { name: 'Apply transform to view only' }))
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-alignment-view', 'DEVELOPMENT')
    expect(screen.getByTestId('height-raster').firstElementChild).toHaveStyle({ transformOrigin: 'center' })
    await user.click(screen.getByRole('button', { name: 'Return to original' }))
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-alignment-view', 'ORIGINAL')

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-alignment-view', 'DEVELOPMENT')
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(screen.getByLabelText('Translation X (pixels)')).toHaveValue(9)
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(screen.getByLabelText('Translation X (pixels)')).toHaveValue(0)
    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    expect(screen.getByLabelText('Translation X (pixels)')).toHaveValue(9)
    await user.click(screen.getByRole('button', { name: 'Reset transform to identity' }))
    expect(screen.getByLabelText('Translation X (pixels)')).toHaveValue(0)
    expect(screen.getByTestId('matrix-0-0')).toHaveTextContent('1')
  })

  it('runs manual flicker safely and cleans its timer when stopped or unmounted', async () => {
    const rendered = renderPage()
    await screen.findByRole('heading', { name: 'View' })
    vi.useFakeTimers()
    const clearInterval = vi.spyOn(window, 'clearInterval')
    fireEvent.click(screen.getByRole('button', { name: 'Start flicker comparison' }))
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-flicker-running', 'true')
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-view-mode', 'Alpha overlay')
    act(() => vi.advanceTimersByTime(400))
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-alignment-view', 'DEVELOPMENT')
    fireEvent.click(within(screen.getByRole('group', { name: 'Vision view modes' })).getByRole('button', { name: 'RGB' }))
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-flicker-running', 'false')
    fireEvent.click(screen.getByRole('button', { name: 'Start flicker comparison' }))
    fireEvent.click(screen.getByRole('button', { name: 'Stop flicker' }))
    expect(screen.getByTestId('vision-canvas')).toHaveAttribute('data-flicker-running', 'false')
    fireEvent.click(screen.getByRole('button', { name: 'Start flicker comparison' }))
    rendered.unmount()
    expect(clearInterval).toHaveBeenCalled()
    clearInterval.mockRestore()
    vi.useRealTimers()
  })

  it('disables flicker when reduced motion is requested', async () => {
    Object.defineProperty(window, 'matchMedia', {
      configurable: true,
      value: vi.fn().mockImplementation(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })),
    })
    renderPage()
    await screen.findByRole('heading', { name: 'View' })
    expect(screen.getByRole('button', { name: 'Start flicker comparison' })).toBeDisabled()
    expect(screen.getByText(/Manual flicker alternates/)).toBeInTheDocument()
  })

  it('labels the display matrix context without calibration or production meaning', async () => {
    renderPage()
    await screen.findByRole('heading', { name: 'Results' })
    expect(screen.getByRole('table', { name: '3 by 3 development affine display matrix' })).toBeInTheDocument()
    expect(screen.getByText('Source: height display coordinates')).toBeInTheDocument()
    expect(screen.getByText('Target: RGB display coordinates')).toBeInTheDocument()
    expect(screen.getByText('Origin: RGB display centre')).toBeInTheDocument()
    expect(screen.getByText('Application: browser view only')).toBeInTheDocument()
  })

  it('selects independent native coordinates directly on the images and clears both crosshairs', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    const rgbRaster = screen.getByTestId('rgb-raster')
    const heightRaster = screen.getByTestId('height-raster')
    setRasterGeometry(rgbRaster, { left: 20, top: 40, width: 320, height: 240 })
    setRasterGeometry(heightRaster, { left: 400, top: 40, width: 320, height: 240 })
    const canvas = screen.getByTestId('vision-canvas')

    fireEvent.pointerDown(canvas, { clientX: 180, clientY: 160, pointerId: 1 })
    expect(screen.getByLabelText('RGB X coordinate')).toHaveValue(320)
    expect(screen.getByLabelText('RGB Y coordinate')).toHaveValue(240)
    expect(screen.getByTestId('rgb-crosshair')).toHaveAccessibleName('RGB selected coordinate X 320, Y 240')
    expect(sampleMock).not.toHaveBeenCalled()

    fireEvent.pointerDown(canvas, { clientX: 480, clientY: 100, pointerId: 2 })
    expect(screen.getByLabelText('Height X coordinate')).toHaveValue(160)
    expect(screen.getByLabelText('Height Y coordinate')).toHaveValue(120)
    expect(screen.getByTestId('height-crosshair')).toHaveAccessibleName('Height selected coordinate X 160, Y 120')

    await user.click(screen.getByRole('button', { name: 'Clear selected coordinates' }))
    expect(screen.queryByTestId('rgb-crosshair')).not.toBeInTheDocument()
    expect(screen.queryByTestId('height-crosshair')).not.toBeInTheDocument()
  })

  it('uses Sample clicks for exactly one GET while preserving the other coordinate space', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    const rgbRaster = screen.getByTestId('rgb-raster')
    const heightRaster = screen.getByTestId('height-raster')
    setRasterGeometry(rgbRaster, { left: 0, top: 0, width: 320, height: 240 })
    setRasterGeometry(heightRaster, { left: 400, top: 0, width: 320, height: 240 })
    await user.clear(screen.getByLabelText('Height X coordinate'))
    await user.type(screen.getByLabelText('Height X coordinate'), '56')
    await user.clear(screen.getByLabelText('Height Y coordinate'))
    await user.type(screen.getByLabelText('Height Y coordinate'), '78')
    await user.click(screen.getByRole('button', { name: /Sample S/ }))
    fireEvent.pointerDown(screen.getByTestId('vision-canvas'), { clientX: 80, clientY: 60, pointerId: 1 })

    await waitFor(() => expect(sampleMock).toHaveBeenCalledTimes(1))
    expect(sampleMock.mock.calls[0]?.[1]).toMatchObject({ rgbX: 160, rgbY: 120, heightX: 56, heightY: 78 })
    expect(screen.getByLabelText('Height X coordinate')).toHaveValue(56)
    expect(screen.getByLabelText('Height Y coordinate')).toHaveValue(78)
  })

  it('cancels stale sample requests and keeps visible request IDs on failures', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    await user.click(screen.getByRole('button', { name: 'Sample values' }))
    await waitFor(() => expect(sampleMock).toHaveBeenCalledTimes(1))
    const firstSignal = sampleMock.mock.calls[0]?.[2]
    await user.click(screen.getByRole('button', { name: 'Sample values' }))
    expect(firstSignal?.aborted).toBe(true)

    sampleMock.mockRejectedValueOnce(new ApiClientError(422, 'SAMPLE_OUT_OF_BOUNDS', 'Sample is outside the raster.', 'request-visible-123'))
    await user.click(screen.getByRole('button', { name: 'Sample values' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Request ID: request-visible-123')
  })

  it('supports one active tool, keyboard shortcuts, coordinate adjustment, and undo/redo', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    const rgbX = screen.getByLabelText('RGB X coordinate')
    await user.clear(rgbX)
    await user.type(rgbX, '10')
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(rgbX).toHaveValue(11)
    expect(screen.getByTestId('rgb-crosshair')).toHaveAccessibleName('RGB selected coordinate X 11, Y 0')
    fireEvent.keyDown(window, { key: 'ArrowDown', shiftKey: true })
    expect(screen.getByLabelText('RGB Y coordinate')).toHaveValue(10)
    expect(screen.getByRole('status', { name: 'Engineering workspace status' })).toHaveTextContent('RGB 11,10')

    for (const [key, label] of [['h', 'Pan'], ['s', 'Sample'], ['c', 'Correspondence'], ['r', 'Rectangle'], ['l', 'Line'], ['v', 'Pointer']] as const) {
      fireEvent.keyDown(window, { key })
      const tools = screen.getByRole('toolbar', { name: 'Engineering tools' })
      expect(within(tools).getByRole('button', { name: new RegExp(label) })).toHaveAttribute('aria-pressed', 'true')
      expect(within(tools).getAllByRole('button').filter((button) => button.getAttribute('aria-pressed') === 'true')).toHaveLength(1)
    }

    fireEvent.keyDown(window, { key: 'z', ctrlKey: true })
    expect(screen.getByLabelText('RGB Y coordinate')).toHaveValue(0)
    fireEvent.keyDown(window, { key: 'y', ctrlKey: true })
    expect(screen.getByLabelText('RGB Y coordinate')).toHaveValue(10)
  })

  it('suppresses shortcuts while typing and Escape cancels an incomplete correspondence', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    const tools = screen.getByRole('toolbar', { name: 'Engineering tools' })
    const rgbX = screen.getByLabelText('RGB X coordinate')
    rgbX.focus()
    fireEvent.keyDown(rgbX, { key: 's' })
    expect(within(tools).getByRole('button', { name: /Pointer/ })).toHaveAttribute('aria-pressed', 'true')

    rgbX.blur()
    fireEvent.keyDown(document.body, { key: '=' })
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('125%')
    fireEvent.keyDown(document.body, { key: '-' })
    expect(screen.getByLabelText('Zoom level')).toHaveTextContent('100%')
    fireEvent.keyDown(document.body, { key: '0' })
    expect(screen.getByRole('button', { name: 'Actual pixels' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.keyDown(document.body, { key: 'f' })
    expect(screen.getByRole('button', { name: 'Fit' })).toHaveAttribute('aria-pressed', 'true')

    fireEvent.keyDown(document.body, { key: 'c' })
    await user.click(screen.getByRole('button', { name: 'Use selected RGB point' }))
    expect(screen.getByText(/Pending P1: RGB/)).toBeInTheDocument()
    fireEvent.keyDown(document.body, { key: 'Escape' })
    expect(screen.queryByText(/Pending P1: RGB/)).not.toBeInTheDocument()
  })

  it('clears session selections and history when the workspace remounts', async () => {
    const first = renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    fireEvent.change(screen.getByLabelText('RGB X coordinate'), { target: { value: '22' } })
    expect(screen.getByTestId('rgb-crosshair')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled()
    first.unmount()

    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    expect(screen.getByLabelText('RGB X coordinate')).toHaveValue(0)
    expect(screen.queryByTestId('rgb-crosshair')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled()
  })

  it('exposes all shortcut help and makes no production decision action available', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    await user.click(screen.getByRole('button', { name: 'Keyboard help' }))
    const help = screen.getByRole('dialog', { name: 'Keyboard shortcuts' })
    expect(help).toHaveTextContent('V Pointer, H Pan, S Sample, C Correspondence, R Rectangle, L Line')
    expect(help).toHaveTextContent('Arrow keys move 1 pixel')
    expect(screen.queryByRole('button', { name: /production|approve|pass|fail/i })).not.toBeInTheDocument()
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByRole('dialog', { name: 'Keyboard shortcuts' })).not.toBeInTheDocument()
  })

  it.each([
    ['desktop', 1440],
    ['laptop', 1100],
    ['tablet', 760],
  ])('keeps all workspace regions available at %s width', async (_name, width) => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: width })
    window.dispatchEvent(new Event('resize'))
    renderPage()
    await screen.findByRole('heading', { name: 'PCB 2D/3D Vision Engineering Workspace' })
    expect(screen.getByRole('navigation', { name: 'Engineering evidence navigator' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Vision canvas workspace' })).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: 'Metadata and pixel inspector' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Persisted workflow record' })).toBeInTheDocument()
  })

  it('rejects malformed UUIDs before requesting engineering evidence', () => {
    renderPage('not-a-uuid')
    expect(screen.getByRole('alert')).toHaveTextContent('Malformed inspection ID')
    expect(viewMock).not.toHaveBeenCalled()
  })
})
