import { beforeEach, describe, expect, it, vi } from 'vitest'
import { buildInspectionFormData, getInspectionAudit, getInspectionReport, runProcessing } from './inspections'
import { recipesResponse } from '../test/fixtures'

describe('inspection API requests', () => {
  beforeEach(() => {
    vi.stubGlobal('crypto', { randomUUID: vi.fn(() => 'request-id') })
  })

  it('builds exact multipart fields, omits empty optional values, and preserves recipe identity', () => {
    const form = buildInspectionFormData({
      boardId: 'BOARD-A',
      recipe: recipesResponse.items[1]!,
      rgbImage: new File(['rgb'], 'board.png', { type: 'image/png' }),
      heightMap: new File(['height'], 'height.npy', { type: 'application/octet-stream' }),
      lotId: '   ',
      operatorId: '',
      stationId: undefined,
    })

    expect([...form.keys()]).toEqual(['board_id', 'recipe_id', 'recipe_version', 'rgb_image', 'height_map'])
    expect(form.get('recipe_id')).toBe('RECIPE-A')
    expect(form.get('recipe_version')).toBe('draft-2')
    expect(form.has('lot_id')).toBe(false)
    expect(form.has('operator_id')).toBe(false)
    expect(form.has('station_id')).toBe(false)
    expect(form.has('rgb_sha256')).toBe(false)
    expect(form.has('height_byte_size')).toBe(false)
  })

  it('sends processing with only the four authoritative policy fields', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)
    await runProcessing('11111111-1111-4111-8111-111111111111')

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit
    const body = JSON.parse(String(init.body)) as Record<string, unknown>
    expect(body).toEqual({
      preprocessing_policy_id: 'synthetic-paired-rgb-height',
      preprocessing_policy_version: '1.0',
      inference_policy_id: 'synthetic-deterministic-mock-inference',
      inference_policy_version: '1.0',
    })
    expect(Object.keys(body)).toHaveLength(4)
    expect(JSON.stringify(body)).not.toMatch(/synthetic_flag|decision|confidence/i)
  })

  it('uses the audit and report paths and preserves an opaque cursor', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', {
      status: 200,
      headers: { 'content-type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)
    await getInspectionAudit('11111111-1111-4111-8111-111111111111', 'opaque.cursor/value', 17)
    await getInspectionReport('11111111-1111-4111-8111-111111111111')
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/v1/inspections/11111111-1111-4111-8111-111111111111/audit?limit=17&cursor=opaque.cursor%2Fvalue')
    expect(fetchMock.mock.calls[1]?.[0]).toBe('/api/v1/inspections/11111111-1111-4111-8111-111111111111/report')
    expect(vi.mocked(crypto.randomUUID)).toHaveBeenCalledTimes(2)
  })
})
