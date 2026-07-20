import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createInspection } from '../api/inspections'
import { getRecipes } from '../api/recipes'
import { ApiClientError } from '../api/errors'
import { INSPECTION_ID, recipesResponse } from '../test/fixtures'
import { NewInspectionPage } from './NewInspectionPage'

vi.mock('../api/recipes', () => ({ getRecipes: vi.fn() }))
vi.mock('../api/inspections', () => ({ createInspection: vi.fn() }))
const recipesMock = vi.mocked(getRecipes)
const createMock = vi.mocked(createInspection)

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/inspections/new']}>
      <Routes>
        <Route path="/inspections/new" element={<NewInspectionPage />} />
        <Route path="/inspections/:inspectionId" element={<p>Created inspection destination</p>} />
      </Routes>
    </MemoryRouter>,
  )
}

async function completeRequiredFields() {
  await userEvent.type(screen.getByLabelText(/Board ID/), 'BOARD-A')
  await userEvent.click(await screen.findByRole('radio', { name: /Board A draft/ }))
  await userEvent.upload(screen.getByLabelText(/RGB image/), new File(['rgb'], 'board.png', { type: 'image/png' }))
  await userEvent.upload(screen.getByLabelText(/Height \/ depth map/), new File(['height'], 'height.npy', { type: 'application/octet-stream' }))
}

describe('new inspection page', () => {
  beforeEach(() => {
    recipesMock.mockResolvedValue({ data: recipesResponse, requestId: 'recipes-request' })
    createMock.mockResolvedValue({
      data: { inspection_id: INSPECTION_ID, status: 'RECEIVED', board_id: 'BOARD-A', recipe_id: 'RECIPE-A', recipe_version: 'draft-2', lot_id: null, request_id: 'intake-request', created_at: '2026-07-20T10:00:00Z', artifacts: [] },
      requestId: 'intake-request',
    })
  })

  it('has labelled required controls and reports missing values accessibly', async () => {
    renderPage()
    await screen.findByText('Board A')
    await userEvent.click(screen.getByRole('button', { name: 'Create inspection' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Board ID is required')
    expect(screen.getByRole('alert')).toHaveTextContent('Select one recipe identity and version')
    expect(screen.getByLabelText(/RGB image/)).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByLabelText(/Height \/ depth map/)).toHaveAttribute('aria-invalid', 'true')
  })

  it('keeps opaque recipe versions separate and displays selected file metadata', async () => {
    renderPage()
    expect(await screen.findAllByRole('radio')).toHaveLength(2)
    expect(screen.getByText(/RECIPE-A · 1.0/)).toBeInTheDocument()
    expect(screen.getByText(/RECIPE-A · draft-2/)).toBeInTheDocument()
    await userEvent.upload(screen.getByLabelText(/RGB image/), new File(['rgb-data'], 'board.PNG', { type: 'image/png' }))
    expect(screen.getByText('board.PNG')).toBeInTheDocument()
    expect(screen.getByText(/8 B · image\/png · .png/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Clear' })).toBeEnabled()
  })

  it('submits an explicitly selected recipe unchanged and navigates after HTTP 201', async () => {
    renderPage()
    await completeRequiredFields()
    await userEvent.click(screen.getByRole('button', { name: 'Create inspection' }))
    await waitFor(() => expect(createMock).toHaveBeenCalledTimes(1))
    expect(createMock.mock.calls[0]?.[0].recipe.recipe_version).toBe('draft-2')
    expect(createMock.mock.calls[0]?.[0].lotId).toBe('')
    expect(await screen.findByText('Created inspection destination')).toBeInTheDocument()
  })

  it('retains entered values after a structured backend failure', async () => {
    createMock.mockRejectedValue(new ApiClientError(422, 'INCOMPLETE_OR_INVALID_MULTIPART_REQUEST', 'Required multipart fields are invalid.', 'intake-error-request'))
    renderPage()
    await completeRequiredFields()
    await userEvent.type(screen.getByLabelText(/Lot ID/), 'LOT-9')
    await userEvent.click(screen.getByRole('button', { name: 'Create inspection' }))
    expect(await screen.findByText('Required multipart fields are invalid.')).toBeInTheDocument()
    expect(screen.getByText('intake-error-request')).toBeInTheDocument()
    expect(screen.getByLabelText(/Board ID/)).toHaveValue('BOARD-A')
    expect(screen.getByLabelText(/Lot ID/)).toHaveValue('LOT-9')
  })

  it('prevents duplicate submit while intake is active', async () => {
    let resolveRequest: (() => void) | undefined
    createMock.mockImplementation(() => new Promise((resolve) => {
      resolveRequest = () => resolve({ data: { inspection_id: INSPECTION_ID, status: 'RECEIVED', board_id: 'BOARD-A', recipe_id: 'RECIPE-A', recipe_version: 'draft-2', lot_id: null, request_id: 'request', created_at: '2026-07-20T10:00:00Z', artifacts: [] }, requestId: 'request' })
    }))
    renderPage()
    await completeRequiredFields()
    const button = screen.getByRole('button', { name: 'Create inspection' })
    await userEvent.click(button)
    expect(screen.getByRole('button', { name: 'Creating inspection…' })).toBeDisabled()
    expect(createMock).toHaveBeenCalledTimes(1)
    resolveRequest?.()
  })
})
