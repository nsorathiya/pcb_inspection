import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getHealth } from '../api/client'
import { Layout } from './Layout'

vi.mock('../api/client', () => ({ getHealth: vi.fn() }))
const healthMock = vi.mocked(getHealth)

describe('operator application layout', () => {
  beforeEach(() => healthMock.mockResolvedValue({ data: { status: 'ok', service: 'pcb-aoi-api', version: '0.1.0', environment: 'development' }, requestId: 'health-request' }))

  it('shows navigation, backend health, environment, and the persistent development warning', async () => {
    render(<MemoryRouter><Routes><Route element={<Layout />}><Route index element={<p>Dashboard</p>} /></Route></Routes></MemoryRouter>)
    expect(screen.getByRole('link', { name: 'Inspection History' })).toHaveAttribute('href', '/')
    expect(screen.getByRole('link', { name: 'New Inspection' })).toHaveAttribute('href', '/inspections/new')
    expect(await screen.findByText('Backend online')).toBeInTheDocument()
    expect(screen.getByText('development')).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent('processing uses deterministic synthetic mock inference')
    expect(screen.getByRole('note')).toHaveTextContent('Results are not production PCB decisions')
  })
})
