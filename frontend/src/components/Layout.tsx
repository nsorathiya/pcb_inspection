import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { getHealth } from '../api/client'
import type { HealthResponse } from '../api/types'
import { DevelopmentWarning } from './DevelopmentWarning'

export function Layout() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthUnavailable, setHealthUnavailable] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    getHealth(controller.signal)
      .then(({ data }) => {
        setHealth(data)
        setHealthUnavailable(false)
      })
      .catch(() => setHealthUnavailable(true))
    return () => controller.abort()
  }, [])

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">AOI</div>
          <div>
            <p className="eyebrow">File-based inspection workflow</p>
            <h1>PCB AOI Operator Console</h1>
          </div>
        </div>
        <div className="system-indicators" aria-label="System status">
          <span className="environment-indicator">{health?.environment ?? 'development'}</span>
          <span className={`health-indicator ${healthUnavailable ? 'unavailable' : health ? 'healthy' : ''}`}>
            <span className="status-dot" aria-hidden="true" />
            {healthUnavailable ? 'Backend unavailable' : health ? 'Backend online' : 'Checking backend'}
          </span>
        </div>
      </header>
      <nav className="primary-nav" aria-label="Operator navigation">
        <NavLink to="/" end>Inspection History</NavLink>
        <NavLink to="/inspections/new">New Inspection</NavLink>
      </nav>
      <DevelopmentWarning />
      <main id="main-content" className="main-content">
        <Outlet />
      </main>
      <footer className="site-footer">
        <span>{health?.service ?? 'pcb-aoi-api'} {health?.version ? `v${health.version}` : ''}</span>
        <span>Manual operator workflow · server state is authoritative</span>
      </footer>
    </div>
  )
}
