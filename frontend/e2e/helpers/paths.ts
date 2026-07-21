import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

export const frontendRoot = path.resolve(fileURLToPath(new URL('../..', import.meta.url)))
export const repositoryRoot = path.resolve(frontendRoot, '..')
export const resultsRoot = path.join(frontendRoot, 'test-results')
export const diagnosticsRoot = path.join(resultsRoot, 'e2e-diagnostics')
export const stateFile = path.join(resultsRoot, 'e2e-runtime-state.json')
export const successMarker = path.join(resultsRoot, 'e2e-success.marker')

export function defaultPythonExecutable(): string {
  if (process.env.PCB_AOI_E2E_PYTHON) return process.env.PCB_AOI_E2E_PYTHON
  const local = process.platform === 'win32'
    ? path.join(repositoryRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(repositoryRoot, '.venv', 'bin', 'python')
  if (existsSync(local)) return local
  return process.platform === 'win32' ? 'python.exe' : 'python'
}
