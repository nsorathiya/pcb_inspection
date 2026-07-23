import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { diagnosticsRoot, stateFile } from './helpers/paths'
import { stopOwnedProcess } from './helpers/processes'
import type { RuntimeState } from './helpers/runtime'

async function removeTemporaryRoot(root: string): Promise<void> {
  let lastError: unknown
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      rmSync(root, { recursive: true, force: true })
      if (!existsSync(root)) return
    } catch (error) {
      lastError = error
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw lastError instanceof Error
    ? lastError
    : new Error(`Isolated E2E temporary root was not removed: ${root}`)
}

export default async function globalTeardown(): Promise<void> {
  if (!existsSync(stateFile)) return
  const state = JSON.parse(readFileSync(stateFile, 'utf8')) as RuntimeState
  await stopOwnedProcess(state.frontendPid)
  await stopOwnedProcess(state.backendPid)

  const sidecars = [`${state.databaseFile}-wal`, `${state.databaseFile}-shm`]
    .filter((candidate) => existsSync(candidate))
    .map((candidate) => path.basename(candidate))
  await removeTemporaryRoot(state.temporaryRoot)
  if (existsSync(state.temporaryRoot)) {
    throw new Error(`Isolated E2E temporary root was not removed: ${state.temporaryRoot}`)
  }

  const summary = {
    frontend_stopped: true,
    backend_stopped: true,
    temporary_root_removed: true,
    database_sidecars_before_root_removal: sidecars,
  }
  writeFileSync(path.join(diagnosticsRoot, 'cleanup.json'), JSON.stringify(summary, null, 2))
  process.stdout.write('Synthetic E2E cleanup: processes stopped and temporary runtime removed.\n')
}
