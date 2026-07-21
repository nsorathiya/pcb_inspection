import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { defaultPythonExecutable, diagnosticsRoot, frontendRoot, repositoryRoot, stateFile, successMarker } from './helpers/paths'
import { availablePort, runChecked, spawnLogged, stopOwnedProcess, waitForUrl } from './helpers/processes'
import type { RuntimeState } from './helpers/runtime'

export default async function globalSetup(): Promise<void> {
  rmSync(diagnosticsRoot, { recursive: true, force: true })
  rmSync(successMarker, { force: true })
  mkdirSync(diagnosticsRoot, { recursive: true })

  const temporaryRoot = mkdtempSync(path.join(os.tmpdir(), 'pcb-aoi-e2e-'))
  const runtimeRoot = path.join(temporaryRoot, 'runtime')
  const fixtureRoot = path.join(temporaryRoot, 'fixtures')
  const databaseFile = path.join(runtimeRoot, 'database', 'pcb_aoi.sqlite3')
  const pythonExecutable = defaultPythonExecutable()
  const backendPort = await availablePort()
  let frontendPort = await availablePort()
  while (frontendPort === backendPort) frontendPort = await availablePort()
  const backendUrl = `http://127.0.0.1:${backendPort}`
  const frontendUrl = `http://127.0.0.1:${frontendPort}`
  const backendLog = path.join(diagnosticsRoot, 'backend.stdout.log')
  const backendErrorLog = path.join(diagnosticsRoot, 'backend.stderr.log')
  const frontendLog = path.join(diagnosticsRoot, 'frontend.stdout.log')
  const frontendErrorLog = path.join(diagnosticsRoot, 'frontend.stderr.log')
  const queryLog = path.join(diagnosticsRoot, 'query-profile.jsonl')
  const state: RuntimeState = {
    temporaryRoot,
    runtimeRoot,
    fixtureRoot,
    databaseFile,
    frontendUrl,
    backendUrl,
    frontendPid: 0,
    backendPid: 0,
    pythonExecutable,
    queryLog,
    fixtureTreeSha256: '',
    browserChannel: process.env.PCB_AOI_E2E_BROWSER_CHANNEL
      ? `Chromium channel ${process.env.PCB_AOI_E2E_BROWSER_CHANNEL}`
      : 'Playwright Chromium',
  }

  try {
    runChecked(pythonExecutable, [
      path.join(repositoryRoot, 'scripts', 'generate_synthetic_inspection_fixtures.py'),
      '--output-root', fixtureRoot,
      '--seed', '20260717',
      '--scenario', 'valid_rgb_png_height_tiff',
      '--scenario', 'valid_different_dimensions',
    ], { cwd: repositoryRoot })
    const manifest = JSON.parse(readFileSync(path.join(fixtureRoot, 'generation_manifest.json'), 'utf8')) as { output_tree_sha256: string }
    state.fixtureTreeSha256 = manifest.output_tree_sha256

    runChecked(pythonExecutable, [
      path.join(frontendRoot, 'e2e', 'support', 'runtime_control.py'),
      'seed-recipes', '--runtime-root', runtimeRoot,
    ], { cwd: repositoryRoot })

    const backendEnvironment: NodeJS.ProcessEnv = {
      PCB_AOI_APPLICATION_NAME: 'pcb-aoi-e2e-api',
      PCB_AOI_APPLICATION_VERSION: '0.1.0-e2e',
      PCB_AOI_ENVIRONMENT: 'synthetic-e2e-development',
      PCB_AOI_RUNTIME_ROOT: runtimeRoot,
      PCB_AOI_LOG_LEVEL: 'WARNING',
      PCB_AOI_ENABLE_SYNTHETIC_PROCESSING_API: 'true',
      PCB_AOI_SYNTHETIC_FIXTURE_ROOT: fixtureRoot,
      PCB_AOI_E2E_BACKEND_HOST: '127.0.0.1',
      PCB_AOI_E2E_BACKEND_PORT: String(backendPort),
      PCB_AOI_E2E_QUERY_LOG: queryLog,
    }
    state.backendPid = spawnLogged(
      pythonExecutable,
      [path.join(frontendRoot, 'e2e', 'support', 'backend_server.py')],
      { cwd: repositoryRoot, env: backendEnvironment, stdout: backendLog, stderr: backendErrorLog },
    )
    writeFileSync(stateFile, JSON.stringify(state, null, 2))
    await waitForUrl(`${backendUrl}/api/v1/health`, 45_000)

    const viteEntrypoint = path.join(frontendRoot, 'node_modules', 'vite', 'bin', 'vite.js')
    state.frontendPid = spawnLogged(
      process.execPath,
      [viteEntrypoint, '--host', '127.0.0.1', '--port', String(frontendPort), '--strictPort'],
      {
        cwd: frontendRoot,
        env: { VITE_DEV_PROXY_TARGET: backendUrl, VITE_API_BASE_URL: undefined },
        stdout: frontendLog,
        stderr: frontendErrorLog,
      },
    )
    writeFileSync(stateFile, JSON.stringify(state, null, 2))
    await waitForUrl(frontendUrl, 45_000)
  } catch (error) {
    writeFileSync(path.join(diagnosticsRoot, 'setup-error.txt'), error instanceof Error ? error.stack ?? error.message : String(error))
    await stopOwnedProcess(state.frontendPid)
    await stopOwnedProcess(state.backendPid)
    rmSync(temporaryRoot, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 })
    throw error
  }
}
