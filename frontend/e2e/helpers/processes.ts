import { closeSync, openSync } from 'node:fs'
import { createServer } from 'node:net'
import { spawn, spawnSync } from 'node:child_process'

export function environmentPathKey(platform: NodeJS.Platform = process.platform): 'Path' | 'PATH' {
  return platform === 'win32' ? 'Path' : 'PATH'
}

export function normalizedEnvironment(
  overrides: NodeJS.ProcessEnv = {},
  sourceEnvironment: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): NodeJS.ProcessEnv {
  const result: NodeJS.ProcessEnv = {}
  const pathKey = environmentPathKey(platform)
  let pathValue: string | undefined
  for (const [key, value] of Object.entries(sourceEnvironment)) {
    if (key.toLowerCase() === 'path') {
      pathValue ??= value
    } else {
      result[key] = value
    }
  }
  if (pathValue) result[pathKey] = pathValue
  for (const [key, value] of Object.entries(overrides)) {
    if (key.toLowerCase() === 'path') {
      delete result.Path
      delete result.PATH
      if (value !== undefined) result[pathKey] = value
    } else if (value === undefined) {
      delete result[key]
    } else {
      result[key] = value
    }
  }
  return result
}

export async function availablePort(): Promise<number> {
  return await new Promise((resolve, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        server.close()
        reject(new Error('Could not reserve a local E2E port.'))
        return
      }
      const port = address.port
      server.close((error) => error ? reject(error) : resolve(port))
    })
  })
}

export function runChecked(
  executable: string,
  args: string[],
  options: { cwd: string; env?: NodeJS.ProcessEnv },
): string {
  const result = spawnSync(executable, args, {
    cwd: options.cwd,
    env: normalizedEnvironment(options.env),
    encoding: 'utf8',
    windowsHide: true,
  })
  if (result.status !== 0) {
    throw new Error([
      `Command failed (${result.status ?? 'no status'}): ${executable} ${args.join(' ')}`,
      result.stdout,
      result.stderr,
    ].filter(Boolean).join('\n'))
  }
  return result.stdout.trim()
}

export function spawnLogged(
  executable: string,
  args: string[],
  options: { cwd: string; env: NodeJS.ProcessEnv; stdout: string; stderr: string },
): number {
  const stdout = openSync(options.stdout, 'a')
  const stderr = openSync(options.stderr, 'a')
  try {
    const child = spawn(executable, args, {
      cwd: options.cwd,
      env: normalizedEnvironment(options.env),
      detached: false,
      windowsHide: true,
      stdio: ['ignore', stdout, stderr],
    })
    if (!child.pid) throw new Error(`Could not start ${executable}.`)
    return child.pid
  } finally {
    closeSync(stdout)
    closeSync(stderr)
  }
}

export async function waitForUrl(url: string, timeoutMs = 30_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  let lastError = 'no response'
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(2_000) })
      if (response.ok) return
      lastError = `HTTP ${response.status}`
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error)
    }
    await new Promise((resolve) => setTimeout(resolve, 200))
  }
  throw new Error(`Readiness timeout for ${url}: ${lastError}`)
}

function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

async function waitForExit(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (!processAlive(pid)) return true
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  return !processAlive(pid)
}

export async function stopOwnedProcess(pid: number | undefined): Promise<void> {
  if (!pid || !processAlive(pid)) return
  try {
    process.kill(pid, 'SIGTERM')
  } catch {
    // The fallback below handles an already-exiting or inaccessible owned process.
  }
  if (await waitForExit(pid, 5_000)) return
  if (process.platform === 'win32') {
    spawnSync('taskkill.exe', ['/PID', String(pid), '/T', '/F'], {
      windowsHide: true,
      encoding: 'utf8',
    })
  } else {
    try {
      process.kill(pid, 'SIGKILL')
    } catch {
      // It may have exited between the liveness check and signal.
    }
  }
  if (!(await waitForExit(pid, 5_000))) {
    process.stderr.write(`Owned E2E process ${pid} did not exit.\n`)
  }
}
