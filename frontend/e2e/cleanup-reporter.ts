import { rmSync } from 'node:fs'
import path from 'node:path'
import type { FullResult, Reporter } from '@playwright/test/reporter'
import { diagnosticsRoot, frontendRoot, stateFile } from './helpers/paths'

export default class SuccessfulRunCleanupReporter implements Reporter {
  private passed = false

  onEnd(result: FullResult): void {
    this.passed = result.status === 'passed'
  }

  async onExit(): Promise<void> {
    if (!this.passed) return
    rmSync(path.join(frontendRoot, 'playwright-report'), { recursive: true, force: true })
    rmSync(path.join(frontendRoot, 'test-results', 'playwright'), { recursive: true, force: true })
    rmSync(diagnosticsRoot, { recursive: true, force: true })
    rmSync(stateFile, { force: true })
  }
}
