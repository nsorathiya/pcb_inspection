import { defineConfig, devices } from '@playwright/test'

const localBrowserChannel = process.env.PCB_AOI_E2E_BROWSER_CHANNEL

export default defineConfig({
  testDir: './e2e/specs',
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 20_000 },
  globalSetup: './e2e/global-setup.ts',
  globalTeardown: './e2e/global-teardown.ts',
  outputDir: './test-results/playwright',
  reporter: [
    ['line'],
    ['html', { outputFolder: './playwright-report', open: 'never' }],
    ['./e2e/cleanup-reporter.ts'],
  ],
  use: {
    ...devices['Desktop Chrome'],
    channel: localBrowserChannel,
    headless: process.env.PCB_AOI_E2E_HEADED !== 'true',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    actionTimeout: 15_000,
  },
})
