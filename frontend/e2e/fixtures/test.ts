import { test as base, expect } from '@playwright/test'

export interface NetworkRecord {
  kind: 'request' | 'response'
  method: string
  url: string
  status?: number
  requestId?: string | null
}

export const test = base.extend<{ networkRecords: NetworkRecord[] }>({
  networkRecords: async ({ page }, provide) => {
    const records: NetworkRecord[] = []
    const request = (value: import('@playwright/test').Request) => {
      if (value.url().includes('/api/')) {
        records.push({
          kind: 'request',
          method: value.method(),
          url: value.url(),
          requestId: value.headers()['x-request-id'] ?? null,
        })
      }
    }
    const response = (value: import('@playwright/test').Response) => {
      if (value.url().includes('/api/')) {
        records.push({
          kind: 'response',
          method: value.request().method(),
          url: value.url(),
          status: value.status(),
          requestId: value.headers()['x-request-id'] ?? null,
        })
      }
    }
    page.on('request', request)
    page.on('response', response)
    await provide(records)
    page.off('request', request)
    page.off('response', response)
  },
})

test.afterEach(async ({ page, networkRecords }, testInfo) => {
  if (testInfo.status === testInfo.expectedStatus) return
  await testInfo.attach('page.html', {
    body: await page.content(),
    contentType: 'text/html',
  })
  await testInfo.attach('network-summary.json', {
    body: Buffer.from(JSON.stringify(networkRecords, null, 2)),
    contentType: 'application/json',
  })
})

export { expect }
