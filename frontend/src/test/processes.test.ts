import { describe, expect, it } from 'vitest'
import { normalizedEnvironment } from '../../e2e/helpers/processes'

describe('E2E child-process environment', () => {
  it('preserves the platform-specific executable search path key', () => {
    const linuxEnvironment = normalizedEnvironment({}, { PATH: '/configured/python/bin' }, 'linux')
    const macEnvironment = normalizedEnvironment({ Path: '/override/bin' }, { PATH: '/base/bin' }, 'darwin')
    const windowsEnvironment = normalizedEnvironment({}, { PATH: 'C:\\configured\\python' }, 'win32')

    expect(linuxEnvironment).toMatchObject({ PATH: '/configured/python/bin' })
    expect(linuxEnvironment.Path).toBeUndefined()
    expect(macEnvironment).toMatchObject({ PATH: '/override/bin' })
    expect(macEnvironment.Path).toBeUndefined()
    expect(windowsEnvironment).toMatchObject({ Path: 'C:\\configured\\python' })
    expect(windowsEnvironment.PATH).toBeUndefined()
  })
})
