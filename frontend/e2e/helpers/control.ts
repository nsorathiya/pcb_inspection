import { readFileSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { frontendRoot, repositoryRoot } from './paths'
import { runChecked } from './processes'
import type { RuntimeState } from './runtime'

export interface RuntimeSnapshot {
  schema_version: number
  foreign_key_failures: unknown[]
  database_fingerprint: string
  row_counts: Record<string, number>
  inspections: Array<Record<string, unknown>>
  processing: Array<Record<string, unknown>>
  audit: Array<Record<string, unknown>>
  artifact_integrity: Array<{
    inspection_id: string
    artifact_type: string
    registered_sha256: string
    registered_byte_size: number
    contained: boolean
    exists: boolean
    actual_sha256: string | null
    actual_byte_size: number | null
    actual_mtime_ns: number | null
  }>
  runtime_files: string[]
  report_files: string[]
  fixture_tree_sha256: string
  fixture_files_verified: boolean
  fixture_integrity: Array<{
    path: string
    sha256: string
    byte_size: number
    mtime_ns: number
  }>
  fixture_control_files: Array<{
    path: string
    sha256: string
    byte_size: number
    mtime_ns: number
  }>
}

export interface QueryProfile {
  method: string
  path: string
  status: number
  select_queries: number
  write_queries: number
  elapsed_ms: number
}

function control(state: RuntimeState, args: string[]): string {
  return runChecked(state.pythonExecutable, [
    path.join(frontendRoot, 'e2e', 'support', 'runtime_control.py'),
    ...args,
    '--runtime-root', state.runtimeRoot,
  ], { cwd: repositoryRoot })
}

export function snapshot(state: RuntimeState): RuntimeSnapshot {
  return JSON.parse(control(state, ['snapshot', '--fixture-root', state.fixtureRoot])) as RuntimeSnapshot
}

export function seedHistory(state: RuntimeState, count = 25): void {
  control(state, ['seed-history', '--count', String(count)])
}

export function tamperRgb(state: RuntimeState, inspectionId: string): void {
  control(state, ['tamper-rgb', '--inspection-id', inspectionId])
}

export function verifyReportEnvelope(state: RuntimeState, rawEnvelope: string): string {
  const envelopeFile = path.join(state.temporaryRoot, 'report-envelope-verification.json')
  writeFileSync(envelopeFile, rawEnvelope, 'utf8')
  try {
    const result = JSON.parse(control(state, ['verify-report', '--envelope-file', envelopeFile])) as {
      calculated_sha256: string
      reported_sha256: string
      matches: boolean
    }
    if (!result.matches || result.calculated_sha256 !== result.reported_sha256) {
      throw new Error('Development report canonical SHA-256 verification failed.')
    }
    return result.calculated_sha256
  } finally {
    rmSync(envelopeFile, { force: true })
  }
}

export function queryProfiles(state: RuntimeState): QueryProfile[] {
  const content = readFileSync(state.queryLog, 'utf8').trim()
  if (!content) return []
  return content.split(/\r?\n/).map((line) => JSON.parse(line) as QueryProfile)
}
