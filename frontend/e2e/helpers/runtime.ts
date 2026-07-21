import { readFileSync } from 'node:fs'
import path from 'node:path'
import { stateFile } from './paths'

export interface RuntimeState {
  temporaryRoot: string
  runtimeRoot: string
  fixtureRoot: string
  databaseFile: string
  frontendUrl: string
  backendUrl: string
  frontendPid: number
  backendPid: number
  pythonExecutable: string
  queryLog: string
  fixtureTreeSha256: string
  browserChannel: string
}

export interface ScenarioArtifact {
  generated_file: string
  media_type: string
  actual_sha256: string
  actual_byte_size: number
}

export interface ScenarioRecord {
  scenario_id: string
  expected_intake_outcome: string
  expected_technical_validation_outcome: string
  expected_finding_codes: string[]
  artifacts: { rgb: ScenarioArtifact; height: ScenarioArtifact }
}

export function runtimeState(): RuntimeState {
  return JSON.parse(readFileSync(stateFile, 'utf8')) as RuntimeState
}

export function scenarioRecord(state: RuntimeState, scenarioId: string): ScenarioRecord {
  const file = path.join(state.fixtureRoot, 'scenarios', scenarioId, 'scenario.json')
  const record = JSON.parse(readFileSync(file, 'utf8')) as ScenarioRecord
  if (record.scenario_id !== scenarioId) throw new Error('Synthetic scenario identity mismatch.')
  return record
}

export function scenarioFiles(state: RuntimeState, scenarioId: string) {
  const record = scenarioRecord(state, scenarioId)
  const root = path.join(state.fixtureRoot, 'scenarios', scenarioId)
  return {
    record,
    rgb: path.join(root, record.artifacts.rgb.generated_file),
    height: path.join(root, record.artifacts.height.generated_file),
  }
}

