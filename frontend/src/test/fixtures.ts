import type {
  InspectionDetailResponse,
  InspectionHistoryResponse,
  InspectionProcessingResponse,
  InspectionValidationResponse,
  RecipeCatalogueResponse,
} from '../api/types'

export const INSPECTION_ID = '11111111-1111-4111-8111-111111111111'

export const recipesResponse: RecipeCatalogueResponse = {
  items: [
    { recipe_id: 'RECIPE-A', recipe_version: '1.0', name: 'Board A', status: 'ACTIVE', created_at: '2026-07-20T10:00:00Z', updated_at: '2026-07-20T10:00:00Z' },
    { recipe_id: 'RECIPE-A', recipe_version: 'draft-2', name: 'Board A draft', status: 'DRAFT', created_at: '2026-07-20T09:00:00Z', updated_at: '2026-07-20T09:00:00Z' },
  ],
  page: { limit: 10, has_more: false, next_cursor: null },
  applied_filters: {},
  request_id: 'recipes-request',
}

export const emptyHistoryResponse: InspectionHistoryResponse = {
  items: [],
  page: { limit: 25, has_more: false, next_cursor: null },
  applied_filters: {},
  request_id: 'history-request',
}

export function detailResponse(status: InspectionDetailResponse['status'] = 'RECEIVED'): InspectionDetailResponse {
  return {
    inspection_id: INSPECTION_ID,
    status,
    board_id: 'BOARD-A',
    recipe_id: 'RECIPE-A',
    recipe_version: '1.0',
    lot_id: null,
    intake_request_id: 'intake-request',
    created_at: '2026-07-20T10:00:00Z',
    started_at: status === 'PROCESSING' ? '2026-07-20T10:01:00Z' : null,
    completed_at: ['PASS', 'FAIL', 'UNCERTAIN', 'ERROR'].includes(status) ? '2026-07-20T10:02:00Z' : null,
    error: status === 'ERROR' ? { code: 'INSPECTION_ERROR', message: 'Inspection processing failed.' } : null,
    artifacts: [
      { artifact_type: 'RGB_RAW', sha256: 'a'.repeat(64), byte_size: 1200, media_type: 'image/png', created_at: '2026-07-20T10:00:00Z' },
      { artifact_type: 'HEIGHT_RAW', sha256: 'b'.repeat(64), byte_size: 2400, media_type: 'image/tiff', created_at: '2026-07-20T10:00:00Z' },
    ],
  }
}

export const validationResponse: InspectionValidationResponse = {
  inspection_id: INSPECTION_ID,
  validation_id: '22222222-2222-4222-8222-222222222222',
  validation_key: 'c'.repeat(64),
  validation_outcome: 'VALIDATION_PASSED',
  inspection_status: 'READY',
  policy: { policy_id: 'development-native-rgb-height', policy_version: '1.0' },
  validator_version: '1.0.0',
  started_at: '2026-07-20T10:00:01Z',
  completed_at: '2026-07-20T10:00:02Z',
  summary: { finding_count: 2, info_count: 1, warning_count: 1, error_count: 0, blocking_count: 0, technically_ready: true, synthetic_example: false },
  artifacts: {
    rgb: { artifact_type: 'RGB_RAW', sha256: 'a'.repeat(64), byte_size: 1200, declared_media_type: 'image/png', detected_format: 'PNG', width: 64, height: 64, channels: 3, bit_depth: 8, storage_data_type: null, readability_status: 'READABLE' },
    height: { artifact_type: 'HEIGHT_RAW', sha256: 'b'.repeat(64), byte_size: 2400, declared_media_type: 'image/tiff', detected_format: 'TIFF', width: 64, height: 64, channels: 1, bit_depth: 16, storage_data_type: 'uint16', readability_status: 'READABLE' },
  },
  findings: [
    { code: 'FIRST_FINDING', severity: 'INFO', category: 'RGB', message: 'First ordered finding.', blocking: false, artifact_type: 'RGB_RAW', field: null, details: null },
    { code: 'SECOND_FINDING', severity: 'WARNING', category: 'HEIGHT', message: 'Second ordered finding.', blocking: false, artifact_type: 'HEIGHT_RAW', field: null, details: null },
  ],
  idempotent_existing: false,
  request_id: 'validation-request',
}

export function processingResponse(decision: 'PASS' | 'FAIL' | 'UNCERTAIN' = 'PASS'): InspectionProcessingResponse {
  return {
    inspection_id: INSPECTION_ID,
    validation_id: validationResponse.validation_id,
    processing_run_id: '33333333-3333-4333-8333-333333333333',
    processing_key: 'd'.repeat(64),
    preprocessing_id: '44444444-4444-4444-8444-444444444444',
    inference_id: '55555555-5555-4555-8555-555555555555',
    preprocessing_outcome: 'PREPROCESSING_SUCCEEDED',
    inference_execution_outcome: 'INFERENCE_SUCCEEDED',
    mock_decision: decision,
    defect_type: decision === 'FAIL' ? 'missing_part' : null,
    inspection_status: decision,
    processing_status: 'COMPLETED',
    synthetic_input_verified: true,
    mock_preprocessing: true,
    mock_inference: true,
    production_approved: false,
    lifecycle_idempotent_existing: false,
    execution_started_now: true,
    started_at: '2026-07-20T10:01:00Z',
    completed_at: '2026-07-20T10:01:01Z',
    preprocessing: {
      preprocessing_id: '44444444-4444-4444-8444-444444444444',
      policy_id: 'synthetic-paired-rgb-height',
      policy_version: '1.0',
      implementation_id: 'synthetic-copy',
      implementation_version: '1.0',
      outcome: 'PREPROCESSING_SUCCEEDED',
      summary: { total_findings: 1, blocking_findings: 0, warnings: 0, errors: 0 },
      findings: [{ code: 'PREPROCESSING_MOCK_USED', severity: 'INFO', category: 'OUTPUT', message: 'Synthetic preprocessing finding.', blocking: false, branch: null, field: null, details: {} }],
    },
    inference: {
      inference_id: '55555555-5555-4555-8555-555555555555',
      policy_id: 'synthetic-deterministic-mock-inference',
      policy_version: '1.0',
      engine_id: 'synthetic-deterministic-mock-engine',
      engine_version: '1.0.0',
      engine_type: 'MOCK',
      execution_outcome: 'INFERENCE_SUCCEEDED',
      decision,
      defect_type: decision === 'FAIL' ? 'missing_part' : null,
      summary: { total_findings: 1, blocking_findings: 0, warnings: 0, errors: 0 },
      findings: [{ code: 'MOCK_DECISION_GENERATED', severity: 'INFO', category: 'DECISION', message: 'Synthetic decision finding.', blocking: false, branch: null, field: null, details: {} }],
    },
    request_id: 'processing-request',
  }
}
