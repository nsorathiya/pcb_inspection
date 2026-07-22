export type InspectionStatus =
  | 'RECEIVED'
  | 'VALIDATION_FAILED'
  | 'READY'
  | 'PROCESSING'
  | 'PASS'
  | 'FAIL'
  | 'UNCERTAIN'
  | 'ERROR'

export type ValidationOutcome =
  | 'VALIDATION_PASSED'
  | 'VALIDATION_FAILED'
  | 'VALIDATION_ERROR'

export type MockDecision = 'PASS' | 'FAIL' | 'UNCERTAIN'
export type ProcessingStatus = 'STARTED' | 'COMPLETED' | 'ERROR'
export type RecipeStatus = 'DRAFT' | 'ACTIVE' | 'RETIRED'
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue }

export interface ApiErrorPayload {
  code: string
  message: string
  request_id: string
}

export interface HealthResponse {
  status: string
  service: string
  version: string
  environment: string
}

export interface DemoInspectionState {
  key: string
  inspection_id: string
  board_id: string
  status: InspectionStatus | null
  validation_outcome: ValidationOutcome | null
  processing_status: ProcessingStatus | null
  preprocessing_outcome: string | null
  mock_decision: MockDecision | null
  complete: boolean
}

export interface DemoWorkspaceResponse {
  enabled: boolean
  available: boolean
  loaded: boolean
  recipes_ready: boolean
  inspections: DemoInspectionState[]
  synthetic: true
  production_approved: false
  idempotent_existing: boolean | null
  request_id: string
}

export interface PageResponse {
  limit: number
  has_more: boolean
  next_cursor: string | null
}

export interface AuditTimelineItem {
  audit_event_id: string
  inspection_id: string
  action: string
  created_at: string
  actor_id: string | null
  request_id: string | null
  details: Record<string, JsonValue>
  details_redacted: boolean
  development_only: boolean | null
  mock_result: MockDecision | null
  production_approved: boolean | null
}

export interface InspectionAuditResponse {
  items: AuditTimelineItem[]
  page: PageResponse
  request_id: string
}

export interface DevelopmentReportArtifact {
  artifact_type: string
  sha256: string
  byte_size: number
  media_type: string | null
  created_at: string
}

export interface DevelopmentReportValidation {
  contract_version: string
  validation_id: string
  validation_key: string
  result_sha256: string
  outcome: ValidationOutcome
  policy: { policy_id: string; policy_version: string }
  validator_version: string
  started_at: string
  completed_at: string
  rgb_technical_summary: Record<string, JsonValue>
  height_technical_summary: Record<string, JsonValue>
  findings: Array<Record<string, JsonValue>>
  summary: Record<string, JsonValue>
}

export interface DevelopmentReportProcessing {
  processing_run_id: string
  validation_id: string
  processing_key: string
  lifecycle_status: ProcessingStatus
  preprocessing_policy: Record<string, string>
  preprocessing_implementation: Record<string, string>
  inference_policy: Record<string, string>
  engine: Record<string, string>
  started_at: string
  completed_at: string | null
  final_decision: MockDecision | null
  error: { code: string; message: string } | null
  preprocessing: Record<string, JsonValue> | null
  inference: Record<string, JsonValue> | null
  synthetic_input: boolean
  mock_preprocessing: boolean
  mock_inference: boolean
  production_approved: false
}

export interface InspectionDevelopmentReport {
  contract_version: string
  inspection_id: string
  development_only: true
  production_approved: false
  synthetic_evidence_present: boolean
  mock_inference_present: boolean
  inspection: {
    created_at: string
    board_id: string
    recipe_id: string
    recipe_version: string
    lot_id: string | null
    operator_id: string | null
    status: InspectionStatus
    error: { code: string; message: string } | null
  }
  artifacts: DevelopmentReportArtifact[]
  validation: DevelopmentReportValidation | null
  processing: DevelopmentReportProcessing | null
  audit: AuditTimelineItem[]
  limitations: string[]
}

export interface InspectionDevelopmentReportResponse {
  report: InspectionDevelopmentReport
  report_sha256: string
  request_id: string
}

export interface RecipeCatalogueItem {
  recipe_id: string
  recipe_version: string
  name: string
  status: RecipeStatus
  created_at: string
  updated_at: string
}

export interface RecipeCatalogueResponse {
  items: RecipeCatalogueItem[]
  page: PageResponse
  applied_filters: Record<string, string>
  request_id: string
}

export interface ArtifactSummary {
  artifact_type: string
  sha256: string
  byte_size: number
  media_type: string | null
}

export interface InspectionIntakeResponse {
  inspection_id: string
  status: InspectionStatus
  board_id: string
  recipe_id: string
  recipe_version: string
  lot_id: string | null
  request_id: string
  created_at: string
  artifacts: ArtifactSummary[]
}

export interface InspectionDetailArtifact extends ArtifactSummary {
  created_at: string
}

export interface InspectionDetailResponse {
  inspection_id: string
  status: InspectionStatus
  board_id: string
  recipe_id: string
  recipe_version: string
  lot_id: string | null
  intake_request_id: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  error: { code: string; message: string } | null
  artifacts: InspectionDetailArtifact[]
}

export interface HistoryValidationSummary {
  validation_id: string
  outcome: ValidationOutcome
  policy_id: string
  policy_version: string
  validator_version: string
  completed_at: string
  total_findings: number
  blocking_findings: number
  warnings: number
  errors: number
}

export interface HistoryProcessingSummary {
  processing_run_id: string
  processing_status: ProcessingStatus
  preprocessing_id: string | null
  preprocessing_outcome: string | null
  inference_id: string | null
  inference_execution_outcome: string | null
  mock_decision: MockDecision | null
  defect_type: string | null
  started_at: string
  completed_at: string | null
  synthetic_input: boolean
  mock_preprocessing: boolean
  mock_inference: boolean
  production_approved: boolean
}

export interface InspectionHistoryItem {
  inspection_id: string
  status: InspectionStatus
  board_id: string
  recipe: { recipe_id: string; recipe_version: string }
  lot_id: string | null
  operator_id: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  technical_error_code: string | null
  validation: HistoryValidationSummary | null
  processing: HistoryProcessingSummary | null
}

export interface InspectionHistoryResponse {
  items: InspectionHistoryItem[]
  page: PageResponse
  applied_filters: Record<string, string | boolean>
  request_id: string
}

export interface ValidationFinding {
  code: string
  severity: string
  category: string
  message: string
  blocking: boolean
  artifact_type: string | null
  field: string | null
  details: Record<string, JsonValue> | null
}

export interface ValidationArtifact {
  artifact_type: string
  sha256: string | null
  byte_size: number | null
  declared_media_type: string | null
  detected_format: string | null
  width: number | null
  height: number | null
  channels: number | null
  bit_depth: number | null
  storage_data_type: string | null
  readability_status: string
}

export interface InspectionValidationResponse {
  inspection_id: string
  validation_id: string
  validation_key: string
  validation_outcome: ValidationOutcome
  inspection_status: InspectionStatus
  policy: { policy_id: string; policy_version: string }
  validator_version: string
  started_at: string
  completed_at: string
  summary: {
    finding_count: number
    info_count: number
    warning_count: number
    error_count: number
    blocking_count: number
    technically_ready: boolean
    synthetic_example: boolean
  }
  artifacts: { rgb: ValidationArtifact; height: ValidationArtifact }
  findings: ValidationFinding[]
  idempotent_existing: boolean
  request_id: string
}

export interface ProcessingFinding {
  code: string
  severity: string
  category: string
  message: string
  blocking: boolean
  branch: string | null
  field: string | null
  details: Record<string, JsonValue>
}

export interface ProcessingSummary {
  total_findings: number
  blocking_findings: number
  warnings: number
  errors: number
}

export interface PreprocessingEvidence {
  preprocessing_id: string
  policy_id: string
  policy_version: string
  implementation_id: string
  implementation_version: string
  outcome: string
  summary: ProcessingSummary
  findings: ProcessingFinding[]
}

export interface InferenceEvidence {
  inference_id: string
  policy_id: string
  policy_version: string
  engine_id: string
  engine_version: string
  engine_type: string
  execution_outcome: string
  decision: MockDecision | null
  defect_type: string | null
  summary: ProcessingSummary
  findings: ProcessingFinding[]
}

export interface InspectionProcessingResponse {
  inspection_id: string
  validation_id: string
  processing_run_id: string
  processing_key: string
  preprocessing_id: string
  inference_id: string | null
  preprocessing_outcome: string
  inference_execution_outcome: string | null
  mock_decision: MockDecision | null
  defect_type: string | null
  inspection_status: InspectionStatus
  processing_status: ProcessingStatus
  synthetic_input_verified: boolean
  mock_preprocessing: boolean
  mock_inference: boolean
  production_approved: boolean
  lifecycle_idempotent_existing: boolean
  execution_started_now: boolean
  started_at: string
  completed_at: string | null
  preprocessing: PreprocessingEvidence
  inference: InferenceEvidence | null
  request_id: string
}

export interface HistoryFilters {
  status?: InspectionStatus | ''
  board_id?: string
  recipe_id?: string
  lot_id?: string
  validation_outcome?: ValidationOutcome | ''
  processing_status?: ProcessingStatus | ''
  mock_decision?: MockDecision | ''
  has_validation?: boolean | ''
  has_processing?: boolean | ''
  created_from?: string
  created_to?: string
}

export interface RecipeFilters {
  recipe_id?: string
  recipe_version?: string
  name?: string
  status?: RecipeStatus | ''
}

export interface EngineeringRasterMetadata {
  artifact_type: 'RGB_RAW' | 'HEIGHT_RAW'
  detected_format: string
  width: number
  height: number
  channels: number
  bit_depth: number
  color_mode: string
  storage_data_type: string | null
  sha256: string
  byte_size: number
}

export interface EngineeringHeightStatistics {
  native_min: number
  native_max: number
  valid_count: number
  invalid_count: number
  histogram: {
    bin_count: 64
    native_min: number
    native_max: number
    counts: number[]
  }
}

export interface EngineeringValidationEvidence {
  available: boolean
  validation_id: string | null
  outcome: string | null
  policy_id: string | null
  policy_version: string | null
  technically_ready: boolean | null
  finding_codes: string[]
}

export interface EngineeringProcessingEvidence {
  available: boolean
  processing_run_id: string | null
  processing_status: string | null
  preprocessing_outcome: string | null
  mock_decision: string | null
  production_approved: boolean | null
  synthetic_input_verified: boolean | null
  finding_codes: string[]
}

export interface EngineeringViewResponse {
  inspection_id: string
  inspection_status: string
  rgb: EngineeringRasterMetadata
  height: EngineeringRasterMetadata
  height_statistics: EngineeringHeightStatistics
  calibration_status: string
  registration_status: string
  physical_height_unit: null
  validation: EngineeringValidationEvidence
  processing: EngineeringProcessingEvidence
  warnings: string[]
  synthetic_input_verified: boolean
  production_approved: false
  request_id: string
}

export interface EngineeringSampleResponse {
  inspection_id: string
  rgb: {
    x: number
    y: number
    storage_data_type: string | null
    values: number[]
  }
  height: {
    x: number
    y: number
    storage_data_type: string | null
    value: number | null
    valid: boolean
    physical_unit: null
  }
  warnings: string[]
  request_id: string
}

export interface EngineeringHeightRoiResponse {
  inspection_id: string
  x: number
  y: number
  width: number
  height: number
  storage_data_type: string | null
  native_min: number
  native_max: number
  native_mean: number
  valid_count: number
  invalid_count: number
  physical_unit: null
  warnings: string[]
  request_id: string
}
