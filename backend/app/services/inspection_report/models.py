from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue


class ReportArtifact(BaseModel):
    artifact_type: str
    sha256: str
    byte_size: int
    media_type: str | None
    created_at: str


class ReportInspection(BaseModel):
    created_at: str
    board_id: str
    recipe_id: str
    recipe_version: str
    lot_id: str | None
    operator_id: str | None
    status: str
    error: dict[str, str] | None


class ReportValidation(BaseModel):
    contract_version: str
    validation_id: str
    validation_key: str
    result_sha256: str
    outcome: str
    policy: dict[str, str]
    validator_version: str
    started_at: str
    completed_at: str
    rgb_technical_summary: dict[str, JsonValue]
    height_technical_summary: dict[str, JsonValue]
    findings: list[dict[str, JsonValue]]
    summary: dict[str, JsonValue]


class ReportProcessing(BaseModel):
    processing_run_id: str
    validation_id: str
    processing_key: str
    lifecycle_status: str
    preprocessing_policy: dict[str, str]
    preprocessing_implementation: dict[str, str]
    inference_policy: dict[str, str]
    engine: dict[str, str]
    started_at: str
    completed_at: str | None
    final_decision: str | None
    error: dict[str, str] | None
    preprocessing: dict[str, JsonValue] | None
    inference: dict[str, JsonValue] | None
    synthetic_input: bool
    mock_preprocessing: bool
    mock_inference: bool
    production_approved: bool


class ReportAuditItem(BaseModel):
    audit_event_id: str
    inspection_id: str
    action: str
    created_at: str
    actor_id: str | None
    request_id: str | None
    details: dict[str, JsonValue]
    details_redacted: bool
    development_only: bool | None
    mock_result: str | None
    production_approved: bool | None


class DevelopmentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str
    inspection_id: str
    development_only: bool
    production_approved: bool
    synthetic_evidence_present: bool
    mock_inference_present: bool
    inspection: ReportInspection
    artifacts: list[ReportArtifact]
    validation: ReportValidation | None
    processing: ReportProcessing | None
    audit: list[ReportAuditItem]
    limitations: list[str]


class DevelopmentReportEnvelope(BaseModel):
    report: DevelopmentReport
    report_sha256: str
    request_id: str
