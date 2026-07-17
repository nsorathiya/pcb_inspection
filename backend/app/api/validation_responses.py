from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from pydantic import BaseModel

from app.db.models import ArtifactType, InspectionStatus
from app.db.validation_types import FindingCategory, FindingSeverity, ValidationOutcome
from app.services.inspection_validation.interfaces import (
    ArtifactTechnicalSummary,
    ReadabilityStatus,
)
from app.services.inspection_validation.orchestrator import ValidationExecutionResult


class ValidationPolicyResponse(BaseModel):
    policy_id: str
    policy_version: str


class ValidationArtifactResponse(BaseModel):
    artifact_type: ArtifactType
    sha256: str | None
    byte_size: int | None
    declared_media_type: str | None
    detected_format: str | None
    width: int | None
    height: int | None
    channels: int | None
    bit_depth: int | None
    storage_data_type: str | None
    readability_status: ReadabilityStatus


class ValidationArtifactsResponse(BaseModel):
    rgb: ValidationArtifactResponse
    height: ValidationArtifactResponse


class ValidationFindingResponse(BaseModel):
    code: str
    severity: FindingSeverity
    category: FindingCategory
    message: str
    blocking: bool
    artifact_type: ArtifactType | None = None
    field: str | None = None
    details: Mapping[str, Any] | None = None


class ValidationSummaryResponse(BaseModel):
    finding_count: int
    info_count: int
    warning_count: int
    error_count: int
    blocking_count: int
    technically_ready: bool
    synthetic_example: bool


class InspectionValidationResponse(BaseModel):
    inspection_id: str
    validation_id: str
    validation_key: str
    validation_outcome: ValidationOutcome
    inspection_status: InspectionStatus
    policy: ValidationPolicyResponse
    validator_version: str
    started_at: datetime
    completed_at: datetime
    summary: ValidationSummaryResponse
    artifacts: ValidationArtifactsResponse
    findings: list[ValidationFindingResponse]
    idempotent_existing: bool
    request_id: str


def _artifact(value: ArtifactTechnicalSummary) -> ValidationArtifactResponse:
    return ValidationArtifactResponse(
        artifact_type=value.artifact_type,
        sha256=value.sha256,
        byte_size=value.byte_size,
        declared_media_type=value.declared_media_type,
        detected_format=value.detected_format,
        width=value.width,
        height=value.height,
        channels=value.channels,
        bit_depth=value.bit_depth,
        storage_data_type=value.storage_data_type,
        readability_status=value.readability_status.value,
    )


def map_validation_response(
    execution: ValidationExecutionResult,
    *,
    request_id: str,
) -> InspectionValidationResponse:
    """Map only path-free typed validation fields into the public API contract."""
    result = execution.result
    return InspectionValidationResponse(
        inspection_id=result.inspection_id,
        validation_id=result.validation_id,
        validation_key=execution.validation_key,
        validation_outcome=result.outcome,
        inspection_status=execution.inspection_status,
        policy=ValidationPolicyResponse(
            policy_id=result.validation_policy_id,
            policy_version=result.validation_policy_version,
        ),
        validator_version=result.validator_version,
        started_at=result.started_at,
        completed_at=result.completed_at,
        summary=ValidationSummaryResponse(
            finding_count=result.summary.finding_count,
            info_count=result.summary.info_count,
            warning_count=result.summary.warning_count,
            error_count=result.summary.error_count,
            blocking_count=result.summary.blocking_count,
            technically_ready=result.summary.technically_ready,
            synthetic_example=result.summary.synthetic_example,
        ),
        artifacts=ValidationArtifactsResponse(
            rgb=_artifact(result.rgb_artifact),
            height=_artifact(result.height_artifact),
        ),
        findings=[
            ValidationFindingResponse(
                code=finding.code,
                severity=finding.severity,
                category=finding.category,
                message=finding.message,
                blocking=finding.blocking,
                artifact_type=finding.artifact_type,
                field=finding.field,
                details=finding.details,
            )
            for finding in result.findings
        ],
        idempotent_existing=execution.idempotent_existing,
        request_id=request_id,
    )
