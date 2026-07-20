from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from app.db.models import InspectionStatus
from app.db.processing_types import ProcessingRunStatus
from app.services.inspection_processing.execution_models import ProcessingExecutionResult


class ProcessingFindingResponse(BaseModel):
    code: str
    severity: str
    category: str
    message: str
    blocking: bool
    branch: str | None
    field: str | None
    details: Mapping[str, Any]


class ProcessingSummaryResponse(BaseModel):
    total_findings: int
    blocking_findings: int
    warnings: int
    errors: int


class PreprocessingEvidenceResponse(BaseModel):
    preprocessing_id: str
    policy_id: str
    policy_version: str
    implementation_id: str
    implementation_version: str
    outcome: str
    summary: ProcessingSummaryResponse
    findings: list[ProcessingFindingResponse]


class InferenceEvidenceResponse(BaseModel):
    inference_id: str
    policy_id: str
    policy_version: str
    engine_id: str
    engine_version: str
    engine_type: str
    execution_outcome: str
    decision: str | None
    defect_type: str | None
    summary: ProcessingSummaryResponse
    findings: list[ProcessingFindingResponse]


class InspectionProcessingResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Development-only persisted synthetic workflow evidence. Mock decisions "
                "are deterministic demonstration values, not image-based predictions or "
                "production PCB dispositions. Confidence is unavailable."
            )
        }
    )

    inspection_id: str
    validation_id: str
    processing_run_id: str
    processing_key: str
    preprocessing_id: str
    inference_id: str | None
    preprocessing_outcome: str
    inference_execution_outcome: str | None
    mock_decision: str | None
    defect_type: str | None
    inspection_status: InspectionStatus
    processing_status: ProcessingRunStatus
    synthetic_input_verified: bool
    mock_preprocessing: bool
    mock_inference: bool
    production_approved: bool
    lifecycle_idempotent_existing: bool
    execution_started_now: bool
    started_at: datetime
    completed_at: datetime | None
    preprocessing: PreprocessingEvidenceResponse
    inference: InferenceEvidenceResponse | None
    request_id: str


def map_processing_response(
    execution: ProcessingExecutionResult,
    *,
    request_id: str,
) -> InspectionProcessingResponse:
    """Map only validated path-free persisted processing evidence."""
    return InspectionProcessingResponse(
        **asdict(execution),
        request_id=request_id,
    )
