from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from app.db.models import InspectionStatus
from app.db.processing_types import ProcessingRunStatus


@dataclass(frozen=True)
class ProcessingFindingResult:
    code: str
    severity: str
    category: str
    message: str
    blocking: bool
    branch: str | None
    field: str | None
    details: Mapping[str, Any]


@dataclass(frozen=True)
class ProcessingSummaryResult:
    total_findings: int
    blocking_findings: int
    warnings: int
    errors: int


@dataclass(frozen=True)
class PreprocessingEvidenceResult:
    preprocessing_id: str
    policy_id: str
    policy_version: str
    implementation_id: str
    implementation_version: str
    outcome: str
    summary: ProcessingSummaryResult
    findings: tuple[ProcessingFindingResult, ...]


@dataclass(frozen=True)
class InferenceEvidenceResult:
    inference_id: str
    policy_id: str
    policy_version: str
    engine_id: str
    engine_version: str
    engine_type: str
    execution_outcome: str
    decision: str | None
    defect_type: str | None
    summary: ProcessingSummaryResult
    findings: tuple[ProcessingFindingResult, ...]


@dataclass(frozen=True)
class ProcessingExecutionResult:
    """Path-free internal result for synthetic mock processing coordination."""

    inspection_id: str
    validation_id: str
    processing_run_id: str
    processing_key: str
    preprocessing_id: str | None
    inference_id: str | None
    preprocessing_outcome: str | None
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
    preprocessing: PreprocessingEvidenceResult
    inference: InferenceEvidenceResult | None
