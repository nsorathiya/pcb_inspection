from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.db.models import InspectionStatus
from app.db.processing_types import ProcessingRunStatus


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
    completed_at: datetime | None

