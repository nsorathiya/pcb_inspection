from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.db.models import InspectionStatus
from app.db.processing_types import ProcessingRunStatus
from app.db.validation_types import ValidationOutcome


@dataclass(frozen=True)
class HistoryFilterInput:
    status: str | None = None
    board_id: str | None = None
    recipe_id: str | None = None
    recipe_version: str | None = None
    lot_id: str | None = None
    operator_id: str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    validation_outcome: str | None = None
    processing_status: str | None = None
    mock_decision: str | None = None
    defect_type: str | None = None
    has_validation: bool | None = None
    has_processing: bool | None = None


@dataclass(frozen=True)
class HistoryFilters:
    status: InspectionStatus | None
    board_id: str | None
    recipe_id: str | None
    recipe_version: str | None
    lot_id: str | None
    operator_id: str | None
    created_from: datetime | None
    created_to: datetime | None
    validation_outcome: ValidationOutcome | None
    processing_status: ProcessingRunStatus | None
    mock_decision: str | None
    defect_type: str | None
    has_validation: bool | None
    has_processing: bool | None


@dataclass(frozen=True)
class CursorBoundary:
    created_at: datetime
    inspection_id: str


@dataclass(frozen=True)
class ValidationSummary:
    validation_id: str
    outcome: str
    policy_id: str
    policy_version: str
    validator_version: str
    completed_at: datetime
    total_findings: int
    blocking_findings: int
    warnings: int
    errors: int


@dataclass(frozen=True)
class ProcessingSummary:
    processing_run_id: str
    processing_status: str
    preprocessing_id: str | None
    preprocessing_outcome: str | None
    inference_id: str | None
    inference_execution_outcome: str | None
    mock_decision: str | None
    defect_type: str | None
    started_at: datetime
    completed_at: datetime | None
    synthetic_input: bool
    mock_preprocessing: bool
    mock_inference: bool
    production_approved: bool


@dataclass(frozen=True)
class HistoryItem:
    inspection_id: str
    status: str
    board_id: str
    recipe_id: str
    recipe_version: str
    lot_id: str | None
    operator_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    technical_error_code: str | None
    validation: ValidationSummary | None
    processing: ProcessingSummary | None


@dataclass(frozen=True)
class HistoryResult:
    items: list[HistoryItem]
    limit: int
    next_cursor: str | None
    has_more: bool
    applied_filters: dict[str, str | bool]
