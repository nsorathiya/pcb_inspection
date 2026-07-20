from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.services.inspection_history.models import HistoryResult


class HistoryValidationSummaryResponse(BaseModel):
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


class HistoryProcessingSummaryResponse(BaseModel):
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


class HistoryRecipeResponse(BaseModel):
    recipe_id: str
    recipe_version: str


class InspectionHistoryItemResponse(BaseModel):
    inspection_id: str
    status: str
    board_id: str
    recipe: HistoryRecipeResponse
    lot_id: str | None
    operator_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    technical_error_code: str | None
    validation: HistoryValidationSummaryResponse | None
    processing: HistoryProcessingSummaryResponse | None


class HistoryPageResponse(BaseModel):
    limit: int
    has_more: bool
    next_cursor: str | None


class InspectionHistoryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "A read-only, cursor-paginated projection of persisted inspection "
                "history. Mock processing fields are non-production workflow evidence."
            )
        }
    )

    items: list[InspectionHistoryItemResponse]
    page: HistoryPageResponse
    applied_filters: dict[str, str | bool]
    request_id: str


def map_history_response(
    result: HistoryResult, *, request_id: str
) -> InspectionHistoryResponse:
    def history_item(item) -> InspectionHistoryItemResponse:
        document = asdict(item)
        recipe_id = document.pop("recipe_id")
        recipe_version = document.pop("recipe_version")
        return InspectionHistoryItemResponse(
            **document,
            recipe={"recipe_id": recipe_id, "recipe_version": recipe_version},
        )

    return InspectionHistoryResponse(
        items=[history_item(item) for item in result.items],
        page=HistoryPageResponse(
            limit=result.limit,
            has_more=result.has_more,
            next_cursor=result.next_cursor,
        ),
        applied_filters=result.applied_filters,
        request_id=request_id,
    )
