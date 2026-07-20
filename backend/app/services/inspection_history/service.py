from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Inspection, InspectionStatus, InspectionValidation
from app.db.processing_types import (
    PersistedInferenceOutcome,
    PersistedPreprocessingOutcome,
    ProcessingFinalDecision,
    ProcessingRunStatus,
)
from app.db.validation_types import ValidationOutcome
from app.services.inspection_inference.mock_engine import DEFAULT_TAXONOMY_PATH

from .cursor import (
    canonical_filter_payload,
    decode_cursor,
    encode_cursor,
    filter_digest,
)
from .exceptions import (
    HistoryConsistencyError,
    HistoryCursorFilterMismatchError,
    HistoryFilterError,
    HistoryRetrievalError,
)
from .models import (
    CursorBoundary,
    HistoryFilterInput,
    HistoryFilters,
    HistoryItem,
    HistoryResult,
    ProcessingSummary,
    ValidationSummary,
)
from .repository import InspectionHistoryRepository, ProcessingEvidence

_SAFE_PROCESSING_ERRORS = frozenset(
    {
        "PREPROCESSING_FAILED",
        "PREPROCESSING_ERROR",
        "INFERENCE_FAILED",
        "INFERENCE_ERROR",
    }
)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class InspectionHistoryService:
    """Read-only, bounded assembly of safe inspection-history projections."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._repository = InspectionHistoryRepository(session_factory)
        try:
            taxonomy = json.loads(DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))
            supported = taxonomy["$defs"]["supported_defect_type"]["enum"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("authoritative defect taxonomy is invalid") from exc
        if (
            not isinstance(supported, list)
            or not supported
            or any(not isinstance(item, str) or not item for item in supported)
            or len(supported) != len(set(supported))
        ):
            raise ValueError("authoritative defect taxonomy is invalid")
        self._supported_defect_types = frozenset(supported)

    @property
    def repository(self) -> InspectionHistoryRepository:
        return self._repository

    @staticmethod
    def _metadata(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            not normalized
            or len(normalized) > 128
            or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
        ):
            raise HistoryFilterError(f"{field} is invalid")
        return normalized

    @staticmethod
    def _enum(enum_type: type, value: str | None, field: str) -> Any:
        if value is None:
            return None
        try:
            return enum_type(value.strip().upper())
        except (ValueError, AttributeError) as exc:
            raise HistoryFilterError(f"{field} is unsupported") from exc

    def normalize_filters(self, source: HistoryFilterInput) -> HistoryFilters:
        created_from = source.created_from
        created_to = source.created_to
        for value, field in ((created_from, "created_from"), (created_to, "created_to")):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise HistoryFilterError(f"{field} must include a UTC offset")
        created_from = _as_utc(created_from) if created_from else None
        created_to = _as_utc(created_to) if created_to else None
        if created_from is not None and created_to is not None and created_from >= created_to:
            raise HistoryFilterError("created_from must be earlier than created_to")
        defect_type = self._metadata(source.defect_type, "defect_type")
        if defect_type is not None and defect_type not in self._supported_defect_types:
            raise HistoryFilterError("defect_type is unsupported")
        mock_decision = self._enum(
            ProcessingFinalDecision, source.mock_decision, "mock_decision"
        )
        return HistoryFilters(
            status=self._enum(InspectionStatus, source.status, "status"),
            board_id=self._metadata(source.board_id, "board_id"),
            recipe_id=self._metadata(source.recipe_id, "recipe_id"),
            recipe_version=self._metadata(source.recipe_version, "recipe_version"),
            lot_id=self._metadata(source.lot_id, "lot_id"),
            operator_id=self._metadata(source.operator_id, "operator_id"),
            created_from=created_from,
            created_to=created_to,
            validation_outcome=self._enum(
                ValidationOutcome, source.validation_outcome, "validation_outcome"
            ),
            processing_status=self._enum(
                ProcessingRunStatus, source.processing_status, "processing_status"
            ),
            mock_decision=mock_decision.value if mock_decision else None,
            defect_type=defect_type,
            has_validation=source.has_validation,
            has_processing=source.has_processing,
        )

    async def list_history(
        self,
        *,
        limit: int,
        cursor: str | None,
        filters: HistoryFilterInput,
    ) -> HistoryResult:
        normalized = self.normalize_filters(filters)
        digest = filter_digest(normalized)
        boundary: CursorBoundary | None = None
        if cursor is not None:
            boundary, cursor_digest = decode_cursor(cursor)
            if cursor_digest != digest:
                raise HistoryCursorFilterMismatchError(
                    "The history cursor does not match the current filters"
                )
        try:
            rows = await self._repository.fetch_page(normalized, boundary, limit + 1)
            has_more = len(rows) > limit
            page_rows = rows[:limit]
            identifiers = [row.id for row in page_rows]
            validations = await self._repository.fetch_latest_validations(identifiers)
            processing = await self._repository.fetch_latest_processing(identifiers)
        except SQLAlchemyError as exc:
            raise HistoryRetrievalError("Inspection history could not be read") from exc

        items = [
            self._assemble(row, validations.get(row.id), processing.get(row.id))
            for row in page_rows
        ]
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_cursor(
                CursorBoundary(_as_utc(last.created_at), last.id), digest
            )
        applied = {
            key: value
            for key, value in canonical_filter_payload(normalized).items()
            if value is not None
        }
        return HistoryResult(
            items=items,
            limit=limit,
            next_cursor=next_cursor,
            has_more=has_more,
            applied_filters=applied,
        )

    @staticmethod
    def _summary(record: InspectionValidation) -> ValidationSummary:
        try:
            document = json.loads(record.summary_json)
            counts = {
                "total_findings": document["finding_count"],
                "blocking_findings": document["blocking_count"],
                "warnings": document["warning_count"],
                "errors": document["error_count"],
            }
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise HistoryConsistencyError("Validation summary is inconsistent") from exc
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise HistoryConsistencyError("Validation summary counts are inconsistent")
        return ValidationSummary(
            validation_id=record.id,
            outcome=record.outcome.value,
            policy_id=record.policy_id,
            policy_version=record.policy_version,
            validator_version=record.validator_version,
            completed_at=_as_utc(record.completed_at),
            **counts,
        )

    def _processing_summary(
        self,
        inspection: Inspection,
        validation: InspectionValidation,
        evidence: ProcessingEvidence,
    ) -> ProcessingSummary:
        run, preprocessing, inference = (
            evidence.run,
            evidence.preprocessing,
            evidence.inference,
        )
        if run.validation_id != validation.id or validation.outcome is not ValidationOutcome.VALIDATION_PASSED:
            raise HistoryConsistencyError("Processing does not reference the latest passed validation")
        if run.status is ProcessingRunStatus.STARTED:
            if (
                inspection.status is not InspectionStatus.PROCESSING
                or preprocessing is not None
                or inference is not None
                or run.completed_at is not None
            ):
                raise HistoryConsistencyError("Started processing state is inconsistent")
        elif run.status is ProcessingRunStatus.COMPLETED:
            if (
                run.final_decision is None
                or inspection.status.value != run.final_decision.value
                or run.completed_at is None
                or preprocessing is None
                or preprocessing.outcome is not PersistedPreprocessingOutcome.SUCCEEDED
                or inference is None
                or inference.execution_outcome is not PersistedInferenceOutcome.SUCCEEDED
                or inference.decision != run.final_decision
                or inference.preprocessing_id != preprocessing.id
            ):
                raise HistoryConsistencyError("Completed processing state is inconsistent")
            if run.final_decision is ProcessingFinalDecision.FAIL:
                defect_is_valid = (
                    inference.defect_type is not None
                    and inference.defect_type in self._supported_defect_types
                )
            else:
                defect_is_valid = inference.defect_type is None
            if not defect_is_valid:
                raise HistoryConsistencyError("Persisted defect type is unsupported")
        elif run.status is ProcessingRunStatus.ERROR:
            pre_succeeded = (
                preprocessing is not None
                and preprocessing.outcome is PersistedPreprocessingOutcome.SUCCEEDED
            )
            if (
                inspection.status is not InspectionStatus.ERROR
                or run.error_code not in _SAFE_PROCESSING_ERRORS
                or run.final_decision is not None
                or run.completed_at is None
                or preprocessing is None
                or (pre_succeeded and inference is None)
                or (not pre_succeeded and inference is not None)
                or (
                    inference is not None
                    and inference.execution_outcome is PersistedInferenceOutcome.SUCCEEDED
                )
                or (inference is not None and inference.defect_type is not None)
            ):
                raise HistoryConsistencyError("Errored processing state is inconsistent")
        else:
            raise HistoryConsistencyError("Processing status is unsupported")
        return ProcessingSummary(
            processing_run_id=run.id,
            processing_status=run.status.value,
            preprocessing_id=preprocessing.id if preprocessing else None,
            preprocessing_outcome=(
                preprocessing.outcome.value if preprocessing else None
            ),
            inference_id=inference.id if inference else None,
            inference_execution_outcome=(
                inference.execution_outcome.value if inference else None
            ),
            mock_decision=run.final_decision.value if run.final_decision else None,
            defect_type=inference.defect_type if inference else None,
            started_at=_as_utc(run.started_at),
            completed_at=_as_utc(run.completed_at) if run.completed_at else None,
            synthetic_input=True,
            mock_preprocessing=preprocessing is not None,
            mock_inference=inference is not None,
            production_approved=False,
        )

    def _assemble(
        self,
        inspection: Inspection,
        validation: InspectionValidation | None,
        processing: ProcessingEvidence | None,
    ) -> HistoryItem:
        if validation is None:
            if processing is not None or inspection.status not in {
                InspectionStatus.RECEIVED,
                InspectionStatus.ERROR,
            }:
                raise HistoryConsistencyError("Inspection lifecycle is inconsistent")
        elif validation.outcome is ValidationOutcome.VALIDATION_FAILED:
            if inspection.status is not InspectionStatus.VALIDATION_FAILED or processing is not None:
                raise HistoryConsistencyError("Failed validation lifecycle is inconsistent")
        elif validation.outcome is ValidationOutcome.VALIDATION_ERROR:
            if inspection.status is not InspectionStatus.ERROR or processing is not None:
                raise HistoryConsistencyError("Errored validation lifecycle is inconsistent")
        elif processing is None and inspection.status is not InspectionStatus.READY:
            raise HistoryConsistencyError("Passed validation lifecycle is inconsistent")

        validation_summary = self._summary(validation) if validation else None
        processing_summary = (
            self._processing_summary(inspection, validation, processing)
            if validation is not None and processing is not None
            else None
        )
        technical_error_code = None
        if processing is not None and processing.run.status is ProcessingRunStatus.ERROR:
            technical_error_code = processing.run.error_code
        elif validation is not None and validation.outcome is ValidationOutcome.VALIDATION_FAILED:
            technical_error_code = "INPUT_VALIDATION_FAILED"
        elif validation is not None and validation.outcome is ValidationOutcome.VALIDATION_ERROR:
            technical_error_code = "VALIDATOR_INTERNAL_ERROR"
        elif inspection.status is InspectionStatus.ERROR:
            technical_error_code = (
                inspection.error_code
                if inspection.error_code == "INSPECTION_INTAKE_FAILED"
                else "INSPECTION_ERROR"
            )
        return HistoryItem(
            inspection_id=inspection.id,
            status=inspection.status.value,
            board_id=inspection.board_id,
            recipe_id=inspection.recipe_id,
            recipe_version=inspection.recipe_version,
            lot_id=inspection.lot_id,
            operator_id=inspection.operator_id,
            created_at=_as_utc(inspection.created_at),
            started_at=_as_utc(inspection.started_at) if inspection.started_at else None,
            completed_at=_as_utc(inspection.completed_at) if inspection.completed_at else None,
            technical_error_code=technical_error_code,
            validation=validation_summary,
            processing=processing_summary,
        )
