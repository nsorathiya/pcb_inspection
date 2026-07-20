from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from app.db.models import (
    Inspection,
    InspectionInferenceResult,
    InspectionPreprocessingResult,
    InspectionProcessingRun,
    InspectionValidation,
)

from .models import CursorBoundary, HistoryFilters


@dataclass(frozen=True)
class ProcessingEvidence:
    run: InspectionProcessingRun
    preprocessing: InspectionPreprocessingResult | None
    inference: InspectionInferenceResult | None


class InspectionHistoryRepository:
    """Projection-only persistence access for inspection history."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _latest_validation_id() -> object:
        candidate = aliased(InspectionValidation)
        return (
            select(candidate.id)
            .where(candidate.inspection_id == Inspection.id)
            .order_by(
                candidate.completed_at.desc(),
                candidate.created_at.desc(),
                candidate.id.asc(),
            )
            .limit(1)
            .correlate(Inspection)
            .scalar_subquery()
        )

    @staticmethod
    def _latest_processing_id() -> object:
        candidate = aliased(InspectionProcessingRun)
        return (
            select(candidate.id)
            .where(candidate.inspection_id == Inspection.id)
            .order_by(
                candidate.started_at.desc(),
                candidate.created_at.desc(),
                candidate.id.asc(),
            )
            .limit(1)
            .correlate(Inspection)
            .scalar_subquery()
        )

    def _page_statement(
        self,
        filters: HistoryFilters,
        boundary: CursorBoundary | None,
        fetch_limit: int,
    ) -> Select[tuple[Inspection]]:
        latest_validation_id = self._latest_validation_id()
        latest_processing_id = self._latest_processing_id()
        statement = select(Inspection)
        conditions = []
        exact_fields = (
            (Inspection.status, filters.status),
            (Inspection.board_id, filters.board_id),
            (Inspection.recipe_id, filters.recipe_id),
            (Inspection.recipe_version, filters.recipe_version),
            (Inspection.lot_id, filters.lot_id),
            (Inspection.operator_id, filters.operator_id),
        )
        conditions.extend(column == value for column, value in exact_fields if value is not None)
        if filters.created_from is not None:
            conditions.append(Inspection.created_at >= filters.created_from)
        if filters.created_to is not None:
            conditions.append(Inspection.created_at < filters.created_to)
        if filters.validation_outcome is not None:
            conditions.append(
                select(InspectionValidation.outcome)
                .where(InspectionValidation.id == latest_validation_id)
                .scalar_subquery()
                == filters.validation_outcome
            )
        if filters.processing_status is not None:
            conditions.append(
                select(InspectionProcessingRun.status)
                .where(InspectionProcessingRun.id == latest_processing_id)
                .scalar_subquery()
                == filters.processing_status
            )
        if filters.mock_decision is not None:
            conditions.append(
                select(InspectionProcessingRun.final_decision)
                .where(InspectionProcessingRun.id == latest_processing_id)
                .scalar_subquery()
                == filters.mock_decision
            )
        if filters.defect_type is not None:
            conditions.append(
                select(InspectionInferenceResult.defect_type)
                .where(InspectionInferenceResult.processing_run_id == latest_processing_id)
                .scalar_subquery()
                == filters.defect_type
            )
        if filters.has_validation is not None:
            conditions.append(
                latest_validation_id.is_not(None)
                if filters.has_validation
                else latest_validation_id.is_(None)
            )
        if filters.has_processing is not None:
            conditions.append(
                latest_processing_id.is_not(None)
                if filters.has_processing
                else latest_processing_id.is_(None)
            )
        if boundary is not None:
            conditions.append(
                or_(
                    Inspection.created_at < boundary.created_at,
                    and_(
                        Inspection.created_at == boundary.created_at,
                        Inspection.id < boundary.inspection_id,
                    ),
                )
            )
        if conditions:
            statement = statement.where(*conditions)
        return statement.order_by(Inspection.created_at.desc(), Inspection.id.desc()).limit(fetch_limit)

    async def fetch_page(
        self,
        filters: HistoryFilters,
        boundary: CursorBoundary | None,
        fetch_limit: int,
    ) -> list[Inspection]:
        async with self._session_factory() as session:
            result = await session.execute(self._page_statement(filters, boundary, fetch_limit))
            return list(result.scalars().all())

    async def fetch_latest_validations(
        self, inspection_ids: list[str]
    ) -> dict[str, InspectionValidation]:
        if not inspection_ids:
            return {}
        ranked = (
            select(
                InspectionValidation.id.label("validation_id"),
                func.row_number()
                .over(
                    partition_by=InspectionValidation.inspection_id,
                    order_by=(
                        InspectionValidation.completed_at.desc(),
                        InspectionValidation.created_at.desc(),
                        InspectionValidation.id.asc(),
                    ),
                )
                .label("row_number"),
            )
            .where(InspectionValidation.inspection_id.in_(inspection_ids))
            .subquery()
        )
        statement = (
            select(InspectionValidation)
            .join(ranked, ranked.c.validation_id == InspectionValidation.id)
            .where(ranked.c.row_number == 1)
        )
        async with self._session_factory() as session:
            result = await session.execute(statement)
            rows = result.scalars().all()
            return {row.inspection_id: row for row in rows}

    async def fetch_latest_processing(
        self, inspection_ids: list[str]
    ) -> dict[str, ProcessingEvidence]:
        if not inspection_ids:
            return {}
        ranked = (
            select(
                InspectionProcessingRun.id.label("processing_run_id"),
                func.row_number()
                .over(
                    partition_by=InspectionProcessingRun.inspection_id,
                    order_by=(
                        InspectionProcessingRun.started_at.desc(),
                        InspectionProcessingRun.created_at.desc(),
                        InspectionProcessingRun.id.asc(),
                    ),
                )
                .label("row_number"),
            )
            .where(InspectionProcessingRun.inspection_id.in_(inspection_ids))
            .subquery()
        )
        statement = (
            select(
                InspectionProcessingRun,
                InspectionPreprocessingResult,
                InspectionInferenceResult,
            )
            .join(ranked, ranked.c.processing_run_id == InspectionProcessingRun.id)
            .outerjoin(
                InspectionPreprocessingResult,
                InspectionPreprocessingResult.processing_run_id == InspectionProcessingRun.id,
            )
            .outerjoin(
                InspectionInferenceResult,
                InspectionInferenceResult.processing_run_id == InspectionProcessingRun.id,
            )
            .where(ranked.c.row_number == 1)
        )
        async with self._session_factory() as session:
            result = await session.execute(statement)
            return {
                run.inspection_id: ProcessingEvidence(run, preprocessing, inference)
                for run, preprocessing, inference in result.all()
            }
