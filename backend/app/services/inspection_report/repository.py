from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.db.models import (
    AuditEvent,
    Inspection,
    InspectionInferenceResult,
    InspectionPreprocessingResult,
    InspectionProcessingRun,
    InspectionValidation,
)


@dataclass(frozen=True)
class InspectionReportRecords:
    inspection: Inspection | None
    validation: InspectionValidation | None
    processing: InspectionProcessingRun | None
    audits: tuple[AuditEvent, ...]


class InspectionReportRepository:
    """Load one report snapshot with a fixed number of relationship queries."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def load(self, inspection_id: str) -> InspectionReportRecords:
        async with self._sessions() as session:
            inspection = await session.scalar(
                select(Inspection)
                .where(Inspection.id == inspection_id)
                .options(selectinload(Inspection.artifacts))
            )
            if inspection is None:
                return InspectionReportRecords(None, None, None, ())
            validation = await session.scalar(
                select(InspectionValidation)
                .where(InspectionValidation.inspection_id == inspection_id)
                .order_by(
                    InspectionValidation.completed_at.desc(),
                    InspectionValidation.created_at.desc(),
                    InspectionValidation.id.asc(),
                )
                .limit(1)
                .options(selectinload(InspectionValidation.findings))
            )
            processing = await session.scalar(
                select(InspectionProcessingRun)
                .where(InspectionProcessingRun.inspection_id == inspection_id)
                .order_by(
                    InspectionProcessingRun.started_at.desc(),
                    InspectionProcessingRun.created_at.desc(),
                    InspectionProcessingRun.id.asc(),
                )
                .limit(1)
                .options(
                    selectinload(InspectionProcessingRun.preprocessing_result).selectinload(
                        InspectionPreprocessingResult.findings
                    ),
                    selectinload(InspectionProcessingRun.inference_result).selectinload(
                        InspectionInferenceResult.findings
                    ),
                )
            )
            audits = tuple(await session.scalars(
                select(AuditEvent).where(
                    AuditEvent.entity_type == "inspection",
                    AuditEvent.entity_id == inspection_id,
                ).order_by(AuditEvent.created_at.asc(), AuditEvent.id.asc())
            ))
        return InspectionReportRecords(inspection, validation, processing, audits)
