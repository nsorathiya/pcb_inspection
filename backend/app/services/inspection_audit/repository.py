from __future__ import annotations

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import AuditEvent, Inspection
from app.services.inspection_audit.models import AuditPosition


class InspectionAuditRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def inspection_exists(self, inspection_id: str) -> bool:
        async with self._sessions() as session:
            return await session.scalar(select(Inspection.id).where(Inspection.id == inspection_id)) is not None

    async def list_page(
        self,
        inspection_id: str,
        *,
        limit: int,
        after: AuditPosition | None,
    ) -> list[AuditEvent]:
        statement = select(AuditEvent).where(
            AuditEvent.entity_type == "inspection",
            AuditEvent.entity_id == inspection_id,
        )
        if after is not None:
            statement = statement.where(or_(
                AuditEvent.created_at > after.created_at,
                and_(AuditEvent.created_at == after.created_at, AuditEvent.id > after.audit_event_id),
            ))
        statement = statement.order_by(AuditEvent.created_at.asc(), AuditEvent.id.asc()).limit(limit + 1)
        async with self._sessions() as session:
            return list(await session.scalars(statement))

    async def list_all(self, inspection_id: str) -> list[AuditEvent]:
        statement = select(AuditEvent).where(
            AuditEvent.entity_type == "inspection",
            AuditEvent.entity_id == inspection_id,
        ).order_by(AuditEvent.created_at.asc(), AuditEvent.id.asc())
        async with self._sessions() as session:
            return list(await session.scalars(statement))
