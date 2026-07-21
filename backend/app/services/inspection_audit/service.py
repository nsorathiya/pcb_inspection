from __future__ import annotations

from app.services.inspection_audit.cursor import decode_cursor, encode_cursor
from app.services.inspection_audit.exceptions import (
    AuditCursorError,
    AuditProjectionError,
    AuditRetrievalError,
)
from app.services.inspection_audit.models import AuditPage, AuditPosition, AuditTimeline
from app.services.inspection_audit.projection import project_audit_event
from app.services.inspection_audit.repository import InspectionAuditRepository


class InspectionAuditService:
    def __init__(self, repository: InspectionAuditRepository) -> None:
        self._repository = repository

    async def inspection_exists(self, inspection_id: str) -> bool:
        try:
            return await self._repository.inspection_exists(inspection_id)
        except Exception as exc:
            raise AuditRetrievalError("inspection existence could not be checked") from exc

    async def get_timeline(self, inspection_id: str, *, limit: int = 50, cursor: str | None = None) -> AuditTimeline:
        if not 1 <= limit <= 200:
            raise AuditCursorError("audit limit is outside the supported range")
        after = None if cursor is None else decode_cursor(cursor, inspection_id)
        try:
            if not await self._repository.inspection_exists(inspection_id):
                raise LookupError("inspection does not exist")
            records = await self._repository.list_page(inspection_id, limit=limit, after=after)
        except LookupError:
            raise
        except Exception as exc:
            raise AuditRetrievalError("audit timeline could not be retrieved") from exc
        has_more = len(records) > limit
        records = records[:limit]
        try:
            items = tuple(project_audit_event(record, inspection_id) for record in records)
        except AuditProjectionError:
            raise
        except Exception as exc:
            raise AuditProjectionError("audit timeline could not be projected") from exc
        next_cursor = None
        if has_more and records:
            last = records[-1]
            next_cursor = encode_cursor(AuditPosition(last.created_at, last.id, inspection_id))
        return AuditTimeline(items, AuditPage(limit, has_more, next_cursor))

    async def project_all(self, inspection_id: str):
        try:
            records = await self._repository.list_all(inspection_id)
            return tuple(project_audit_event(record, inspection_id) for record in records)
        except AuditProjectionError:
            raise
        except Exception as exc:
            raise AuditRetrievalError("audit timeline could not be retrieved") from exc
