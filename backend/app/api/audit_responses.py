from datetime import datetime

from pydantic import BaseModel, JsonValue

from app.services.inspection_audit.models import AuditTimeline


class AuditItemResponse(BaseModel):
    audit_event_id: str
    inspection_id: str
    action: str
    created_at: datetime
    actor_id: str | None
    request_id: str | None
    details: dict[str, JsonValue]
    details_redacted: bool
    development_only: bool | None
    mock_result: str | None
    production_approved: bool | None


class AuditPageResponse(BaseModel):
    limit: int
    has_more: bool
    next_cursor: str | None


class AuditTimelineResponse(BaseModel):
    items: list[AuditItemResponse]
    page: AuditPageResponse
    request_id: str


def map_audit_timeline(value: AuditTimeline, request_id: str) -> AuditTimelineResponse:
    return AuditTimelineResponse(
        items=[AuditItemResponse(
            audit_event_id=item.audit_event_id,
            inspection_id=item.inspection_id,
            action=item.action,
            created_at=item.created_at,
            actor_id=item.actor_id,
            request_id=item.request_id,
            details=dict(item.details),
            details_redacted=item.details_redacted,
            development_only=item.development_only,
            mock_result=item.mock_result,
            production_approved=item.production_approved,
        ) for item in value.items],
        page=AuditPageResponse(
            limit=value.page.limit,
            has_more=value.page.has_more,
            next_cursor=value.page.next_cursor,
        ),
        request_id=request_id,
    )
