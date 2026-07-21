from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class AuditPosition:
    created_at: datetime
    audit_event_id: str
    inspection_id: str


@dataclass(frozen=True)
class SafeAuditItem:
    audit_event_id: str
    inspection_id: str
    action: str
    created_at: datetime
    actor_id: str | None
    request_id: str | None
    details: Mapping[str, Any]
    details_redacted: bool
    development_only: bool | None
    mock_result: str | None
    production_approved: bool | None


@dataclass(frozen=True)
class AuditPage:
    limit: int
    has_more: bool
    next_cursor: str | None


@dataclass(frozen=True)
class AuditTimeline:
    items: tuple[SafeAuditItem, ...]
    page: AuditPage
