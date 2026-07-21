from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime, timezone
from uuid import UUID

from app.services.inspection_audit.exceptions import (
    AuditCursorError,
    AuditCursorInspectionMismatchError,
    AuditCursorVersionError,
)
from app.services.inspection_audit.models import AuditPosition

CURSOR_VERSION = 1


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise AuditCursorError("audit cursor UUID is invalid")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise AuditCursorError("audit cursor UUID is invalid") from exc
    if str(parsed) != value:
        raise AuditCursorError("audit cursor UUID is not canonical")
    return value


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        # SQLite returns persisted timezone-aware values as naive datetimes.
        # Database timestamps are defined as UTC throughout this application.
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def encode_cursor(position: AuditPosition) -> str:
    payload = {
        "created_at": _timestamp(position.created_at),
        "event_id": _canonical_uuid(position.audit_event_id),
        "inspection_id": _canonical_uuid(position.inspection_id),
        "version": CURSOR_VERSION,
    }
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(value: str, inspection_id: str) -> AuditPosition:
    if not value or not isinstance(value, str):
        raise AuditCursorError("audit cursor is empty")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AuditCursorError("audit cursor is malformed") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "version", "created_at", "event_id", "inspection_id"
    }:
        raise AuditCursorError("audit cursor fields are invalid")
    if payload["version"] != CURSOR_VERSION:
        raise AuditCursorVersionError("audit cursor version is unsupported")
    event_id = _canonical_uuid(payload["event_id"])
    cursor_inspection_id = _canonical_uuid(payload["inspection_id"])
    if cursor_inspection_id != inspection_id:
        raise AuditCursorInspectionMismatchError("audit cursor belongs to another inspection")
    timestamp = payload["created_at"]
    if not isinstance(timestamp, str):
        raise AuditCursorError("audit cursor timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditCursorError("audit cursor timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AuditCursorError("audit cursor timestamp has no timezone")
    return AuditPosition(parsed.astimezone(timezone.utc), event_id, cursor_inspection_id)
