from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import UUID

from .exceptions import HistoryCursorError, HistoryCursorVersionError
from .models import CursorBoundary, HistoryFilters

CURSOR_CONTRACT_VERSION = "pcb-aoi-inspection-history-cursor/1.0"
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ENCODED_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_CURSOR_KEYS = {
    "contract_version",
    "created_at",
    "filter_digest",
    "inspection_id",
}


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_filter_payload(filters: HistoryFilters) -> dict[str, object]:
    return {
        "board_id": filters.board_id,
        "created_from": canonical_utc(filters.created_from) if filters.created_from else None,
        "created_to": canonical_utc(filters.created_to) if filters.created_to else None,
        "defect_type": filters.defect_type,
        "has_processing": filters.has_processing,
        "has_validation": filters.has_validation,
        "lot_id": filters.lot_id,
        "mock_decision": filters.mock_decision,
        "operator_id": filters.operator_id,
        "processing_status": filters.processing_status.value if filters.processing_status else None,
        "recipe_id": filters.recipe_id,
        "recipe_version": filters.recipe_version,
        "status": filters.status.value if filters.status else None,
        "validation_outcome": filters.validation_outcome.value if filters.validation_outcome else None,
    }


def filter_digest(filters: HistoryFilters) -> str:
    encoded = json.dumps(
        canonical_filter_payload(filters),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_cursor(boundary: CursorBoundary, digest: str) -> str:
    payload = {
        "contract_version": CURSOR_CONTRACT_VERSION,
        "created_at": canonical_utc(boundary.created_at),
        "filter_digest": digest,
        "inspection_id": boundary.inspection_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(value: str) -> tuple[CursorBoundary, str]:
    if not value or len(value) > 2048 or not _ENCODED_PATTERN.fullmatch(value):
        raise HistoryCursorError("The history cursor is malformed")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HistoryCursorError("The history cursor is malformed") from exc
    if not isinstance(payload, dict) or set(payload) != _CURSOR_KEYS:
        raise HistoryCursorError("The history cursor shape is invalid")
    if payload["contract_version"] != CURSOR_CONTRACT_VERSION:
        raise HistoryCursorVersionError("The history cursor version is unsupported")
    created_at_raw = payload["created_at"]
    inspection_id = payload["inspection_id"]
    digest = payload["filter_digest"]
    if not all(isinstance(item, str) for item in (created_at_raw, inspection_id, digest)):
        raise HistoryCursorError("The history cursor contains invalid values")
    if not created_at_raw.endswith("Z"):
        raise HistoryCursorError("The history cursor timestamp must be UTC")
    try:
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        parsed_id = UUID(inspection_id)
    except ValueError as exc:
        raise HistoryCursorError("The history cursor contains invalid values") from exc
    if canonical_utc(created_at) != created_at_raw or str(parsed_id) != inspection_id:
        raise HistoryCursorError("The history cursor is not canonical")
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise HistoryCursorError("The history cursor filter digest is invalid")
    return CursorBoundary(created_at=created_at, inspection_id=inspection_id), digest
