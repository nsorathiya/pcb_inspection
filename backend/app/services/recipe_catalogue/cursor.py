from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import UUID

from .exceptions import (
    RecipeCatalogueCursorError,
    RecipeCatalogueCursorVersionError,
)
from .models import RecipeCatalogueCursorBoundary, RecipeCatalogueFilters

CURSOR_CONTRACT_VERSION = "pcb-aoi-recipe-catalogue-cursor/1.0"
_CURSOR_KEYS = {
    "contract_version",
    "created_at",
    "filter_digest",
    "recipe_row_id",
}
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ENCODED_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_filter_payload(
    filters: RecipeCatalogueFilters,
) -> dict[str, str | None]:
    return {
        "name": filters.name,
        "recipe_id": filters.recipe_id,
        "recipe_version": filters.recipe_version,
        "status": filters.status.value if filters.status else None,
    }


def filter_digest(filters: RecipeCatalogueFilters) -> str:
    encoded = json.dumps(
        canonical_filter_payload(filters),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_cursor(
    boundary: RecipeCatalogueCursorBoundary,
    digest: str,
) -> str:
    try:
        canonical_id = str(UUID(boundary.row_id))
    except ValueError as exc:
        raise RecipeCatalogueCursorError("The recipe cursor row ID is invalid") from exc
    if canonical_id != boundary.row_id or not _DIGEST_PATTERN.fullmatch(digest):
        raise RecipeCatalogueCursorError("The recipe cursor boundary is invalid")
    payload = {
        "contract_version": CURSOR_CONTRACT_VERSION,
        "created_at": canonical_utc(boundary.created_at),
        "filter_digest": digest,
        "recipe_row_id": boundary.row_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(
    value: str,
) -> tuple[RecipeCatalogueCursorBoundary, str]:
    if not value or len(value) > 2048 or not _ENCODED_PATTERN.fullmatch(value):
        raise RecipeCatalogueCursorError("The recipe cursor is malformed")
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecipeCatalogueCursorError("The recipe cursor is malformed") from exc
    if not isinstance(payload, dict) or set(payload) != _CURSOR_KEYS:
        raise RecipeCatalogueCursorError("The recipe cursor shape is invalid")
    if payload["contract_version"] != CURSOR_CONTRACT_VERSION:
        raise RecipeCatalogueCursorVersionError(
            "The recipe cursor version is unsupported"
        )
    created_at_raw = payload["created_at"]
    row_id = payload["recipe_row_id"]
    digest = payload["filter_digest"]
    if not all(isinstance(item, str) for item in (created_at_raw, row_id, digest)):
        raise RecipeCatalogueCursorError("The recipe cursor contains invalid values")
    if not created_at_raw.endswith("Z"):
        raise RecipeCatalogueCursorError("The recipe cursor timestamp must be UTC")
    try:
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        parsed_id = UUID(row_id)
    except ValueError as exc:
        raise RecipeCatalogueCursorError(
            "The recipe cursor contains invalid values"
        ) from exc
    if canonical_utc(created_at) != created_at_raw or str(parsed_id) != row_id:
        raise RecipeCatalogueCursorError("The recipe cursor is not canonical")
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise RecipeCatalogueCursorError("The recipe cursor digest is invalid")
    return RecipeCatalogueCursorBoundary(created_at, row_id), digest
