from __future__ import annotations

import json
import re
from datetime import timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from app.db.models import AuditEvent
from app.services.audit_actions import (
    INSPECTION_INTAKE_FAILED,
    INSPECTION_MOCK_RESULT_FAIL,
    INSPECTION_MOCK_RESULT_PASS,
    INSPECTION_MOCK_RESULT_UNCERTAIN,
    INSPECTION_PROCESSING_ERROR,
    INSPECTION_PROCESSING_STARTED,
    INSPECTION_RECEIVED,
    MOCK_RESULT_BY_ACTION,
    PROCESSING_ACTIONS,
    VALIDATION_ACTIONS,
)
from app.services.inspection_audit.exceptions import AuditProjectionError
from app.services.inspection_audit.models import SafeAuditItem

_UNSAFE_KEY = re.compile(r"(?:path|filename|file_name|root|manifest|traceback|stack|exception|sql|content|bytes)", re.I)
_ALLOWLISTS = {
    INSPECTION_RECEIVED: frozenset({"artifact_types", "byte_sizes", "station_id"}),
    INSPECTION_INTAKE_FAILED: frozenset({"failure_category", "compensation_complete"}),
    **{action: frozenset({
        "validation_id", "validation_key", "validation_outcome", "policy_id",
        "policy_version", "validator_version", "result_sha256", "finding_count",
        "blocking_finding_count", "warning_count", "inspection_status",
    }) for action in VALIDATION_ACTIONS},
    INSPECTION_PROCESSING_STARTED: frozenset({
        "processing_run_id", "processing_key", "validation_id",
        "preprocessing_policy_id", "preprocessing_policy_version",
        "preprocessing_implementation_id", "preprocessing_implementation_version",
        "inference_policy_id", "inference_policy_version", "engine_id",
        "engine_version", "engine_type",
    }),
    **{action: frozenset({
        "processing_run_id", "processing_key", "preprocessing_id", "inference_id",
        "preprocessing_outcome", "inference_execution_outcome",
        "preprocessing_result_sha256", "inference_result_sha256",
        "preprocessing_finding_count", "preprocessing_blocking_finding_count",
        "preprocessing_warning_count", "inference_finding_count",
        "inference_blocking_finding_count", "inference_warning_count",
        "final_inspection_status", "preprocessing_policy_id",
        "preprocessing_policy_version", "preprocessing_implementation_id",
        "preprocessing_implementation_version", "inference_policy_id",
        "inference_policy_version", "engine_id", "engine_version", "engine_type",
        "mock_inference", "production_approved",
    }) for action in (
        INSPECTION_MOCK_RESULT_PASS, INSPECTION_MOCK_RESULT_FAIL,
        INSPECTION_MOCK_RESULT_UNCERTAIN, INSPECTION_PROCESSING_ERROR,
    )},
}


def _unsafe_value(value: Any, key: str) -> bool:
    if _UNSAFE_KEY.search(key):
        return True
    if isinstance(value, Mapping):
        return any(_unsafe_value(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_unsafe_value(child, key) for child in value)
    if isinstance(value, str):
        return (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or "\\" in value
            or ".." in PurePosixPath(value).parts
            or "traceback" in value.lower()
            or "select " in value.lower()
        )
    return not isinstance(value, (str, int, float, bool, type(None)))


def _safe_details(action: str, details_json: str) -> tuple[dict[str, Any], bool]:
    try:
        raw = json.loads(details_json)
    except (TypeError, json.JSONDecodeError):
        return {}, True
    if not isinstance(raw, dict):
        return {}, True
    allowlist = _ALLOWLISTS.get(action)
    if allowlist is None:
        return {}, bool(raw)
    safe: dict[str, Any] = {}
    redacted = False
    for key, value in raw.items():
        if key not in allowlist or _unsafe_value(value, key):
            redacted = True
            continue
        # Current intake aggregate values are useful, but nested structures get
        # an explicit scalar-only projection instead of a blind pass-through.
        if key == "artifact_types" and isinstance(value, list) and all(isinstance(item, str) for item in value):
            safe[key] = list(value)
        elif key == "byte_sizes" and isinstance(value, dict) and all(
            isinstance(child_key, str) and isinstance(child, int) and not isinstance(child, bool)
            for child_key, child in value.items()
        ):
            safe[key] = dict(sorted(value.items()))
        elif isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        else:
            redacted = True
    return safe, redacted


def project_audit_event(record: AuditEvent, inspection_id: str) -> SafeAuditItem:
    if record.entity_type != "inspection" or record.entity_id != inspection_id:
        raise AuditProjectionError("audit event ownership is inconsistent")
    details, redacted = _safe_details(record.action, record.details_json)
    development_only = (
        True
        if record.action in PROCESSING_ACTIONS
        or (
            record.action in VALIDATION_ACTIONS
            and details.get("policy_id") == "development-native-rgb-height"
        )
        else None
    )
    production_approved = False if record.action in PROCESSING_ACTIONS else None
    created_at = record.created_at
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = created_at.astimezone(timezone.utc)
    return SafeAuditItem(
        audit_event_id=record.id,
        inspection_id=inspection_id,
        action=record.action,
        created_at=created_at,
        actor_id=record.actor_id,
        request_id=record.request_id,
        details=details,
        details_redacted=redacted,
        development_only=development_only,
        mock_result=MOCK_RESULT_BY_ACTION.get(record.action),
        production_approved=production_approved,
    )
