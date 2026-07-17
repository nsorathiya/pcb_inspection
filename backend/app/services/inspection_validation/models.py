from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from app.services.inspection_validation.interfaces import (
    ArtifactTechnicalSummary,
    InspectionValidationResult,
    ValidationFinding,
)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("validation timestamps must include timezone information")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact(value: ArtifactTechnicalSummary) -> dict[str, Any]:
    return {
        "artifact_type": value.artifact_type.value,
        "sha256": value.sha256,
        "byte_size": value.byte_size,
        "declared_media_type": value.declared_media_type,
        "detected_format": value.detected_format,
        "width": value.width,
        "height": value.height,
        "channels": value.channels,
        "bit_depth": value.bit_depth,
        "storage_data_type": value.storage_data_type,
        "readability_status": value.readability_status.value,
    }


def _finding(value: ValidationFinding) -> dict[str, Any]:
    result: dict[str, Any] = {
        "code": value.code,
        "severity": value.severity.value,
        "category": value.category.value,
        "message": value.message,
        "blocking": value.blocking,
    }
    if value.artifact_type is not None:
        result["artifact_type"] = value.artifact_type.value
    if value.field is not None:
        result["field"] = value.field
    if value.details is not None:
        result["details"] = dict(value.details)
    return result


def result_to_dict(value: InspectionValidationResult) -> dict[str, Any]:
    summary = value.summary
    return {
        "contract_version": value.contract_version,
        "validation_id": value.validation_id,
        "inspection_id": value.inspection_id,
        "validation_policy_id": value.validation_policy_id,
        "validation_policy_version": value.validation_policy_version,
        "outcome": value.outcome.value,
        "started_at": _timestamp(value.started_at),
        "completed_at": _timestamp(value.completed_at),
        "validator_version": value.validator_version,
        "rgb_artifact": _artifact(value.rgb_artifact),
        "height_artifact": _artifact(value.height_artifact),
        "findings": [_finding(item) for item in value.findings],
        "summary": {
            "finding_count": summary.finding_count,
            "info_count": summary.info_count,
            "warning_count": summary.warning_count,
            "error_count": summary.error_count,
            "blocking_count": summary.blocking_count,
            "technically_ready": summary.technically_ready,
            "synthetic_example": summary.synthetic_example,
        },
    }


def result_json(value: InspectionValidationResult) -> str:
    return canonical_result_bytes(value).decode("utf-8")


def canonical_result_bytes(value: InspectionValidationResult) -> bytes:
    return json.dumps(
        result_to_dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_result_sha256(value: InspectionValidationResult) -> str:
    return sha256(canonical_result_bytes(value)).hexdigest()
