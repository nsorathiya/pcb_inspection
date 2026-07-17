from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping

from app.db.models import ArtifactType
from app.db.validation_types import FindingCategory, FindingSeverity, ValidationOutcome

from app.services.inspection_validation.interfaces import (
    ArtifactTechnicalSummary,
    InspectionValidationResult,
    ReadabilityStatus,
    ValidationFinding,
    ValidationSummary,
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


def _exact_object(
    value: Any,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("persisted validation value must be an object")
    keys = set(value)
    if not required.issubset(keys) or keys.difference(required | optional):
        raise ValueError("persisted validation object fields are inconsistent")
    return value


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("persisted validation timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("persisted validation timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("persisted validation timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _artifact_from_dict(value: Any) -> ArtifactTechnicalSummary:
    fields = {
        "artifact_type",
        "sha256",
        "byte_size",
        "declared_media_type",
        "detected_format",
        "width",
        "height",
        "channels",
        "bit_depth",
        "storage_data_type",
        "readability_status",
    }
    item = _exact_object(value, required=fields)
    return ArtifactTechnicalSummary(
        artifact_type=ArtifactType(item["artifact_type"]),
        sha256=item["sha256"],
        byte_size=item["byte_size"],
        declared_media_type=item["declared_media_type"],
        detected_format=item["detected_format"],
        width=item["width"],
        height=item["height"],
        channels=item["channels"],
        bit_depth=item["bit_depth"],
        storage_data_type=item["storage_data_type"],
        readability_status=ReadabilityStatus(item["readability_status"]),
    )


def _finding_from_dict(value: Any) -> ValidationFinding:
    required = {"code", "severity", "category", "message", "blocking"}
    optional = {"artifact_type", "field", "details"}
    item = _exact_object(value, required=required, optional=optional)
    details = item.get("details")
    if details is not None and not isinstance(details, Mapping):
        raise ValueError("persisted validation finding details are invalid")
    return ValidationFinding(
        code=item["code"],
        severity=FindingSeverity(item["severity"]),
        category=FindingCategory(item["category"]),
        message=item["message"],
        blocking=item["blocking"],
        artifact_type=(
            None
            if item.get("artifact_type") is None
            else ArtifactType(item["artifact_type"])
        ),
        field=item.get("field"),
        details=None if details is None else dict(details),
    )


def result_from_dict(value: Mapping[str, Any]) -> InspectionValidationResult:
    """Reconstruct the authoritative typed result from canonical persistence JSON."""
    fields = {
        "contract_version",
        "validation_id",
        "inspection_id",
        "validation_policy_id",
        "validation_policy_version",
        "outcome",
        "started_at",
        "completed_at",
        "validator_version",
        "rgb_artifact",
        "height_artifact",
        "findings",
        "summary",
    }
    item = _exact_object(value, required=fields)
    findings_value = item["findings"]
    if not isinstance(findings_value, list):
        raise ValueError("persisted validation findings are invalid")
    summary_fields = {
        "finding_count",
        "info_count",
        "warning_count",
        "error_count",
        "blocking_count",
        "technically_ready",
        "synthetic_example",
    }
    summary = _exact_object(item["summary"], required=summary_fields)
    return InspectionValidationResult(
        contract_version=item["contract_version"],
        validation_id=item["validation_id"],
        inspection_id=item["inspection_id"],
        validation_policy_id=item["validation_policy_id"],
        validation_policy_version=item["validation_policy_version"],
        outcome=ValidationOutcome(item["outcome"]),
        started_at=_parse_timestamp(item["started_at"]),
        completed_at=_parse_timestamp(item["completed_at"]),
        validator_version=item["validator_version"],
        rgb_artifact=_artifact_from_dict(item["rgb_artifact"]),
        height_artifact=_artifact_from_dict(item["height_artifact"]),
        findings=tuple(_finding_from_dict(finding) for finding in findings_value),
        summary=ValidationSummary(**dict(summary)),
    )
