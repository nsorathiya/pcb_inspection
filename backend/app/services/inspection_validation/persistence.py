from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    ArtifactType,
    InspectionValidation,
    InspectionValidationFinding,
)
from app.db.validation_types import FindingCategory, FindingSeverity, ValidationOutcome
from app.services.inspection_validation.findings import FindingFactory
from app.services.inspection_validation.interfaces import (
    ArtifactTechnicalSummary,
    InspectionValidationResult,
    ReadabilityStatus,
    ValidationFinding,
    ValidationPersistenceResult,
)
from app.services.inspection_validation.models import (
    canonical_result_bytes,
    canonical_result_sha256,
    result_to_dict,
)

SUPPORTED_RESULT_CONTRACT_VERSION = "pcb-aoi-inspection-validation/1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_FORBIDDEN_DETAIL_KEYS = {
    "path", "absolute_path", "relative_path", "filename", "source_filename"
}


class ValidationPersistenceError(Exception):
    """Base error for immutable validation-result persistence."""


class ValidationPersistenceConflictError(ValidationPersistenceError):
    """An idempotency key already identifies different canonical result bytes."""


class ValidationPersistenceIntegrityError(ValidationPersistenceError):
    """The database rejected an immutable validation result or finding."""


@dataclass(frozen=True)
class PersistedInspectionValidation:
    validation_id: str
    inspection_id: str
    contract_version: str
    policy_id: str
    policy_version: str
    validator_version: str
    validation_key: str
    outcome: ValidationOutcome
    started_at: datetime
    completed_at: datetime
    rgb_summary: Mapping[str, Any]
    height_summary: Mapping[str, Any]
    summary: Mapping[str, Any]
    result: Mapping[str, Any]
    result_sha256: str
    created_at: datetime


@dataclass(frozen=True)
class PersistedValidationFinding:
    finding_id: str
    validation_id: str
    ordinal: int
    code: str
    severity: FindingSeverity
    category: FindingCategory
    message: str
    artifact_type: ArtifactType | None
    field: str | None
    blocking: bool
    details: Mapping[str, Any]
    created_at: datetime


def _canonical_uuid(value: str, field: str) -> str:
    try:
        canonical = str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID string") from exc
    if canonical != value:
        raise ValueError(f"{field} must be a canonical UUID string")
    return value


def _aware_utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include timezone information")
    return value.astimezone(timezone.utc)


def _retrieved_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _json_object_text(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("persisted JSON value must be an object")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("persisted JSON value must be canonical JSON") from exc


def _json_object(value: str) -> Mapping[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValidationPersistenceIntegrityError("stored JSON is not an object")
    return loaded


def _path_shaped(value: Any, *, key: str | None = None) -> bool:
    if key is not None and (key in _FORBIDDEN_DETAIL_KEYS or key.endswith("_path")):
        return True
    if isinstance(value, Mapping):
        return any(_path_shaped(child, key=str(child_key).lower()) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_path_shaped(child) for child in value)
    if isinstance(value, str):
        return (
            PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or "\\" in value
            or ".." in PurePosixPath(value).parts
        )
    return False


def _validate_artifact_summary(summary: ArtifactTechnicalSummary, expected: ArtifactType) -> None:
    if not isinstance(summary, ArtifactTechnicalSummary) or summary.artifact_type is not expected:
        raise ValueError(f"{expected.value} technical summary is invalid")
    if summary.sha256 is not None and not _SHA256.fullmatch(summary.sha256):
        raise ValueError(f"{expected.value} summary SHA-256 is invalid")
    if summary.byte_size is not None and (
        not isinstance(summary.byte_size, int)
        or isinstance(summary.byte_size, bool)
        or summary.byte_size < 0
    ):
        raise ValueError(f"{expected.value} summary byte size is invalid")
    if not isinstance(summary.readability_status, ReadabilityStatus):
        raise ValueError(f"{expected.value} readability status is invalid")


class InspectionValidationRepository:
    """Append/read-only repository for completed typed validation results."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        findings: FindingFactory | None = None,
        clock: Callable[[], datetime] | None = None,
        finding_id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._findings = findings or FindingFactory()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._finding_id = finding_id_generator or (lambda: str(uuid4()))

    async def save_validation_result(
        self,
        inspection_id: str,
        result: InspectionValidationResult,
        validation_key: str,
    ) -> ValidationPersistenceResult:
        _canonical_uuid(inspection_id, "inspection_id")
        if not _SHA256.fullmatch(validation_key):
            raise ValueError("validation_key must be a lowercase SHA-256")
        self._validate_result(inspection_id, result)
        canonical_bytes = canonical_result_bytes(result)
        result_sha256 = canonical_result_sha256(result)
        canonical_document = result_to_dict(result)
        created_at = _aware_utc(self._clock(), "created_at")

        existing = await self.get_by_inspection_and_key(inspection_id, validation_key)
        if existing is not None:
            return self._idempotent_response(existing, result_sha256)

        record = InspectionValidation(
            id=result.validation_id,
            inspection_id=inspection_id,
            contract_version=result.contract_version,
            policy_id=result.validation_policy_id,
            policy_version=result.validation_policy_version,
            validator_version=result.validator_version,
            validation_key=validation_key,
            outcome=result.outcome,
            started_at=_aware_utc(result.started_at, "started_at"),
            completed_at=_aware_utc(result.completed_at, "completed_at"),
            rgb_summary_json=_json_object_text(canonical_document["rgb_artifact"]),
            height_summary_json=_json_object_text(canonical_document["height_artifact"]),
            summary_json=_json_object_text(canonical_document["summary"]),
            result_json=canonical_bytes.decode("utf-8"),
            result_sha256=result_sha256,
            created_at=created_at,
        )
        finding_records = self._finding_records(result, created_at)
        try:
            async with self._sessions() as session, session.begin():
                session.add(record)
                await session.flush()
                session.add_all(finding_records)
                await session.flush()
        except IntegrityError as exc:
            existing = await self.get_by_inspection_and_key(inspection_id, validation_key)
            if existing is not None:
                return self._idempotent_response(existing, result_sha256)
            raise ValidationPersistenceIntegrityError(
                "validation result and findings were not persisted"
            ) from exc
        return ValidationPersistenceResult(
            validation_id=record.id,
            inspection_id=record.inspection_id,
            validation_key=record.validation_key,
            result_sha256=record.result_sha256,
            outcome=record.outcome,
            created_at=created_at,
            idempotent_existing=False,
        )

    async def get_by_validation_id(self, validation_id: str) -> PersistedInspectionValidation | None:
        _canonical_uuid(validation_id, "validation_id")
        async with self._sessions() as session:
            record = await session.get(InspectionValidation, validation_id)
        return None if record is None else self._persisted(record)

    async def get_by_inspection_and_key(self, inspection_id: str, validation_key: str) -> PersistedInspectionValidation | None:
        _canonical_uuid(inspection_id, "inspection_id")
        if not _SHA256.fullmatch(validation_key):
            raise ValueError("validation_key must be a lowercase SHA-256")
        statement = select(InspectionValidation).where(
            InspectionValidation.inspection_id == inspection_id,
            InspectionValidation.validation_key == validation_key,
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
        return None if record is None else self._persisted(record)

    async def get_latest_for_inspection(self, inspection_id: str) -> PersistedInspectionValidation | None:
        _canonical_uuid(inspection_id, "inspection_id")
        statement = (
            select(InspectionValidation)
            .where(InspectionValidation.inspection_id == inspection_id)
            .order_by(
                InspectionValidation.completed_at.desc(),
                InspectionValidation.created_at.desc(),
                InspectionValidation.id.asc(),
            )
            .limit(1)
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
        return None if record is None else self._persisted(record)

    async def list_findings(self, validation_id: str) -> tuple[PersistedValidationFinding, ...]:
        _canonical_uuid(validation_id, "validation_id")
        statement = (
            select(InspectionValidationFinding)
            .where(InspectionValidationFinding.validation_id == validation_id)
            .order_by(InspectionValidationFinding.ordinal.asc())
        )
        async with self._sessions() as session:
            records = list(await session.scalars(statement))
        return tuple(self._persisted_finding(item) for item in records)

    def _validate_result(self, inspection_id: str, result: InspectionValidationResult) -> None:
        if not isinstance(result, InspectionValidationResult):
            raise TypeError("persistence requires an InspectionValidationResult")
        if result.inspection_id != inspection_id:
            raise ValueError("result inspection_id does not match persistence target")
        _canonical_uuid(result.validation_id, "validation_id")
        _canonical_uuid(result.inspection_id, "result inspection_id")
        if result.contract_version != SUPPORTED_RESULT_CONTRACT_VERSION:
            raise ValueError("result contract version is unsupported")
        if not _IDENTIFIER.fullmatch(result.validation_policy_id):
            raise ValueError("validation policy identity is invalid")
        if not _VERSION.fullmatch(result.validation_policy_version) or not _VERSION.fullmatch(result.validator_version):
            raise ValueError("policy and validator versions must be explicit")
        if not isinstance(result.outcome, ValidationOutcome):
            raise ValueError("validation outcome is invalid")
        started = _aware_utc(result.started_at, "started_at")
        completed = _aware_utc(result.completed_at, "completed_at")
        if completed < started:
            raise ValueError("completed_at must not precede started_at")
        _validate_artifact_summary(result.rgb_artifact, ArtifactType.RGB_RAW)
        _validate_artifact_summary(result.height_artifact, ArtifactType.HEIGHT_RAW)

        findings = tuple(result.findings)
        for finding in findings:
            self._validate_finding(finding)
        try:
            ordered = self._findings.sort(findings)
        except KeyError as exc:
            raise ValueError("validation result contains an unknown finding code") from exc
        if ordered != findings:
            raise ValueError("validation findings are not in deterministic catalogue order")

        summary = result.summary
        counts = {
            "finding_count": len(findings),
            "info_count": sum(item.severity is FindingSeverity.INFO for item in findings),
            "warning_count": sum(item.severity is FindingSeverity.WARNING for item in findings),
            "error_count": sum(item.severity is FindingSeverity.ERROR for item in findings),
            "blocking_count": sum(item.blocking for item in findings),
        }
        for field, expected in counts.items():
            if getattr(summary, field) != expected:
                raise ValueError("validation summary counts do not match findings")
        blocking = [item for item in findings if item.blocking]
        if result.outcome is ValidationOutcome.VALIDATION_PASSED:
            if blocking or not summary.technically_ready:
                raise ValueError("VALIDATION_PASSED cannot contain blocking findings")
        elif result.outcome is ValidationOutcome.VALIDATION_FAILED:
            if not blocking or summary.technically_ready:
                raise ValueError("VALIDATION_FAILED requires a blocking finding")
        else:
            internal = [item for item in findings if item.code == "VALIDATOR_INTERNAL_ERROR"]
            if not internal or not any(item.severity is FindingSeverity.ERROR for item in findings):
                raise ValueError("VALIDATION_ERROR requires VALIDATOR_INTERNAL_ERROR")
            if not any(item.blocking for item in internal) or summary.technically_ready:
                raise ValueError("VALIDATION_ERROR internal finding must be blocking")

    def _validate_finding(self, finding: ValidationFinding) -> None:
        if not isinstance(finding, ValidationFinding) or not self._findings.is_known(finding.code):
            raise ValueError("validation result contains an unknown finding code")
        definition = self._findings.definition(finding.code)
        if not isinstance(finding.severity, FindingSeverity) or finding.severity is not definition.severity:
            raise ValueError("validation finding severity is invalid")
        if not isinstance(finding.category, FindingCategory) or finding.category is not definition.category:
            raise ValueError("validation finding category is invalid")
        if finding.artifact_type is not None and not isinstance(finding.artifact_type, ArtifactType):
            raise ValueError("validation finding artifact type is invalid")
        if not isinstance(finding.message, str) or not 1 <= len(finding.message) <= 512:
            raise ValueError("validation finding message is invalid")
        if finding.field is not None and (
            not isinstance(finding.field, str) or not _FIELD.fullmatch(finding.field)
        ):
            raise ValueError("validation finding field is invalid")
        if not isinstance(finding.blocking, bool):
            raise ValueError("validation finding blocking flag is invalid")
        if finding.details is not None:
            if not isinstance(finding.details, Mapping):
                raise ValueError("validation finding details must be an object")
            if _path_shaped(finding.details):
                raise ValueError("validation finding details must not contain paths")
            _json_object_text(finding.details)

    def _finding_records(self, result: InspectionValidationResult, created_at: datetime) -> list[InspectionValidationFinding]:
        records = []
        for ordinal, finding in enumerate(result.findings):
            finding_id = _canonical_uuid(self._finding_id(), "finding_id")
            records.append(InspectionValidationFinding(
                id=finding_id,
                validation_id=result.validation_id,
                ordinal=ordinal,
                code=finding.code,
                severity=finding.severity,
                category=finding.category,
                message=finding.message,
                artifact_type=finding.artifact_type,
                field=finding.field,
                blocking=finding.blocking,
                details_json=_json_object_text(finding.details or {}),
                created_at=created_at,
            ))
        return records

    @staticmethod
    def _idempotent_response(existing: PersistedInspectionValidation, result_sha256: str) -> ValidationPersistenceResult:
        if existing.result_sha256 != result_sha256:
            raise ValidationPersistenceConflictError(
                "validation key already identifies a different canonical result"
            )
        return ValidationPersistenceResult(
            validation_id=existing.validation_id,
            inspection_id=existing.inspection_id,
            validation_key=existing.validation_key,
            result_sha256=existing.result_sha256,
            outcome=existing.outcome,
            created_at=existing.created_at,
            idempotent_existing=True,
        )

    @staticmethod
    def _persisted(record: InspectionValidation) -> PersistedInspectionValidation:
        return PersistedInspectionValidation(
            validation_id=record.id,
            inspection_id=record.inspection_id,
            contract_version=record.contract_version,
            policy_id=record.policy_id,
            policy_version=record.policy_version,
            validator_version=record.validator_version,
            validation_key=record.validation_key,
            outcome=record.outcome,
            started_at=_retrieved_utc(record.started_at),
            completed_at=_retrieved_utc(record.completed_at),
            rgb_summary=_json_object(record.rgb_summary_json),
            height_summary=_json_object(record.height_summary_json),
            summary=_json_object(record.summary_json),
            result=_json_object(record.result_json),
            result_sha256=record.result_sha256,
            created_at=_retrieved_utc(record.created_at),
        )

    @staticmethod
    def _persisted_finding(record: InspectionValidationFinding) -> PersistedValidationFinding:
        return PersistedValidationFinding(
            finding_id=record.id,
            validation_id=record.validation_id,
            ordinal=record.ordinal,
            code=record.code,
            severity=record.severity,
            category=record.category,
            message=record.message,
            artifact_type=record.artifact_type,
            field=record.field,
            blocking=record.blocking,
            details=_json_object(record.details_json),
            created_at=_retrieved_utc(record.created_at),
        )
