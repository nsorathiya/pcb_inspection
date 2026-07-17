from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import AuditEvent, Inspection, InspectionStatus
from app.db.validation_types import ValidationOutcome
from app.services.inspection_validation.interfaces import InspectionValidationResult
from app.services.inspection_validation.persistence import (
    InspectionValidationRepository,
    ValidationPersistenceConflictError,
    _PreparedValidation,
    _aware_utc,
    _canonical_uuid,
    _json_object_text,
    _retrieved_utc,
)

AUDIT_ACTION_VALIDATION_PASSED = "INSPECTION_VALIDATION_PASSED"
AUDIT_ACTION_VALIDATION_FAILED = "INSPECTION_VALIDATION_FAILED"
AUDIT_ACTION_VALIDATION_ERROR = "INSPECTION_VALIDATION_ERROR"

INPUT_VALIDATION_FAILED = "INPUT_VALIDATION_FAILED"
VALIDATOR_INTERNAL_ERROR = "VALIDATOR_INTERNAL_ERROR"


class ValidationCommitError(Exception):
    """Base public error for an atomic validation lifecycle commit."""


class InspectionNotFoundError(ValidationCommitError):
    """The completed result references no persisted inspection."""


class ValidationCommitConflictError(ValidationCommitError):
    """The lifecycle commit conflicts with current immutable database state."""


class InvalidInspectionTransitionError(ValidationCommitConflictError):
    """A new validation lifecycle can begin only from RECEIVED."""


class ValidationLifecycleConsistencyError(ValidationCommitConflictError):
    """Existing validation evidence and inspection lifecycle state disagree."""


@dataclass(frozen=True)
class ValidationCommitResult:
    validation_id: str
    inspection_id: str
    validation_key: str
    result_sha256: str
    validation_outcome: ValidationOutcome
    inspection_status: InspectionStatus
    persistence_existing: bool
    lifecycle_idempotent_existing: bool
    lifecycle_committed_now: bool
    audit_action: str | None
    committed_at: datetime


@dataclass(frozen=True)
class _Transition:
    status: InspectionStatus
    action: str
    completed: bool
    error_code: str | None
    error_message: str | None


_TRANSITIONS = {
    ValidationOutcome.VALIDATION_PASSED: _Transition(
        status=InspectionStatus.READY,
        action=AUDIT_ACTION_VALIDATION_PASSED,
        completed=False,
        error_code=None,
        error_message=None,
    ),
    ValidationOutcome.VALIDATION_FAILED: _Transition(
        status=InspectionStatus.VALIDATION_FAILED,
        action=AUDIT_ACTION_VALIDATION_FAILED,
        completed=True,
        error_code=INPUT_VALIDATION_FAILED,
        error_message="Inspection input validation failed.",
    ),
    ValidationOutcome.VALIDATION_ERROR: _Transition(
        status=InspectionStatus.ERROR,
        action=AUDIT_ACTION_VALIDATION_ERROR,
        completed=True,
        error_code=VALIDATOR_INTERNAL_ERROR,
        error_message="Inspection validation could not complete.",
    ),
}


def _optional_audit_identity(value: str | None, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


class ValidationCommitService:
    """Atomically adopt completed validation evidence into inspection lifecycle."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        validation_repository: InspectionValidationRepository | None = None,
        clock: Callable[[], datetime] | None = None,
        audit_id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._validations = validation_repository or InspectionValidationRepository(
            session_factory
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._audit_id = audit_id_generator or (lambda: str(uuid4()))

    async def commit_validation(
        self,
        result: InspectionValidationResult,
        validation_key: str,
        actor_id: str | None = None,
        request_id: str | None = None,
    ) -> ValidationCommitResult:
        if not isinstance(result, InspectionValidationResult):
            raise TypeError("validation lifecycle requires an InspectionValidationResult")
        actor_id = _optional_audit_identity(actor_id, "actor_id", 128)
        request_id = _optional_audit_identity(request_id, "request_id", 256)
        prepared = self._validations._prepare_validation(
            result.inspection_id,
            result,
            validation_key,
        )
        committed_at = _aware_utc(self._clock(), "committed_at")

        async with self._sessions() as session:
            try:
                # SQLite writer serialization makes the read/decision/update sequence
                # deterministic across processes; the conditional UPDATE remains the
                # authoritative lifecycle guard.
                await session.execute(text("BEGIN IMMEDIATE"))
                response = await self._commit_in_session(
                    session,
                    result,
                    prepared,
                    actor_id,
                    request_id,
                    committed_at,
                )
                await session.commit()
                return response
            except (ValidationCommitError, ValidationPersistenceConflictError):
                await session.rollback()
                raise
            except Exception as exc:
                await session.rollback()
                raise ValidationCommitError(
                    "validation lifecycle commit did not complete"
                ) from exc

    async def _commit_in_session(
        self,
        session: AsyncSession,
        result: InspectionValidationResult,
        prepared: _PreparedValidation,
        actor_id: str | None,
        request_id: str | None,
        committed_at: datetime,
    ) -> ValidationCommitResult:
        inspection = await session.get(Inspection, result.inspection_id)
        if inspection is None:
            raise InspectionNotFoundError("inspection does not exist")

        transition = _TRANSITIONS[result.outcome]
        persistence = await self._validations._persist_prepared(session, prepared)

        if persistence.idempotent_existing:
            if inspection.status is transition.status:
                audit = await self._find_lifecycle_audit(
                    session,
                    result.inspection_id,
                    transition.action,
                    prepared.validation_key,
                    persistence.result_sha256,
                )
                return ValidationCommitResult(
                    validation_id=persistence.validation_id,
                    inspection_id=persistence.inspection_id,
                    validation_key=persistence.validation_key,
                    result_sha256=persistence.result_sha256,
                    validation_outcome=result.outcome,
                    inspection_status=transition.status,
                    persistence_existing=True,
                    lifecycle_idempotent_existing=True,
                    lifecycle_committed_now=False,
                    audit_action=None if audit is None else audit.action,
                    committed_at=(
                        persistence.created_at
                        if audit is None
                        else _retrieved_utc(audit.created_at)
                    ),
                )
            if inspection.status is not InspectionStatus.RECEIVED:
                raise ValidationLifecycleConsistencyError(
                    "existing validation is incompatible with inspection status"
                )
        elif inspection.status is not InspectionStatus.RECEIVED:
            raise InvalidInspectionTransitionError(
                "new validation lifecycle requires RECEIVED inspection status"
            )

        updated = await self._apply_transition(
            session,
            result.inspection_id,
            transition,
            result.completed_at,
        )
        if updated != 1:
            raise InvalidInspectionTransitionError(
                "inspection is no longer eligible for validation transition"
            )

        await self._append_audit(
            session,
            result,
            prepared.validation_key,
            persistence.result_sha256,
            transition,
            actor_id,
            request_id,
            committed_at,
        )
        return ValidationCommitResult(
            validation_id=persistence.validation_id,
            inspection_id=persistence.inspection_id,
            validation_key=persistence.validation_key,
            result_sha256=persistence.result_sha256,
            validation_outcome=result.outcome,
            inspection_status=transition.status,
            persistence_existing=persistence.idempotent_existing,
            lifecycle_idempotent_existing=False,
            lifecycle_committed_now=True,
            audit_action=transition.action,
            committed_at=committed_at,
        )

    async def _apply_transition(
        self,
        session: AsyncSession,
        inspection_id: str,
        transition: _Transition,
        validation_completed_at: datetime,
    ) -> int:
        values = {
            "status": transition.status,
            "completed_at": (
                _aware_utc(validation_completed_at, "completed_at")
                if transition.completed
                else None
            ),
            "error_code": transition.error_code,
            "error_message": transition.error_message,
        }
        statement = (
            update(Inspection)
            .where(
                Inspection.id == inspection_id,
                Inspection.status == InspectionStatus.RECEIVED,
            )
            .values(**values)
        )
        outcome = await session.execute(statement)
        return outcome.rowcount

    async def _append_audit(
        self,
        session: AsyncSession,
        result: InspectionValidationResult,
        validation_key: str,
        result_sha256: str,
        transition: _Transition,
        actor_id: str | None,
        request_id: str | None,
        committed_at: datetime,
    ) -> None:
        details = {
            "blocking_finding_count": result.summary.blocking_count,
            "finding_count": result.summary.finding_count,
            "inspection_status": transition.status.value,
            "policy_id": result.validation_policy_id,
            "policy_version": result.validation_policy_version,
            "result_sha256": result_sha256,
            "validation_id": result.validation_id,
            "validation_key": validation_key,
            "validation_outcome": result.outcome.value,
            "validator_version": result.validator_version,
            "warning_count": result.summary.warning_count,
        }
        session.add(AuditEvent(
            id=_canonical_uuid(self._audit_id(), "audit_id"),
            entity_type="inspection",
            entity_id=result.inspection_id,
            action=transition.action,
            actor_id=actor_id,
            request_id=request_id,
            details_json=_json_object_text(details),
            created_at=committed_at,
        ))
        await session.flush()

    @staticmethod
    async def _find_lifecycle_audit(
        session: AsyncSession,
        inspection_id: str,
        action: str,
        validation_key: str,
        result_sha256: str,
    ) -> AuditEvent | None:
        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.entity_type == "inspection",
                AuditEvent.entity_id == inspection_id,
                AuditEvent.action == action,
            )
            .order_by(AuditEvent.created_at.asc(), AuditEvent.id.asc())
        )
        for event in await session.scalars(statement):
            try:
                details = json.loads(event.details_json)
            except (TypeError, ValueError):
                continue
            if (
                isinstance(details, dict)
                and details.get("validation_key") == validation_key
                and details.get("result_sha256") == result_sha256
            ):
                return event
        return None
