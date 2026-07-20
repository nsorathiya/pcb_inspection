from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    AuditEvent,
    Inspection,
    InspectionProcessingRun,
    InspectionStatus,
    InspectionValidation,
)
from app.db.processing_types import ProcessingFinalDecision, ProcessingRunStatus
from app.db.validation_types import ValidationOutcome
from app.services.inspection_inference.models import (
    InferenceExecutionOutcome,
    InspectionInferenceResult,
)
from app.services.inspection_preprocessing.models import (
    InspectionPreprocessingResult,
    PreprocessingOutcome,
)
from app.services.inspection_processing.models import (
    BeginProcessingResult,
    CompleteProcessingResult,
    ProcessingStartIdentity,
    ProcessingKeyArtifact,
    canonical_uuid,
    generate_processing_key,
    lowercase_sha256,
)
from app.services.inspection_processing.persistence import (
    InspectionProcessingRepository,
    ProcessingPersistenceConflictError,
    aware_utc,
    json_object_text,
    retrieved_utc,
)

AUDIT_PROCESSING_STARTED = "INSPECTION_PROCESSING_STARTED"
AUDIT_MOCK_PASS = "INSPECTION_MOCK_RESULT_PASS"
AUDIT_MOCK_FAIL = "INSPECTION_MOCK_RESULT_FAIL"
AUDIT_MOCK_UNCERTAIN = "INSPECTION_MOCK_RESULT_UNCERTAIN"
AUDIT_PROCESSING_ERROR = "INSPECTION_PROCESSING_ERROR"


class ProcessingLifecycleError(Exception):
    pass


class ProcessingInspectionNotFoundError(ProcessingLifecycleError):
    pass


class ProcessingValidationNotFoundError(ProcessingLifecycleError):
    pass


class ProcessingLifecycleConflictError(ProcessingLifecycleError):
    pass


class InvalidProcessingTransitionError(ProcessingLifecycleConflictError):
    pass


class ProcessingLifecycleConsistencyError(ProcessingLifecycleConflictError):
    pass


def _audit_identity(value: str | None, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{field} is invalid")
    return value


class ProcessingLifecycleService:
    """Coordinate persistence only; preprocessing and inference never execute here."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        repository: InspectionProcessingRepository | None = None,
        clock: Callable[[], datetime] | None = None,
        audit_id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._sessions = session_factory
        self._repository = repository or InspectionProcessingRepository(session_factory)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._audit_id = audit_id_generator or (lambda: str(uuid4()))

    async def begin_processing(
        self,
        identity: ProcessingStartIdentity,
        processing_key: str,
        actor_id: str | None = None,
        request_id: str | None = None,
    ) -> BeginProcessingResult:
        if not isinstance(identity, ProcessingStartIdentity):
            raise TypeError("begin_processing requires ProcessingStartIdentity")
        lowercase_sha256(processing_key, "processing_key")
        if processing_key != identity.generated_key():
            raise ValueError("processing_key does not match the canonical processing identity")
        actor_id = _audit_identity(actor_id, "actor_id", 128)
        request_id = _audit_identity(request_id, "request_id", 256)
        started_at = aware_utc(self._clock(), "started_at")
        async with self._sessions() as session:
            try:
                await session.execute(text("BEGIN IMMEDIATE"))
                result = await self._begin(session, identity, processing_key, actor_id, request_id, started_at)
                await session.commit()
                return result
            except (ProcessingLifecycleError, TypeError, ValueError):
                await session.rollback()
                raise
            except Exception as exc:
                await session.rollback()
                raise ProcessingLifecycleError("processing start did not complete") from exc

    async def _begin(
        self,
        session: AsyncSession,
        identity: ProcessingStartIdentity,
        processing_key: str,
        actor_id: str | None,
        request_id: str | None,
        started_at: datetime,
    ) -> BeginProcessingResult:
        inspection = await session.get(Inspection, identity.inspection_id)
        if inspection is None:
            raise ProcessingInspectionNotFoundError("inspection does not exist")
        existing = await session.scalar(select(InspectionProcessingRun).where(
            InspectionProcessingRun.inspection_id == identity.inspection_id,
            InspectionProcessingRun.processing_key == processing_key,
        ))
        if existing is not None:
            self._require_run_identity(existing, identity)
            expected_status = self._inspection_status_for_run(existing)
            if inspection.status is not expected_status:
                raise ProcessingLifecycleConsistencyError(
                    "existing processing run is incompatible with inspection status"
                )
            return BeginProcessingResult(
                existing.id, existing.inspection_id, existing.validation_id,
                existing.processing_key, inspection.status, existing.status, True,
                retrieved_utc(existing.started_at),
            )
        if inspection.status is not InspectionStatus.READY:
            raise InvalidProcessingTransitionError(
                "new processing lifecycle requires READY inspection status"
            )
        validation = await session.get(InspectionValidation, identity.validation_id)
        if validation is None:
            raise ProcessingValidationNotFoundError("validation does not exist")
        if validation.inspection_id != identity.inspection_id:
            raise ProcessingLifecycleConflictError("validation does not belong to inspection")
        if validation.outcome is not ValidationOutcome.VALIDATION_PASSED:
            raise ProcessingLifecycleConflictError("processing requires a passed validation")
        if validation.result_sha256 != identity.validation_result_sha256:
            raise ProcessingLifecycleConflictError("validation result identity does not match")

        run = InspectionProcessingRun(
            id=identity.processing_run_id, inspection_id=identity.inspection_id,
            validation_id=identity.validation_id, processing_key=processing_key,
            status=ProcessingRunStatus.STARTED,
            preprocessing_policy_id=identity.preprocessing_policy_id,
            preprocessing_policy_version=identity.preprocessing_policy_version,
            preprocessing_implementation_id=identity.preprocessing_implementation_id,
            preprocessing_implementation_version=identity.preprocessing_implementation_version,
            inference_policy_id=identity.inference_policy_id,
            inference_policy_version=identity.inference_policy_version,
            engine_id=identity.engine_id, engine_version=identity.engine_version,
            engine_type=identity.engine_type, started_at=started_at, created_at=started_at,
        )
        session.add(run)
        await session.flush()
        updated = await self._begin_transition(session, identity.inspection_id)
        if updated != 1:
            raise InvalidProcessingTransitionError("inspection is no longer eligible to begin processing")
        await self._append_audit(
            session, identity.inspection_id, AUDIT_PROCESSING_STARTED, actor_id,
            request_id, started_at,
            {
                "engine_id": identity.engine_id,
                "engine_type": identity.engine_type,
                "engine_version": identity.engine_version,
                "inference_policy_id": identity.inference_policy_id,
                "inference_policy_version": identity.inference_policy_version,
                "preprocessing_implementation_id": identity.preprocessing_implementation_id,
                "preprocessing_implementation_version": identity.preprocessing_implementation_version,
                "preprocessing_policy_id": identity.preprocessing_policy_id,
                "preprocessing_policy_version": identity.preprocessing_policy_version,
                "processing_key": processing_key,
                "processing_run_id": identity.processing_run_id,
                "validation_id": identity.validation_id,
            },
        )
        return BeginProcessingResult(
            run.id, run.inspection_id, run.validation_id, run.processing_key,
            InspectionStatus.PROCESSING, run.status, False, started_at,
        )

    async def complete_processing(
        self,
        processing_run_id: str,
        preprocessing_result: InspectionPreprocessingResult,
        inference_result: InspectionInferenceResult | None,
        actor_id: str | None = None,
        request_id: str | None = None,
    ) -> CompleteProcessingResult:
        canonical_uuid(processing_run_id, "processing_run_id")
        actor_id = _audit_identity(actor_id, "actor_id", 128)
        request_id = _audit_identity(request_id, "request_id", 256)
        async with self._sessions() as session:
            try:
                await session.execute(text("BEGIN IMMEDIATE"))
                result = await self._complete(
                    session, processing_run_id, preprocessing_result, inference_result,
                    actor_id, request_id,
                )
                await session.commit()
                return result
            except (
                ProcessingLifecycleError,
                ProcessingPersistenceConflictError,
                TypeError,
                ValueError,
            ):
                await session.rollback()
                raise
            except Exception as exc:
                await session.rollback()
                raise ProcessingLifecycleError("processing completion did not complete") from exc

    async def _complete(
        self,
        session: AsyncSession,
        processing_run_id: str,
        preprocessing_result: InspectionPreprocessingResult,
        inference_result: InspectionInferenceResult | None,
        actor_id: str | None,
        request_id: str | None,
    ) -> CompleteProcessingResult:
        run = await session.get(InspectionProcessingRun, processing_run_id)
        if run is None:
            raise ProcessingLifecycleConflictError("processing run does not exist")
        inspection = await session.get(Inspection, run.inspection_id)
        if inspection is None:
            raise ProcessingLifecycleConsistencyError("processing run inspection does not exist")
        prepared_pre = self._repository._prepare_preprocessing(run, preprocessing_result)
        validation = await session.get(InspectionValidation, run.validation_id)
        if validation is None:
            raise ProcessingLifecycleConsistencyError("processing validation no longer exists")
        evidence = tuple(
            ProcessingKeyArtifact(item.artifact_type, item.sha256, item.byte_size)
            for item in (
                preprocessing_result.validity_mask_input,
                preprocessing_result.calibration_input,
            )
            if item is not None
        )
        expected_key = generate_processing_key(
            inspection_id=run.inspection_id,
            validation_id=run.validation_id,
            validation_result_sha256=validation.result_sha256,
            rgb_artifact=ProcessingKeyArtifact(
                preprocessing_result.rgb_input.artifact_type,
                preprocessing_result.rgb_input.sha256,
                preprocessing_result.rgb_input.byte_size,
            ),
            height_artifact=ProcessingKeyArtifact(
                preprocessing_result.height_input.artifact_type,
                preprocessing_result.height_input.sha256,
                preprocessing_result.height_input.byte_size,
            ),
            preprocessing_policy_id=run.preprocessing_policy_id,
            preprocessing_policy_version=run.preprocessing_policy_version,
            preprocessing_implementation_id=run.preprocessing_implementation_id,
            preprocessing_implementation_version=run.preprocessing_implementation_version,
            inference_policy_id=run.inference_policy_id,
            inference_policy_version=run.inference_policy_version,
            engine_id=run.engine_id,
            engine_version=run.engine_version,
            engine_type=run.engine_type,
            evidence_artifacts=evidence,
        )
        if expected_key != run.processing_key:
            raise ProcessingLifecycleConflictError(
                "preprocessing input identities do not match processing key"
            )
        run_started_at = retrieved_utc(run.started_at)
        if aware_utc(preprocessing_result.started_at, "preprocessing started_at") < run_started_at:
            raise ProcessingLifecycleConflictError(
                "preprocessing result cannot precede processing start"
            )
        if preprocessing_result.outcome is PreprocessingOutcome.SUCCEEDED:
            if inference_result is None:
                raise ProcessingLifecycleConflictError(
                    "successful preprocessing requires an inference result for completion"
                )
            prepared_inf = self._repository._prepare_inference(run, preprocessing_result, inference_result)
            if aware_utc(inference_result.started_at, "inference started_at") < aware_utc(
                preprocessing_result.completed_at, "preprocessing completed_at"
            ):
                raise ProcessingLifecycleConflictError(
                    "inference result cannot precede preprocessing completion"
                )
        else:
            if inference_result is not None:
                raise ProcessingLifecycleConflictError(
                    "failed or error preprocessing cannot contain an inference result"
                )
            prepared_inf = None

        transition = self._completion_transition(preprocessing_result, inference_result)
        if transition[4] < run_started_at:
            raise ProcessingLifecycleConflictError(
                "processing completion cannot precede processing start"
            )
        persisted = await self._repository._persist_results(session, run, prepared_pre, prepared_inf)
        if run.status is not ProcessingRunStatus.STARTED:
            self._require_completed_consistency(run, inspection, transition)
            if not persisted.idempotent_existing:
                raise ProcessingLifecycleConsistencyError("completed run was missing immutable results")
            audit = await self._find_audit(session, run.inspection_id, transition[3], run.id)
            if audit is None:
                raise ProcessingLifecycleConsistencyError("completed processing audit is missing")
            return self._completion_response(
                run, preprocessing_result, inference_result, transition, True,
                retrieved_utc(run.completed_at), transition[3],
            )
        if inspection.status is not InspectionStatus.PROCESSING:
            raise ProcessingLifecycleConsistencyError(
                "started processing run is incompatible with inspection status"
            )
        run_updated = await self._complete_run_transition(session, run.id, transition)
        if run_updated != 1:
            raise InvalidProcessingTransitionError("processing run is no longer eligible for completion")
        inspection_updated = await self._complete_inspection_transition(session, run.inspection_id, transition)
        if inspection_updated != 1:
            raise InvalidProcessingTransitionError("inspection is no longer eligible for processing completion")
        details = self._completion_audit_details(
            run, preprocessing_result, inference_result, transition,
            persisted.preprocessing_sha256, persisted.inference_sha256,
        )
        await self._append_audit(
            session, run.inspection_id, transition[3], actor_id, request_id,
            transition[4], details,
        )
        return self._completion_response(
            run, preprocessing_result, inference_result, transition, False,
            transition[4], transition[3],
        )

    @staticmethod
    def _completion_transition(preprocessing, inference):
        if preprocessing.outcome is not PreprocessingOutcome.SUCCEEDED:
            code = "PREPROCESSING_FAILED" if preprocessing.outcome is PreprocessingOutcome.FAILED else "PREPROCESSING_ERROR"
            return (
                ProcessingRunStatus.ERROR, InspectionStatus.ERROR, None,
                AUDIT_PROCESSING_ERROR, aware_utc(preprocessing.completed_at, "completed_at"),
                code, "Inspection preprocessing did not complete successfully.",
            )
        if inference.execution_outcome is not InferenceExecutionOutcome.SUCCEEDED:
            code = "INFERENCE_FAILED" if inference.execution_outcome is InferenceExecutionOutcome.FAILED else "INFERENCE_ERROR"
            return (
                ProcessingRunStatus.ERROR, InspectionStatus.ERROR, None,
                AUDIT_PROCESSING_ERROR, aware_utc(inference.completed_at, "completed_at"),
                code, "Inspection inference did not complete successfully.",
            )
        decision = ProcessingFinalDecision(inference.decision.value)
        actions = {
            ProcessingFinalDecision.PASS: AUDIT_MOCK_PASS,
            ProcessingFinalDecision.FAIL: AUDIT_MOCK_FAIL,
            ProcessingFinalDecision.UNCERTAIN: AUDIT_MOCK_UNCERTAIN,
        }
        return (
            ProcessingRunStatus.COMPLETED, InspectionStatus(decision.value), decision,
            actions[decision], aware_utc(inference.completed_at, "completed_at"), None, None,
        )

    @staticmethod
    async def _begin_transition(session: AsyncSession, inspection_id: str) -> int:
        result = await session.execute(
            update(Inspection).where(
                Inspection.id == inspection_id, Inspection.status == InspectionStatus.READY
            ).values(status=InspectionStatus.PROCESSING)
        )
        return result.rowcount

    @staticmethod
    async def _complete_run_transition(session: AsyncSession, run_id: str, transition) -> int:
        result = await session.execute(
            update(InspectionProcessingRun).where(
                InspectionProcessingRun.id == run_id,
                InspectionProcessingRun.status == ProcessingRunStatus.STARTED,
            ).values(
                status=transition[0], completed_at=transition[4],
                final_decision=transition[2], error_code=transition[5], error_message=transition[6],
            )
        )
        return result.rowcount

    @staticmethod
    async def _complete_inspection_transition(session: AsyncSession, inspection_id: str, transition) -> int:
        result = await session.execute(
            update(Inspection).where(
                Inspection.id == inspection_id,
                Inspection.status == InspectionStatus.PROCESSING,
            ).values(
                status=transition[1], completed_at=transition[4],
                error_code=transition[5], error_message=transition[6], confidence=None,
            )
        )
        return result.rowcount

    async def _append_audit(
        self, session, inspection_id, action, actor_id, request_id, created_at, details
    ) -> None:
        session.add(AuditEvent(
            id=canonical_uuid(self._audit_id(), "audit_id"), entity_type="inspection",
            entity_id=inspection_id, action=action, actor_id=actor_id,
            request_id=request_id, details_json=json_object_text(details), created_at=created_at,
        ))
        await session.flush()

    @staticmethod
    async def _find_audit(session, inspection_id, action, run_id):
        records = await session.scalars(select(AuditEvent).where(
            AuditEvent.entity_type == "inspection", AuditEvent.entity_id == inspection_id,
            AuditEvent.action == action,
        ).order_by(AuditEvent.created_at.asc(), AuditEvent.id.asc()))
        for record in records:
            try:
                details = json.loads(record.details_json)
            except (TypeError, ValueError):
                continue
            if isinstance(details, dict) and details.get("processing_run_id") == run_id:
                return record
        return None

    @staticmethod
    def _require_run_identity(run, identity):
        expected = (
            identity.validation_id,
            identity.preprocessing_policy_id, identity.preprocessing_policy_version,
            identity.preprocessing_implementation_id, identity.preprocessing_implementation_version,
            identity.inference_policy_id, identity.inference_policy_version,
            identity.engine_id, identity.engine_version, identity.engine_type,
        )
        actual = (
            run.validation_id, run.preprocessing_policy_id,
            run.preprocessing_policy_version, run.preprocessing_implementation_id,
            run.preprocessing_implementation_version, run.inference_policy_id,
            run.inference_policy_version, run.engine_id, run.engine_version, run.engine_type,
        )
        if actual != expected:
            raise ProcessingLifecycleConsistencyError(
                "processing key is associated with inconsistent lifecycle identity"
            )

    @staticmethod
    def _inspection_status_for_run(run):
        if run.status is ProcessingRunStatus.STARTED:
            return InspectionStatus.PROCESSING
        if run.status is ProcessingRunStatus.ERROR:
            return InspectionStatus.ERROR
        if run.final_decision is None:
            raise ProcessingLifecycleConsistencyError("completed run has no final decision")
        return InspectionStatus(run.final_decision.value)

    @staticmethod
    def _require_completed_consistency(run, inspection, transition):
        if (
            run.status is not transition[0]
            or inspection.status is not transition[1]
            or run.final_decision is not transition[2]
            or run.error_code != transition[5]
            or run.error_message != transition[6]
            or retrieved_utc(run.completed_at) != transition[4]
        ):
            raise ProcessingLifecycleConsistencyError(
                "completed processing result is incompatible with lifecycle state"
            )

    @staticmethod
    def _completion_audit_details(run, preprocessing, inference, transition, pre_hash, inf_hash):
        return {
            "engine_id": run.engine_id,
            "engine_type": run.engine_type,
            "engine_version": run.engine_version,
            "final_inspection_status": transition[1].value,
            "inference_execution_outcome": None if inference is None else inference.execution_outcome.value,
            "inference_finding_count": 0 if inference is None else inference.summary.total_findings,
            "inference_blocking_finding_count": 0 if inference is None else inference.summary.blocking_findings,
            "inference_warning_count": 0 if inference is None else inference.summary.warnings,
            "inference_id": None if inference is None else inference.inference_id,
            "inference_policy_id": run.inference_policy_id,
            "inference_policy_version": run.inference_policy_version,
            "inference_result_sha256": inf_hash,
            "mock_inference": True,
            "preprocessing_finding_count": preprocessing.summary.total_findings,
            "preprocessing_blocking_finding_count": preprocessing.summary.blocking_findings,
            "preprocessing_warning_count": preprocessing.summary.warnings,
            "preprocessing_id": preprocessing.preprocessing_id,
            "preprocessing_implementation_id": run.preprocessing_implementation_id,
            "preprocessing_implementation_version": run.preprocessing_implementation_version,
            "preprocessing_outcome": preprocessing.outcome.value,
            "preprocessing_policy_id": run.preprocessing_policy_id,
            "preprocessing_policy_version": run.preprocessing_policy_version,
            "preprocessing_result_sha256": pre_hash,
            "processing_key": run.processing_key,
            "processing_run_id": run.id,
            "production_approved": False,
        }

    @staticmethod
    def _completion_response(run, preprocessing, inference, transition, idempotent, completed_at, action):
        return CompleteProcessingResult(
            run.id, run.inspection_id, preprocessing.preprocessing_id,
            None if inference is None else inference.inference_id,
            transition[0], transition[1], transition[2], preprocessing.outcome.value,
            None if inference is None else inference.execution_outcome.value,
            idempotent, completed_at, action,
        )
