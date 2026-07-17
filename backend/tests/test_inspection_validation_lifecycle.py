import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.runtime_paths import RuntimePaths
from app.db.database import Database
from app.db.models import (
    ArtifactType,
    AuditEvent,
    InspectionArtifact,
    InspectionStatus,
    InspectionValidation,
    InspectionValidationFinding,
    ModelCompatibilityStatus,
    ModelStatus,
    ModelVersion,
)
from app.db.repositories import (
    AuditEventCreate,
    InspectionArtifactCreate,
    InspectionCreate,
    ModelVersionCreate,
    Repositories,
)
from app.main import create_app
from app.services.inspection_validation import (
    AUDIT_ACTION_VALIDATION_ERROR,
    AUDIT_ACTION_VALIDATION_FAILED,
    AUDIT_ACTION_VALIDATION_PASSED,
    ArtifactTechnicalSummary,
    FindingCategory,
    FindingSeverity,
    InspectionNotFoundError,
    InspectionValidationRepository,
    InspectionValidationResult,
    InspectionValidationService,
    InvalidInspectionTransitionError,
    ReadabilityStatus,
    ValidationCommitError,
    ValidationCommitResult,
    ValidationCommitService,
    ValidationFinding,
    ValidationKeyArtifact,
    ValidationLifecycleConsistencyError,
    ValidationOutcome,
    ValidationPersistenceConflictError,
    ValidationSummary,
    generate_validation_key,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "contracts" / "examples"
FIXED_COMMITTED_AT = datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc)


def _timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _typed_example(name, *, inspection_id=None, validation_id=None):
    value = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    artifacts = []
    for field in ("rgb_artifact", "height_artifact"):
        item = value[field]
        artifacts.append(ArtifactTechnicalSummary(
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
        ))
    findings = tuple(ValidationFinding(
        code=item["code"],
        severity=FindingSeverity(item["severity"]),
        category=FindingCategory(item["category"]),
        message=item["message"],
        blocking=item["blocking"],
        artifact_type=(
            ArtifactType(item["artifact_type"])
            if item.get("artifact_type")
            else None
        ),
        field=item.get("field"),
        details=item.get("details"),
    ) for item in value["findings"])
    return InspectionValidationResult(
        contract_version=value["contract_version"],
        validation_id=validation_id or value["validation_id"],
        inspection_id=inspection_id or value["inspection_id"],
        validation_policy_id=value["validation_policy_id"],
        validation_policy_version=value["validation_policy_version"],
        outcome=ValidationOutcome(value["outcome"]),
        started_at=_timestamp(value["started_at"]),
        completed_at=_timestamp(value["completed_at"]),
        validator_version=value["validator_version"],
        rgb_artifact=artifacts[0],
        height_artifact=artifacts[1],
        findings=findings,
        summary=ValidationSummary(**value["summary"]),
    )


def _key(result, *, rgb_hash="a" * 64):
    return generate_validation_key(
        inspection_id=result.inspection_id,
        rgb_artifact=ValidationKeyArtifact(ArtifactType.RGB_RAW, rgb_hash, 100),
        height_artifact=ValidationKeyArtifact(ArtifactType.HEIGHT_RAW, "b" * 64, 200),
        contract_version=result.contract_version,
        policy_id=result.validation_policy_id,
        policy_version=result.validation_policy_version,
        validator_version=result.validator_version,
    )


async def _database(runtime_root):
    paths = RuntimePaths.from_root(runtime_root)
    paths.create_directories()
    database = Database(paths.database_file, busy_timeout_ms=5000)
    await database.initialize()
    repositories = Repositories.from_session_factory(database.session_factory)
    return database, repositories, paths


async def _inspection(repositories, inspection_id, status=InspectionStatus.RECEIVED):
    completed_at = (
        FIXED_COMMITTED_AT
        if status in {InspectionStatus.PASS, InspectionStatus.FAIL, InspectionStatus.UNCERTAIN}
        else None
    )
    return await repositories.inspections.create(InspectionCreate(
        id=inspection_id,
        status=status,
        board_id="BOARD-1",
        recipe_id="RECIPE-1",
        recipe_version="1.0",
        model_id="MODEL-UNCHANGED",
        model_version="1.0",
        confidence=0.5,
        processing_ms=12.5,
        completed_at=completed_at,
    ))


async def _counts(database):
    async with database.session() as session:
        counts = []
        for model in (
            InspectionValidation,
            InspectionValidationFinding,
            AuditEvent,
            ModelVersion,
        ):
            counts.append(
                await session.scalar(select(func.count()).select_from(model))
            )
        return tuple(counts)


def _service(database, **overrides):
    return ValidationCommitService(
        database.session_factory,
        clock=lambda: FIXED_COMMITTED_AT,
        **overrides,
    )


@pytest.mark.parametrize(
    ("example", "expected_status", "expected_action", "error_code", "error_message"),
    [
        (
            "inspection_validation_result.passed.json",
            InspectionStatus.READY,
            AUDIT_ACTION_VALIDATION_PASSED,
            None,
            None,
        ),
        (
            "inspection_validation_result.failed.json",
            InspectionStatus.VALIDATION_FAILED,
            AUDIT_ACTION_VALIDATION_FAILED,
            "INPUT_VALIDATION_FAILED",
            "Inspection input validation failed.",
        ),
        (
            "inspection_validation_result.error.json",
            InspectionStatus.ERROR,
            AUDIT_ACTION_VALIDATION_ERROR,
            "VALIDATOR_INTERNAL_ERROR",
            "Inspection validation could not complete.",
        ),
    ],
)
def test_successful_outcomes_commit_result_findings_transition_and_safe_audit(
    tmp_path, example, expected_status, expected_action, error_code, error_message
):
    async def scenario():
        result = _typed_example(example)
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            artifact = await repositories.artifacts.create(InspectionArtifactCreate(
                inspection_id=result.inspection_id,
                artifact_type=ArtifactType.RGB_RAW,
                relative_path=f"raw_uploads/{result.inspection_id}/rgb/rgb_raw.png",
                sha256="a" * 64,
                byte_size=100,
            ))
            artifact_before = await repositories.artifacts.get(artifact.id)
            committed = await _service(database).commit_validation(
                result,
                _key(result),
                actor_id="operator-1",
                request_id="request-1",
            )

            inspection = await repositories.inspections.get(result.inspection_id)
            persisted = await repositories.validations.get_by_validation_id(
                result.validation_id
            )
            findings = await repositories.validations.list_findings(result.validation_id)
            audits = await repositories.audit_events.list_for_entity(
                "inspection", result.inspection_id
            )
            artifact_after = await repositories.artifacts.get(artifact.id)

            assert isinstance(committed, ValidationCommitResult)
            assert committed.inspection_status is expected_status
            assert committed.lifecycle_committed_now
            assert not committed.persistence_existing
            assert not committed.lifecycle_idempotent_existing
            assert committed.audit_action == expected_action
            assert committed.committed_at == FIXED_COMMITTED_AT
            assert persisted.result == result.to_dict()
            assert [item.code for item in findings] == [item.code for item in result.findings]
            assert [item.ordinal for item in findings] == list(range(len(findings)))

            assert inspection.status is expected_status
            if expected_status is InspectionStatus.READY:
                assert inspection.completed_at is None
            else:
                assert inspection.completed_at is not None
                assert inspection.completed_at.replace(tzinfo=timezone.utc) == result.completed_at
            assert inspection.error_code == error_code
            assert inspection.error_message == error_message
            assert inspection.board_id == "BOARD-1"
            assert (inspection.recipe_id, inspection.recipe_version) == ("RECIPE-1", "1.0")
            assert (inspection.model_id, inspection.model_version) == ("MODEL-UNCHANGED", "1.0")
            assert (inspection.confidence, inspection.processing_ms) == (0.5, 12.5)
            assert (
                artifact_before.relative_path,
                artifact_before.sha256,
                artifact_before.byte_size,
            ) == (
                artifact_after.relative_path,
                artifact_after.sha256,
                artifact_after.byte_size,
            )

            assert len(audits) == 1
            audit = audits[0]
            assert audit.action == expected_action
            assert audit.actor_id == "operator-1"
            assert audit.request_id == "request-1"
            details = json.loads(audit.details_json)
            assert details == {
                "blocking_finding_count": result.summary.blocking_count,
                "finding_count": result.summary.finding_count,
                "inspection_status": expected_status.value,
                "policy_id": result.validation_policy_id,
                "policy_version": result.validation_policy_version,
                "result_sha256": committed.result_sha256,
                "validation_id": result.validation_id,
                "validation_key": committed.validation_key,
                "validation_outcome": result.outcome.value,
                "validator_version": result.validator_version,
                "warning_count": result.summary.warning_count,
            }
            assert not {
                "path", "relative_path", "filename", "details", "exception", "sql"
            }.intersection(details)
            assert all(item.message not in inspection.error_message for item in result.findings) if inspection.error_message else True
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_exact_replay_is_fully_idempotent(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.failed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            service = _service(database)
            first = await service.commit_validation(result, _key(result))
            second = await service.commit_validation(result, _key(result))
            assert first.lifecycle_committed_now
            assert second.persistence_existing
            assert second.lifecycle_idempotent_existing
            assert not second.lifecycle_committed_now
            assert second.audit_action == AUDIT_ACTION_VALIDATION_FAILED
            assert second.committed_at == first.committed_at
            validations, findings, audits, _ = await _counts(database)
            assert validations == 1
            assert findings == len(result.findings)
            assert audits == 1
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_standalone_persistence_on_received_is_adopted_without_duplication(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            await repositories.validations.save_validation_result(
                result.inspection_id, result, _key(result)
            )
            committed = await _service(database).commit_validation(result, _key(result))
            assert committed.persistence_existing
            assert committed.lifecycle_committed_now
            assert not committed.lifecycle_idempotent_existing
            assert (await repositories.inspections.get(result.inspection_id)).status is InspectionStatus.READY
            validations, findings, audits, _ = await _counts(database)
            assert (validations, findings, audits) == (1, len(result.findings), 1)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_existing_identical_result_and_expected_status_is_idempotent(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id, InspectionStatus.READY)
            await repositories.validations.save_validation_result(
                result.inspection_id, result, _key(result)
            )
            committed = await _service(database).commit_validation(result, _key(result))
            assert committed.persistence_existing
            assert committed.lifecycle_idempotent_existing
            assert not committed.lifecycle_committed_now
            assert committed.audit_action is None
            assert (await _counts(database))[:3] == (1, len(result.findings), 0)
        finally:
            await database.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status",
    [
        InspectionStatus.VALIDATION_FAILED,
        InspectionStatus.ERROR,
        InspectionStatus.PROCESSING,
        InspectionStatus.PASS,
        InspectionStatus.FAIL,
        InspectionStatus.UNCERTAIN,
    ],
)
def test_existing_identical_result_in_incompatible_status_is_rejected(tmp_path, status):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id, status)
            await repositories.validations.save_validation_result(
                result.inspection_id, result, _key(result)
            )
            with pytest.raises(ValidationLifecycleConsistencyError):
                await _service(database).commit_validation(result, _key(result))
            assert (await repositories.inspections.get(result.inspection_id)).status is status
            assert (await _counts(database))[:3] == (1, len(result.findings), 0)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_same_key_with_different_result_hash_is_rejected_without_lifecycle_change(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        changed = replace(result, validation_id=str(uuid4()), validator_version="1.0.1")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            await repositories.validations.save_validation_result(
                result.inspection_id, result, _key(result)
            )
            with pytest.raises(ValidationPersistenceConflictError):
                await _service(database).commit_validation(changed, _key(result))
            inspection = await repositories.inspections.get(result.inspection_id)
            assert inspection.status is InspectionStatus.RECEIVED
            assert (await _counts(database))[:3] == (1, len(result.findings), 0)
        finally:
            await database.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status",
    [
        InspectionStatus.READY,
        InspectionStatus.VALIDATION_FAILED,
        InspectionStatus.ERROR,
        InspectionStatus.PROCESSING,
        InspectionStatus.PASS,
        InspectionStatus.FAIL,
        InspectionStatus.UNCERTAIN,
    ],
)
def test_new_validation_key_after_received_is_rejected_and_rolled_back(tmp_path, status):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id, status)
            with pytest.raises(InvalidInspectionTransitionError):
                await _service(database).commit_validation(result, _key(result))
            assert (await repositories.inspections.get(result.inspection_id)).status is status
            assert (await _counts(database))[:3] == (0, 0, 0)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_missing_inspection_and_invalid_result_change_nothing(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            with pytest.raises(InspectionNotFoundError):
                await _service(database).commit_validation(result, _key(result))
            await _inspection(repositories, result.inspection_id)
            invalid = replace(result, completed_at=result.completed_at.replace(tzinfo=None))
            with pytest.raises(ValueError, match="timezone"):
                await _service(database).commit_validation(invalid, _key(result))
            assert (await repositories.inspections.get(result.inspection_id)).status is InspectionStatus.RECEIVED
            assert (await _counts(database))[:3] == (0, 0, 0)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_result_insertion_failure_rolls_back_every_new_effect(tmp_path):
    async def scenario():
        original = _typed_example("inspection_validation_result.passed.json")
        other_id = str(uuid4())
        target_id = str(uuid4())
        existing = replace(original, inspection_id=other_id)
        target = replace(original, inspection_id=target_id)
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, other_id)
            await _inspection(repositories, target_id)
            await repositories.validations.save_validation_result(
                other_id, existing, _key(existing)
            )
            baseline = await _counts(database)
            with pytest.raises(ValidationCommitError):
                await _service(database).commit_validation(target, _key(target))
            assert await _counts(database) == baseline
            assert (await repositories.inspections.get(target_id)).status is InspectionStatus.RECEIVED
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_finding_insertion_failure_rolls_back_result_status_and_audit(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.failed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        duplicate = str(uuid4())
        repository = InspectionValidationRepository(
            database.session_factory,
            finding_id_generator=lambda: duplicate,
        )
        try:
            await _inspection(repositories, result.inspection_id)
            service = _service(database, validation_repository=repository)
            with pytest.raises(ValidationCommitError):
                await service.commit_validation(result, _key(result))
            assert (await _counts(database))[:3] == (0, 0, 0)
            assert (await repositories.inspections.get(result.inspection_id)).status is InspectionStatus.RECEIVED
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_conditional_transition_failure_rolls_back_new_persistence(tmp_path, monkeypatch):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            service = _service(database)

            async def no_transition(*_args, **_kwargs):
                return 0

            monkeypatch.setattr(service, "_apply_transition", no_transition)
            with pytest.raises(InvalidInspectionTransitionError):
                await service.commit_validation(result, _key(result))
            assert (await _counts(database))[:3] == (0, 0, 0)
            assert (await repositories.inspections.get(result.inspection_id)).status is InspectionStatus.RECEIVED
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_audit_insertion_failure_rolls_back_result_findings_and_status(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        duplicate_id = str(uuid4())
        try:
            await _inspection(repositories, result.inspection_id)
            await repositories.audit_events.append(AuditEventCreate(
                id=duplicate_id,
                entity_type="test",
                entity_id="baseline",
                action="BASELINE",
                details={},
            ))
            baseline = await _counts(database)
            service = _service(database, audit_id_generator=lambda: duplicate_id)
            with pytest.raises(ValidationCommitError):
                await service.commit_validation(result, _key(result))
            assert await _counts(database) == baseline
            assert (await repositories.inspections.get(result.inspection_id)).status is InspectionStatus.RECEIVED
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_commit_failure_rolls_back_and_never_reports_success(tmp_path):
    class FailingCommitSession(AsyncSession):
        async def commit(self):
            await self.rollback()
            raise RuntimeError("simulated commit failure")

    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            failing_factory = async_sessionmaker(
                database.engine,
                class_=FailingCommitSession,
                expire_on_commit=False,
            )
            service = ValidationCommitService(
                failing_factory,
                clock=lambda: FIXED_COMMITTED_AT,
            )
            with pytest.raises(ValidationCommitError, match="did not complete"):
                await service.commit_validation(result, _key(result))
            assert (await _counts(database))[:3] == (0, 0, 0)
            assert (await repositories.inspections.get(result.inspection_id)).status is InspectionStatus.RECEIVED
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_concurrent_identical_commits_produce_one_lifecycle(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.failed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            first_service = _service(database)
            second_service = _service(database)
            responses = await asyncio.gather(
                first_service.commit_validation(result, _key(result)),
                second_service.commit_validation(result, _key(result)),
            )
            assert sum(item.lifecycle_committed_now for item in responses) == 1
            assert sum(item.lifecycle_idempotent_existing for item in responses) == 1
            assert (await _counts(database))[:3] == (1, len(result.findings), 1)
            assert (await repositories.inspections.get(result.inspection_id)).status is InspectionStatus.VALIDATION_FAILED
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_concurrent_different_keys_produce_one_winner_and_no_partial_loser(tmp_path):
    async def scenario():
        inspection_id = str(uuid4())
        first = _typed_example(
            "inspection_validation_result.passed.json",
            inspection_id=inspection_id,
            validation_id=str(uuid4()),
        )
        second = replace(
            first,
            validation_id=str(uuid4()),
            validator_version="1.0.1",
        )
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, inspection_id)
            responses = await asyncio.gather(
                _service(database).commit_validation(first, _key(first)),
                _service(database).commit_validation(second, _key(second)),
                return_exceptions=True,
            )
            successes = [item for item in responses if isinstance(item, ValidationCommitResult)]
            failures = [item for item in responses if isinstance(item, Exception)]
            assert len(successes) == len(failures) == 1
            assert isinstance(failures[0], InvalidInspectionTransitionError)
            assert (await _counts(database))[:3] == (1, len(first.findings), 1)
            assert (await repositories.inspections.get(inspection_id)).status is InspectionStatus.READY
            stored_ids = await repositories.validations.get_latest_for_inspection(inspection_id)
            assert stored_ids.validation_id == successes[0].validation_id
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_coordinator_has_no_engine_route_model_or_filesystem_side_effects(tmp_path, monkeypatch):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, paths = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            raw_file = paths.raw_uploads / result.inspection_id / "rgb" / "rgb_raw.png"
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            raw_file.write_bytes(b"unchanged-raw-bytes")
            before_file = (raw_file.read_bytes(), raw_file.stat().st_mtime_ns)
            artifact = await repositories.artifacts.create(InspectionArtifactCreate(
                inspection_id=result.inspection_id,
                artifact_type=ArtifactType.RGB_RAW,
                relative_path=f"raw_uploads/{result.inspection_id}/rgb/rgb_raw.png",
                sha256="a" * 64,
                byte_size=100,
            ))
            await repositories.models.register(ModelVersionCreate(
                model_id="MODEL-1",
                model_version="1.0",
                engine_type="MOCK",
                class_label_contract_version="1.0",
                defect_taxonomy_version="1.0",
                compatibility_status=ModelCompatibilityStatus.UNVERIFIED,
                status=ModelStatus.BLOCKED,
            ))
            artifact_before = await repositories.artifacts.get(artifact.id)
            models_before = (await _counts(database))[3]

            async def engine_must_not_run(*_args, **_kwargs):
                raise AssertionError("semantic-validation engine was called")

            monkeypatch.setattr(
                InspectionValidationService,
                "validate_inspection_pair",
                engine_must_not_run,
            )
            await _service(database).commit_validation(result, _key(result))

            artifact_after = await repositories.artifacts.get(artifact.id)
            assert before_file == (raw_file.read_bytes(), raw_file.stat().st_mtime_ns)
            assert (
                artifact_before.id,
                artifact_before.inspection_id,
                artifact_before.artifact_type,
                artifact_before.relative_path,
                artifact_before.sha256,
                artifact_before.byte_size,
                artifact_before.media_type,
                artifact_before.created_at,
            ) == (
                artifact_after.id,
                artifact_after.inspection_id,
                artifact_after.artifact_type,
                artifact_after.relative_path,
                artifact_after.sha256,
                artifact_after.byte_size,
                artifact_after.media_type,
                artifact_after.created_at,
            )
            assert (await _counts(database))[3] == models_before == 1
            assert list(paths.previews.iterdir()) == []
            assert list(paths.reports.iterdir()) == []
            assert list(paths.results.iterdir()) == []
        finally:
            await database.dispose()

    asyncio.run(scenario())
    application = create_app(Settings(_env_file=None, runtime_root=tmp_path / "app-runtime"))
    assert not any("validation" in route for route in application.openapi()["paths"])
