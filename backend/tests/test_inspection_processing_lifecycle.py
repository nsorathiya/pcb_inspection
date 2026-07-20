import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, inspect, select, text

from app.core.runtime_paths import RuntimePaths
from app.db.database import Database
from app.db.migrations.runner import DEFAULT_MIGRATIONS, Migration, MigrationRunner
from app.db.models import (
    AuditEvent,
    InspectionInferenceResult,
    InspectionPreprocessingResult,
    InspectionProcessingRun,
    InspectionStatus,
    InspectionValidation,
    InspectionValidationFinding,
    SCHEMA_VERSION,
)
from app.db.repositories import InspectionCreate, Repositories
from app.services.inspection_inference.models import InferenceExecutionOutcome
from app.services.inspection_processing import (
    AUDIT_MOCK_FAIL,
    AUDIT_MOCK_PASS,
    AUDIT_MOCK_UNCERTAIN,
    AUDIT_PROCESSING_ERROR,
    AUDIT_PROCESSING_STARTED,
    InspectionProcessingRepository,
    InvalidProcessingTransitionError,
    ProcessingKeyArtifact,
    ProcessingLifecycleConflictError,
    ProcessingLifecycleService,
    ProcessingPersistenceConflictError,
    ProcessingStartIdentity,
    ProcessingValidationNotFoundError,
    canonical_inference_result_bytes,
    canonical_inference_result_sha256,
    canonical_preprocessing_result_bytes,
    canonical_preprocessing_result_sha256,
    generate_processing_key,
)
from app.services.inspection_preprocessing.models import PreprocessingOutcome
from tests.processing_result_helpers import (
    inference_result,
    preprocessing_result,
    validation_result,
)

FIXED_TIME = datetime(2026, 7, 17, 11, 0, tzinfo=timezone.utc)


async def _database(root):
    paths = RuntimePaths.from_root(root)
    paths.create_directories()
    database = Database(paths.database_file, busy_timeout_ms=5000)
    await database.initialize()
    return database, Repositories.from_session_factory(database.session_factory), paths


async def _ready(database, repositories):
    inspection_id, validation_id = str(uuid4()), str(uuid4())
    await repositories.inspections.create(InspectionCreate(
        id=inspection_id, status=InspectionStatus.READY, board_id="BOARD-1",
        recipe_id="RECIPE-1", recipe_version="1.0",
    ))
    validation = validation_result(inspection_id=inspection_id, validation_id=validation_id)
    persisted = await repositories.validations.save_validation_result(
        inspection_id, validation, "c" * 64
    )
    return inspection_id, validation_id, persisted.result_sha256


def _results(inspection_id, validation_id, decision="PASS"):
    preprocessing_id, inference_id = str(uuid4()), str(uuid4())
    pre = preprocessing_result(
        "inspection_preprocessing_result.succeeded.json",
        inspection_id=inspection_id, validation_id=validation_id,
        preprocessing_id=preprocessing_id,
    )
    inf = inference_result(
        f"inspection_inference_result.{decision.lower()}.json",
        inspection_id=inspection_id, validation_id=validation_id,
        preprocessing_id=preprocessing_id, inference_id=inference_id,
    )
    return pre, inf


def _identity(inspection_id, validation_id, validation_hash, pre, inf, **changes):
    values = dict(
        processing_run_id=str(uuid4()), inspection_id=inspection_id,
        validation_id=validation_id, validation_result_sha256=validation_hash,
        rgb_artifact=ProcessingKeyArtifact("RGB_RAW", pre.rgb_input.sha256, pre.rgb_input.byte_size),
        height_artifact=ProcessingKeyArtifact("HEIGHT_RAW", pre.height_input.sha256, pre.height_input.byte_size),
        preprocessing_policy_id=pre.policy_id, preprocessing_policy_version=pre.policy_version,
        preprocessing_implementation_id=pre.implementation_id,
        preprocessing_implementation_version=pre.implementation_version,
        inference_policy_id=inf.policy_id, inference_policy_version=inf.policy_version,
        engine_id=inf.engine_id, engine_version=inf.engine_version, engine_type=inf.engine_type,
    )
    values.update(changes)
    return ProcessingStartIdentity(**values)


def _service(database, **changes):
    return ProcessingLifecycleService(
        database.session_factory, clock=lambda: FIXED_TIME, **changes
    )


async def _started(database, repositories, decision="PASS"):
    inspection_id, validation_id, validation_hash = await _ready(database, repositories)
    pre, inf = _results(inspection_id, validation_id, decision)
    identity = _identity(inspection_id, validation_id, validation_hash, pre, inf)
    started = await _service(database).begin_processing(identity, identity.generated_key())
    return identity, pre, inf, started


def test_schema_three_new_and_version_two_migration_preserve_validation(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "new")
        try:
            async with database.engine.connect() as connection:
                tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
                version = await connection.scalar(text("SELECT version FROM schema_version WHERE id=1"))
                pragmas = (
                    await connection.scalar(text("PRAGMA foreign_keys")),
                    await connection.scalar(text("PRAGMA journal_mode")),
                    await connection.scalar(text("PRAGMA busy_timeout")),
                )
            assert version == SCHEMA_VERSION == 3
            assert {
                "inspection_processing_runs", "inspection_preprocessing_results",
                "inspection_preprocessing_result_findings", "inspection_inference_results",
                "inspection_inference_result_findings",
            } <= tables
            assert pragmas[0] == 1 and str(pragmas[1]).lower() == "wal" and pragmas[2] == 5000
        finally:
            await database.dispose()

        paths = RuntimePaths.from_root(tmp_path / "migrate")
        paths.create_directories()
        old = Database(paths.database_file, busy_timeout_ms=5000)
        await MigrationRunner(old.engine).run(target_version=2)
        repos = Repositories.from_session_factory(old.session_factory)
        inspection_id, validation_id = str(uuid4()), str(uuid4())
        await repos.inspections.create(InspectionCreate(
            id=inspection_id, status=InspectionStatus.READY, board_id="B",
            recipe_id="R", recipe_version="1",
        ))
        validation = validation_result(inspection_id=inspection_id, validation_id=validation_id)
        await repos.validations.save_validation_result(inspection_id, validation, "d" * 64)
        await old.initialize()
        async with old.session() as session:
            assert await session.scalar(text("SELECT version FROM schema_version WHERE id=1")) == 3
            assert await session.scalar(select(func.count()).select_from(InspectionValidation)) == 1
            assert await session.scalar(select(func.count()).select_from(InspectionValidationFinding)) == len(validation.findings)
        await old.initialize()
        await old.dispose()
    asyncio.run(scenario())


def test_failed_migration_does_not_report_version_three(tmp_path):
    async def scenario():
        paths = RuntimePaths.from_root(tmp_path / "runtime")
        paths.create_directories()
        database = Database(paths.database_file, busy_timeout_ms=5000)
        await MigrationRunner(database.engine).run(target_version=2)

        async def fail(connection):
            await DEFAULT_MIGRATIONS[2].upgrade(connection)
            raise RuntimeError("simulated migration-three failure")

        migration = Migration(3, "003_failure", DEFAULT_MIGRATIONS[2].required_table_names, fail)
        with pytest.raises(RuntimeError, match="simulated"):
            await MigrationRunner(database.engine, (*DEFAULT_MIGRATIONS[:2], migration)).run()
        async with database.engine.connect() as connection:
            assert await connection.scalar(text("SELECT version FROM schema_version WHERE id=1")) == 2
        await database.dispose()
    asyncio.run(scenario())


def test_processing_key_and_canonical_hashes_are_deterministic_and_sensitive():
    inspection_id, validation_id = str(uuid4()), str(uuid4())
    pre, inf = _results(inspection_id, validation_id)
    identity = _identity(inspection_id, validation_id, "a" * 64, pre, inf)
    assert identity.generated_key() == identity.generated_key()
    changed = replace(identity, engine_version="different")
    assert identity.generated_key() != changed.generated_key()
    assert canonical_preprocessing_result_sha256(pre) == hashlib.sha256(
        canonical_preprocessing_result_bytes(pre)
    ).hexdigest()
    assert canonical_inference_result_sha256(inf) == hashlib.sha256(
        canonical_inference_result_bytes(inf)
    ).hexdigest()
    assert json.loads(canonical_preprocessing_result_bytes(pre)) == pre.to_dict()


def test_begin_transitions_and_exact_replay_are_atomic_and_idempotent(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            inspection_id, validation_id, validation_hash = await _ready(database, repositories)
            pre, inf = _results(inspection_id, validation_id)
            identity = _identity(inspection_id, validation_id, validation_hash, pre, inf)
            service = _service(database)
            first = await service.begin_processing(identity, identity.generated_key(), "operator", "request")
            replay = await service.begin_processing(
                replace(identity, processing_run_id=str(uuid4())), identity.generated_key()
            )
            assert first.inspection_status is InspectionStatus.PROCESSING
            assert not first.idempotent_existing and replay.idempotent_existing
            assert first.processing_run_id == replay.processing_run_id
            audits = await repositories.audit_events.list_for_entity("inspection", inspection_id)
            assert [item.action for item in audits] == [AUDIT_PROCESSING_STARTED]
            assert json.loads(audits[0].details_json)["processing_key"] == identity.generated_key()
        finally:
            await database.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("decision", "status", "action"),
    [
        ("PASS", InspectionStatus.PASS, AUDIT_MOCK_PASS),
        ("FAIL", InspectionStatus.FAIL, AUDIT_MOCK_FAIL),
        ("UNCERTAIN", InspectionStatus.UNCERTAIN, AUDIT_MOCK_UNCERTAIN),
    ],
)
def test_successful_completion_persists_ordered_results_and_final_state(
    tmp_path, decision, status, action
):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / decision)
        try:
            identity, pre, inf, _ = await _started(database, repositories, decision)
            completed = await _service(database).complete_processing(identity.processing_run_id, pre, inf)
            replay = await _service(database).complete_processing(identity.processing_run_id, pre, inf)
            assert completed.inspection_status is status and completed.audit_action == action
            assert not completed.idempotent_existing and replay.idempotent_existing
            persisted_pre = await repositories.processing.get_preprocessing_result(identity.processing_run_id)
            persisted_inf = await repositories.processing.get_inference_result(identity.processing_run_id)
            assert persisted_pre.result_sha256 == canonical_preprocessing_result_sha256(pre)
            assert persisted_inf.result_sha256 == canonical_inference_result_sha256(inf)
            pre_findings = await repositories.processing.list_preprocessing_findings(pre.preprocessing_id)
            inf_findings = await repositories.processing.list_inference_findings(inf.inference_id)
            assert [item.ordinal for item in pre_findings] == list(range(len(pre.findings)))
            assert [item.code for item in inf_findings] == [item.code for item in inf.findings]
            audits = await repositories.audit_events.list_for_entity("inspection", identity.inspection_id)
            assert [item.action for item in audits] == [AUDIT_PROCESSING_STARTED, action]
            details = json.loads(audits[-1].details_json)
            assert details["mock_inference"] is True and details["production_approved"] is False
            assert "confidence" not in details and "accuracy" not in json.dumps(details).lower()
        finally:
            await database.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("pre_name", "inference_outcome", "error_code"),
    [
        ("inspection_preprocessing_result.failed.json", None, "PREPROCESSING_FAILED"),
        ("inspection_preprocessing_result.error.json", None, "PREPROCESSING_ERROR"),
        (None, InferenceExecutionOutcome.FAILED, "INFERENCE_FAILED"),
        (None, InferenceExecutionOutcome.ERROR, "INFERENCE_ERROR"),
    ],
)
def test_technical_results_map_only_to_error(tmp_path, pre_name, inference_outcome, error_code):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / error_code)
        try:
            identity, succeeded, inf, _ = await _started(database, repositories)
            if pre_name:
                pre = preprocessing_result(
                    pre_name, inspection_id=identity.inspection_id,
                    validation_id=identity.validation_id, preprocessing_id=succeeded.preprocessing_id,
                )
                pre = replace(
                    pre, rgb_input=succeeded.rgb_input, height_input=succeeded.height_input
                )
                inference = None
            else:
                pre = succeeded
                error = inference_result(
                    "inspection_inference_result.error.json",
                    inspection_id=identity.inspection_id, validation_id=identity.validation_id,
                    preprocessing_id=pre.preprocessing_id, inference_id=inf.inference_id,
                )
                inference = replace(error, execution_outcome=inference_outcome)
            completed = await _service(database).complete_processing(identity.processing_run_id, pre, inference)
            inspection = await repositories.inspections.get(identity.inspection_id)
            run = await repositories.processing.get_run_by_id(identity.processing_run_id)
            assert completed.inspection_status is InspectionStatus.ERROR
            assert completed.final_decision is None and completed.audit_action == AUDIT_PROCESSING_ERROR
            assert inspection.error_code == run.error_code == error_code
            assert run.final_decision is None
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_invalid_completion_identity_confidence_and_conflicting_replay_are_rejected(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            identity, pre, inf, _ = await _started(database, repositories)
            with pytest.raises(ValueError):
                await _service(database).complete_processing(
                    identity.processing_run_id, replace(pre, validation_id=str(uuid4())), inf
                )
            with pytest.raises(ValueError):
                await _service(database).complete_processing(
                    identity.processing_run_id, pre, replace(inf, confidence=0.5)
                )
            mismatched_input = replace(
                pre, rgb_input=replace(pre.rgb_input, sha256="0" * 64)
            )
            with pytest.raises(ProcessingLifecycleConflictError, match="processing key"):
                await _service(database).complete_processing(
                    identity.processing_run_id, mismatched_input, inf
                )
            await _service(database).complete_processing(identity.processing_run_id, pre, inf)
            changed = replace(inf, inference_id=str(uuid4()))
            with pytest.raises(ProcessingPersistenceConflictError):
                await _service(database).complete_processing(identity.processing_run_id, pre, changed)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_begin_safety_and_concurrent_winner_rules(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            inspection_id, validation_id, validation_hash = await _ready(database, repositories)
            pre, inf = _results(inspection_id, validation_id)
            identity = _identity(inspection_id, validation_id, validation_hash, pre, inf)
            calls = await asyncio.gather(
                _service(database).begin_processing(identity, identity.generated_key()),
                _service(database).begin_processing(replace(identity, processing_run_id=str(uuid4())), identity.generated_key()),
            )
            assert sum(not item.idempotent_existing for item in calls) == 1
            async with database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 1

            other_inspection, other_validation, other_hash = await _ready(database, repositories)
            other_pre, other_inf = _results(other_inspection, other_validation)
            first = _identity(other_inspection, other_validation, other_hash, other_pre, other_inf)
            second = replace(first, processing_run_id=str(uuid4()), engine_version="different")
            outcomes = await asyncio.gather(
                _service(database).begin_processing(first, first.generated_key()),
                _service(database).begin_processing(second, second.generated_key()),
                return_exceptions=True,
            )
            assert sum(not isinstance(item, Exception) for item in outcomes) == 1
            assert any(isinstance(item, InvalidProcessingTransitionError) for item in outcomes)
            async with database.session() as session:
                count = await session.scalar(select(func.count()).select_from(InspectionProcessingRun).where(
                    InspectionProcessingRun.inspection_id == other_inspection
                ))
                assert count == 1
        finally:
            await database.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize(
    "status",
    [
        InspectionStatus.RECEIVED,
        InspectionStatus.VALIDATION_FAILED,
        InspectionStatus.ERROR,
        InspectionStatus.PROCESSING,
        InspectionStatus.PASS,
        InspectionStatus.FAIL,
        InspectionStatus.UNCERTAIN,
    ],
)
def test_only_ready_inspections_can_begin(tmp_path, status):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / status.value)
        inspection_id, validation_id = str(uuid4()), str(uuid4())
        try:
            await repositories.inspections.create(InspectionCreate(
                id=inspection_id, status=status, board_id="B", recipe_id="R",
                recipe_version="1", completed_at=(
                    FIXED_TIME if status in {
                        InspectionStatus.PASS, InspectionStatus.FAIL,
                        InspectionStatus.UNCERTAIN,
                    } else None
                ),
            ))
            pre, inf = _results(inspection_id, validation_id)
            identity = _identity(inspection_id, validation_id, "a" * 64, pre, inf)
            with pytest.raises(InvalidProcessingTransitionError):
                await _service(database).begin_processing(identity, identity.generated_key())
            async with database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionProcessingRun)) == 0
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_missing_wrong_owner_and_not_passed_validation_cannot_begin(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            inspection_id = str(uuid4())
            await repositories.inspections.create(InspectionCreate(
                id=inspection_id, status=InspectionStatus.READY, board_id="B",
                recipe_id="R", recipe_version="1",
            ))
            missing_id = str(uuid4())
            pre, inf = _results(inspection_id, missing_id)
            missing = _identity(inspection_id, missing_id, "a" * 64, pre, inf)
            with pytest.raises(ProcessingValidationNotFoundError):
                await _service(database).begin_processing(missing, missing.generated_key())

            owner_id, validation_id, validation_hash = await _ready(database, repositories)
            pre, inf = _results(inspection_id, validation_id)
            wrong_owner = _identity(inspection_id, validation_id, validation_hash, pre, inf)
            with pytest.raises(ProcessingLifecycleConflictError, match="belong"):
                await _service(database).begin_processing(wrong_owner, wrong_owner.generated_key())

            async with database.session_factory.begin() as session:
                validation = await session.get(InspectionValidation, validation_id)
                validation.inspection_id = inspection_id
                validation.outcome = "VALIDATION_FAILED"
            not_passed = _identity(inspection_id, validation_id, validation_hash, pre, inf)
            with pytest.raises(ProcessingLifecycleConflictError, match="passed"):
                await _service(database).begin_processing(not_passed, not_passed.generated_key())
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_completion_shape_rules_and_invalid_catalogue_or_taxonomy_are_rejected(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            identity, pre, inf, _ = await _started(database, repositories)
            with pytest.raises(ProcessingLifecycleConflictError, match="requires an inference"):
                await _service(database).complete_processing(identity.processing_run_id, pre, None)
            failed = preprocessing_result(
                "inspection_preprocessing_result.failed.json",
                inspection_id=identity.inspection_id, validation_id=identity.validation_id,
                preprocessing_id=pre.preprocessing_id,
            )
            failed = replace(failed, rgb_input=pre.rgb_input, height_input=pre.height_input)
            with pytest.raises(ProcessingLifecycleConflictError, match="cannot contain"):
                await _service(database).complete_processing(identity.processing_run_id, failed, inf)
            bad_finding = replace(
                pre, findings=(replace(pre.findings[0], code="UNKNOWN_FINDING"),)
            )
            with pytest.raises(ValueError, match="contract-valid"):
                await _service(database).complete_processing(identity.processing_run_id, bad_finding, inf)
            failed_inf = _results(identity.inspection_id, identity.validation_id, "FAIL")[1]
            failed_inf = replace(
                failed_inf, preprocessing_id=pre.preprocessing_id,
                inference_id=inf.inference_id, defect_type="invented_defect",
            )
            with pytest.raises(ValueError, match="contract-valid"):
                await _service(database).complete_processing(identity.processing_run_id, pre, failed_inf)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_completion_audit_failure_rolls_back_all_completion_effects(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            identity, pre, inf, _ = await _started(database, repositories)
            audits = await repositories.audit_events.list_for_entity("inspection", identity.inspection_id)
            service = _service(database, audit_id_generator=lambda: audits[0].id)
            from app.services.inspection_processing import ProcessingLifecycleError
            with pytest.raises(ProcessingLifecycleError):
                await service.complete_processing(identity.processing_run_id, pre, inf)
            inspection = await repositories.inspections.get(identity.inspection_id)
            run = await repositories.processing.get_run_by_id(identity.processing_run_id)
            assert inspection.status is InspectionStatus.PROCESSING
            assert run.status.value == "STARTED"
            assert await repositories.processing.get_preprocessing_result(identity.processing_run_id) is None
            assert await repositories.processing.get_inference_result(identity.processing_run_id) is None
            assert len(await repositories.audit_events.list_for_entity("inspection", identity.inspection_id)) == 1
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_conditional_completion_failure_rolls_back_results_run_and_audit(tmp_path, monkeypatch):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            identity, pre, inf, _ = await _started(database, repositories)
            service = _service(database)

            async def no_transition(*_args):
                return 0

            monkeypatch.setattr(service, "_complete_inspection_transition", no_transition)
            with pytest.raises(InvalidProcessingTransitionError):
                await service.complete_processing(identity.processing_run_id, pre, inf)
            assert (await repositories.inspections.get(identity.inspection_id)).status is InspectionStatus.PROCESSING
            assert (await repositories.processing.get_run_by_id(identity.processing_run_id)).status.value == "STARTED"
            assert await repositories.processing.get_preprocessing_result(identity.processing_run_id) is None
            assert await repositories.processing.get_inference_result(identity.processing_run_id) is None
            assert len(await repositories.audit_events.list_for_entity("inspection", identity.inspection_id)) == 1
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_concurrent_identical_and_conflicting_completion_are_serialized(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            identity, pre, inf, _ = await _started(database, repositories)
            outcomes = await asyncio.gather(
                _service(database).complete_processing(identity.processing_run_id, pre, inf),
                _service(database).complete_processing(identity.processing_run_id, pre, inf),
            )
            assert sum(not item.idempotent_existing for item in outcomes) == 1
            async with database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionPreprocessingResult)) == 1
                assert await session.scalar(select(func.count()).select_from(InspectionInferenceResult)) == 1
                assert await session.scalar(select(func.count()).select_from(AuditEvent)) == 2

            identity2, pre2, inf2, _ = await _started(database, repositories)
            changed = replace(inf2, inference_id=str(uuid4()))
            outcomes = await asyncio.gather(
                _service(database).complete_processing(identity2.processing_run_id, pre2, inf2),
                _service(database).complete_processing(identity2.processing_run_id, pre2, changed),
                return_exceptions=True,
            )
            assert sum(not isinstance(item, Exception) for item in outcomes) == 1
            assert any(isinstance(item, ProcessingPersistenceConflictError) for item in outcomes)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_repository_is_narrow_and_coordinator_has_no_execution_or_file_side_effects(tmp_path, monkeypatch):
    public = {name for name in dir(InspectionProcessingRepository) if not name.startswith("_")}
    assert public == {
        "get_run_by_id", "get_run_by_inspection_and_key", "get_latest_run_for_inspection",
        "get_preprocessing_result", "get_inference_result",
        "list_preprocessing_findings", "list_inference_findings",
    }

    async def scenario():
        database, repositories, paths = await _database(tmp_path / "runtime")
        raw = paths.raw_uploads / "unchanged.bin"
        raw.write_bytes(b"unchanged")
        before = (raw.read_bytes(), raw.stat().st_mtime_ns)
        try:
            identity, pre, inf, _ = await _started(database, repositories)

            async def must_not_execute(*_args, **_kwargs):
                raise AssertionError("execution service called")

            monkeypatch.setattr(
                "app.services.inspection_preprocessing.service.SyntheticInspectionPreprocessingService.preprocess_inspection",
                must_not_execute,
            )
            monkeypatch.setattr(
                "app.services.inspection_inference.service.SyntheticMockInferenceService.run_inference",
                must_not_execute,
            )
            await _service(database).complete_processing(identity.processing_run_id, pre, inf)
            assert before == (raw.read_bytes(), raw.stat().st_mtime_ns)
            assert list(paths.previews.iterdir()) == []
            assert list(paths.reports.iterdir()) == []
            assert list(paths.results.iterdir()) == []
        finally:
            await database.dispose()
    asyncio.run(scenario())
