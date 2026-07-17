import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.core.runtime_paths import RuntimePaths
from app.db.database import Database
from app.db.migrations.runner import DEFAULT_MIGRATIONS, Migration, MigrationRunner
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
    Recipe,
    SCHEMA_VERSION,
)
from app.db.repositories import (
    AuditEventCreate,
    InspectionArtifactCreate,
    InspectionCreate,
    ModelVersionCreate,
    RecipeCreate,
    Repositories,
)
from app.services.inspection_validation import (
    ArtifactTechnicalSummary,
    FindingCategory,
    FindingSeverity,
    InspectionValidationRepository,
    InspectionValidationResult,
    ReadabilityStatus,
    ValidationFinding,
    ValidationKeyArtifact,
    ValidationOutcome,
    ValidationPersistenceConflictError,
    ValidationPersistenceIntegrityError,
    ValidationSummary,
    canonical_result_bytes,
    canonical_result_sha256,
    generate_validation_key,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "contracts" / "examples"
FIXED_CREATED_AT = datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc)


def _timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _typed_example(name, *, inspection_id=None, validation_id=None):
    value = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    artifact_values = []
    for field in ("rgb_artifact", "height_artifact"):
        item = value[field]
        artifact_values.append(ArtifactTechnicalSummary(
            artifact_type=ArtifactType(item["artifact_type"]),
            sha256=item["sha256"], byte_size=item["byte_size"],
            declared_media_type=item["declared_media_type"],
            detected_format=item["detected_format"], width=item["width"],
            height=item["height"], channels=item["channels"],
            bit_depth=item["bit_depth"], storage_data_type=item["storage_data_type"],
            readability_status=ReadabilityStatus(item["readability_status"]),
        ))
    findings = tuple(ValidationFinding(
        code=item["code"], severity=FindingSeverity(item["severity"]),
        category=FindingCategory(item["category"]), message=item["message"],
        blocking=item["blocking"],
        artifact_type=ArtifactType(item["artifact_type"]) if item.get("artifact_type") else None,
        field=item.get("field"), details=item.get("details"),
    ) for item in value["findings"])
    summary = ValidationSummary(**value["summary"])
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
        rgb_artifact=artifact_values[0], height_artifact=artifact_values[1],
        findings=findings, summary=summary,
    )


def _key(result, *, rgb_hash="a" * 64, height_hash="b" * 64, policy_version=None, validator_version=None):
    return generate_validation_key(
        inspection_id=result.inspection_id,
        rgb_artifact=ValidationKeyArtifact(ArtifactType.RGB_RAW, rgb_hash, 100),
        height_artifact=ValidationKeyArtifact(ArtifactType.HEIGHT_RAW, height_hash, 200),
        contract_version=result.contract_version,
        policy_id=result.validation_policy_id,
        policy_version=policy_version or result.validation_policy_version,
        validator_version=validator_version or result.validator_version,
    )


async def _database(runtime):
    paths = RuntimePaths.from_root(runtime)
    paths.create_directories()
    database = Database(paths.database_file, busy_timeout_ms=4321)
    await database.initialize()
    return database, Repositories.from_session_factory(database.session_factory), paths


async def _inspection(repositories, inspection_id, status=InspectionStatus.RECEIVED):
    return await repositories.inspections.create(InspectionCreate(
        id=inspection_id, status=status, board_id="BOARD-1",
        recipe_id="RECIPE-1", recipe_version="1.0",
    ))


def test_new_database_initializes_at_schema_two_with_required_tables_and_pragmas(tmp_path):
    async def scenario():
        database, _, _ = await _database(tmp_path / "runtime")
        try:
            async with database.engine.connect() as connection:
                tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
                version = await connection.scalar(text("SELECT version FROM schema_version WHERE id=1"))
                foreign_keys = await connection.scalar(text("PRAGMA foreign_keys"))
                journal = await connection.scalar(text("PRAGMA journal_mode"))
                timeout = await connection.scalar(text("PRAGMA busy_timeout"))
            assert version == SCHEMA_VERSION == 2
            assert {"inspection_validations", "inspection_validation_findings"} <= tables
            assert foreign_keys == 1 and str(journal).lower() == "wal" and timeout == 4321
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_existing_version_one_database_migrates_and_preserves_all_existing_entities(tmp_path):
    async def scenario():
        paths = RuntimePaths.from_root(tmp_path / "runtime")
        paths.create_directories()
        database = Database(paths.database_file, busy_timeout_ms=5000)
        await MigrationRunner(database.engine).run(target_version=1)
        repositories = Repositories.from_session_factory(database.session_factory)
        inspection_id = str(uuid4())
        await _inspection(repositories, inspection_id)
        artifact = await repositories.artifacts.create(InspectionArtifactCreate(
            inspection_id=inspection_id, artifact_type=ArtifactType.RGB_RAW,
            relative_path=f"raw_uploads/{inspection_id}/rgb/rgb_raw.png",
            sha256="a" * 64, byte_size=10, media_type="image/png",
        ))
        await repositories.recipes.register(RecipeCreate(
            recipe_id="RECIPE-1", recipe_version="1.0", name="Recipe",
            configuration={"policy": "development"},
        ))
        await repositories.models.register(ModelVersionCreate(
            model_id="MODEL-1", model_version="1.0", engine_type="MOCK",
            class_label_contract_version="1.0", defect_taxonomy_version="1.0",
            compatibility_status=ModelCompatibilityStatus.UNVERIFIED,
            status=ModelStatus.BLOCKED,
        ))
        event_record = await repositories.audit_events.append(AuditEventCreate(
            entity_type="inspection", entity_id=inspection_id,
            action="CREATED", details={"source": "migration-test"},
        ))
        await database.initialize()
        try:
            async with database.session() as session:
                version = await session.scalar(text("SELECT version FROM schema_version WHERE id=1"))
                counts = [await session.scalar(select(func.count()).select_from(model)) for model in (Recipe, ModelVersion, AuditEvent)]
            assert version == 2
            assert (await repositories.inspections.get(inspection_id)).id == inspection_id
            assert (await repositories.artifacts.get(artifact.id)).sha256 == "a" * 64
            assert counts == [1, 1, 1]
            assert (await repositories.audit_events.get(event_record.id)).id == event_record.id
            await database.initialize()
            async with database.session() as session:
                assert await session.scalar(text("SELECT version FROM schema_version WHERE id=1")) == 2
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_unknown_schema_version_fails_safely(tmp_path):
    async def scenario():
        paths = RuntimePaths.from_root(tmp_path / "runtime")
        paths.create_directories()
        database = Database(paths.database_file, busy_timeout_ms=5000)
        await MigrationRunner(database.engine).run(target_version=1)
        async with database.engine.begin() as connection:
            await connection.execute(text("UPDATE schema_version SET version=99 WHERE id=1"))
        with pytest.raises(RuntimeError, match="Unsupported database schema version 99"):
            await database.initialize()
        await database.dispose()
    asyncio.run(scenario())


def test_failed_migration_does_not_report_version_two(tmp_path):
    async def scenario():
        paths = RuntimePaths.from_root(tmp_path / "runtime")
        paths.create_directories()
        database = Database(paths.database_file, busy_timeout_ms=5000)
        await MigrationRunner(database.engine).run(target_version=1)

        async def fail_after_ddl(connection):
            await DEFAULT_MIGRATIONS[1].upgrade(connection)
            raise RuntimeError("simulated migration failure")

        failing = Migration(2, "002_validation_results_failure_test", DEFAULT_MIGRATIONS[1].required_table_names, fail_after_ddl)
        runner = MigrationRunner(database.engine, (DEFAULT_MIGRATIONS[0], failing))
        with pytest.raises(RuntimeError, match="simulated migration failure"):
            await runner.run()
        async with database.engine.connect() as connection:
            version = await connection.scalar(text("SELECT version FROM schema_version WHERE id=1"))
        assert version == 1
        await database.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize("example", [
    "inspection_validation_result.passed.json",
    "inspection_validation_result.failed.json",
    "inspection_validation_result.error.json",
])
def test_passed_failed_and_error_results_persist_and_retrieve(example, tmp_path):
    async def scenario():
        result = _typed_example(example)
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            saved = await repositories.validations.save_validation_result(result.inspection_id, result, _key(result))
            retrieved = await repositories.validations.get_by_validation_id(result.validation_id)
            by_key = await repositories.validations.get_by_inspection_and_key(result.inspection_id, saved.validation_key)
            findings = await repositories.validations.list_findings(result.validation_id)
            assert saved.outcome is result.outcome and not saved.idempotent_existing
            assert retrieved.result == result.to_dict() == by_key.result
            assert [item.ordinal for item in findings] == list(range(len(result.findings)))
            assert [item.code for item in findings] == [item.code for item in result.findings]
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_canonical_result_serialization_and_sha256_are_exact_and_deterministic():
    result = _typed_example("inspection_validation_result.passed.json")
    first = canonical_result_bytes(result)
    second = canonical_result_bytes(result)
    assert first == second
    assert canonical_result_sha256(result) == hashlib.sha256(first).hexdigest()
    assert json.loads(first) == result.to_dict()


def test_validation_key_is_deterministic_order_independent_for_evidence_and_sensitive_to_required_inputs():
    result = _typed_example("inspection_validation_result.passed.json")
    mask = ValidationKeyArtifact(ArtifactType.VALIDITY_MASK, "c" * 64, 30)
    calibration = ValidationKeyArtifact(ArtifactType.CALIBRATION, "d" * 64, 40)
    base = dict(
        inspection_id=result.inspection_id,
        rgb_artifact=ValidationKeyArtifact(ArtifactType.RGB_RAW, "a" * 64, 100),
        height_artifact=ValidationKeyArtifact(ArtifactType.HEIGHT_RAW, "b" * 64, 200),
        contract_version=result.contract_version, policy_id=result.validation_policy_id,
        policy_version=result.validation_policy_version, validator_version=result.validator_version,
    )
    first = generate_validation_key(**base, evidence_artifacts=(mask, calibration))
    assert first == generate_validation_key(**base, evidence_artifacts=(calibration, mask))
    assert first != generate_validation_key(**{**base, "rgb_artifact": replace(base["rgb_artifact"], sha256="e" * 64)}, evidence_artifacts=(mask, calibration))
    assert first != generate_validation_key(**{**base, "height_artifact": replace(base["height_artifact"], sha256="e" * 64)}, evidence_artifacts=(mask, calibration))
    assert first != generate_validation_key(**{**base, "policy_version": "2.0"}, evidence_artifacts=(mask, calibration))
    assert first != generate_validation_key(**{**base, "validator_version": "2.0.0"}, evidence_artifacts=(mask, calibration))


def test_idempotent_replay_returns_existing_without_duplicate_findings(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.failed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            key = _key(result)
            first = await repositories.validations.save_validation_result(result.inspection_id, result, key)
            second = await repositories.validations.save_validation_result(result.inspection_id, result, key)
            findings = await repositories.validations.list_findings(result.validation_id)
            assert not first.idempotent_existing and second.idempotent_existing
            assert first.validation_id == second.validation_id
            assert len(findings) == len(result.findings)
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_same_key_with_different_result_raises_conflict_without_overwrite(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            key = _key(result)
            await repositories.validations.save_validation_result(result.inspection_id, result, key)
            changed = replace(result, validation_id=str(uuid4()), validator_version="1.0.1")
            with pytest.raises(ValidationPersistenceConflictError):
                await repositories.validations.save_validation_result(result.inspection_id, changed, key)
            persisted = await repositories.validations.get_by_inspection_and_key(result.inspection_id, key)
            assert persisted.validation_id == result.validation_id
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_latest_validation_order_is_completed_created_then_id(tmp_path):
    async def scenario():
        inspection_id = str(uuid4())
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, inspection_id)
            older = _typed_example("inspection_validation_result.passed.json", inspection_id=inspection_id, validation_id=str(uuid4()))
            newer = replace(
                older,
                validation_id=str(uuid4()),
                validator_version="1.0.1",
                started_at=older.started_at + timedelta(minutes=1),
                completed_at=older.completed_at + timedelta(minutes=1),
            )
            await repositories.validations.save_validation_result(inspection_id, older, _key(older))
            await repositories.validations.save_validation_result(inspection_id, newer, _key(newer))
            latest = await repositories.validations.get_latest_for_inspection(inspection_id)
            assert latest.validation_id == newer.validation_id
        finally:
            await database.dispose()
    asyncio.run(scenario())


def _recount(findings, technically_ready):
    return ValidationSummary(
        finding_count=len(findings),
        info_count=sum(item.severity is FindingSeverity.INFO for item in findings),
        warning_count=sum(item.severity is FindingSeverity.WARNING for item in findings),
        error_count=sum(item.severity is FindingSeverity.ERROR for item in findings),
        blocking_count=sum(item.blocking for item in findings),
        technically_ready=technically_ready,
        synthetic_example=True,
    )


@pytest.mark.parametrize("mutation", [
    "invalid_outcome", "unknown_code", "invalid_severity", "invalid_category",
    "invalid_artifact", "noncanonical_uuid", "naive_time", "reversed_time",
    "bad_summary", "passed_blocking", "failed_nonblocking", "error_no_internal",
    "unsafe_details",
])
def test_invalid_typed_results_are_rejected_before_insertion(tmp_path, mutation):
    async def scenario():
        base_name = "inspection_validation_result.passed.json"
        if mutation == "failed_nonblocking":
            base_name = "inspection_validation_result.failed.json"
        elif mutation == "error_no_internal":
            base_name = "inspection_validation_result.error.json"
        result = _typed_example(base_name)
        if mutation == "invalid_outcome":
            result = replace(result, outcome="PASS")
        elif mutation in {"unknown_code", "invalid_severity", "invalid_category", "invalid_artifact", "unsafe_details"}:
            finding = result.findings[0]
            changes = {
                "unknown_code": {"code": "UNKNOWN"},
                "invalid_severity": {"severity": "ERROR"},
                "invalid_category": {"category": "PAIR"},
                "invalid_artifact": {"artifact_type": "RGB_RAW"},
                "unsafe_details": {"details": {"absolute_path": "C:\\secret\\file"}},
            }[mutation]
            result = replace(result, findings=(replace(finding, **changes),))
            result = replace(result, summary=_recount(result.findings, result.outcome is ValidationOutcome.VALIDATION_PASSED))
        elif mutation == "noncanonical_uuid":
            result = replace(result, validation_id="AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA")
        elif mutation == "naive_time":
            result = replace(result, started_at=result.started_at.replace(tzinfo=None))
        elif mutation == "reversed_time":
            result = replace(result, completed_at=result.started_at - timedelta(seconds=1))
        elif mutation == "bad_summary":
            result = replace(result, summary=replace(result.summary, finding_count=999))
        elif mutation == "passed_blocking":
            finding = replace(result.findings[0], blocking=True)
            result = replace(result, findings=(finding,), summary=_recount((finding,), True))
        elif mutation == "failed_nonblocking":
            findings = tuple(replace(item, blocking=False) for item in result.findings)
            result = replace(result, findings=findings, summary=_recount(findings, False))
        elif mutation == "error_no_internal":
            finding = ValidationFinding("FILE_UNREADABLE", FindingSeverity.ERROR, FindingCategory.FORMAT, "Unreadable", True)
            result = replace(result, findings=(finding,), summary=_recount((finding,), False))

        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            with pytest.raises((TypeError, ValueError)):
                await repositories.validations.save_validation_result(result.inspection_id, result, "a" * 64)
            async with database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionValidation)) == 0
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_finding_insertion_failure_rolls_back_validation_result(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.failed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        await _inspection(repositories, result.inspection_id)
        duplicate_id = str(uuid4())
        repository = InspectionValidationRepository(
            database.session_factory,
            finding_id_generator=lambda: duplicate_id,
        )
        try:
            with pytest.raises(ValidationPersistenceIntegrityError):
                await repository.save_validation_result(result.inspection_id, result, _key(result))
            async with database.session() as session:
                validations = await session.scalar(select(func.count()).select_from(InspectionValidation))
                findings = await session.scalar(select(func.count()).select_from(InspectionValidationFinding))
            assert validations == findings == 0
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_missing_inspection_foreign_key_fails_without_partial_rows(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            with pytest.raises(ValidationPersistenceIntegrityError):
                await repositories.validations.save_validation_result(result.inspection_id, result, _key(result))
            async with database.session() as session:
                assert await session.scalar(select(func.count()).select_from(InspectionValidation)) == 0
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_database_finding_constraints_reject_negative_and_duplicate_ordinals(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id)
            await repositories.validations.save_validation_result(result.inspection_id, result, _key(result))
            common = dict(
                validation_id=result.validation_id, code="FILE_UNREADABLE",
                severity=FindingSeverity.ERROR, category=FindingCategory.FORMAT,
                message="Unreadable", blocking=True, details_json="{}",
                created_at=FIXED_CREATED_AT,
            )
            with pytest.raises(IntegrityError):
                async with database.session_factory() as session, session.begin():
                    session.add(InspectionValidationFinding(id=str(uuid4()), ordinal=-1, **common))
                    await session.flush()
            with pytest.raises(IntegrityError):
                async with database.session_factory() as session, session.begin():
                    session.add(InspectionValidationFinding(id=str(uuid4()), ordinal=0, **common))
                    await session.flush()
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_database_rejects_finding_without_validation_parent(tmp_path):
    async def scenario():
        database, _, _ = await _database(tmp_path / "runtime")
        try:
            async with database.session_factory.begin() as session:
                session.add(InspectionValidationFinding(
                    id=str(uuid4()), validation_id=str(uuid4()), ordinal=0,
                    code="FILE_UNREADABLE", severity=FindingSeverity.ERROR,
                    category=FindingCategory.FORMAT, message="Unreadable",
                    artifact_type=ArtifactType.RGB_RAW, field=None, blocking=True,
                    details_json="{}", created_at=datetime.now(timezone.utc),
                ))
                with pytest.raises(IntegrityError):
                    await session.flush()
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_validation_repository_exposes_append_and_read_operations_only(tmp_path):
    async def scenario():
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            public_methods = {
                name for name in dir(repositories.validations)
                if not name.startswith("_")
            }
            assert public_methods == {
                "get_by_inspection_and_key",
                "get_by_validation_id",
                "get_latest_for_inspection",
                "list_findings",
                "save_validation_result",
            }
        finally:
            await database.dispose()
    asyncio.run(scenario())


def test_persistence_leaves_inspection_audit_and_artifacts_unchanged_and_returns_no_paths(tmp_path):
    async def scenario():
        result = _typed_example("inspection_validation_result.passed.json")
        database, repositories, _ = await _database(tmp_path / "runtime")
        try:
            await _inspection(repositories, result.inspection_id, InspectionStatus.READY)
            artifact = await repositories.artifacts.create(InspectionArtifactCreate(
                inspection_id=result.inspection_id, artifact_type=ArtifactType.RGB_RAW,
                relative_path=f"raw_uploads/{result.inspection_id}/rgb/rgb_raw.png",
                sha256="a" * 64, byte_size=100,
            ))
            before = await repositories.artifacts.get(artifact.id)
            async with database.session() as session:
                audit_before = await session.scalar(select(func.count()).select_from(AuditEvent))
            saved = await repositories.validations.save_validation_result(result.inspection_id, result, _key(result))
            inspection = await repositories.inspections.get(result.inspection_id)
            after = await repositories.artifacts.get(artifact.id)
            async with database.session() as session:
                audit_after = await session.scalar(select(func.count()).select_from(AuditEvent))
            assert inspection.status is InspectionStatus.READY
            assert (before.relative_path, before.sha256, before.byte_size) == (after.relative_path, after.sha256, after.byte_size)
            assert audit_before == audit_after == 0
            assert "path" not in repr(saved).lower()
        finally:
            await database.dispose()
    asyncio.run(scenario())
