import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.core.logging import APPLICATION_LOGGER_NAME
from app.core.runtime_paths import RuntimePaths
from app.db.database import Database
from app.db.models import (
    ArtifactType,
    ModelCompatibilityStatus,
    ModelStatus,
    ModelVersion,
    SCHEMA_VERSION,
    InspectionStatus,
)
from app.db.repositories import (
    AuditEventCreate,
    InspectionArtifactCreate,
    InspectionCreate,
    ModelVersionCreate,
    RecipeCreate,
    Repositories,
)
from app.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _settings(runtime_root: Path, **overrides) -> Settings:
    return Settings(_env_file=None, runtime_root=runtime_root, **overrides)


async def _initialized_database(
    runtime_root: Path,
    *,
    busy_timeout_ms: int = 4321,
) -> tuple[Database, Repositories, RuntimePaths]:
    paths = RuntimePaths.from_root(runtime_root)
    paths.create_directories()
    database = Database(
        paths.database_file,
        busy_timeout_ms=busy_timeout_ms,
    )
    await database.initialize()
    return (
        database,
        Repositories.from_session_factory(database.session_factory),
        paths,
    )


def test_importing_application_creates_no_database_file(tmp_path) -> None:
    runtime_root = tmp_path / "import-only-runtime"
    environment = os.environ.copy()
    environment["PCB_AOI_RUNTIME_ROOT"] = str(runtime_root)
    environment["PYTHONPATH"] = str(BACKEND_ROOT)

    subprocess.run(
        [sys.executable, "-c", "from app.main import app"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not runtime_root.exists()


def test_app_lifespan_creates_database_below_runtime_root(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    application = create_app(_settings(runtime_root))

    assert not application.state.runtime_paths.database_file.exists()
    with TestClient(application):
        assert application.state.runtime_paths.database_file.is_file()
    assert application.state.runtime_paths.database_file == (
        runtime_root / "database" / "pcb_aoi.sqlite3"
    ).resolve()


def test_required_tables_and_schema_version_exist(tmp_path) -> None:
    async def scenario() -> None:
        database, _, _ = await _initialized_database(tmp_path / "runtime")
        try:
            async with database.engine.connect() as connection:
                table_names = await connection.run_sync(
                    lambda sync_connection: set(
                        inspect(sync_connection).get_table_names()
                    )
                )
                version = await connection.scalar(
                    text("SELECT version FROM schema_version WHERE id = 1")
                )
            assert {
                "schema_version",
                "inspections",
                "inspection_artifacts",
                "recipes",
                "model_versions",
                "audit_events",
            } <= table_names
            assert version == SCHEMA_VERSION
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_sqlite_pragmas_are_effective(tmp_path) -> None:
    async def scenario() -> None:
        database, _, _ = await _initialized_database(
            tmp_path / "runtime",
            busy_timeout_ms=4321,
        )
        try:
            async with database.engine.connect() as connection:
                foreign_keys = await connection.scalar(text("PRAGMA foreign_keys"))
                journal_mode = await connection.scalar(text("PRAGMA journal_mode"))
                busy_timeout = await connection.scalar(text("PRAGMA busy_timeout"))
            assert foreign_keys == 1
            assert str(journal_mode).lower() == "wal"
            assert busy_timeout == 4321
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_repeated_startup_is_idempotent_and_content_persists(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())

    first_app = create_app(_settings(runtime_root))
    with TestClient(first_app):
        created = asyncio.run(
            first_app.state.repositories.inspections.create(
                InspectionCreate(
                    id=inspection_id,
                    status=InspectionStatus.RECEIVED,
                    board_id="BOARD-1",
                    recipe_id="RECIPE-1",
                    recipe_version="1.0",
                )
            )
        )
        assert created.id == inspection_id

    second_app = create_app(_settings(runtime_root))
    with TestClient(second_app):
        retrieved = asyncio.run(
            second_app.state.repositories.inspections.get(inspection_id)
        )
        assert retrieved is not None
        assert retrieved.board_id == "BOARD-1"


def test_inspection_create_get_ordering_and_pagination(tmp_path) -> None:
    async def scenario() -> None:
        database, repositories, _ = await _initialized_database(tmp_path / "runtime")
        try:
            base_time = datetime(2026, 7, 17, tzinfo=timezone.utc)
            ids = [str(uuid4()) for _ in range(3)]
            for index, inspection_id in enumerate(ids):
                await repositories.inspections.create(
                    InspectionCreate(
                        id=inspection_id,
                        status=InspectionStatus.RECEIVED,
                        board_id=f"BOARD-{index}",
                        recipe_id="RECIPE-1",
                        recipe_version="1.0",
                        created_at=base_time + timedelta(seconds=index),
                    )
                )
            retrieved = await repositories.inspections.get(ids[1])
            assert retrieved is not None
            assert retrieved.board_id == "BOARD-1"
            first_page = await repositories.inspections.list(limit=2, offset=0)
            second_page = await repositories.inspections.list(limit=2, offset=2)
            assert [record.id for record in first_page] == [ids[2], ids[1]]
            assert [record.id for record in second_page] == [ids[0]]
            with pytest.raises(ValueError, match="limit"):
                await repositories.inspections.list(limit=0)
            with pytest.raises(ValueError, match="offset"):
                await repositories.inspections.list(offset=-1)
        finally:
            await database.dispose()

    asyncio.run(scenario())


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_invalid_confidence_is_rejected(confidence) -> None:
    with pytest.raises(ValueError, match="confidence"):
        InspectionCreate(
            status=InspectionStatus.RECEIVED,
            board_id="BOARD-1",
            recipe_id="RECIPE-1",
            recipe_version="1.0",
            confidence=confidence,
        )


def test_negative_processing_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="processing_ms"):
        InspectionCreate(
            status=InspectionStatus.RECEIVED,
            board_id="BOARD-1",
            recipe_id="RECIPE-1",
            recipe_version="1.0",
            processing_ms=-1,
        )


def test_final_inspection_requires_completed_at() -> None:
    with pytest.raises(ValueError, match="completed_at"):
        InspectionCreate(
            status=InspectionStatus.PASS,
            board_id="BOARD-1",
            recipe_id="RECIPE-1",
            recipe_version="1.0",
        )


@pytest.mark.parametrize(
    "relative_path",
    ["C:/absolute/rgb.png", "/absolute/rgb.png", "../escape.png", "raw/../escape.png"],
)
def test_unsafe_artifact_paths_are_rejected(relative_path) -> None:
    with pytest.raises(ValueError, match="path"):
        InspectionArtifactCreate(
            inspection_id=str(uuid4()),
            artifact_type=ArtifactType.RGB_RAW,
            relative_path=relative_path,
            sha256="a" * 64,
            byte_size=1,
        )


def test_invalid_artifact_sha256_is_rejected() -> None:
    with pytest.raises(ValueError, match="sha256"):
        InspectionArtifactCreate(
            inspection_id=str(uuid4()),
            artifact_type=ArtifactType.RGB_RAW,
            relative_path="raw/rgb.png",
            sha256="INVALID",
            byte_size=1,
        )


def test_artifact_foreign_key_integrity_is_enforced(tmp_path) -> None:
    async def scenario() -> None:
        database, repositories, _ = await _initialized_database(tmp_path / "runtime")
        try:
            with pytest.raises(IntegrityError):
                await repositories.artifacts.create(
                    InspectionArtifactCreate(
                        inspection_id=str(uuid4()),
                        artifact_type=ArtifactType.RGB_RAW,
                        relative_path="raw/rgb.png",
                        sha256="a" * 64,
                        byte_size=100,
                    )
                )
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_duplicate_recipe_version_is_rejected(tmp_path) -> None:
    async def scenario() -> None:
        database, repositories, _ = await _initialized_database(tmp_path / "runtime")
        recipe = RecipeCreate(
            recipe_id="RECIPE-1",
            recipe_version="1.0",
            name="Recipe one",
            configuration={"threshold": 0.5},
        )
        try:
            await repositories.recipes.register(recipe)
            with pytest.raises(IntegrityError):
                await repositories.recipes.register(recipe)
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_duplicate_model_version_is_rejected_and_legacy_model_is_not_registered(
    tmp_path,
) -> None:
    async def scenario() -> None:
        database, repositories, _ = await _initialized_database(tmp_path / "runtime")
        model = ModelVersionCreate(
            model_id="MODEL-1",
            model_version="1.0",
            engine_type="ONNX_RUNTIME",
            class_label_contract_version="1.0",
            defect_taxonomy_version="pcb-aoi-defects/1.0",
            compatibility_status=ModelCompatibilityStatus.UNVERIFIED,
            status=ModelStatus.BLOCKED,
        )
        try:
            async with database.session() as session:
                initial_count = await session.scalar(
                    select(func.count()).select_from(ModelVersion)
                )
            assert initial_count == 0
            await repositories.models.register(model)
            with pytest.raises(IntegrityError):
                await repositories.models.register(model)
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_audit_event_is_append_only_repository_data_and_retrievable(tmp_path) -> None:
    async def scenario() -> None:
        database, repositories, _ = await _initialized_database(tmp_path / "runtime")
        try:
            event = await repositories.audit_events.append(
                AuditEventCreate(
                    entity_type="inspection",
                    entity_id="inspection-1",
                    action="CREATED",
                    actor_id="operator-1",
                    request_id="request-1",
                    details={"source": "test"},
                )
            )
            retrieved = await repositories.audit_events.get(event.id)
            listed = await repositories.audit_events.list_for_entity(
                "inspection", "inspection-1"
            )
            assert retrieved is not None
            assert json.loads(retrieved.details_json) == {"source": "test"}
            assert [record.id for record in listed] == [event.id]
        finally:
            await database.dispose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "filename",
    ["../outside.sqlite3", "sub/database.sqlite3", "C:\\temp\\database.sqlite3"],
)
def test_database_filename_cannot_escape_runtime_root(filename) -> None:
    with pytest.raises(ValidationError, match="database filename"):
        Settings(_env_file=None, database_filename=filename)


@pytest.mark.parametrize("timeout", [0, 60001])
def test_busy_timeout_has_bounded_positive_validation(timeout) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, sqlite_busy_timeout_ms=timeout)


def test_database_environment_overrides_are_respected(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PCB_AOI_DATABASE_FILENAME", "custom.sqlite3")
    monkeypatch.setenv("PCB_AOI_SQLITE_BUSY_TIMEOUT_MS", "8765")
    monkeypatch.setenv("PCB_AOI_DATABASE_ECHO", "true")

    settings = Settings(_env_file=None, runtime_root=tmp_path / "runtime")
    paths = RuntimePaths.from_root(
        settings.runtime_root,
        settings.database_filename,
    )

    assert settings.database_filename == "custom.sqlite3"
    assert settings.sqlite_busy_timeout_ms == 8765
    assert settings.database_echo is True
    assert paths.database_file == (
        tmp_path / "runtime" / "database" / "custom.sqlite3"
    ).resolve()


def test_public_health_contract_remains_unchanged(tmp_path) -> None:
    with TestClient(create_app(_settings(tmp_path / "runtime"))) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "pcb-aoi-api",
        "version": "0.1.0",
        "environment": "development",
    }


def test_startup_fails_and_logs_when_database_creation_is_impossible(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    database_directory = runtime_root / "database"
    database_directory.mkdir(parents=True)
    blocked_database_path = database_directory / "pcb_aoi.sqlite3"
    blocked_database_path.mkdir()
    application = create_app(_settings(runtime_root))
    logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    records: list[logging.LogRecord] = []

    class RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = RecordingHandler()
    logger.addHandler(handler)
    try:
        with pytest.raises(Exception):
            with TestClient(application):
                pytest.fail("startup should not yield when database initialization fails")
    finally:
        logger.removeHandler(handler)
        handler.close()

    error_records = [record for record in records if record.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert error_records[0].getMessage() == "Database initialization failed"
    assert not any(
        record.getMessage().startswith("Application startup") for record in records
    )
