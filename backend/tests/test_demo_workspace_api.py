import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.core.config import Settings
from app.db.models import (
    AuditEvent,
    Inspection,
    InspectionArtifact,
    InspectionInferenceResult,
    InspectionPreprocessingResult,
    InspectionProcessingRun,
    InspectionStatus,
    InspectionValidation,
    Recipe,
    SCHEMA_VERSION,
    SchemaVersion,
)
from app.db.repositories import InspectionCreate
from app.main import create_app

EXPECTED = {
    "mock_pass": ("PASS", "VALIDATION_PASSED", "COMPLETED", "PASS"),
    "mock_fail": ("FAIL", "VALIDATION_PASSED", "COMPLETED", "FAIL"),
    "mock_uncertain": (
        "UNCERTAIN", "VALIDATION_PASSED", "COMPLETED", "UNCERTAIN"
    ),
    "technical_error": ("ERROR", "VALIDATION_PASSED", "ERROR", None),
    "validation_failure": (
        "VALIDATION_FAILED", "VALIDATION_FAILED", None, None
    ),
}


def _application(tmp_path: Path, *, enabled: bool = True, configured: bool = True):
    return create_app(
        Settings(
            _env_file=None,
            runtime_root=tmp_path / "runtime",
            enable_demo_workspace=enabled,
            synthetic_fixture_root=(tmp_path / "fixtures") if configured else None,
        )
    )


async def _counts(application) -> tuple[int, ...]:
    models = (
        Recipe,
        Inspection,
        InspectionArtifact,
        InspectionValidation,
        InspectionProcessingRun,
        InspectionPreprocessingResult,
        InspectionInferenceResult,
        AuditEvent,
    )
    async with application.state.database.session() as session:
        counts = []
        for model in models:
            value = await session.scalar(select(func.count()).select_from(model))
            counts.append(int(value or 0))
        return tuple(counts)


async def _schema_version(application) -> int:
    async with application.state.database.session() as session:
        return int(await session.scalar(select(SchemaVersion.version)) or 0)


def _assert_expected(payload: dict) -> None:
    assert payload["enabled"] is True
    assert payload["available"] is True
    assert payload["loaded"] is True
    assert payload["recipes_ready"] is True
    assert payload["synthetic"] is True
    assert payload["production_approved"] is False
    assert len(payload["inspections"]) == 5
    for item in payload["inspections"]:
        assert item["complete"] is True
        assert (
            item["status"],
            item["validation_outcome"],
            item["processing_status"],
            item["mock_decision"],
        ) == EXPECTED[item["key"]]
    technical = next(
        item for item in payload["inspections"] if item["key"] == "technical_error"
    )
    assert technical["preprocessing_outcome"] == "PREPROCESSING_ERROR"


def test_first_load_uses_real_lifecycles_and_returns_expected_outcomes(tmp_path):
    application = _application(tmp_path)
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/development/demo-workspace/load",
            headers={"X-Request-ID": "demo-first-load"},
        )
        status = client.get("/api/v1/development/demo-workspace")

    assert response.status_code == status.status_code == 200, response.text
    _assert_expected(response.json())
    _assert_expected(status.json())
    assert response.json()["idempotent_existing"] is False
    assert status.json()["idempotent_existing"] is None
    assert response.json()["request_id"] == "demo-first-load"
    assert "relative_path" not in response.text
    assert str(tmp_path.resolve()) not in response.text
    assert asyncio.run(_schema_version(application)) == SCHEMA_VERSION


def test_exact_retry_is_idempotent_and_creates_no_duplicate_records(tmp_path):
    application = _application(tmp_path)
    with TestClient(application) as client:
        first = client.post("/api/v1/development/demo-workspace/load")
        before = asyncio.run(_counts(application))
        retry = client.post("/api/v1/development/demo-workspace/load")
        after = asyncio.run(_counts(application))

    assert first.status_code == retry.status_code == 200
    assert retry.json()["idempotent_existing"] is True
    assert before == after
    assert before[:7] == (2, 5, 10, 5, 4, 4, 3)


def test_concurrent_loads_converge_on_one_workspace(tmp_path):
    application = _application(tmp_path)

    async def load_twice():
        return await asyncio.gather(
            application.state.demo_workspace.load(request_id="concurrent-one"),
            application.state.demo_workspace.load(request_id="concurrent-two"),
        )

    with TestClient(application):
        results = asyncio.run(load_twice())
        counts = asyncio.run(_counts(application))

    assert [result.loaded for result in results] == [True, True]
    assert sorted(result.idempotent_existing for result in results) == [False, True]
    assert counts[:7] == (2, 5, 10, 5, 4, 4, 3)


def test_existing_data_is_preserved(tmp_path):
    application = _application(tmp_path)
    existing_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    with TestClient(application) as client:
        asyncio.run(
            application.state.repositories.inspections.create(
                InspectionCreate(
                    id=existing_id,
                    status=InspectionStatus.RECEIVED,
                    board_id="VISION-TEAM-EXISTING",
                    recipe_id="existing-recipe",
                    recipe_version="7.0",
                )
            )
        )
        response = client.post("/api/v1/development/demo-workspace/load")
        existing = asyncio.run(
            application.state.repositories.inspections.get(existing_id)
        )
        counts = asyncio.run(_counts(application))

    assert response.status_code == 200, response.text
    assert existing is not None
    assert existing.board_id == "VISION-TEAM-EXISTING"
    assert existing.status is InspectionStatus.RECEIVED
    assert counts[1] == 6


def test_disabled_and_missing_fixture_configuration_fail_safely(tmp_path):
    disabled = _application(tmp_path / "disabled", enabled=False)
    with TestClient(disabled) as client:
        status = client.get("/api/v1/development/demo-workspace")
        load = client.post("/api/v1/development/demo-workspace/load")
    assert status.status_code == 200
    assert status.json()["available"] is False
    assert status.json()["inspections"] == []
    assert load.status_code == 404
    assert load.json()["code"] == "DEMO_WORKSPACE_DISABLED"

    unconfigured = _application(
        tmp_path / "unconfigured", enabled=True, configured=False
    )
    with TestClient(unconfigured) as client:
        status = client.get("/api/v1/development/demo-workspace")
        load = client.post("/api/v1/development/demo-workspace/load")
    assert status.status_code == 200
    assert status.json()["enabled"] is True
    assert status.json()["available"] is False
    assert load.status_code == 503
    assert load.json()["code"] == "DEMO_WORKSPACE_NOT_CONFIGURED"
    assert "fixture" not in json.dumps(load.json()).lower()
