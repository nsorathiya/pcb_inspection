import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.core.config import Settings
from app.core.request_context import REQUEST_ID_HEADER
from app.db.models import ArtifactType, InspectionStatus
from app.db.repositories import (
    AuditEventCreate,
    InspectionArtifactCreate,
    InspectionCreate,
)
from app.main import create_app


def _settings(runtime_root: Path) -> Settings:
    return Settings(_env_file=None, runtime_root=runtime_root)


def _create_inspection(application, **overrides):
    values = {
        "status": InspectionStatus.RECEIVED,
        "board_id": "PCB_A",
        "recipe_id": "PCB_A",
        "recipe_version": "1.0",
        "request_id": "intake-request-123",
    }
    values.update(overrides)
    return asyncio.run(
        application.state.repositories.inspections.create(InspectionCreate(**values))
    )


def _register_artifact(application, inspection_id: str, artifact_type: ArtifactType):
    position = tuple(ArtifactType).index(artifact_type) + 1
    return asyncio.run(
        application.state.repositories.artifacts.create(
            InspectionArtifactCreate(
                inspection_id=inspection_id,
                artifact_type=artifact_type,
                relative_path=f"private/{artifact_type.value.lower()}.bin",
                sha256=f"{position:064x}",
                byte_size=position * 100,
                media_type="application/octet-stream",
            )
        )
    )


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_received_inspection_returns_safe_metadata_and_distinct_request_ids(
    tmp_path,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    intake_request_id = "intake-request-123"
    get_request_id = "details-request-456"

    with TestClient(application) as client:
        intake = client.post(
            "/api/v1/inspections",
            data={
                "board_id": "PCB_A",
                "recipe_id": "PCB_A",
                "recipe_version": "1.0",
                "lot_id": "LOT-7",
                "operator_id": "operator-1",
                "station_id": "station-1",
            },
            files={
                "rgb_image": ("board.png", b"rgb bytes", "image/png"),
                "height_map": ("height.tiff", b"height bytes", "image/tiff"),
            },
            headers={REQUEST_ID_HEADER: intake_request_id},
        )
        assert intake.status_code == 201, intake.json()

        response = client.get(
            f"/api/v1/inspections/{intake.json()['inspection_id']}",
            headers={REQUEST_ID_HEADER: get_request_id},
        )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == get_request_id
    payload = response.json()
    assert payload == {
        "inspection_id": intake.json()["inspection_id"],
        "status": "RECEIVED",
        "board_id": "PCB_A",
        "recipe_id": "PCB_A",
        "recipe_version": "1.0",
        "lot_id": "LOT-7",
        "intake_request_id": intake_request_id,
        "created_at": payload["created_at"],
        "started_at": None,
        "completed_at": None,
        "error": None,
        "artifacts": payload["artifacts"],
    }
    assert [item["artifact_type"] for item in payload["artifacts"]] == [
        "RGB_RAW",
        "HEIGHT_RAW",
    ]
    assert [item["byte_size"] for item in payload["artifacts"]] == [9, 12]
    assert [item["media_type"] for item in payload["artifacts"]] == [
        "image/png",
        "image/tiff",
    ]
    assert all(len(item["sha256"]) == 64 for item in payload["artifacts"])
    assert _parse_timestamp(payload["created_at"]).tzinfo is not None
    assert all(
        _parse_timestamp(item["created_at"]).tzinfo is not None
        for item in payload["artifacts"]
    )
    assert _all_keys(payload).isdisjoint(
        {
            "id",
            "relative_path",
            "absolute_path",
            "filename",
            "content",
            "operator_id",
            "model_id",
            "model_version",
            "confidence",
            "classification",
        }
    )
    assert "private" not in response.text
    assert get_request_id != payload["intake_request_id"]


def test_optional_artifacts_follow_authoritative_enum_order(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        inspection = _create_inspection(application)
        for artifact_type in reversed(tuple(ArtifactType)):
            _register_artifact(application, inspection.id, artifact_type)

        response = client.get(f"/api/v1/inspections/{inspection.id}")

    assert response.status_code == 200
    assert [item["artifact_type"] for item in response.json()["artifacts"]] == [
        artifact_type.value for artifact_type in ArtifactType
    ]
    assert str(UUID(response.headers[REQUEST_ID_HEADER])) == response.headers[
        REQUEST_ID_HEADER
    ]


def test_inspection_without_artifacts_is_retrieved(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        inspection = _create_inspection(application, request_id=None)
        response = client.get(f"/api/v1/inspections/{inspection.id}")

    assert response.status_code == 200
    assert response.json()["artifacts"] == []
    assert response.json()["intake_request_id"] is None


@pytest.mark.parametrize(
    "inspection_id",
    ["not-a-uuid", "A" * 36, "{00000000-0000-0000-0000-000000000000}"],
)
def test_malformed_uuid_returns_400_without_repository_query(
    monkeypatch,
    tmp_path,
    inspection_id,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    queried = False

    async def unexpected_query(_inspection_id):
        nonlocal queried
        queried = True
        raise AssertionError("repository must not be queried")

    monkeypatch.setattr(
        application.state.repositories.inspections,
        "get",
        unexpected_query,
    )
    with TestClient(application) as client:
        response = client.get(
            f"/api/v1/inspections/{inspection_id}",
            headers={REQUEST_ID_HEADER: "bad-id-request"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "code": "INVALID_INSPECTION_ID",
        "message": "Inspection ID must be a canonical UUID.",
        "request_id": "bad-id-request",
    }
    assert queried is False


def test_unknown_canonical_uuid_returns_structured_404(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    missing_id = str(uuid4())
    with TestClient(application) as client:
        response = client.get(f"/api/v1/inspections/{missing_id}")

    assert response.status_code == 404
    assert response.json()["code"] == "INSPECTION_NOT_FOUND"
    assert response.json()["message"] == "Inspection was not found."
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_error_inspection_returns_only_known_safe_persisted_error(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        inspection = _create_inspection(
            application,
            status=InspectionStatus.ERROR,
            error_code="INSPECTION_INTAKE_FAILED",
            error_message="Paired artifact intake did not complete.",
            completed_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        )
        response = client.get(f"/api/v1/inspections/{inspection.id}")

    assert response.status_code == 200
    assert response.json()["status"] == "ERROR"
    assert response.json()["error"] == {
        "code": "INSPECTION_INTAKE_FAILED",
        "message": "Paired artifact intake did not complete.",
    }


def test_unknown_persisted_error_details_are_replaced_with_safe_values(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    secret = r"sqlite SELECT * FROM audit_events at C:\runtime\database\private.sqlite3"
    with TestClient(application) as client:
        inspection = _create_inspection(
            application,
            status=InspectionStatus.VALIDATION_FAILED,
            error_code="RAW_EXCEPTION",
            error_message=secret,
            completed_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        )
        response = client.get(f"/api/v1/inspections/{inspection.id}")

    assert response.status_code == 200
    assert response.json()["error"] == {
        "code": "INSPECTION_VALIDATION_FAILED",
        "message": "Inspection validation failed.",
    }
    assert secret not in response.text
    assert "SELECT" not in response.text
    assert "private.sqlite3" not in response.text


def test_repository_read_failure_returns_safe_structured_500(
    monkeypatch,
    tmp_path,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    inspection_id = str(uuid4())
    secret = r"SELECT failed at C:\runtime\database\private.sqlite3"

    async def failed_read(_inspection_id):
        raise RuntimeError(secret)

    monkeypatch.setattr(application.state.repositories.inspections, "get", failed_read)
    with TestClient(application) as client:
        response = client.get(
            f"/api/v1/inspections/{inspection_id}",
            headers={REQUEST_ID_HEADER: "failed-read-request"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "code": "INSPECTION_READ_FAILED",
        "message": "Inspection details could not be retrieved.",
        "request_id": "failed-read-request",
    }
    assert secret not in response.text
    assert "SELECT" not in response.text
    assert "private.sqlite3" not in response.text


def test_get_executes_only_reads_and_appends_no_audit_event(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        inspection = _create_inspection(application)
        event_record = asyncio.run(
            application.state.repositories.audit_events.append(
                AuditEventCreate(
                    entity_type="inspection",
                    entity_id=inspection.id,
                    action="PREEXISTING_EVENT",
                )
            )
        )
        statements: list[str] = []

        def record_statement(
            _connection,
            _cursor,
            statement,
            _parameters,
            _context,
            _many,
        ):
            statements.append(statement)

        sync_engine = application.state.database.engine.sync_engine
        event.listen(sync_engine, "before_cursor_execute", record_statement)
        try:
            response = client.get(f"/api/v1/inspections/{inspection.id}")
        finally:
            event.remove(sync_engine, "before_cursor_execute", record_statement)

        events = asyncio.run(
            application.state.repositories.audit_events.list_for_entity(
                "inspection", inspection.id
            )
        )
        persisted = asyncio.run(
            application.state.repositories.inspections.get(inspection.id)
        )

    assert response.status_code == 200
    assert statements
    assert all(
        statement.lstrip().upper().startswith("SELECT") for statement in statements
    )
    assert [record.id for record in events] == [event_record.id]
    assert persisted is not None
    assert persisted.status is InspectionStatus.RECEIVED


@pytest.mark.parametrize("inspection_status", list(InspectionStatus))
def test_every_persisted_status_is_reported_without_inference(
    tmp_path,
    inspection_status,
) -> None:
    application = create_app(_settings(tmp_path / inspection_status.value))
    completed_at = (
        datetime(2026, 7, 17, tzinfo=timezone.utc)
        if inspection_status
        in {
            InspectionStatus.PASS,
            InspectionStatus.FAIL,
            InspectionStatus.UNCERTAIN,
        }
        else None
    )
    with TestClient(application) as client:
        inspection = _create_inspection(
            application,
            status=inspection_status,
            completed_at=completed_at,
        )
        response = client.get(f"/api/v1/inspections/{inspection.id}")

    assert response.status_code == 200
    assert response.json()["status"] == inspection_status.value
    assert "confidence" not in response.json()
    assert "classification" not in response.json()


def test_openapi_documents_detail_and_collection_get(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        schema = client.get("/openapi.json").json()

    detail_operation = schema["paths"]["/api/v1/inspections/{inspection_id}"]["get"]
    success_schema = detail_operation["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert success_schema["$ref"].endswith("/InspectionDetailResponse")
    assert set(schema["paths"]["/api/v1/inspections"]) == {"post", "get"}
