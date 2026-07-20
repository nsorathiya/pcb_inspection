import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.core.config import Settings
from app.core.request_context import REQUEST_ID_HEADER
from app.db.models import SCHEMA_VERSION, ArtifactType, InspectionStatus
from app.main import create_app

RGB_BYTES = b"rgb-exact-bytes"
HEIGHT_BYTES = b"\x93NUMPY-height-exact-bytes"


def _settings(runtime_root: Path, **overrides) -> Settings:
    return Settings(_env_file=None, runtime_root=runtime_root, **overrides)


def _files(**overrides):
    values = {
        "rgb_image": ("board.PNG", RGB_BYTES, "image/png"),
        "height_map": ("height.NPY", HEIGHT_BYTES, "application/octet-stream"),
    }
    values.update(overrides)
    return values


def _data(**overrides):
    values = {
        "board_id": " PCB_A ",
        "recipe_id": " RECIPE_A ",
        "recipe_version": " 1.0 ",
        "lot_id": " LOT_1 ",
        "operator_id": " operator-1 ",
        "station_id": " station-1 ",
    }
    values.update(overrides)
    return values


def _required_data(**overrides):
    values = {
        "board_id": " PCB_A ",
        "recipe_id": " RECIPE_A ",
        "recipe_version": " 1.0 ",
    }
    values.update(overrides)
    return values


def _post(client: TestClient, *, files=None, data=None, headers=None):
    return client.post(
        "/api/v1/inspections",
        files=_files() if files is None else files,
        data=_data() if data is None else data,
        headers=headers,
    )


def _inspections(application):
    return asyncio.run(application.state.repositories.inspections.list(limit=100))


def _artifacts(application, inspection_id: str):
    return asyncio.run(
        application.state.repositories.artifacts.list_for_inspection(inspection_id)
    )


def _audit_events(application, inspection_id: str):
    return asyncio.run(
        application.state.repositories.audit_events.list_for_entity(
            "inspection", inspection_id
        )
    )


def _assert_no_persisted_intake(application) -> None:
    assert _inspections(application) == []
    assert not list(application.state.runtime_paths.raw_uploads.rglob("*.*"))


def _assert_failed_intake(application):
    inspections = _inspections(application)
    assert len(inspections) == 1
    inspection = inspections[0]
    assert inspection.status is InspectionStatus.ERROR
    assert inspection.error_code == "INSPECTION_INTAKE_FAILED"
    assert inspection.error_message == "Paired artifact intake did not complete."
    assert inspection.completed_at is not None
    assert _artifacts(application, inspection.id) == []
    assert not list(application.state.runtime_paths.raw_uploads.rglob("*.*"))
    events = _audit_events(application, inspection.id)
    assert [event.action for event in events] == ["INSPECTION_INTAKE_FAILED"]
    details = json.loads(events[0].details_json)
    assert "failure_category" in details
    assert "compensation_complete" in details
    assert str(application.state.runtime_paths.root) not in events[0].details_json
    return inspection, events[0]


def test_successful_pair_returns_201_and_persists_exact_pair(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    application = create_app(_settings(runtime_root))

    with TestClient(application) as client:
        response = _post(client)

        assert response.status_code == 201
        payload = response.json()
        inspection_id = payload["inspection_id"]
        assert str(UUID(inspection_id)) == inspection_id
        assert payload["status"] == "RECEIVED"
        assert payload["board_id"] == "PCB_A"
        assert payload["recipe_id"] == "RECIPE_A"
        assert payload["recipe_version"] == "1.0"
        assert payload["lot_id"] == "LOT_1"
        created_at = payload["created_at"].replace("Z", "+00:00")
        assert datetime.fromisoformat(created_at).tzinfo is not None
        assert "confidence" not in payload
        assert "classification" not in payload
        assert "model_id" not in payload
        assert str(runtime_root.resolve()) not in response.text
        assert "relative_path" not in response.text

        inspections = _inspections(application)
        assert len(inspections) == 1
        inspection = inspections[0]
        assert inspection.id == inspection_id
        assert inspection.status is InspectionStatus.RECEIVED
        assert inspection.lot_id == "LOT_1"
        assert inspection.operator_id == "operator-1"
        assert inspection.model_id is None
        assert inspection.model_version is None
        assert inspection.confidence is None
        assert inspection.processing_ms is None

        artifacts = _artifacts(application, inspection_id)
        assert {artifact.artifact_type for artifact in artifacts} == {
            ArtifactType.RGB_RAW,
            ArtifactType.HEIGHT_RAW,
        }
        assert len(artifacts) == 2
        by_type = {artifact.artifact_type: artifact for artifact in artifacts}
        assert by_type[ArtifactType.RGB_RAW].sha256 == hashlib.sha256(
            RGB_BYTES
        ).hexdigest()
        assert by_type[ArtifactType.RGB_RAW].byte_size == len(RGB_BYTES)
        assert by_type[ArtifactType.HEIGHT_RAW].sha256 == hashlib.sha256(
            HEIGHT_BYTES
        ).hexdigest()
        assert by_type[ArtifactType.HEIGHT_RAW].byte_size == len(HEIGHT_BYTES)
        for artifact in artifacts:
            stored = runtime_root.joinpath(*artifact.relative_path.split("/"))
            assert stored.is_file()


def test_generated_request_id_is_stored_and_returned(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client)
        request_id = response.headers[REQUEST_ID_HEADER]

        assert response.status_code == 201
        assert response.json()["request_id"] == request_id
        assert str(UUID(request_id)) == request_id
        assert _inspections(application)[0].request_id == request_id


def test_supplied_request_id_is_preserved_and_stored(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    supplied = "caller-intake-123"
    with TestClient(application) as client:
        response = _post(client, headers={REQUEST_ID_HEADER: supplied})

        assert response.status_code == 201
        assert response.headers[REQUEST_ID_HEADER] == supplied
        assert response.json()["request_id"] == supplied
        assert _inspections(application)[0].request_id == supplied


def test_matching_expected_hashes_and_sizes_succeed(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    expectations = _data(
        rgb_sha256=hashlib.sha256(RGB_BYTES).hexdigest(),
        height_sha256=hashlib.sha256(HEIGHT_BYTES).hexdigest(),
        rgb_byte_size=str(len(RGB_BYTES)),
        height_byte_size=str(len(HEIGHT_BYTES)),
    )
    with TestClient(application) as client:
        response = _post(client, data=expectations)

    assert response.status_code == 201
    assert response.json()["status"] == "RECEIVED"


def test_all_optional_metadata_can_be_omitted_and_absence_is_consistent(
    tmp_path,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client, data=_required_data())
        assert response.status_code == 201, response.json()
        payload = response.json()
        inspection_id = payload["inspection_id"]
        detail = client.get(f"/api/v1/inspections/{inspection_id}")
        history = client.get("/api/v1/inspections")

    assert payload["lot_id"] is None
    inspection = _inspections(application)[0]
    assert inspection.status is InspectionStatus.RECEIVED
    assert inspection.lot_id is None
    assert inspection.operator_id is None
    assert len(_artifacts(application, inspection_id)) == 2
    event = _audit_events(application, inspection_id)[0]
    assert event.actor_id is None
    assert json.loads(event.details_json)["station_id"] is None
    assert detail.status_code == 200
    assert detail.json()["lot_id"] is None
    assert history.status_code == 200
    history_item = history.json()["items"][0]
    assert history_item["inspection_id"] == inspection_id
    assert history_item["lot_id"] is None
    assert history_item["operator_id"] is None


@pytest.mark.parametrize("empty_value", ["", "   "])
def test_empty_browser_optional_values_normalize_to_null(tmp_path, empty_value) -> None:
    case_name = "empty" if not empty_value else "spaces"
    application = create_app(_settings(tmp_path / case_name))
    optional_fields = {
        "lot_id": empty_value,
        "operator_id": empty_value,
        "station_id": empty_value,
        "rgb_sha256": empty_value,
        "height_sha256": empty_value,
        "rgb_byte_size": empty_value,
        "height_byte_size": empty_value,
    }
    with TestClient(application) as client:
        response = _post(
            client,
            data=_required_data(**optional_fields),
        )

    assert response.status_code == 201, response.json()
    assert response.json()["lot_id"] is None
    inspection = _inspections(application)[0]
    assert inspection.lot_id is None
    assert inspection.operator_id is None
    event = _audit_events(application, inspection.id)[0]
    assert event.actor_id is None
    assert json.loads(event.details_json)["station_id"] is None
    assert all(
        placeholder not in response.text
        for placeholder in ('"N/A"', '"unknown"', '"-"', '"null"')
    )


def test_literal_null_identifier_is_preserved_as_text(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client, data=_required_data(lot_id="null"))

    assert response.status_code == 201, response.json()
    assert response.json()["lot_id"] == "null"
    assert _inspections(application)[0].lot_id == "null"


@pytest.mark.parametrize(
    "omitted_field",
    [
        "lot_id",
        "operator_id",
        "station_id",
        "rgb_sha256",
        "height_sha256",
        "rgb_byte_size",
        "height_byte_size",
    ],
)
def test_each_optional_field_can_be_omitted_independently(
    tmp_path,
    omitted_field,
) -> None:
    application = create_app(_settings(tmp_path / omitted_field))
    data = _data(
        rgb_sha256=hashlib.sha256(RGB_BYTES).hexdigest(),
        height_sha256=hashlib.sha256(HEIGHT_BYTES).hexdigest(),
        rgb_byte_size=str(len(RGB_BYTES)),
        height_byte_size=str(len(HEIGHT_BYTES)),
    )
    data.pop(omitted_field)
    with TestClient(application) as client:
        response = _post(client, data=data)

    assert response.status_code == 201, response.json()
    assert response.json()["status"] == "RECEIVED"


@pytest.mark.parametrize("missing_field", ["rgb_image", "height_map"])
def test_missing_pair_file_returns_structured_422_before_inspection(
    tmp_path,
    missing_field,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    files = _files()
    files.pop(missing_field)
    with TestClient(application) as client:
        response = _post(client, files=files)

        assert response.status_code == 422
        assert response.json()["code"] == "INCOMPLETE_OR_INVALID_MULTIPART_REQUEST"
        assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]
        _assert_no_persisted_intake(application)


@pytest.mark.parametrize(
    "missing_field",
    ["board_id", "recipe_id", "recipe_version"],
)
def test_missing_required_identifier_returns_structured_422(
    tmp_path,
    missing_field,
) -> None:
    application = create_app(_settings(tmp_path / missing_field))
    data = _required_data()
    data.pop(missing_field)
    with TestClient(application) as client:
        response = _post(client, data=data)

    assert response.status_code == 422
    assert response.json()["code"] == "INCOMPLETE_OR_INVALID_MULTIPART_REQUEST"
    _assert_no_persisted_intake(application)


@pytest.mark.parametrize("field", ["board_id", "recipe_id", "recipe_version"])
def test_empty_required_identifier_returns_400_before_inspection(tmp_path, field) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    data = _data(**{field: " \t "})
    with TestClient(application) as client:
        response = _post(client, data=data)

        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_INTAKE_METADATA"
        _assert_no_persisted_intake(application)


def test_control_character_and_overlong_identifiers_are_rejected(tmp_path) -> None:
    for board_id in ("PCB\x00A", "X" * 129):
        application = create_app(_settings(tmp_path / str(len(board_id))))
        with TestClient(application) as client:
            response = _post(client, data=_data(board_id=board_id))
            assert response.status_code == 400
            assert response.json()["code"] == "INVALID_INTAKE_METADATA"
            _assert_no_persisted_intake(application)


@pytest.mark.parametrize("field", ["lot_id", "operator_id", "station_id"])
@pytest.mark.parametrize("value", ["valid\x07value", "X" * 129])
def test_invalid_nonempty_optional_identifiers_are_rejected(
    tmp_path,
    field,
    value,
) -> None:
    application = create_app(_settings(tmp_path / f"{field}-{len(value)}"))
    with TestClient(application) as client:
        response = _post(client, data=_data(**{field: value}))

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_INTAKE_METADATA"
    _assert_no_persisted_intake(application)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rgb_sha256", "0" * 64 + "\x07"),
        ("height_sha256", "0" * 64 + "\x07"),
        ("rgb_byte_size", "1\x07"),
        ("height_byte_size", "1\x07"),
    ],
)
def test_control_characters_in_optional_expectations_are_rejected(
    tmp_path,
    field,
    value,
) -> None:
    application = create_app(_settings(tmp_path / field))
    with TestClient(application) as client:
        response = _post(client, data=_data(**{field: value}))

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_INTAKE_METADATA"
    _assert_no_persisted_intake(application)


@pytest.mark.parametrize("field", ["rgb_sha256", "height_sha256"])
def test_invalid_expected_hash_returns_400_before_inspection(tmp_path, field) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client, data=_data(**{field: "ABC123"}))

        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_INTAKE_METADATA"
        _assert_no_persisted_intake(application)


@pytest.mark.parametrize("value", ["-1", "1.5", "not-an-integer"])
def test_invalid_expected_byte_size_returns_400_before_inspection(
    tmp_path,
    value,
) -> None:
    application = create_app(_settings(tmp_path / value.replace(".", "_")))
    with TestClient(application) as client:
        response = _post(client, data=_data(rgb_byte_size=value))

        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_INTAKE_METADATA"
        _assert_no_persisted_intake(application)


@pytest.mark.parametrize(
    ("field", "file_value"),
    [
        ("rgb_image", ("rgb.gif", b"gif", "image/gif")),
        ("height_map", ("height.exr", b"exr", "application/octet-stream")),
    ],
)
def test_unsupported_extension_returns_400_before_inspection(
    tmp_path,
    field,
    file_value,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    files = _files(**{field: file_value})
    with TestClient(application) as client:
        response = _post(client, files=files)

        assert response.status_code == 400
        assert response.json()["code"] == "UNSUPPORTED_INTAKE_FORMAT"
        _assert_no_persisted_intake(application)


def test_media_type_must_match_the_conservative_intake_gate(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(
            client,
            files=_files(rgb_image=("rgb.png", RGB_BYTES, "text/plain")),
        )
        assert response.status_code == 400
        assert response.json()["code"] == "UNSUPPORTED_INTAKE_FORMAT"
        _assert_no_persisted_intake(application)


def test_duplicate_multipart_file_field_is_rejected(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    files = [
        ("rgb_image", ("first.png", RGB_BYTES, "image/png")),
        ("rgb_image", ("second.png", RGB_BYTES, "image/png")),
        (
            "height_map",
            ("height.npy", HEIGHT_BYTES, "application/octet-stream"),
        ),
    ]
    with TestClient(application) as client:
        response = _post(client, files=files)

        assert response.status_code == 400
        assert response.json()["code"] == "DUPLICATE_MULTIPART_FIELD"
        _assert_no_persisted_intake(application)


def test_client_inspection_id_is_rejected_before_creation(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client, data=_data(inspection_id=str(UUID(int=0))))

        assert response.status_code == 400
        assert response.json()["code"] == "CLIENT_INSPECTION_ID_NOT_ALLOWED"
        _assert_no_persisted_intake(application)


@pytest.mark.parametrize(
    ("settings_override", "expected_category"),
    [
        ({"max_rgb_bytes": len(RGB_BYTES) - 1}, "SIZE_LIMIT_EXCEEDED"),
        ({"max_height_bytes": len(HEIGHT_BYTES) - 1}, "SIZE_LIMIT_EXCEEDED"),
    ],
)
def test_size_limit_failure_never_leaves_a_received_partial_pair(
    tmp_path,
    settings_override,
    expected_category,
) -> None:
    application = create_app(_settings(tmp_path / "runtime", **settings_override))
    with TestClient(application) as client:
        response = _post(client)

        assert response.status_code == 413
        assert response.json()["code"] == "ARTIFACT_SIZE_LIMIT_EXCEEDED"
        inspection, event = _assert_failed_intake(application)
        assert inspection.status is InspectionStatus.ERROR
        assert json.loads(event.details_json)["failure_category"] == expected_category


@pytest.mark.parametrize(
    "hash_field",
    ["rgb_sha256", "height_sha256"],
)
def test_expected_hash_mismatch_is_compensated_and_audited(
    tmp_path,
    hash_field,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client, data=_data(**{hash_field: "0" * 64}))

        assert response.status_code == 400
        assert response.json()["code"] == "ARTIFACT_INTEGRITY_MISMATCH"
        _, event = _assert_failed_intake(application)
        assert json.loads(event.details_json)["failure_category"] == (
            "INTEGRITY_MISMATCH"
        )


@pytest.mark.parametrize(
    ("size_field", "wrong_size"),
    [
        ("rgb_byte_size", len(RGB_BYTES) + 1),
        ("height_byte_size", len(HEIGHT_BYTES) + 1),
    ],
)
def test_expected_size_mismatch_is_compensated_and_audited(
    tmp_path,
    size_field,
    wrong_size,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client, data=_data(**{size_field: str(wrong_size)}))

        assert response.status_code == 400
        assert response.json()["code"] == "ARTIFACT_INTEGRITY_MISMATCH"
        _, event = _assert_failed_intake(application)
        assert json.loads(event.details_json)["failure_category"] == (
            "INTEGRITY_MISMATCH"
        )


@pytest.mark.parametrize("failed_type", [ArtifactType.RGB_RAW, ArtifactType.HEIGHT_RAW])
def test_database_registration_failure_is_compensated_and_never_succeeds(
    tmp_path,
    monkeypatch,
    failed_type,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    repository = application.state.repositories.artifacts
    original_create = repository.create

    async def fail_selected(data):
        if data.artifact_type is failed_type:
            raise RuntimeError("simulated database registration failure")
        return await original_create(data)

    monkeypatch.setattr(repository, "create", fail_selected)
    with TestClient(application) as client:
        response = _post(client)

        assert response.status_code == 500
        assert response.json()["code"] == "ARTIFACT_REGISTRATION_FAILED"
        _, event = _assert_failed_intake(application)
        assert json.loads(event.details_json)["failure_category"] == (
            "DATABASE_REGISTRATION_FAILED"
        )


def test_successful_intake_emits_safe_received_audit_event(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client)
        inspection_id = response.json()["inspection_id"]
        events = _audit_events(application, inspection_id)

        assert response.status_code == 201
        assert [event.action for event in events] == ["INSPECTION_RECEIVED"]
        event = events[0]
        assert event.actor_id == "operator-1"
        assert event.request_id == response.json()["request_id"]
        details = json.loads(event.details_json)
        assert details == {
            "artifact_types": ["RGB_RAW", "HEIGHT_RAW"],
            "byte_sizes": {
                "HEIGHT_RAW": len(HEIGHT_BYTES),
                "RGB_RAW": len(RGB_BYTES),
            },
            "station_id": "station-1",
        }
        assert str(application.state.runtime_paths.root) not in event.details_json


def test_upload_streams_are_closed_after_success(tmp_path, monkeypatch) -> None:
    closed_filenames: list[str | None] = []
    original_close = StarletteUploadFile.close

    async def tracking_close(upload):
        closed_filenames.append(upload.filename)
        await original_close(upload)

    monkeypatch.setattr(StarletteUploadFile, "close", tracking_close)
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        response = _post(client)

    assert response.status_code == 201
    assert {"board.PNG", "height.NPY"} <= set(closed_filenames)


def test_structured_error_contains_no_internal_path_or_classification(tmp_path) -> None:
    runtime_root = tmp_path / "private-runtime"
    application = create_app(_settings(runtime_root))
    with TestClient(application) as client:
        response = _post(client, data=_data(rgb_sha256="0" * 64))

    payload = response.json()
    assert set(payload) == {"code", "message", "request_id"}
    assert str(runtime_root.resolve()) not in response.text
    assert "sql" not in response.text.lower()
    assert "traceback" not in response.text.lower()
    assert "confidence" not in response.text.lower()
    assert "pass" not in response.text.lower()


def test_repeated_application_startup_and_health_contract_remain_valid(tmp_path) -> None:
    settings = _settings(tmp_path / "runtime")
    for _ in range(2):
        application = create_app(settings)
        with TestClient(application) as client:
            health = client.get("/api/v1/health")
            assert health.status_code == 200
            assert health.json() == {
                "status": "ok",
                "service": "pcb-aoi-api",
                "version": "0.1.0",
                "environment": "development",
            }


def test_openapi_documents_multipart_pair_and_intake_only_semantics(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    schema = application.openapi()
    operation = schema["paths"]["/api/v1/inspections"]["post"]
    request_schema = operation["requestBody"]["content"]["multipart/form-data"][
        "schema"
    ]
    component_name = request_schema["$ref"].rsplit("/", 1)[-1]
    component = schema["components"]["schemas"][component_name]

    required_fields = {
        "rgb_image",
        "height_map",
        "board_id",
        "recipe_id",
        "recipe_version",
    }
    optional_fields = {
        "lot_id",
        "operator_id",
        "station_id",
        "rgb_sha256",
        "height_sha256",
        "rgb_byte_size",
        "height_byte_size",
    }
    assert set(component["required"]) == required_fields
    assert optional_fields <= set(component["properties"])
    assert optional_fields.isdisjoint(component["required"])
    assert all(
        {variant.get("type") for variant in component["properties"][field]["anyOf"]}
        == {"string", "null"}
        for field in optional_fields
    )
    assert SCHEMA_VERSION == 3
    assert "not decoded" in operation["description"]
    assert "alignment is not proven" in operation["description"]
    assert "RECEIVED does not mean PASS" in operation["description"]
