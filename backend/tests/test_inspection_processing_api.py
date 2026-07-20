import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, event, func, select

from app.core.config import Settings
from app.core.request_context import REQUEST_ID_HEADER
from app.db.models import (
    ArtifactType,
    AuditEvent,
    InspectionInferenceResult,
    InspectionPreprocessingResult,
    InspectionProcessingRun,
    InspectionStatus,
)
from app.db.validation_types import ValidationOutcome
from app.main import create_app
from app.services.artifact_storage import ArtifactInput
from app.services.inspection_validation import ValidationCommitService, ValidationSummary
from app.testing.synthetic_aoi import generate_fixtures

PREPROCESSING_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
INFERENCE_ID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
VALIDATION_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
VALIDATION_POLICY = {
    "policy_id": "development-native-rgb-height",
    "policy_version": "1.0",
}
PROCESSING_POLICY = {
    "preprocessing_policy_id": "synthetic-paired-rgb-height",
    "preprocessing_policy_version": "1.0",
    "inference_policy_id": "synthetic-deterministic-mock-inference",
    "inference_policy_version": "1.0",
}


def _application(tmp_path: Path, scenario_id: str, *, enabled: bool = True):
    fixture_root = tmp_path / "fixtures"
    generate_fixtures(fixture_root, scenario_ids=(scenario_id,))
    settings = Settings(
        _env_file=None,
        runtime_root=tmp_path / "runtime",
        enable_synthetic_processing_api=enabled,
        synthetic_fixture_root=fixture_root,
    )
    return create_app(settings), fixture_root


def _scenario_files(fixture_root: Path, scenario_id: str):
    root = fixture_root / "scenarios" / scenario_id
    record = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    rgb = record["artifacts"]["rgb"]
    height = record["artifacts"]["height"]
    return {
        "rgb_image": (
            rgb["generated_file"],
            (root / rgb["generated_file"]).read_bytes(),
            rgb["media_type"],
        ),
        "height_map": (
            height["generated_file"],
            (root / height["generated_file"]).read_bytes(),
            height["media_type"],
        ),
    }


def _prepare(
    application,
    client: TestClient,
    fixture_root: Path,
    scenario_id: str,
    monkeypatch,
    *,
    inspection_id: str = "00000000-0000-4000-8000-000000000003",
) -> str:
    monkeypatch.setattr(
        "app.services.inspection_intake.uuid4",
        lambda: UUID(inspection_id),
    )
    application.state.inspection_validation._engine._validation_id = (
        lambda: VALIDATION_ID
    )
    orchestrator = application.state.inspection_processing._orchestrator
    orchestrator._preprocess._preprocessing_id = lambda: PREPROCESSING_ID
    orchestrator._infer._inference_id = lambda: INFERENCE_ID
    intake = client.post(
        "/api/v1/inspections",
        data={
            "board_id": "SYNTHETIC",
            "recipe_id": "development-native-rgb-height",
            "recipe_version": "1.0",
            "lot_id": "synthetic-lot",
            "operator_id": "synthetic-operator",
            "station_id": "synthetic-station",
        },
        files=_scenario_files(fixture_root, scenario_id),
    )
    assert intake.status_code == 201, intake.json()
    assert intake.json()["inspection_id"] == inspection_id
    if scenario_id == "valid_different_dimensions":
        validation_orchestrator = application.state.inspection_validation
        policy = validation_orchestrator._policies.load(
            VALIDATION_POLICY["policy_id"], VALIDATION_POLICY["policy_version"]
        )
        _inspection, artifacts = asyncio.run(
            validation_orchestrator._inspection_and_artifacts(inspection_id)
        )
        validation_key = validation_orchestrator._validation_key(
            inspection_id, artifacts, policy
        )
        technical = asyncio.run(
            validation_orchestrator._engine.validate_inspection_pair(
                inspection_id, policy
            )
        )
        passed = replace(
            technical,
            outcome=ValidationOutcome.VALIDATION_PASSED,
            findings=(),
            summary=ValidationSummary(0, 0, 0, 0, 0, True, False),
        )
        asyncio.run(
            ValidationCommitService(
                application.state.database.session_factory,
                validation_repository=application.state.repositories.validations,
            ).commit_validation(passed, validation_key)
        )
    else:
        validation = client.post(
            f"/api/v1/inspections/{inspection_id}/validate",
            json=VALIDATION_POLICY,
        )
        assert validation.status_code == 200, validation.json()
        assert validation.json()["validation_outcome"] == "VALIDATION_PASSED"
    return inspection_id


def _processing_counts(application, inspection_id: str):
    async def read():
        async with application.state.database.session() as session:
            return (
                await session.scalar(
                    select(func.count())
                    .select_from(InspectionProcessingRun)
                    .where(InspectionProcessingRun.inspection_id == inspection_id)
                ),
                await session.scalar(
                    select(func.count()).select_from(InspectionPreprocessingResult)
                ),
                await session.scalar(
                    select(func.count()).select_from(InspectionInferenceResult)
                ),
                await session.scalar(
                    select(func.count())
                    .select_from(AuditEvent)
                    .where(
                        AuditEvent.entity_id == inspection_id,
                        (
                            AuditEvent.action.like("INSPECTION_PROCESSING_%")
                            | AuditEvent.action.like("INSPECTION_MOCK_RESULT_%")
                        ),
                    )
                ),
            )

    return asyncio.run(read())


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


@pytest.mark.parametrize(
    ("inspection_id", "expected"),
    [
        ("00000000-0000-4000-8000-000000000003", "PASS"),
        ("00000000-0000-4000-8000-000000000001", "FAIL"),
        ("00000000-0000-4000-8000-000000000006", "UNCERTAIN"),
    ],
)
def test_post_returns_safe_mock_decisions_and_persisted_evidence(
    tmp_path, monkeypatch, inspection_id, expected
):
    scenario_id = "valid_rgb_png_height_tiff"
    application, fixtures = _application(tmp_path, scenario_id)
    with TestClient(application) as client:
        prepared = _prepare(
            application,
            client,
            fixtures,
            scenario_id,
            monkeypatch,
            inspection_id=inspection_id,
        )
        response = client.post(
            f"/api/v1/inspections/{prepared}/process",
            json=PROCESSING_POLICY,
            headers={REQUEST_ID_HEADER: "processing-request"},
        )

    assert response.status_code == 200, response.json()
    payload = response.json()
    assert response.headers[REQUEST_ID_HEADER] == payload["request_id"] == "processing-request"
    assert payload["mock_decision"] == expected
    assert payload["inspection_status"] == expected
    assert payload["processing_status"] == "COMPLETED"
    assert payload["synthetic_input_verified"] is True
    assert payload["mock_preprocessing"] is True
    assert payload["mock_inference"] is True
    assert payload["production_approved"] is False
    assert payload["execution_started_now"] is True
    assert payload["lifecycle_idempotent_existing"] is False
    assert payload["preprocessing"]["policy_id"] == PROCESSING_POLICY["preprocessing_policy_id"]
    assert payload["inference"]["engine_type"] == "MOCK"
    assert {
        "MOCK_INFERENCE_USED",
        "MOCK_DECISION_GENERATED",
        "CONFIDENCE_UNAVAILABLE",
    } <= {item["code"] for item in payload["inference"]["findings"]}
    assert payload["defect_type"] is not None if expected == "FAIL" else payload["defect_type"] is None
    assert "confidence" not in _all_keys(payload)
    assert "path" not in _all_keys(payload)
    assert "buffer" not in _all_keys(payload)
    assert str(fixtures.resolve()) not in response.text
    assert _processing_counts(application, prepared) == (1, 1, 1, 2)


@pytest.mark.parametrize(
    "scenario_id",
    [
        "valid_rgb_png_height_tiff",
        "valid_rgb_tiff_height_png16",
        "valid_rgb_png_height_npy_float32",
    ],
)
def test_post_supports_all_trusted_synthetic_format_pairs(
    tmp_path, monkeypatch, scenario_id
):
    application, fixtures = _application(tmp_path, scenario_id)
    with TestClient(application) as client:
        inspection_id = _prepare(
            application, client, fixtures, scenario_id, monkeypatch
        )
        response = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )
    assert response.status_code == 200, response.json()
    assert response.json()["preprocessing_outcome"] == "PREPROCESSING_SUCCEEDED"
    assert response.json()["inference_execution_outcome"] == "INFERENCE_SUCCEEDED"


def test_exact_retry_and_get_share_persisted_mapping_without_file_reads(
    tmp_path, monkeypatch
):
    scenario_id = "valid_rgb_png_height_tiff"
    application, fixtures = _application(tmp_path, scenario_id)
    with TestClient(application) as client:
        inspection_id = _prepare(
            application, client, fixtures, scenario_id, monkeypatch
        )
        orchestrator = application.state.inspection_processing._orchestrator
        first = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )
        before = _processing_counts(application, inspection_id)

        async def must_not_execute(*_args, **_kwargs):
            raise AssertionError("execution must not run for replay or GET")

        def must_not_verify(*_args, **_kwargs):
            raise AssertionError("manifest verification must not run")

        monkeypatch.setattr(orchestrator._preprocess, "preprocess_inspection", must_not_execute)
        monkeypatch.setattr(orchestrator._infer, "run_inference", must_not_execute)
        monkeypatch.setattr(orchestrator._preflight, "resolve_and_verify", must_not_execute)
        monkeypatch.setattr(orchestrator._provenance, "verify", must_not_verify)
        retry = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )
        application.state.inspection_processing._orchestrator = None
        retrieved = client.get(
            f"/api/v1/inspections/{inspection_id}/processing",
            headers={REQUEST_ID_HEADER: "processing-get"},
        )

    assert first.status_code == retry.status_code == retrieved.status_code == 200
    for field in ("processing_run_id", "preprocessing_id", "inference_id", "mock_decision"):
        assert retry.json()[field] == retrieved.json()[field] == first.json()[field]
    assert retry.json()["lifecycle_idempotent_existing"] is True
    assert retry.json()["execution_started_now"] is False
    assert retrieved.json()["request_id"] == "processing-get"
    assert retrieved.json()["preprocessing"]["findings"] == first.json()["preprocessing"]["findings"]
    assert retrieved.json()["inference"]["findings"] == first.json()["inference"]["findings"]
    assert _processing_counts(application, inspection_id) == before == (1, 1, 1, 2)


def test_different_dimensions_complete_and_replay_as_technical_error(
    tmp_path, monkeypatch
):
    scenario_id = "valid_different_dimensions"
    application, fixtures = _application(tmp_path, scenario_id)
    with TestClient(application) as client:
        inspection_id = _prepare(
            application, client, fixtures, scenario_id, monkeypatch
        )
        first = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )
        for path in fixtures.rglob("*"):
            if path.is_file():
                path.unlink()
        retry = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )

    assert first.status_code == retry.status_code == 200
    assert first.json()["inspection_status"] == "ERROR"
    assert first.json()["processing_status"] == "ERROR"
    assert first.json()["preprocessing_outcome"] == "PREPROCESSING_FAILED"
    assert first.json()["inference"] is None
    assert retry.json()["processing_run_id"] == first.json()["processing_run_id"]
    assert retry.json()["lifecycle_idempotent_existing"] is True
    assert _processing_counts(application, inspection_id) == (1, 1, 0, 2)


@pytest.mark.parametrize(
    ("enabled", "root_configured"),
    [(False, True), (True, False)],
)
def test_post_requires_explicit_enabled_configuration(
    tmp_path, enabled, root_configured
):
    settings = Settings(
        _env_file=None,
        runtime_root=tmp_path / "runtime",
        enable_synthetic_processing_api=enabled,
        synthetic_fixture_root=(tmp_path / "fixtures") if root_configured else None,
    )
    application = create_app(settings)
    with TestClient(application) as client:
        response = client.post(
            "/api/v1/inspections/00000000-0000-4000-8000-000000000003/process",
            json=PROCESSING_POLICY,
        )
    assert response.status_code == 503
    assert response.json()["code"] == "SYNTHETIC_PROCESSING_NOT_CONFIGURED"
    assert "fixture" not in response.text.lower()
    assert str(tmp_path.resolve()) not in response.text


def test_processing_request_and_identifier_errors_are_structured(tmp_path):
    application, _ = _application(tmp_path, "valid_rgb_png_height_tiff")
    with TestClient(application) as client:
        malformed = client.post("/api/v1/inspections/not-a-uuid/process", json=PROCESSING_POLICY)
        missing = client.post(
            "/api/v1/inspections/00000000-0000-4000-8000-000000000003/process"
        )
        extra = client.post(
            "/api/v1/inspections/00000000-0000-4000-8000-000000000003/process",
            json={**PROCESSING_POLICY, "confidence": 0.9},
        )
        blank = client.post(
            "/api/v1/inspections/00000000-0000-4000-8000-000000000003/process",
            json={**PROCESSING_POLICY, "preprocessing_policy_id": " \n"},
        )
        unknown_pre = client.post(
            "/api/v1/inspections/00000000-0000-4000-8000-000000000003/process",
            json={**PROCESSING_POLICY, "preprocessing_policy_id": "unknown"},
        )
        unknown_inf = client.post(
            "/api/v1/inspections/00000000-0000-4000-8000-000000000003/process",
            json={**PROCESSING_POLICY, "inference_policy_version": "9.9"},
        )

    assert malformed.status_code == 400
    assert missing.status_code == extra.status_code == 422
    assert missing.json()["code"] == extra.json()["code"] == "INVALID_PROCESSING_REQUEST"
    assert blank.status_code == 400
    assert blank.json()["code"] == "INVALID_PROCESSING_POLICY_SELECTION"
    assert unknown_pre.status_code == 404
    assert unknown_pre.json()["code"] == "PREPROCESSING_POLICY_NOT_FOUND"
    assert unknown_inf.status_code == 404
    assert unknown_inf.json()["code"] == "INFERENCE_POLICY_VERSION_UNSUPPORTED"
    for response in (malformed, missing, extra, blank, unknown_pre, unknown_inf):
        assert set(response.json()) == {"code", "message", "request_id"}
        assert "traceback" not in response.text.lower()
        assert "select " not in response.text.lower()


def test_get_missing_and_inconsistent_results_are_safe(tmp_path, monkeypatch):
    scenario_id = "valid_rgb_png_height_tiff"
    application, fixtures = _application(tmp_path, scenario_id)
    with TestClient(application) as client:
        missing_inspection = client.get(
            "/api/v1/inspections/00000000-0000-4000-8000-000000000009/processing"
        )
        inspection_id = _prepare(
            application, client, fixtures, scenario_id, monkeypatch
        )
        missing_result = client.get(
            f"/api/v1/inspections/{inspection_id}/processing"
        )
        completed = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )
        assert completed.status_code == 200

        async def corrupt():
            async with application.state.database.session_factory.begin() as session:
                await session.execute(delete(InspectionInferenceResult))
                await session.execute(delete(InspectionPreprocessingResult))

        asyncio.run(corrupt())
        inconsistent = client.get(
            f"/api/v1/inspections/{inspection_id}/processing"
        )

    assert missing_inspection.status_code == 404
    assert missing_inspection.json()["code"] == "INSPECTION_NOT_FOUND"
    assert missing_result.status_code == 404
    assert missing_result.json()["code"] == "INSPECTION_PROCESSING_NOT_FOUND"
    assert inconsistent.status_code == 500
    assert inconsistent.json()["code"] == "PROCESSING_DATA_INCONSISTENT"
    assert str(tmp_path.resolve()) not in inconsistent.text


def test_get_is_select_only_and_adds_no_audit(tmp_path, monkeypatch):
    scenario_id = "valid_rgb_png_height_tiff"
    application, fixtures = _application(tmp_path, scenario_id)
    with TestClient(application) as client:
        inspection_id = _prepare(
            application, client, fixtures, scenario_id, monkeypatch
        )
        processed = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )
        assert processed.status_code == 200
        before = _processing_counts(application, inspection_id)
        statements: list[str] = []

        def record_sql(_connection, _cursor, statement, *_args):
            statements.append(statement)

        sync_engine = application.state.database.engine.sync_engine
        event.listen(sync_engine, "before_cursor_execute", record_sql)
        try:
            response = client.get(
                f"/api/v1/inspections/{inspection_id}/processing"
            )
        finally:
            event.remove(sync_engine, "before_cursor_execute", record_sql)

    assert response.status_code == 200
    assert statements
    assert all(value.lstrip().upper().startswith("SELECT") for value in statements)
    assert _processing_counts(application, inspection_id) == before


def test_optional_evidence_is_rejected_before_lifecycle_begin(tmp_path, monkeypatch):
    scenario_id = "valid_with_mask_and_calibration_evidence"
    application, fixtures = _application(tmp_path, scenario_id)
    with TestClient(application) as client:
        inspection_id = _prepare(
            application, client, fixtures, scenario_id, monkeypatch
        )
        scenario_root = fixtures / "scenarios" / scenario_id
        record = json.loads((scenario_root / "scenario.json").read_text(encoding="utf-8"))
        for role, artifact_type in (
            ("mask", ArtifactType.VALIDITY_MASK),
            ("calibration", ArtifactType.CALIBRATION),
        ):
            item = record["artifacts"][role]
            asyncio.run(
                application.state.artifact_registration.store_and_register(
                    ArtifactInput(
                        inspection_id=inspection_id,
                        artifact_type=artifact_type,
                        source=(scenario_root / item["generated_file"]).read_bytes(),
                        original_filename=item["generated_file"],
                        media_type=item["media_type"],
                        expected_sha256=item["actual_sha256"],
                        expected_byte_size=item["actual_byte_size"],
                    )
                )
            )
        response = client.post(
            f"/api/v1/inspections/{inspection_id}/process",
            json=PROCESSING_POLICY,
        )

    assert response.status_code == 409
    assert response.json()["code"] == "OPTIONAL_EVIDENCE_PROCESSING_UNSUPPORTED"
    assert _processing_counts(application, inspection_id) == (0, 0, 0, 0)
    inspection = asyncio.run(application.state.repositories.inspections.get(inspection_id))
    assert inspection.status is InspectionStatus.READY


def test_openapi_documents_development_only_mock_contract(tmp_path):
    application, _ = _application(tmp_path, "valid_rgb_png_height_tiff")
    schema = application.openapi()
    post = schema["paths"]["/api/v1/inspections/{inspection_id}/process"]["post"]
    get = schema["paths"]["/api/v1/inspections/{inspection_id}/processing"]["get"]
    description = post["description"]
    assert "Development-only" in description
    assert "trusted synthetic processing orchestrator" in description
    assert "not image-based predictions" in description
    assert "confidence is unavailable" in description
    assert "Reprocessing is not supported" in description
    assert "without rerunning" in get["description"]
    assert "/api/v1/inspections/{inspection_id}/validate" in schema["paths"]
    assert "/api/v1/inspections" in schema["paths"]
