import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select, update

from app.core.config import Settings
from app.core.request_context import REQUEST_ID_HEADER
from app.db.models import (
    AuditEvent,
    Inspection,
    InspectionArtifact,
    InspectionStatus,
    InspectionValidation,
    InspectionValidationFinding,
)
from app.main import create_app
from app.services.inspection_validation import (
    ArtifactTechnicalSummary,
    FindingFactory,
    InspectionValidationResult,
    ReadabilityStatus,
    ValidationFinding,
    ValidationOutcome,
    PolicyLoadError,
    ValidationPolicyLoader,
    ValidationSummary,
)
from app.services.inspection_validation.policy_loader import (
    DEFAULT_DEVELOPMENT_POLICY_PATH,
)
from app.services.inspection_validation.service import DEFAULT_VALIDATOR_VERSION
from app.testing.synthetic_aoi import generate_fixtures

POLICY = {
    "policy_id": "development-native-rgb-height",
    "policy_version": "1.0",
}


def _settings(runtime_root: Path) -> Settings:
    return Settings(_env_file=None, runtime_root=runtime_root)


def _scenario_files(tmp_path: Path, scenario_id: str):
    generated = tmp_path / "generated"
    generate_fixtures(generated, scenario_ids=(scenario_id,))
    root = generated / "scenarios" / scenario_id
    record = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
    rgb = record["artifacts"]["rgb"]
    height = record["artifacts"]["height"]
    return {
        "rgb_image": (
            Path(rgb["generated_file"]).name,
            (root / rgb["generated_file"]).read_bytes(),
            rgb["media_type"],
        ),
        "height_map": (
            Path(height["generated_file"]).name,
            (root / height["generated_file"]).read_bytes(),
            height["media_type"],
        ),
    }


def _intake(client: TestClient, tmp_path: Path, scenario_id: str) -> str:
    response = client.post(
        "/api/v1/inspections",
        data={
            "board_id": "SYNTHETIC",
            "recipe_id": "development-native-rgb-height",
            "recipe_version": "1.0",
            "lot_id": "synthetic-lot",
            "operator_id": "synthetic-operator",
            "station_id": "synthetic-station",
        },
        files=_scenario_files(tmp_path, scenario_id),
    )
    assert response.status_code == 201, response.json()
    return response.json()["inspection_id"]


def _counts(application, inspection_id: str):
    async def read():
        async with application.state.database.session() as session:
            validations = await session.scalar(
                select(func.count())
                .select_from(InspectionValidation)
                .where(InspectionValidation.inspection_id == inspection_id)
            )
            findings = await session.scalar(
                select(func.count())
                .select_from(InspectionValidationFinding)
                .join(InspectionValidation)
                .where(InspectionValidation.inspection_id == inspection_id)
            )
            audits = await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.entity_type == "inspection",
                    AuditEvent.entity_id == inspection_id,
                    AuditEvent.action.like("INSPECTION_VALIDATION_%"),
                )
            )
            return validations, findings, audits

    return asyncio.run(read())


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_post_pass_retry_and_get_use_one_safe_persisted_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        inspection_id = _intake(
            client,
            tmp_path,
            "valid_rgb_png_height_tiff",
        )
        first = client.post(
            f"/api/v1/inspections/{inspection_id}/validate",
            json=POLICY,
            headers={REQUEST_ID_HEADER: "validation-request-1"},
        )

        assert first.status_code == 200, first.json()
        payload = first.json()
        assert payload["validation_outcome"] == "VALIDATION_PASSED"
        assert payload["inspection_status"] == "READY"
        assert payload["policy"] == POLICY
        assert payload["idempotent_existing"] is False
        assert payload["request_id"] == "validation-request-1"
        assert first.headers[REQUEST_ID_HEADER] == "validation-request-1"
        assert payload["artifacts"]["rgb"]["readability_status"] == "READABLE"
        assert payload["artifacts"]["height"]["readability_status"] == "READABLE"
        assert payload["artifacts"]["height"]["storage_data_type"] == "uint16"
        assert _all_keys(payload).isdisjoint(
            {
                "path",
                "relative_path",
                "absolute_path",
                "filename",
                "model_id",
                "model_version",
                "confidence",
                "classification",
            }
        )
        before = _counts(application, inspection_id)
        assert before == (1, payload["summary"]["finding_count"], 1)

        async def must_not_execute(*_args, **_kwargs):
            raise AssertionError("semantic engine must not execute on replay or GET")

        monkeypatch.setattr(
            application.state.inspection_validation._engine,
            "validate_inspection_pair",
            must_not_execute,
        )
        retry = client.post(
            f"/api/v1/inspections/{inspection_id}/validate",
            json=POLICY,
        )
        assert retry.status_code == 200, retry.json()
        assert retry.json()["validation_id"] == payload["validation_id"]
        assert retry.json()["validation_key"] == payload["validation_key"]
        assert retry.json()["idempotent_existing"] is True
        assert _counts(application, inspection_id) == before
        report = client.get(f"/api/v1/inspections/{inspection_id}/report")
        assert report.status_code == 200, report.json()
        assert report.json()["report"]["inspection"]["status"] == "READY"
        assert report.json()["report"]["validation"]["outcome"] == "VALIDATION_PASSED"
        assert report.json()["report"]["processing"] is None

        statements: list[str] = []

        def record_sql(_connection, _cursor, statement, *_args):
            statements.append(statement)

        sync_engine = application.state.database.engine.sync_engine
        event.listen(sync_engine, "before_cursor_execute", record_sql)
        try:
            retrieved = client.get(
                f"/api/v1/inspections/{inspection_id}/validation",
                headers={REQUEST_ID_HEADER: "validation-get-2"},
            )
        finally:
            event.remove(sync_engine, "before_cursor_execute", record_sql)

        assert retrieved.status_code == 200, retrieved.json()
        assert retrieved.json()["validation_id"] == payload["validation_id"]
        assert retrieved.json()["findings"] == payload["findings"]
        assert retrieved.json()["inspection_status"] == "READY"
        assert retrieved.json()["request_id"] == "validation-get-2"
        assert retrieved.headers[REQUEST_ID_HEADER] == "validation-get-2"
        assert statements
        assert all(item.lstrip().upper().startswith("SELECT") for item in statements)
        assert _counts(application, inspection_id) == before


def test_completed_validation_failed_returns_200_and_failed_status(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    with TestClient(application) as client:
        inspection_id = _intake(client, tmp_path, "height_png_uint8")
        response = client.post(
            f"/api/v1/inspections/{inspection_id}/validate",
            json=POLICY,
        )
        report = client.get(f"/api/v1/inspections/{inspection_id}/report")

    assert response.status_code == 200, response.json()
    assert report.status_code == 200, report.json()
    assert report.json()["report"]["inspection"]["status"] == "VALIDATION_FAILED"
    assert report.json()["report"]["validation"]["outcome"] == "VALIDATION_FAILED"
    assert response.json()["validation_outcome"] == "VALIDATION_FAILED"
    assert response.json()["inspection_status"] == "VALIDATION_FAILED"
    assert response.json()["summary"]["blocking_count"] > 0
    assert "HEIGHT_BIT_DEPTH_TOO_LOW" in {
        item["code"] for item in response.json()["findings"]
    }


def test_completed_validation_error_returns_200_and_error_status(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))

    class ErrorEngine:
        async def validate_inspection_pair(self, inspection_id, policy):
            finding = FindingFactory().create("VALIDATOR_INTERNAL_ERROR")
            now = datetime.now(timezone.utc)
            blank = lambda artifact_type: ArtifactTechnicalSummary(
                artifact_type=artifact_type,
                sha256=None,
                byte_size=None,
                declared_media_type=None,
                detected_format=None,
                width=None,
                height=None,
                channels=None,
                bit_depth=None,
                storage_data_type=None,
                readability_status=ReadabilityStatus.UNINSPECTED,
            )
            from app.db.models import ArtifactType

            return InspectionValidationResult(
                contract_version="pcb-aoi-inspection-validation/1.0",
                validation_id=str(uuid4()),
                inspection_id=inspection_id,
                validation_policy_id=policy.policy_id,
                validation_policy_version=policy.policy_version,
                outcome=ValidationOutcome.VALIDATION_ERROR,
                started_at=now,
                completed_at=now,
                validator_version=DEFAULT_VALIDATOR_VERSION,
                rgb_artifact=blank(ArtifactType.RGB_RAW),
                height_artifact=blank(ArtifactType.HEIGHT_RAW),
                findings=(finding,),
                summary=ValidationSummary(
                    finding_count=1,
                    info_count=0,
                    warning_count=0,
                    error_count=1,
                    blocking_count=1,
                    technically_ready=False,
                    synthetic_example=False,
                ),
            )

    application.state.inspection_validation._engine = ErrorEngine()
    with TestClient(application) as client:
        inspection_id = _intake(
            client,
            tmp_path,
            "valid_rgb_png_height_tiff",
        )
        response = client.post(
            f"/api/v1/inspections/{inspection_id}/validate",
            json=POLICY,
        )
        report = client.get(f"/api/v1/inspections/{inspection_id}/report")

    assert response.status_code == 200, response.json()
    assert report.status_code == 200, report.json()
    assert report.json()["report"]["inspection"]["status"] == "ERROR"
    assert report.json()["report"]["validation"]["outcome"] == "VALIDATION_ERROR"
    assert response.json()["validation_outcome"] == "VALIDATION_ERROR"
    assert response.json()["inspection_status"] == "ERROR"
    assert [item["code"] for item in response.json()["findings"]] == [
        "VALIDATOR_INTERNAL_ERROR"
    ]
    assert "traceback" not in response.text.lower()
    assert str(tmp_path.resolve()) not in response.text


def test_standalone_persisted_result_is_adopted_without_engine_rerun(
    tmp_path,
    monkeypatch,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    orchestrator = application.state.inspection_validation
    with TestClient(application) as client:
        inspection_id = _intake(
            client,
            tmp_path,
            "valid_rgb_png_height_tiff",
        )
        policy = orchestrator._policies.load(**{
            "policy_id": POLICY["policy_id"],
            "policy_version": POLICY["policy_version"],
        })
        inspection, artifacts = asyncio.run(
            orchestrator._inspection_and_artifacts(inspection_id)
        )
        key = orchestrator._validation_key(inspection_id, artifacts, policy)
        result = asyncio.run(
            orchestrator._engine.validate_inspection_pair(inspection_id, policy)
        )
        asyncio.run(
            application.state.repositories.validations.save_validation_result(
                inspection_id,
                result,
                key,
            )
        )

        async def must_not_execute(*_args, **_kwargs):
            raise AssertionError("standalone evidence must be adopted")

        monkeypatch.setattr(
            orchestrator._engine,
            "validate_inspection_pair",
            must_not_execute,
        )
        response = client.post(
            f"/api/v1/inspections/{inspection_id}/validate",
            json=POLICY,
        )

    assert response.status_code == 200, response.json()
    assert response.json()["validation_id"] == result.validation_id
    assert response.json()["idempotent_existing"] is True
    assert response.json()["inspection_status"] == "READY"
    assert _counts(application, inspection_id) == (1, len(result.findings), 1)


def test_concurrent_identical_executions_converge_on_one_lifecycle(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    orchestrator = application.state.inspection_validation
    with TestClient(application) as client:
        inspection_id = _intake(
            client,
            tmp_path,
            "valid_rgb_png_height_tiff",
        )

        async def execute_both():
            return await asyncio.gather(
                orchestrator.execute_validation(
                    inspection_id,
                    POLICY["policy_id"],
                    POLICY["policy_version"],
                    actor_id=None,
                    request_id="concurrent-1",
                ),
                orchestrator.execute_validation(
                    inspection_id,
                    POLICY["policy_id"],
                    POLICY["policy_version"],
                    actor_id=None,
                    request_id="concurrent-2",
                ),
            )

        responses = asyncio.run(execute_both())

    assert responses[0].result.validation_id == responses[1].result.validation_id
    assert responses[0].validation_key == responses[1].validation_key
    assert sum(item.idempotent_existing for item in responses) == 1
    assert _counts(application, inspection_id) == (
        1,
        len(responses[0].result.findings),
        1,
    )


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
def test_non_received_status_without_exact_replay_returns_409(
    tmp_path,
    status,
) -> None:
    application = create_app(_settings(tmp_path / status.value))
    with TestClient(application) as client:
        inspection_id = _intake(
            client,
            tmp_path / status.value,
            "valid_rgb_png_height_tiff",
        )

        async def set_status():
            values = {"status": status}
            if status in {
                InspectionStatus.PASS,
                InspectionStatus.FAIL,
                InspectionStatus.UNCERTAIN,
            }:
                values["completed_at"] = datetime.now(timezone.utc)
            async with application.state.database.session_factory() as session:
                async with session.begin():
                    await session.execute(
                        update(Inspection)
                        .where(Inspection.id == inspection_id)
                        .values(**values)
                    )

        asyncio.run(set_status())
        response = client.post(
            f"/api/v1/inspections/{inspection_id}/validate",
            json=POLICY,
        )

    assert response.status_code == 409
    assert response.json()["code"] == "INSPECTION_NOT_ELIGIBLE_FOR_VALIDATION"
    assert _counts(application, inspection_id) == (0, 0, 0)


def test_post_and_get_failures_are_structured_safe_and_do_not_execute_engine(
    tmp_path,
    monkeypatch,
) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    calls = 0

    async def count_engine(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("engine must not run")

    monkeypatch.setattr(
        application.state.inspection_validation._engine,
        "validate_inspection_pair",
        count_engine,
    )
    missing_id = str(uuid4())
    with TestClient(application) as client:
        malformed = client.post("/api/v1/inspections/not-a-uuid/validate", json=POLICY)
        missing = client.post(
            f"/api/v1/inspections/{missing_id}/validate",
            json=POLICY,
        )
        missing_body = client.post(
            f"/api/v1/inspections/{missing_id}/validate"
        )
        blank_policy = client.post(
            f"/api/v1/inspections/{missing_id}/validate",
            json={"policy_id": "", "policy_version": "1.0"},
        )
        unknown_policy = client.post(
            f"/api/v1/inspections/{missing_id}/validate",
            json={"policy_id": "unknown", "policy_version": "1.0"},
        )
        unsupported = client.post(
            f"/api/v1/inspections/{missing_id}/validate",
            json={
                "policy_id": POLICY["policy_id"],
                "policy_version": "999",
            },
        )
        unknown_get = client.get(
            f"/api/v1/inspections/{missing_id}/validation"
        )
        inspection_id = _intake(
            client,
            tmp_path,
            "valid_rgb_png_height_tiff",
        )
        no_result = client.get(
            f"/api/v1/inspections/{inspection_id}/validation",
            headers={REQUEST_ID_HEADER: "get-no-result"},
        )

    assert malformed.status_code == 400
    assert missing.status_code == 404
    assert missing_body.status_code == 422
    assert missing_body.json()["code"] == "INVALID_VALIDATION_REQUEST"
    assert blank_policy.status_code == 400
    assert unknown_policy.status_code == 404
    assert unknown_policy.json()["code"] == "VALIDATION_POLICY_NOT_FOUND"
    assert unsupported.status_code == 404
    assert unsupported.json()["code"] == "VALIDATION_POLICY_VERSION_UNSUPPORTED"
    assert unknown_get.status_code == 404
    assert no_result.status_code == 404
    assert no_result.json() == {
        "code": "INSPECTION_VALIDATION_NOT_FOUND",
        "message": "No validation result exists for this inspection.",
        "request_id": "get-no-result",
    }
    assert calls == 0
    for response in (
        malformed,
        missing,
        missing_body,
        blank_policy,
        unknown_policy,
        unsupported,
        unknown_get,
        no_result,
    ):
        assert set(response.json()) == {"code", "message", "request_id"}
        assert str(tmp_path.resolve()) not in response.text
        assert "traceback" not in response.text.lower()
        assert "select " not in response.text.lower()


def test_malformed_registered_application_policy_fails_during_assembly(tmp_path) -> None:
    malformed = tmp_path / "malformed-policy.json"
    malformed.write_text('{"contract_version":"wrong"}', encoding="utf-8")
    loader = ValidationPolicyLoader(
        registry={(POLICY["policy_id"], POLICY["policy_version"]): malformed}
    )
    with pytest.raises(PolicyLoadError):
        create_app(
            _settings(tmp_path / "runtime"),
            validation_policy_loader=loader,
        )
    assert DEFAULT_DEVELOPMENT_POLICY_PATH.is_file()


def test_unexpected_orchestrator_failure_returns_safe_500(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))

    class FailingOrchestrator:
        async def execute_validation(self, *_args, **_kwargs):
            raise RuntimeError(r"SELECT secret FROM C:\private\database.sqlite3")

    application.state.inspection_validation = FailingOrchestrator()
    with TestClient(application) as client:
        response = client.post(
            f"/api/v1/inspections/{uuid4()}/validate",
            json=POLICY,
            headers={REQUEST_ID_HEADER: "failed-orchestration"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "code": "VALIDATION_ORCHESTRATION_FAILED",
        "message": "Inspection validation could not be completed reliably.",
        "request_id": "failed-orchestration",
    }
    assert "SELECT" not in response.text
    assert "private" not in response.text


def test_openapi_documents_both_technical_validation_routes(tmp_path) -> None:
    application = create_app(_settings(tmp_path / "runtime"))
    schema = application.openapi()
    post = schema["paths"][
        "/api/v1/inspections/{inspection_id}/validate"
    ]["post"]
    get = schema["paths"][
        "/api/v1/inspections/{inspection_id}/validation"
    ]["get"]

    assert "does not perform PCB defect classification" in post["description"]
    assert "does not run AI inference" in post["description"]
    assert "technically ready for future preprocessing only" in post["description"]
    assert "Revalidation is not supported" in post["description"]
    assert "system-idempotent" in post["description"]
    assert "does not rerun validation" in get["description"]
    assert "does not classify the PCB" in get["description"]
