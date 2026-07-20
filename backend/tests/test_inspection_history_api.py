import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select

from app.core.config import Settings
from app.db.models import (
    AuditEvent,
    Inspection,
    InspectionArtifact,
    InspectionInferenceResult,
    InspectionInferenceResultFinding,
    InspectionPreprocessingResult,
    InspectionPreprocessingResultFinding,
    InspectionProcessingRun,
    InspectionStatus,
    InspectionValidation,
    InspectionValidationFinding,
)
from app.db.processing_types import (
    PersistedInferenceOutcome,
    PersistedPreprocessingOutcome,
    ProcessingFinalDecision,
    ProcessingRunStatus,
)
from app.db.validation_types import ValidationOutcome
from app.main import create_app

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
SUMMARY = json.dumps(
    {
        "finding_count": 2,
        "info_count": 0,
        "warning_count": 1,
        "error_count": 1,
        "blocking_count": 1,
        "pair_valid": False,
        "technically_ready": False,
    }
)


def _app(tmp_path: Path):
    return create_app(Settings(_env_file=None, runtime_root=tmp_path / "runtime"))


def _uuid(group: int, number: int) -> str:
    return f"{group:08x}-0000-4000-8000-{number:012x}"


def _add_inspection(
    application,
    number: int,
    *,
    created_at: datetime,
    status: InspectionStatus = InspectionStatus.RECEIVED,
    board_id: str = "board-a",
    recipe_id: str = "recipe-a",
    recipe_version: str = "1.0",
    lot_id: str | None = "lot-a",
    operator_id: str | None = "operator-a",
    error_code: str | None = None,
) -> str:
    inspection_id = _uuid(1, number)

    async def insert() -> None:
        async with application.state.database.session() as session:
            session.add(
                Inspection(
                    id=inspection_id,
                    status=status,
                    board_id=board_id,
                    recipe_id=recipe_id,
                    recipe_version=recipe_version,
                    lot_id=lot_id,
                    operator_id=operator_id,
                    request_id=f"intake-{number}",
                    error_code=error_code,
                    error_message="private failure detail" if error_code else None,
                    created_at=created_at,
                    started_at=created_at if status is InspectionStatus.PROCESSING else None,
                    completed_at=(
                        created_at + timedelta(seconds=2)
                        if status in {InspectionStatus.PASS, InspectionStatus.FAIL, InspectionStatus.UNCERTAIN}
                        else None
                    ),
                )
            )
            await session.commit()

    asyncio.run(insert())
    return inspection_id


def _add_validation(
    application,
    inspection_id: str,
    number: int,
    *,
    outcome: ValidationOutcome,
    completed_at: datetime,
) -> str:
    validation_id = _uuid(2, number)

    async def insert() -> None:
        async with application.state.database.session() as session:
            session.add(
                InspectionValidation(
                    id=validation_id,
                    inspection_id=inspection_id,
                    contract_version="pcb-aoi-inspection-validation/1.0",
                    policy_id="development-native-rgb-height",
                    policy_version="1.0",
                    validator_version="1.0.0",
                    validation_key=f"{number:064x}",
                    outcome=outcome,
                    started_at=completed_at - timedelta(seconds=1),
                    completed_at=completed_at,
                    rgb_summary_json="{}",
                    height_summary_json="{}",
                    summary_json=SUMMARY,
                    result_json="{}",
                    result_sha256="a" * 64,
                    created_at=completed_at,
                )
            )
            await session.commit()

    asyncio.run(insert())
    return validation_id


def _add_processing(
    application,
    inspection_id: str,
    validation_id: str,
    number: int,
    *,
    started_at: datetime,
    decision: ProcessingFinalDecision = ProcessingFinalDecision.PASS,
    defect_type: str | None = None,
) -> str:
    run_id = _uuid(3, number)
    preprocessing_id = _uuid(4, number)
    inference_id = _uuid(5, number)
    completed_at = started_at + timedelta(seconds=1)

    async def insert() -> None:
        async with application.state.database.session() as session:
            session.add(
                InspectionProcessingRun(
                    id=run_id,
                    inspection_id=inspection_id,
                    validation_id=validation_id,
                    processing_key=f"{number:064x}",
                    status=ProcessingRunStatus.COMPLETED,
                    preprocessing_policy_id="synthetic-paired-rgb-height",
                    preprocessing_policy_version="1.0",
                    preprocessing_implementation_id="synthetic-copy",
                    preprocessing_implementation_version="1.0",
                    inference_policy_id="synthetic-deterministic-mock-inference",
                    inference_policy_version="1.0",
                    engine_id="synthetic-deterministic-mock-engine",
                    engine_version="1.0.0",
                    engine_type="MOCK",
                    started_at=started_at,
                    completed_at=completed_at,
                    final_decision=decision,
                    created_at=started_at,
                )
            )
            session.add(
                InspectionPreprocessingResult(
                    id=preprocessing_id,
                    processing_run_id=run_id,
                    contract_version="pcb-aoi-inspection-preprocessing/1.0",
                    policy_id="synthetic-paired-rgb-height",
                    policy_version="1.0",
                    implementation_id="synthetic-copy",
                    implementation_version="1.0",
                    outcome=PersistedPreprocessingOutcome.SUCCEEDED,
                    started_at=started_at,
                    completed_at=completed_at,
                    summary_json="{}",
                    result_json="{}",
                    result_sha256="b" * 64,
                    created_at=started_at,
                )
            )
            session.add(
                InspectionInferenceResult(
                    id=inference_id,
                    processing_run_id=run_id,
                    preprocessing_id=preprocessing_id,
                    contract_version="pcb-aoi-inspection-inference/1.0",
                    policy_id="synthetic-deterministic-mock-inference",
                    policy_version="1.0",
                    engine_id="synthetic-deterministic-mock-engine",
                    engine_version="1.0.0",
                    engine_type="MOCK",
                    execution_outcome=PersistedInferenceOutcome.SUCCEEDED,
                    decision=decision,
                    defect_type=defect_type,
                    confidence=None,
                    decision_basis="DETERMINISTIC_HASH_BUCKET",
                    decision_digest="c" * 64,
                    started_at=started_at,
                    completed_at=completed_at,
                    summary_json="{}",
                    result_json="{}",
                    result_sha256="d" * 64,
                    created_at=started_at,
                )
            )
            await session.commit()

    asyncio.run(insert())
    return run_id


def _add_processing_error(
    application,
    inspection_id: str,
    validation_id: str,
    number: int,
    *,
    started_at: datetime,
) -> str:
    run_id = _uuid(3, number)
    preprocessing_id = _uuid(4, number)
    completed_at = started_at + timedelta(seconds=1)

    async def insert() -> None:
        async with application.state.database.session() as session:
            session.add(
                InspectionProcessingRun(
                    id=run_id,
                    inspection_id=inspection_id,
                    validation_id=validation_id,
                    processing_key=f"{number:064x}",
                    status=ProcessingRunStatus.ERROR,
                    preprocessing_policy_id="synthetic-paired-rgb-height",
                    preprocessing_policy_version="1.0",
                    preprocessing_implementation_id="synthetic-copy",
                    preprocessing_implementation_version="1.0",
                    inference_policy_id="synthetic-deterministic-mock-inference",
                    inference_policy_version="1.0",
                    engine_id="synthetic-deterministic-mock-engine",
                    engine_version="1.0.0",
                    engine_type="MOCK",
                    started_at=started_at,
                    completed_at=completed_at,
                    error_code="PREPROCESSING_FAILED",
                    error_message="private processing failure",
                    created_at=started_at,
                )
            )
            session.add(
                InspectionPreprocessingResult(
                    id=preprocessing_id,
                    processing_run_id=run_id,
                    contract_version="pcb-aoi-inspection-preprocessing/1.0",
                    policy_id="synthetic-paired-rgb-height",
                    policy_version="1.0",
                    implementation_id="synthetic-copy",
                    implementation_version="1.0",
                    outcome=PersistedPreprocessingOutcome.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    summary_json="{}",
                    result_json="{}",
                    result_sha256="e" * 64,
                    created_at=started_at,
                )
            )
            await session.commit()

    asyncio.run(insert())
    return run_id


def _ids(response) -> list[str]:
    return [item["inspection_id"] for item in response.json()["items"]]


def _cursor(document: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    ).decode().rstrip("=")


def test_empty_and_base_history_contract_order_and_request_id(tmp_path) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        empty = client.get("/api/v1/inspections", headers={"X-Request-ID": "history-request"})
        assert empty.status_code == 200
        assert empty.json() == {
            "items": [],
            "page": {"limit": 25, "has_more": False, "next_cursor": None},
            "applied_filters": {},
            "request_id": "history-request",
        }
        first = _add_inspection(application, 1, created_at=NOW)
        second = _add_inspection(application, 2, created_at=NOW)
        older = _add_inspection(application, 3, created_at=NOW - timedelta(seconds=1))
        response = client.get("/api/v1/inspections")

    assert response.status_code == 200
    assert _ids(response) == [second, first, older]
    item = response.json()["items"][0]
    assert item["validation"] is None and item["processing"] is None
    assert item["technical_error_code"] is None
    assert "request_id" not in item
    forbidden = {"relative_path", "filename", "confidence", "result_json", "findings"}
    assert forbidden.isdisjoint(item)
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_cursor_pagination_is_complete_stable_and_limit_is_not_bound(tmp_path) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        expected = [
            _add_inspection(application, number, created_at=NOW - timedelta(seconds=number))
            for number in range(31)
        ]
        expected.reverse()
        # Construction is oldest ID first but timestamps make number 0 newest.
        expected = list(reversed(expected))
        seen = []
        cursor = None
        limit = 7
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
                limit = 5  # limit is deliberately excluded from cursor binding
                params["limit"] = limit
            response = client.get("/api/v1/inspections", params=params)
            assert response.status_code == 200, response.json()
            seen.extend(_ids(response))
            cursor = response.json()["page"]["next_cursor"]
            if cursor is None:
                break
        assert seen == [_uuid(1, number) for number in range(31)]
        assert len(seen) == len(set(seen))

        first_page = client.get("/api/v1/inspections", params={"limit": 4}).json()
        _add_inspection(application, 99, created_at=NOW + timedelta(days=1))
        continuation = client.get(
            "/api/v1/inspections",
            params={"limit": 100, "cursor": first_page["page"]["next_cursor"]},
        )
        assert _uuid(1, 99) not in _ids(continuation)


def test_cursor_and_query_validation_fail_closed(tmp_path) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        _add_inspection(application, 1, created_at=NOW, board_id="board-a")
        page = client.get("/api/v1/inspections", params={"limit": 1, "board_id": "board-a"})
        cursor = page.json()["page"]["next_cursor"]
        # Ensure a next cursor exists.
        _add_inspection(application, 2, created_at=NOW - timedelta(seconds=1), board_id="board-a")
        page = client.get("/api/v1/inspections", params={"limit": 1, "board_id": "board-a"})
        cursor = page.json()["page"]["next_cursor"]
        assert cursor
        mismatch = client.get(
            "/api/v1/inspections",
            params={"cursor": cursor, "board_id": "board-b"},
        )
        malformed = client.get("/api/v1/inspections", params={"cursor": cursor[:-1] + "!"})
        low = client.get("/api/v1/inspections", params={"limit": 0})
        high = client.get("/api/v1/inspections", params={"limit": 101})
        naive = client.get("/api/v1/inspections", params={"created_from": "2026-01-01T00:00:00"})
        reversed_range = client.get(
            "/api/v1/inspections",
            params={
                "created_from": "2026-01-02T00:00:00Z",
                "created_to": "2026-01-01T00:00:00Z",
            },
        )
    assert mismatch.status_code == 400
    assert mismatch.json()["code"] == "HISTORY_CURSOR_FILTER_MISMATCH"
    assert malformed.status_code == 400 and malformed.json()["code"] == "INVALID_HISTORY_CURSOR"
    assert {low.status_code, high.status_code} == {422}
    assert low.json()["code"] == high.json()["code"] == "INVALID_INSPECTION_HISTORY_QUERY"
    assert naive.status_code == 400 and reversed_range.status_code == 400


def test_cursor_rejects_malformed_json_version_uuid_and_timestamp(tmp_path) -> None:
    application = _app(tmp_path)
    base = {
        "contract_version": "pcb-aoi-inspection-history-cursor/1.0",
        "created_at": "2026-07-20T12:00:00Z",
        "filter_digest": "0" * 64,
        "inspection_id": _uuid(1, 1),
    }
    variants = [
        base64.urlsafe_b64encode(b"not-json").decode().rstrip("="),
        _cursor({**base, "contract_version": "future/9.0"}),
        _cursor(
            {
                **base,
                "inspection_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
            }
        ),
        _cursor({**base, "created_at": "2026-07-20T12:00:00"}),
    ]
    with TestClient(application) as client:
        responses = [
            client.get("/api/v1/inspections", params={"cursor": value})
            for value in variants
        ]
    assert [response.status_code for response in responses] == [400, 400, 400, 400]
    assert responses[1].json()["code"] == "UNSUPPORTED_HISTORY_CURSOR_VERSION"
    assert all("traceback" not in json.dumps(response.json()).lower() for response in responses)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("status", "NOT_A_STATUS"),
        ("validation_outcome", "NOT_AN_OUTCOME"),
        ("processing_status", "NOT_A_STATUS"),
        ("mock_decision", "NOT_A_DECISION"),
        ("defect_type", "invented_defect"),
        ("board_id", "   "),
        ("recipe_id", "bad\u0007value"),
    ],
)
def test_invalid_filters_are_rejected(tmp_path, parameter, value) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        response = client.get("/api/v1/inspections", params={parameter: value})
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_HISTORY_FILTER"


def test_all_supported_filters_use_latest_children_and_and_semantics(tmp_path) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        received = _add_inspection(
            application, 1, created_at=NOW, board_id="board-r", lot_id=None, operator_id=None
        )
        ready = _add_inspection(
            application,
            2,
            created_at=NOW + timedelta(minutes=1),
            status=InspectionStatus.READY,
            board_id="board-ready",
            recipe_id="recipe-b",
            recipe_version="2.0",
            lot_id="lot-b",
            operator_id="operator-b",
        )
        ready_validation = _add_validation(
            application,
            ready,
            2,
            outcome=ValidationOutcome.VALIDATION_PASSED,
            completed_at=NOW + timedelta(minutes=1),
        )
        completed = _add_inspection(
            application,
            3,
            created_at=NOW + timedelta(minutes=2),
            status=InspectionStatus.FAIL,
            board_id="board-fail",
            recipe_id="recipe-c",
            recipe_version="3.0",
            lot_id="lot-c",
            operator_id="operator-c",
        )
        completed_validation = _add_validation(
            application,
            completed,
            3,
            outcome=ValidationOutcome.VALIDATION_PASSED,
            completed_at=NOW + timedelta(minutes=2),
        )
        _add_processing(
            application,
            completed,
            completed_validation,
            3,
            started_at=NOW + timedelta(minutes=3),
            decision=ProcessingFinalDecision.FAIL,
            defect_type="missing_part",
        )

        cases = [
            ({"status": "RECEIVED"}, [received]),
            ({"board_id": "board-ready"}, [ready]),
            ({"recipe_id": "recipe-b", "recipe_version": "2.0"}, [ready]),
            ({"lot_id": "lot-c", "operator_id": "operator-c"}, [completed]),
            ({"validation_outcome": "VALIDATION_PASSED"}, [completed, ready]),
            ({"processing_status": "COMPLETED"}, [completed]),
            ({"mock_decision": "FAIL"}, [completed]),
            ({"defect_type": "missing_part"}, [completed]),
            ({"has_validation": "false"}, [received]),
            ({"has_processing": "true"}, [completed]),
            (
                {
                    "created_from": "2026-07-20T12:01:00Z",
                    "created_to": "2026-07-20T12:02:00Z",
                },
                [ready],
            ),
            ({"status": "FAIL", "board_id": "board-ready"}, []),
        ]
        for params, expected in cases:
            response = client.get("/api/v1/inspections", params=params)
            assert response.status_code == 200, (params, response.json())
            assert _ids(response) == expected
            assert response.json()["applied_filters"]
        detail = client.get("/api/v1/inspections", params={"board_id": "board-r"})

    assert ready_validation
    assert detail.json()["items"][0]["lot_id"] is None


def test_latest_summaries_and_nonproduction_flags_are_compact(tmp_path) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        ready = _add_inspection(
            application, 1, created_at=NOW, status=InspectionStatus.READY
        )
        older = _add_validation(
            application,
            ready,
            1,
            outcome=ValidationOutcome.VALIDATION_FAILED,
            completed_at=NOW,
        )
        latest = _add_validation(
            application,
            ready,
            2,
            outcome=ValidationOutcome.VALIDATION_PASSED,
            completed_at=NOW + timedelta(seconds=1),
        )
        completed = _add_inspection(
            application,
            2,
            created_at=NOW + timedelta(minutes=1),
            status=InspectionStatus.PASS,
        )
        validation = _add_validation(
            application,
            completed,
            3,
            outcome=ValidationOutcome.VALIDATION_PASSED,
            completed_at=NOW + timedelta(minutes=1),
        )
        run = _add_processing(
            application,
            completed,
            validation,
            3,
            started_at=NOW + timedelta(minutes=2),
        )
        response = client.get("/api/v1/inspections")

    assert response.status_code == 200, response.json()
    by_id = {item["inspection_id"]: item for item in response.json()["items"]}
    assert by_id[ready]["validation"]["validation_id"] == latest
    assert by_id[ready]["validation"]["validation_id"] != older
    assert by_id[ready]["validation"]["total_findings"] == 2
    processing = by_id[completed]["processing"]
    assert processing["processing_run_id"] == run
    assert processing["processing_status"] == "COMPLETED"
    assert processing["preprocessing_id"]
    assert processing["preprocessing_outcome"] == "PREPROCESSING_SUCCEEDED"
    assert processing["inference_id"]
    assert processing["inference_execution_outcome"] == "INFERENCE_SUCCEEDED"
    assert processing["mock_decision"] == "PASS"
    assert processing["defect_type"] is None
    assert processing["synthetic_input"] is True
    assert processing["mock_preprocessing"] is True
    assert processing["mock_inference"] is True
    assert processing["production_approved"] is False


def test_uncertain_and_technical_processing_error_summaries(tmp_path) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        uncertain = _add_inspection(
            application,
            10,
            created_at=NOW,
            status=InspectionStatus.UNCERTAIN,
        )
        uncertain_validation = _add_validation(
            application,
            uncertain,
            10,
            outcome=ValidationOutcome.VALIDATION_PASSED,
            completed_at=NOW,
        )
        _add_processing(
            application,
            uncertain,
            uncertain_validation,
            10,
            started_at=NOW + timedelta(seconds=1),
            decision=ProcessingFinalDecision.UNCERTAIN,
        )
        errored = _add_inspection(
            application,
            11,
            created_at=NOW + timedelta(minutes=1),
            status=InspectionStatus.ERROR,
            error_code="PREPROCESSING_FAILED",
        )
        error_validation = _add_validation(
            application,
            errored,
            11,
            outcome=ValidationOutcome.VALIDATION_PASSED,
            completed_at=NOW + timedelta(minutes=1),
        )
        _add_processing_error(
            application,
            errored,
            error_validation,
            11,
            started_at=NOW + timedelta(minutes=1, seconds=1),
        )
        response = client.get("/api/v1/inspections")

    assert response.status_code == 200, response.json()
    by_id = {item["inspection_id"]: item for item in response.json()["items"]}
    assert by_id[uncertain]["processing"]["mock_decision"] == "UNCERTAIN"
    assert by_id[uncertain]["processing"]["defect_type"] is None
    assert by_id[errored]["status"] == "ERROR"
    assert by_id[errored]["technical_error_code"] == "PREPROCESSING_FAILED"
    assert by_id[errored]["processing"]["processing_status"] == "ERROR"
    assert by_id[errored]["processing"]["mock_inference"] is False


@pytest.mark.parametrize("page_size", [1, 25, 100])
def test_query_count_is_bounded_and_history_is_read_only(
    tmp_path, page_size, monkeypatch
) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        for number in range(page_size):
            _add_inspection(application, number + 1, created_at=NOW - timedelta(seconds=number))

        async def counts():
            async with application.state.database.session() as session:
                models = (
                    Inspection,
                    InspectionArtifact,
                    InspectionValidation,
                    InspectionValidationFinding,
                    InspectionProcessingRun,
                    InspectionPreprocessingResult,
                    InspectionPreprocessingResultFinding,
                    InspectionInferenceResult,
                    InspectionInferenceResultFinding,
                    AuditEvent,
                )
                values = []
                for model in models:
                    values.append(
                        await session.scalar(select(func.count()).select_from(model))
                    )
                return tuple(values)

        before = asyncio.run(counts())
        runtime_root = application.state.runtime_paths.root
        files_before = {
            str(path.relative_to(runtime_root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in runtime_root.rglob("*")
            if path.is_file()
        }
        statements = []

        async def forbidden_service(*_args, **_kwargs):
            raise AssertionError("history must not execute workflow services")

        def forbidden_file_read(*_args, **_kwargs):
            raise AssertionError("history must not read runtime or manifest files")

        monkeypatch.setattr(
            application.state.inspection_validation,
            "execute_validation",
            forbidden_service,
        )
        monkeypatch.setattr(
            application.state.inspection_processing,
            "execute_processing",
            forbidden_service,
        )
        monkeypatch.setattr(Path, "open", forbidden_file_read)
        monkeypatch.setattr(Path, "read_text", forbidden_file_read)
        monkeypatch.setattr(Path, "read_bytes", forbidden_file_read)

        def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement.strip().split(None, 1)[0].upper())

        event.listen(application.state.database.engine.sync_engine, "before_cursor_execute", capture)
        try:
            response = client.get("/api/v1/inspections", params={"limit": page_size})
        finally:
            event.remove(application.state.database.engine.sync_engine, "before_cursor_execute", capture)
        after = asyncio.run(counts())
        files_after = {
            str(path.relative_to(runtime_root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in runtime_root.rglob("*")
            if path.is_file()
        }

    assert response.status_code == 200
    assert before == after
    assert files_before == files_after
    assert statements and set(statements) == {"SELECT"}
    assert len(statements) == 3


def test_inconsistent_lifecycle_and_repository_failure_return_safe_500(tmp_path, monkeypatch) -> None:
    application = _app(tmp_path)
    with TestClient(application) as client:
        _add_inspection(application, 1, created_at=NOW, status=InspectionStatus.READY)
        inconsistent = client.get("/api/v1/inspections")
        assert inconsistent.status_code == 500
        assert inconsistent.json()["code"] == "INSPECTION_HISTORY_INCONSISTENT"
        assert "lifecycle" not in inconsistent.json()["message"].lower()

        async def fail(*_args, **_kwargs):
            raise RuntimeError("secret database failure")

        monkeypatch.setattr(application.state.inspection_history.repository, "fetch_page", fail)
        failed = client.get("/api/v1/inspections")
    assert failed.status_code == 500
    assert failed.json()["code"] == "INSPECTION_HISTORY_READ_FAILED"
    assert "secret" not in json.dumps(failed.json()).lower()


def test_openapi_documents_history_contract_without_station_or_total(tmp_path) -> None:
    application = _app(tmp_path)
    schema = application.openapi()
    operation = schema["paths"]["/api/v1/inspections"]["get"]
    parameters = {item["name"]: item for item in operation["parameters"]}
    assert set(parameters) >= {
        "limit",
        "cursor",
        "status",
        "board_id",
        "recipe_id",
        "recipe_version",
        "lot_id",
        "operator_id",
        "created_from",
        "created_to",
        "validation_outcome",
        "processing_status",
        "mock_decision",
        "defect_type",
        "has_validation",
        "has_processing",
    }
    assert "station_id" not in parameters
    assert "total" not in json.dumps(operation).lower()
    assert "read" in operation["description"].lower()
