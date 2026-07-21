import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy import event, update

from app.core.config import Settings
from app.core.request_context import REQUEST_ID_HEADER
from app.db.models import Inspection, InspectionStatus
from app.db.repositories import InspectionArtifactCreate, InspectionCreate
from app.main import create_app


def _app(tmp_path):
    return create_app(Settings(_env_file=None, runtime_root=tmp_path / "runtime"))


def _inspection(application, status=InspectionStatus.RECEIVED):
    return asyncio.run(application.state.repositories.inspections.create(InspectionCreate(
        status=status,
        board_id="BOARD-A",
        recipe_id="RECIPE-A",
        recipe_version="1.0",
        operator_id="operator-1",
        request_id="intake-request",
        completed_at=datetime.now(timezone.utc) if status in {
            InspectionStatus.PASS, InspectionStatus.FAIL, InspectionStatus.UNCERTAIN
        } else None,
    )))


def test_received_report_is_deterministic_schema_valid_path_free_and_read_only(tmp_path, monkeypatch):
    application = _app(tmp_path)
    with TestClient(application) as client:
        inspection = _inspection(application)
        asyncio.run(application.state.repositories.artifacts.create(InspectionArtifactCreate(
            inspection_id=inspection.id,
            artifact_type="RGB_RAW",
            relative_path="private/never-opened.png",
            sha256="a" * 64,
            byte_size=123,
            media_type="image/png",
        )))
        writes = []
        def record(_c, _cur, statement, _parameters, _context, _many):
            if not statement.lstrip().upper().startswith("SELECT"):
                writes.append(statement)
        engine = application.state.database.engine.sync_engine
        event.listen(engine, "before_cursor_execute", record)
        original_open = Path.open
        monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("report GET opened a file")))
        try:
            first = client.get(f"/api/v1/inspections/{inspection.id}/report", headers={REQUEST_ID_HEADER: "report-request-1"})
            second = client.get(f"/api/v1/inspections/{inspection.id}/report", headers={REQUEST_ID_HEADER: "report-request-2"})
        finally:
            event.remove(engine, "before_cursor_execute", record)
            monkeypatch.setattr(Path, "open", original_open)
    assert first.status_code == second.status_code == 200, first.json()
    assert first.json()["report"] == second.json()["report"]
    assert first.json()["report_sha256"] == second.json()["report_sha256"]
    assert first.json()["request_id"] != second.json()["request_id"]
    assert writes == []
    report = first.json()["report"]
    assert report["validation"] is None and report["processing"] is None
    assert report["development_only"] is True and report["production_approved"] is False
    assert set(report["artifacts"][0]) == {"artifact_type", "sha256", "byte_size", "media_type", "created_at"}
    assert "relative_path" not in first.text and "never-opened" not in first.text
    assert "confidence" not in json.dumps(report).lower() or "No confidence was produced." in report["limitations"]
    schema = json.loads((Path(__file__).parents[2] / "contracts/inspection_development_report.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)


def test_persisted_change_changes_report_hash(tmp_path):
    application = _app(tmp_path)
    with TestClient(application) as client:
        inspection = _inspection(application)
        before = client.get(f"/api/v1/inspections/{inspection.id}/report").json()
        async def mutate():
            async with application.state.database.session_factory() as session:
                await session.execute(update(Inspection).where(Inspection.id == inspection.id).values(board_id="BOARD-B"))
                await session.commit()
        asyncio.run(mutate())
        after = client.get(f"/api/v1/inspections/{inspection.id}/report").json()
    assert before["report_sha256"] != after["report_sha256"]
    assert after["report"]["inspection"]["board_id"] == "BOARD-B"


def test_missing_malformed_and_inconsistent_report_errors_are_safe(tmp_path):
    application = _app(tmp_path)
    with TestClient(application) as client:
        malformed = client.get("/api/v1/inspections/not-a-uuid/report")
        from uuid import uuid4
        missing = client.get(f"/api/v1/inspections/{uuid4()}/report")
        inconsistent = _inspection(application, InspectionStatus.PASS)
        response = client.get(f"/api/v1/inspections/{inconsistent.id}/report", headers={REQUEST_ID_HEADER: "consistency-request"})
    assert malformed.status_code == 400
    assert missing.status_code == 404
    assert response.status_code == 500
    assert response.json() == {
        "code": "DEVELOPMENT_REPORT_INCONSISTENT",
        "message": "Persisted report evidence is internally inconsistent.",
        "request_id": "consistency-request",
    }


def test_report_query_count_is_bounded_and_openapi_adds_both_gets(tmp_path):
    application = _app(tmp_path)
    with TestClient(application) as client:
        inspection = _inspection(application)
        statements = []
        def record(_c, _cur, statement, _parameters, _context, _many):
            statements.append(statement)
        engine = application.state.database.engine.sync_engine
        event.listen(engine, "before_cursor_execute", record)
        try:
            response = client.get(f"/api/v1/inspections/{inspection.id}/report")
        finally:
            event.remove(engine, "before_cursor_execute", record)
        schema = client.get("/openapi.json").json()
    assert response.status_code == 200
    assert 4 <= len([item for item in statements if item.lstrip().upper().startswith("SELECT")]) <= 10
    assert set(schema["paths"]["/api/v1/inspections/{inspection_id}/audit"]) == {"get"}
    assert set(schema["paths"]["/api/v1/inspections/{inspection_id}/report"]) == {"get"}
