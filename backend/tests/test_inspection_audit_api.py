import asyncio
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.core.config import Settings
from app.core.request_context import REQUEST_ID_HEADER
from app.db.models import AuditEvent, InspectionStatus
from app.db.repositories import AuditEventCreate, InspectionCreate
from app.main import create_app
from app.services.inspection_audit.cursor import encode_cursor
from app.services.inspection_audit.models import AuditPosition


def _app(tmp_path):
    return create_app(Settings(_env_file=None, runtime_root=tmp_path / "runtime"))


def _inspection(application):
    return asyncio.run(application.state.repositories.inspections.create(InspectionCreate(
        status=InspectionStatus.RECEIVED,
        board_id="BOARD-A",
        recipe_id="RECIPE-A",
        recipe_version="1.0",
    )))


def _insert(application, inspection_id, *, event_id, action, created_at, details=None):
    async def add():
        async with application.state.database.session_factory() as session:
            session.add(AuditEvent(
                id=event_id,
                entity_type="inspection",
                entity_id=inspection_id,
                action=action,
                actor_id="operator-1",
                request_id="historical-request",
                details_json=json.dumps(details or {}, sort_keys=True, separators=(",", ":")),
                created_at=created_at,
            ))
            await session.commit()
    asyncio.run(add())


def test_unknown_and_empty_inspection_timelines(tmp_path):
    application = _app(tmp_path)
    with TestClient(application) as client:
        missing = client.get(f"/api/v1/inspections/{uuid4()}/audit")
        inspection = _inspection(application)
        empty = client.get(f"/api/v1/inspections/{inspection.id}/audit")
    assert missing.status_code == 404
    assert missing.json()["code"] == "INSPECTION_NOT_FOUND"
    assert empty.status_code == 200
    assert empty.json()["items"] == []
    assert empty.json()["page"] == {"limit": 50, "has_more": False, "next_cursor": None}


def test_ordering_pagination_ownership_and_no_write(tmp_path):
    application = _app(tmp_path)
    moment = datetime(2026, 7, 21, 8, tzinfo=timezone.utc)
    with TestClient(application) as client:
        inspection = _inspection(application)
        other = _inspection(application)
        ids = [
            "00000000-0000-4000-8000-000000000003",
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
        ]
        _insert(application, inspection.id, event_id=ids[0], action="THIRD", created_at=moment + timedelta(seconds=1))
        _insert(application, inspection.id, event_id=ids[1], action="FIRST", created_at=moment)
        _insert(application, inspection.id, event_id=ids[2], action="SECOND", created_at=moment)
        _insert(application, other.id, event_id=str(uuid4()), action="OTHER", created_at=moment)
        seen = []
        cursor = None
        while True:
            response = client.get(
                f"/api/v1/inspections/{inspection.id}/audit",
                params={"limit": 1, **({"cursor": cursor} if cursor else {})},
            )
            assert response.status_code == 200, response.json()
            payload = response.json()
            seen.extend(item["audit_event_id"] for item in payload["items"])
            if not payload["page"]["has_more"]:
                break
            cursor = payload["page"]["next_cursor"]
        persisted = asyncio.run(application.state.repositories.audit_events.list_for_entity("inspection", inspection.id))
    assert seen == [ids[1], ids[2], ids[0]]
    assert len(seen) == len(set(seen)) == 3
    assert len(persisted) == 3


@pytest.mark.parametrize("cursor", ["not-base64", "e30", "eyJ2ZXJzaW9uIjoyfQ"])
def test_malformed_or_unsupported_cursor_is_rejected(tmp_path, cursor):
    application = _app(tmp_path)
    with TestClient(application) as client:
        inspection = _inspection(application)
        response = client.get(f"/api/v1/inspections/{inspection.id}/audit", params={"cursor": cursor})
    assert response.status_code == 400
    assert response.json()["code"] in {"INVALID_AUDIT_CURSOR", "UNSUPPORTED_AUDIT_CURSOR_VERSION"}


def test_cursor_inspection_mismatch_and_timezone_less_timestamp(tmp_path):
    application = _app(tmp_path)
    with TestClient(application) as client:
        inspection = _inspection(application)
        other = _inspection(application)
        mismatch = encode_cursor(AuditPosition(datetime.now(timezone.utc), str(uuid4()), other.id))
        response = client.get(f"/api/v1/inspections/{inspection.id}/audit", params={"cursor": mismatch})
        raw = json.dumps({
            "version": 1, "created_at": "2026-07-21T08:00:00",
            "event_id": str(uuid4()), "inspection_id": inspection.id,
        }, sort_keys=True, separators=(",", ":")).encode()
        import base64
        naive = base64.urlsafe_b64encode(raw).decode().rstrip("=")
        naive_response = client.get(f"/api/v1/inspections/{inspection.id}/audit", params={"cursor": naive})
    assert response.status_code == 400
    assert response.json()["code"] == "AUDIT_CURSOR_INSPECTION_MISMATCH"
    assert naive_response.status_code == 400
    assert naive_response.json()["code"] == "INVALID_AUDIT_CURSOR"


def test_safe_projection_redacts_unknown_path_and_nested_values(tmp_path):
    application = _app(tmp_path)
    now = datetime.now(timezone.utc)
    with TestClient(application) as client:
        inspection = _inspection(application)
        _insert(application, inspection.id, event_id=str(uuid4()), action="INSPECTION_VALIDATION_PASSED", created_at=now, details={
            "validation_id": str(uuid4()), "inspection_status": "READY",
            "unknown": "secret", "result_path": "C:\\private\\result.json",
            "policy_id": "development-native-rgb-height",
            "validator_version": {"nested_path": "/private/value"},
        })
        _insert(application, inspection.id, event_id=str(uuid4()), action="UNKNOWN_ACTION", created_at=now + timedelta(seconds=1), details={"arbitrary": "secret"})
        response = client.get(
            f"/api/v1/inspections/{inspection.id}/audit",
            headers={REQUEST_ID_HEADER: "current-request"},
        )
    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == response.json()["request_id"] == "current-request"
    known, unknown = response.json()["items"]
    assert set(known["details"]) == {"validation_id", "inspection_status", "policy_id"}
    assert known["details_redacted"] is True
    assert known["development_only"] is True
    assert unknown["details"] == {}
    assert unknown["details_redacted"] is True
    assert "private" not in response.text
    assert "secret" not in response.text


def test_audit_query_count_is_bounded(tmp_path):
    application = _app(tmp_path)
    with TestClient(application) as client:
        inspection = _inspection(application)
        for index in range(60):
            asyncio.run(application.state.repositories.audit_events.append(AuditEventCreate(
                entity_type="inspection", entity_id=inspection.id, action="EVENT",
                details={"index": index},
            )))
        statements = []
        def record(_c, _cur, statement, _parameters, _context, _many):
            statements.append(statement)
        engine = application.state.database.engine.sync_engine
        event.listen(engine, "before_cursor_execute", record)
        try:
            response = client.get(f"/api/v1/inspections/{inspection.id}/audit")
        finally:
            event.remove(engine, "before_cursor_execute", record)
    assert response.status_code == 200
    assert len([item for item in statements if item.lstrip().upper().startswith("SELECT")]) == 2
    assert response.json()["page"]["has_more"] is True


def test_audit_limit_validation_and_malformed_inspection_id(tmp_path):
    application = _app(tmp_path)
    with TestClient(application) as client:
        inspection = _inspection(application)
        assert client.get(f"/api/v1/inspections/{inspection.id}/audit", params={"limit": 201}).status_code == 422
        malformed = client.get("/api/v1/inspections/not-a-uuid/audit")
    assert malformed.status_code == 400
    assert malformed.json()["code"] == "INVALID_INSPECTION_ID"
