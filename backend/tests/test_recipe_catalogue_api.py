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
    ArtifactType,
    AuditEvent,
    Inspection,
    InspectionArtifact,
    InspectionInferenceResult,
    InspectionPreprocessingResult,
    InspectionProcessingRun,
    InspectionStatus,
    InspectionValidation,
    Recipe,
    RecipeStatus,
)
from app.db.repositories import RecipeCreate
from app.main import create_app

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def _application(tmp_path: Path):
    return create_app(Settings(_env_file=None, runtime_root=tmp_path / "runtime"))


def _uuid(number: int) -> str:
    return f"60000000-0000-4000-8000-{number:012x}"


def _add_recipe(
    application,
    number: int,
    *,
    recipe_id: str = "RECIPE-A",
    recipe_version: str = "1.0",
    name: str = "Recipe A",
    status: RecipeStatus = RecipeStatus.DRAFT,
    created_at: datetime = NOW,
    updated_at: datetime | None = None,
) -> str:
    row_id = _uuid(number)
    asyncio.run(
        application.state.repositories.recipes.register(
            RecipeCreate(
                id=row_id,
                recipe_id=recipe_id,
                recipe_version=recipe_version,
                name=name,
                status=status,
                configuration={
                    "private_configuration": "must-not-be-loaded-or-returned",
                    "path": "C:/private/recipe.json",
                    "model_path": "C:/private/model.onnx",
                },
                created_at=created_at,
                updated_at=updated_at or created_at,
            )
        )
    )
    return row_id


def _identities(response) -> list[tuple[str, str]]:
    return [
        (item["recipe_id"], item["recipe_version"])
        for item in response.json()["items"]
    ]


def _encoded_cursor(document: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    ).decode().rstrip("=")


def test_empty_catalogue_and_request_id_contract(tmp_path) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        response = client.get(
            "/api/v1/recipes",
            headers={"X-Request-ID": "recipe-catalogue-request"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "page": {"limit": 25, "has_more": False, "next_cursor": None},
        "applied_filters": {},
        "request_id": "recipe-catalogue-request",
    }
    assert response.headers["X-Request-ID"] == "recipe-catalogue-request"


def test_listing_uses_stable_order_and_keeps_versions_separate(tmp_path) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        older = _add_recipe(
            application,
            1,
            recipe_id="RECIPE-A",
            recipe_version="1.0",
            created_at=NOW - timedelta(seconds=1),
        )
        tied_lower = _add_recipe(
            application,
            2,
            recipe_id="RECIPE-A",
            recipe_version="2.0",
            name="Recipe A version 2",
            status=RecipeStatus.ACTIVE,
            created_at=NOW,
        )
        tied_higher = _add_recipe(
            application,
            3,
            recipe_id="RECIPE-B",
            recipe_version="draft-7",
            name="Recipe B",
            status=RecipeStatus.RETIRED,
            created_at=NOW,
        )
        response = client.get("/api/v1/recipes")

    assert response.status_code == 200
    assert _identities(response) == [
        ("RECIPE-B", "draft-7"),
        ("RECIPE-A", "2.0"),
        ("RECIPE-A", "1.0"),
    ]
    assert older != tied_lower != tied_higher
    item = response.json()["items"][0]
    assert set(item) == {
        "recipe_id",
        "recipe_version",
        "name",
        "status",
        "created_at",
        "updated_at",
    }
    assert item["status"] == "RETIRED"
    forbidden = {
        "id",
        "configuration",
        "configuration_json",
        "path",
        "model",
        "active",
        "approved",
    }
    assert forbidden.isdisjoint(item)
    assert "private" not in response.text.lower()


def test_cursor_pagination_is_complete_deterministic_and_limit_independent(
    tmp_path,
) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        for number in range(31):
            _add_recipe(
                application,
                number,
                recipe_id=f"RECIPE-{number:03d}",
                recipe_version=f"build-{number}",
                created_at=NOW - timedelta(seconds=number),
            )

        first = client.get("/api/v1/recipes", params={"limit": 7})
        repeated = client.get("/api/v1/recipes", params={"limit": 7})
        assert first.status_code == repeated.status_code == 200
        assert first.json()["page"]["next_cursor"] == repeated.json()["page"][
            "next_cursor"
        ]
        assert first.json()["page"]["has_more"] is True

        seen = _identities(first)
        cursor = first.json()["page"]["next_cursor"]
        next_limit = 5
        while cursor is not None:
            page = client.get(
                "/api/v1/recipes",
                params={"limit": next_limit, "cursor": cursor},
            )
            assert page.status_code == 200, page.json()
            seen.extend(_identities(page))
            cursor = page.json()["page"]["next_cursor"]
            next_limit = 100

    assert seen == [(f"RECIPE-{number:03d}", f"build-{number}") for number in range(31)]
    assert len(seen) == len(set(seen)) == 31
    assert page.json()["page"] == {
        "limit": 100,
        "has_more": False,
        "next_cursor": None,
    }


def test_cursor_validation_and_filter_binding_fail_closed(tmp_path) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        for number in range(2):
            _add_recipe(
                application,
                number,
                recipe_id="RECIPE-A",
                recipe_version=str(number),
                created_at=NOW - timedelta(seconds=number),
            )
        page = client.get(
            "/api/v1/recipes",
            params={"limit": 1, "recipe_id": "RECIPE-A"},
        )
        cursor = page.json()["page"]["next_cursor"]
        mismatch = client.get(
            "/api/v1/recipes",
            params={"cursor": cursor, "recipe_id": "RECIPE-B"},
        )
        malformed_base64 = client.get(
            "/api/v1/recipes", params={"cursor": "not+urlsafe"}
        )
        malformed_json = client.get(
            "/api/v1/recipes",
            params={
                "cursor": base64.urlsafe_b64encode(b"not-json").decode().rstrip("=")
            },
        )

    assert mismatch.status_code == 400
    assert mismatch.json()["code"] == "RECIPE_CURSOR_FILTER_MISMATCH"
    assert malformed_base64.status_code == 400
    assert malformed_json.status_code == 400
    assert malformed_base64.json()["code"] == "INVALID_RECIPE_CURSOR"
    assert malformed_json.json()["code"] == "INVALID_RECIPE_CURSOR"


def test_cursor_rejects_version_missing_fields_identity_and_timestamp(tmp_path) -> None:
    application = _application(tmp_path)
    base = {
        "contract_version": "pcb-aoi-recipe-catalogue-cursor/1.0",
        "created_at": "2026-07-20T12:00:00Z",
        "filter_digest": "0" * 64,
        "recipe_row_id": _uuid(1),
    }
    variants = [
        _encoded_cursor({**base, "contract_version": "future/9.0"}),
        _encoded_cursor({key: value for key, value in base.items() if key != "recipe_row_id"}),
        _encoded_cursor(
            {**base, "recipe_row_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"}
        ),
        _encoded_cursor({**base, "created_at": "2026-07-20T12:00:00"}),
        _encoded_cursor({**base, "created_at": "not-a-timestamp"}),
    ]
    with TestClient(application) as client:
        responses = [
            client.get("/api/v1/recipes", params={"cursor": cursor})
            for cursor in variants
        ]

    assert [response.status_code for response in responses] == [400] * 5
    assert responses[0].json()["code"] == "UNSUPPORTED_RECIPE_CURSOR_VERSION"
    assert all("traceback" not in response.text.lower() for response in responses)


def test_exact_filters_and_and_semantics(tmp_path) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        _add_recipe(
            application,
            1,
            recipe_id="RECIPE-A",
            recipe_version="1.0",
            name="Alpha",
            status=RecipeStatus.DRAFT,
            created_at=NOW,
        )
        _add_recipe(
            application,
            2,
            recipe_id="RECIPE-A",
            recipe_version="2.0",
            name="Alpha 2",
            status=RecipeStatus.ACTIVE,
            created_at=NOW + timedelta(seconds=1),
        )
        _add_recipe(
            application,
            3,
            recipe_id="RECIPE-B",
            recipe_version="1.0",
            name="Beta",
            status=RecipeStatus.RETIRED,
            created_at=NOW + timedelta(seconds=2),
        )
        cases = [
            ({"recipe_id": "RECIPE-A"}, [("RECIPE-A", "2.0"), ("RECIPE-A", "1.0")]),
            ({"recipe_version": "1.0"}, [("RECIPE-B", "1.0"), ("RECIPE-A", "1.0")]),
            ({"name": "Alpha 2"}, [("RECIPE-A", "2.0")]),
            ({"status": "ACTIVE"}, [("RECIPE-A", "2.0")]),
            (
                {"recipe_id": "RECIPE-A", "recipe_version": "1.0", "status": "DRAFT"},
                [("RECIPE-A", "1.0")],
            ),
            ({"recipe_id": "UNKNOWN"}, []),
            ({"name": "Alpha"}, [("RECIPE-A", "1.0")]),
        ]
        for parameters, expected in cases:
            response = client.get("/api/v1/recipes", params=parameters)
            assert response.status_code == 200, (parameters, response.json())
            assert _identities(response) == expected
            assert response.json()["applied_filters"] == {
                key: value.upper() if key == "status" else value
                for key, value in parameters.items()
            }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recipe_id", "   "),
        ("recipe_version", "bad\u0007value"),
        ("recipe_id", "X" * 129),
        ("recipe_version", "X" * 129),
        ("name", "X" * 257),
        ("status", "APPROVED"),
    ],
)
def test_invalid_filters_are_rejected(tmp_path, field, value) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        response = client.get("/api/v1/recipes", params={field: value})
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_RECIPE_FILTER"


def test_limit_validation_uses_structured_422(tmp_path) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        low = client.get("/api/v1/recipes", params={"limit": 0})
        high = client.get("/api/v1/recipes", params={"limit": 101})
    assert low.status_code == high.status_code == 422
    assert low.json()["code"] == high.json()["code"] == (
        "INVALID_RECIPE_CATALOGUE_QUERY"
    )


@pytest.mark.parametrize("page_size", [0, 1, 25, 100])
def test_query_is_one_projected_select_and_catalogue_is_fully_read_only(
    tmp_path,
    page_size,
    monkeypatch,
) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        for number in range(page_size):
            _add_recipe(
                application,
                number,
                recipe_id=f"RECIPE-{number}",
                recipe_version="1.0",
                created_at=NOW - timedelta(seconds=number),
            )

        async def counts():
            async with application.state.database.session() as session:
                values = []
                for model in (
                    Recipe,
                    Inspection,
                    InspectionArtifact,
                    InspectionValidation,
                    InspectionProcessingRun,
                    InspectionPreprocessingResult,
                    InspectionInferenceResult,
                    AuditEvent,
                ):
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

        async def forbidden_service(*_args, **_kwargs):
            raise AssertionError("recipe catalogue must not execute workflow services")

        def forbidden_file_read(*_args, **_kwargs):
            raise AssertionError("recipe catalogue must not read files")

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

        statements = []

        def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement)

        event.listen(application.state.database.engine.sync_engine, "before_cursor_execute", capture)
        try:
            response = client.get("/api/v1/recipes", params={"limit": max(page_size, 1)})
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
    assert len(statements) == 1
    assert statements[0].lstrip().upper().startswith("SELECT")
    assert "configuration_json" not in statements[0].lower()


def test_returned_identity_is_accepted_unchanged_by_inspection_intake(tmp_path) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        _add_recipe(
            application,
            1,
            recipe_id="RECIPE_EXACT_17",
            recipe_version="2026.07",
            name="Exact identity recipe",
            status=RecipeStatus.DRAFT,
        )
        catalogue = client.get("/api/v1/recipes")
        selected = catalogue.json()["items"][0]
        intake = client.post(
            "/api/v1/inspections",
            data={
                "board_id": "BOARD-1",
                "recipe_id": selected["recipe_id"],
                "recipe_version": selected["recipe_version"],
            },
            files={
                "rgb_image": ("board.png", b"rgb-bytes", "image/png"),
                "height_map": (
                    "height.npy",
                    b"\x93NUMPY-height-bytes",
                    "application/octet-stream",
                ),
            },
        )

    assert catalogue.status_code == 200
    assert intake.status_code == 201, intake.json()
    assert intake.json()["lot_id"] is None
    assert intake.json()["recipe_id"] == selected["recipe_id"] == "RECIPE_EXACT_17"
    assert intake.json()["recipe_version"] == selected["recipe_version"] == (
        "2026.07"
    )


def test_persisted_unsafe_identity_and_repository_failure_are_safe_500(
    tmp_path,
    monkeypatch,
) -> None:
    application = _application(tmp_path)
    with TestClient(application) as client:
        async def insert_unsafe():
            async with application.state.database.session() as session:
                session.add(
                    Recipe(
                        id=_uuid(1),
                        recipe_id=" RECIPE-A ",
                        recipe_version="1.0",
                        name="Unsafe",
                        configuration_json="{}",
                        status=RecipeStatus.DRAFT,
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                await session.commit()

        asyncio.run(insert_unsafe())
        inconsistent = client.get("/api/v1/recipes")
        assert inconsistent.status_code == 500
        assert inconsistent.json()["code"] == "RECIPE_CATALOGUE_INCONSISTENT"

        async def fail(*_args, **_kwargs):
            raise RuntimeError("private database path and SQL")

        monkeypatch.setattr(application.state.recipe_catalogue.repository, "fetch_page", fail)
        failed = client.get("/api/v1/recipes")

    assert failed.status_code == 500
    assert failed.json()["code"] == "RECIPE_CATALOGUE_READ_FAILED"
    assert "private" not in failed.text.lower()
    assert "sql" not in failed.text.lower()


def test_openapi_documents_read_only_intake_catalogue_and_preserves_routes(tmp_path) -> None:
    schema = _application(tmp_path).openapi()
    operation = schema["paths"]["/api/v1/recipes"]["get"]
    parameters = {parameter["name"] for parameter in operation["parameters"]}
    assert parameters == {
        "limit",
        "cursor",
        "recipe_id",
        "recipe_version",
        "name",
        "status",
    }
    description = operation["description"].lower()
    assert "inspection-intake selection" in description
    assert "does not expose configuration json" in description
    assert "does not" in description and "mutate recipes" in description
    assert "production readiness" in description
    assert set(schema["paths"]["/api/v1/recipes"]) == {"get"}
    assert "/api/v1/inspections" in schema["paths"]
    assert "/api/v1/inspections/{inspection_id}/validate" in schema["paths"]
    assert "/api/v1/inspections/{inspection_id}/process" in schema["paths"]
