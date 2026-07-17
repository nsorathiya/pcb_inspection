import logging
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import APPLICATION_LOGGER_NAME, is_application_handler
from app.core.request_context import REQUEST_ID_HEADER
from app.main import create_app


def default_settings(monkeypatch, runtime_root: Path) -> Settings:
    for name in (
        "PCB_AOI_APPLICATION_NAME",
        "PCB_AOI_APPLICATION_VERSION",
        "PCB_AOI_ENVIRONMENT",
        "PCB_AOI_API_PREFIX",
        "PCB_AOI_DEBUG",
        "PCB_AOI_LOG_LEVEL",
        "PCB_AOI_LOG_FORMAT",
        "PCB_AOI_RUNTIME_ROOT",
        "PCB_AOI_DATABASE_FILENAME",
        "PCB_AOI_SQLITE_BUSY_TIMEOUT_MS",
        "PCB_AOI_DATABASE_ECHO",
    ):
        monkeypatch.delenv(name, raising=False)

    return Settings(_env_file=None, runtime_root=runtime_root)


def test_create_app_does_not_duplicate_logging_handlers(monkeypatch, tmp_path) -> None:
    settings = default_settings(monkeypatch, tmp_path / "runtime")

    first_application = create_app(settings)
    second_application = create_app(settings)

    logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    managed_handlers = [
        handler for handler in logger.handlers if is_application_handler(handler)
    ]
    assert len(managed_handlers) == 1
    assert first_application.state.logger is logger
    assert second_application.state.logger is logger


def test_startup_smoke_and_health_endpoint(monkeypatch, tmp_path) -> None:
    settings = default_settings(monkeypatch, tmp_path / "runtime")

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"]
    assert payload["version"]
    assert payload["environment"]
    assert payload == {
        "status": "ok",
        "service": "pcb-aoi-api",
        "version": "0.1.0",
        "environment": "development",
    }


def test_lifecycle_logs_include_startup_context_and_shutdown(
    monkeypatch,
    tmp_path,
) -> None:
    settings = default_settings(monkeypatch, tmp_path / "runtime")
    application = create_app(settings)
    logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    records: list[logging.LogRecord] = []

    class RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = RecordingHandler()
    logger.addHandler(handler)
    try:
        with TestClient(application):
            pass
    finally:
        logger.removeHandler(handler)
        handler.close()

    messages = [record.getMessage() for record in records]
    assert messages == [
        "Application startup service=pcb-aoi-api version=0.1.0 "
        f"environment=development runtime_root={settings.runtime_root}",
        "Application shutdown",
    ]


def test_supplied_request_id_is_returned_unchanged(monkeypatch, tmp_path) -> None:
    settings = default_settings(monkeypatch, tmp_path / "runtime")
    supplied_request_id = "operator-request-123"

    with TestClient(create_app(settings)) as client:
        response = client.get(
            "/api/v1/health",
            headers={REQUEST_ID_HEADER: supplied_request_id},
        )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == supplied_request_id


def test_missing_request_id_is_generated_as_uuid(monkeypatch, tmp_path) -> None:
    settings = default_settings(monkeypatch, tmp_path / "runtime")

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    generated_request_id = response.headers[REQUEST_ID_HEADER]
    assert str(UUID(generated_request_id)) == generated_request_id
