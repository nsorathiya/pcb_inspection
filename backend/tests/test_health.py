from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_health_endpoint(monkeypatch) -> None:
    for name in (
        "PCB_AOI_APPLICATION_NAME",
        "PCB_AOI_APPLICATION_VERSION",
        "PCB_AOI_ENVIRONMENT",
        "PCB_AOI_API_PREFIX",
        "PCB_AOI_DEBUG",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

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
