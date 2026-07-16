import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import APPLICATION_LOGGER_NAME
from app.core.runtime_paths import DEFAULT_RUNTIME_ROOT, RuntimePaths
from app.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def test_default_runtime_paths_resolve_independently_of_cwd(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("PCB_AOI_RUNTIME_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    first_settings = Settings(_env_file=None)
    second_settings = Settings(_env_file=None)

    assert first_settings.runtime_root == DEFAULT_RUNTIME_ROOT
    assert second_settings.runtime_root == DEFAULT_RUNTIME_ROOT
    assert (
        RuntimePaths.from_root(first_settings.runtime_root).root
        == DEFAULT_RUNTIME_ROOT
    )


def test_runtime_root_environment_override_is_respected(
    monkeypatch,
    tmp_path,
) -> None:
    overridden_root = tmp_path / "custom-runtime"
    monkeypatch.setenv("PCB_AOI_RUNTIME_ROOT", str(overridden_root))

    settings = Settings(_env_file=None)

    assert settings.runtime_root == overridden_root.resolve()


def test_lifespan_creates_expected_runtime_directories(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    settings = Settings(_env_file=None, runtime_root=runtime_root)
    application = create_app(settings)
    runtime_paths: RuntimePaths = application.state.runtime_paths

    assert not runtime_root.exists()
    with TestClient(application):
        assert all(directory.is_dir() for directory in runtime_paths.directories)


def test_repeated_application_startup_is_idempotent(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    settings = Settings(_env_file=None, runtime_root=runtime_root)

    for _ in range(2):
        with TestClient(create_app(settings)):
            pass

    runtime_paths = RuntimePaths.from_root(runtime_root)
    assert all(directory.is_dir() for directory in runtime_paths.directories)


def test_importing_application_does_not_create_runtime_directories(tmp_path) -> None:
    runtime_root = tmp_path / "import-only-runtime"
    environment = os.environ.copy()
    environment["PCB_AOI_RUNTIME_ROOT"] = str(runtime_root)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(BACKEND_ROOT)
    if existing_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + existing_pythonpath

    subprocess.run(
        [sys.executable, "-c", "from app.main import app"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not runtime_root.exists()


def test_startup_fails_and_logs_error_when_runtime_root_is_a_file(
    tmp_path,
) -> None:
    blocked_root = tmp_path / "blocked-runtime"
    blocked_root.write_text("not a directory", encoding="utf-8")
    settings = Settings(_env_file=None, runtime_root=blocked_root)
    application = create_app(settings)
    logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    records: list[logging.LogRecord] = []

    class RecordingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = RecordingHandler()
    logger.addHandler(handler)
    try:
        with pytest.raises(OSError):
            with TestClient(application):
                pass
    finally:
        logger.removeHandler(handler)
        handler.close()

    error_records = [record for record in records if record.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert "Runtime directory initialization failed" in error_records[0].getMessage()
    assert str(blocked_root.resolve()) in error_records[0].getMessage()
    assert not any(
        record.getMessage().startswith("Application startup") for record in records
    )
