import asyncio
import hashlib
import io
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, select

from app.core.config import Settings
from app.core.runtime_paths import RuntimePaths
from app.db.database import Database
from app.db.models import ArtifactType, InspectionArtifact, InspectionStatus
from app.db.repositories import InspectionCreate, Repositories
from app.services.artifact_storage import (
    ArtifactConflictError,
    ArtifactHashMismatchError,
    ArtifactInput,
    ArtifactPathPolicy,
    ArtifactPathRedirectError,
    ArtifactRegistrationError,
    ArtifactRegistrationService,
    ArtifactSizeLimitError,
    ArtifactSizeLimits,
    ArtifactSizeMismatchError,
    ArtifactStorageResult,
    ArtifactStorageService,
    InvalidArtifactInputError,
    UnsupportedArtifactExtensionError,
)
from app.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIMIT = 1024 * 1024


def _limits(**overrides: int) -> ArtifactSizeLimits:
    values = {
        "rgb_bytes": DEFAULT_LIMIT,
        "height_bytes": DEFAULT_LIMIT,
        "mask_bytes": DEFAULT_LIMIT,
        "calibration_bytes": DEFAULT_LIMIT,
        "generated_artifact_bytes": DEFAULT_LIMIT,
    }
    values.update(overrides)
    return ArtifactSizeLimits(**values)


def _storage(
    runtime_root: Path,
    *,
    limits: ArtifactSizeLimits | None = None,
) -> ArtifactStorageService:
    paths = RuntimePaths.from_root(runtime_root)
    return ArtifactStorageService(
        ArtifactPathPolicy(paths),
        limits or _limits(),
    )


def _input(
    inspection_id: str,
    *,
    artifact_type: ArtifactType = ArtifactType.RGB_RAW,
    source=b"artifact-bytes",
    original_filename: str | None = "source.PNG",
    media_type: str | None = "image/png",
    expected_sha256: str | None = None,
    expected_byte_size: int | None = None,
) -> ArtifactInput:
    return ArtifactInput(
        inspection_id=inspection_id,
        artifact_type=artifact_type,
        source=source,
        original_filename=original_filename,
        media_type=media_type,
        expected_sha256=expected_sha256,
        expected_byte_size=expected_byte_size,
    )


def _absolute(runtime_root: Path, result: ArtifactStorageResult) -> Path:
    return runtime_root.resolve().joinpath(*result.relative_path.split("/"))


def _temporary_files(runtime_root: Path) -> list[Path]:
    if not runtime_root.exists():
        return []
    return list(runtime_root.rglob("*.tmp"))


async def _initialized_database(
    runtime_root: Path,
) -> tuple[Database, Repositories, RuntimePaths]:
    paths = RuntimePaths.from_root(runtime_root)
    paths.create_directories()
    database = Database(paths.database_file, busy_timeout_ms=5000)
    await database.initialize()
    return database, Repositories.from_session_factory(database.session_factory), paths


async def _create_inspection(repositories: Repositories, inspection_id: str) -> None:
    await repositories.inspections.create(
        InspectionCreate(
            id=inspection_id,
            status=InspectionStatus.RECEIVED,
            board_id="BOARD-1",
            recipe_id="RECIPE-1",
            recipe_version="1.0",
        )
    )


def test_importing_storage_modules_creates_no_directories(tmp_path) -> None:
    runtime_root = tmp_path / "import-only-runtime"
    environment = os.environ.copy()
    environment["PCB_AOI_RUNTIME_ROOT"] = str(runtime_root)
    environment["PYTHONPATH"] = str(BACKEND_ROOT)

    subprocess.run(
        [sys.executable, "-c", "import app.services.artifact_storage"],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not runtime_root.exists()


def test_storing_bytes_succeeds_with_expected_result_shape(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())
    result = _storage(runtime_root).store(_input(inspection_id))

    assert _absolute(runtime_root, result).read_bytes() == b"artifact-bytes"
    assert {field.name for field in fields(result)} == {
        "artifact_type",
        "relative_path",
        "sha256",
        "byte_size",
        "media_type",
        "original_filename",
        "idempotent_existing",
    }


def test_storing_binary_stream_succeeds_without_requiring_all_bytes_in_memory(
    tmp_path,
) -> None:
    runtime_root = tmp_path / "runtime"
    result = _storage(runtime_root).store(
        _input(str(uuid4()), source=io.BytesIO(b"streamed-content"))
    )

    assert _absolute(runtime_root, result).read_bytes() == b"streamed-content"


def test_sha256_is_calculated_from_exact_bytes(tmp_path) -> None:
    content = b"exact\x00binary\xffcontent"
    result = _storage(tmp_path / "runtime").store(
        _input(str(uuid4()), source=content)
    )

    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.sha256 == result.sha256.lower()


def test_byte_size_is_calculated_during_write(tmp_path) -> None:
    content = b"1234567890"
    result = _storage(tmp_path / "runtime").store(
        _input(str(uuid4()), source=io.BytesIO(content))
    )

    assert result.byte_size == len(content)


def test_expected_hash_match_succeeds(tmp_path) -> None:
    content = b"verified"
    expected = hashlib.sha256(content).hexdigest()
    result = _storage(tmp_path / "runtime").store(
        _input(str(uuid4()), source=content, expected_sha256=expected)
    )

    assert result.sha256 == expected


def test_expected_hash_mismatch_fails_and_cleans_files(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    with pytest.raises(ArtifactHashMismatchError, match="SHA-256"):
        _storage(runtime_root).store(
            _input(str(uuid4()), expected_sha256="0" * 64)
        )

    assert not list(runtime_root.rglob("rgb_raw.png"))
    assert _temporary_files(runtime_root) == []


def test_expected_byte_size_mismatch_fails_and_cleans_files(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    with pytest.raises(ArtifactSizeMismatchError, match="byte size"):
        _storage(runtime_root).store(
            _input(str(uuid4()), expected_byte_size=999)
        )

    assert not list(runtime_root.rglob("rgb_raw.png"))
    assert _temporary_files(runtime_root) == []


class _CountingStream:
    def __init__(self) -> None:
        self.read_count = 0

    def read(self, _size: int) -> bytes:
        self.read_count += 1
        return b"too-large" if self.read_count == 1 else b"should-not-be-read"


def test_per_artifact_size_limit_stops_streaming_and_cleans_temporary_file(
    tmp_path,
) -> None:
    runtime_root = tmp_path / "runtime"
    stream = _CountingStream()
    storage = _storage(runtime_root, limits=_limits(rgb_bytes=4))

    with pytest.raises(ArtifactSizeLimitError, match="4-byte"):
        storage.store(_input(str(uuid4()), source=stream))

    assert stream.read_count == 1
    assert _temporary_files(runtime_root) == []
    assert not list(runtime_root.rglob("rgb_raw.png"))


class _FailingStream:
    def __init__(self) -> None:
        self._first = True

    def read(self, _size: int) -> bytes:
        if self._first:
            self._first = False
            return b"partial"
        raise OSError("simulated source failure")


def test_temporary_file_is_removed_when_stream_read_fails(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    with pytest.raises(OSError, match="simulated"):
        _storage(runtime_root).store(
            _input(str(uuid4()), source=_FailingStream())
        )

    assert _temporary_files(runtime_root) == []
    assert not list(runtime_root.rglob("rgb_raw.png"))


class _BlockingStream:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release
        self._complete = False

    def read(self, _size: int) -> bytes:
        if self._complete:
            return b""
        self._started.set()
        if not self._release.wait(timeout=5):
            raise TimeoutError("test did not release stream")
        self._complete = True
        return b"atomic-content"


def test_final_file_is_not_visible_before_atomic_finalization(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())
    expected_final = (
        runtime_root / "raw_uploads" / inspection_id / "rgb" / "rgb_raw.png"
    )
    started = threading.Event()
    release = threading.Event()
    storage = _storage(runtime_root)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            storage.store,
            _input(
                inspection_id,
                source=_BlockingStream(started, release),
            ),
        )
        assert started.wait(timeout=5)
        assert not expected_final.exists()
        assert len(_temporary_files(runtime_root)) == 1
        release.set()
        result = future.result(timeout=5)

    assert _absolute(runtime_root, result) == expected_final.resolve()
    assert expected_final.read_bytes() == b"atomic-content"
    assert _temporary_files(runtime_root) == []


@pytest.mark.parametrize(
    "source_filename",
    [
        "../../escape.PNG",
        "C:\\Windows\\System32\\payload.PNG",
        "CON.PNG",
        ".hidden.PNG",
    ],
)
def test_client_filename_never_controls_stored_path(tmp_path, source_filename) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())
    result = _storage(runtime_root).store(
        _input(inspection_id, original_filename=source_filename)
    )

    assert result.relative_path == (
        f"raw_uploads/{inspection_id}/rgb/rgb_raw.png"
    )
    assert source_filename not in result.relative_path


@pytest.mark.parametrize("filename", ["payload.exe", "image.png.exe", "script.ps1"])
def test_dangerous_or_unsupported_suffix_is_rejected(tmp_path, filename) -> None:
    runtime_root = tmp_path / "runtime"
    with pytest.raises(UnsupportedArtifactExtensionError):
        _storage(runtime_root).store(
            _input(str(uuid4()), original_filename=filename)
        )
    assert not list(runtime_root.rglob("*.*"))


def test_database_relative_path_is_portable_and_has_no_traversal(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    result = _storage(runtime_root).store(_input(str(uuid4())))

    assert not Path(result.relative_path).is_absolute()
    assert ".." not in result.relative_path.split("/")
    assert "\\" not in result.relative_path
    assert ":" not in result.relative_path


@pytest.mark.parametrize(
    ("artifact_type", "filename", "expected_prefix"),
    [
        (ArtifactType.RGB_RAW, "rgb.png", "raw_uploads/{id}/rgb/"),
        (ArtifactType.HEIGHT_RAW, "height.npy", "raw_uploads/{id}/height/"),
        (ArtifactType.VALIDITY_MASK, "mask.png", "raw_uploads/{id}/masks/"),
        (
            ArtifactType.CALIBRATION,
            "calibration.json",
            "raw_uploads/{id}/calibration/",
        ),
        (ArtifactType.RGB_PREVIEW, "rgb.png", "previews/{id}/"),
        (ArtifactType.HEIGHT_PREVIEW, "height.png", "previews/{id}/"),
        (ArtifactType.RESULT_OVERLAY, "overlay.png", "results/{id}/"),
        (ArtifactType.REPORT, "report.pdf", "reports/{id}/"),
    ],
)
def test_artifact_types_route_to_expected_runtime_category(
    tmp_path,
    artifact_type,
    filename,
    expected_prefix,
) -> None:
    inspection_id = str(uuid4())
    result = _storage(tmp_path / "runtime").store(
        _input(
            inspection_id,
            artifact_type=artifact_type,
            original_filename=filename,
        )
    )

    assert result.relative_path.startswith(expected_prefix.format(id=inspection_id))


def test_unknown_artifact_type_fails() -> None:
    with pytest.raises(InvalidArtifactInputError, match="unknown artifact type"):
        ArtifactInput(
            inspection_id=str(uuid4()),
            artifact_type="UNKNOWN",
            source=b"data",
        )


def test_existing_identical_artifact_is_idempotent(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())
    storage = _storage(runtime_root)

    first = storage.store(_input(inspection_id))
    second = storage.store(_input(inspection_id))

    assert first.idempotent_existing is False
    assert second.idempotent_existing is True
    assert first.relative_path == second.relative_path
    assert first.sha256 == second.sha256


def test_existing_different_content_raises_conflict_without_overwrite(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())
    storage = _storage(runtime_root)
    first = storage.store(_input(inspection_id, source=b"original"))

    with pytest.raises(ArtifactConflictError, match="different content"):
        storage.store(_input(inspection_id, source=b"replacement"))

    assert _absolute(runtime_root, first).read_bytes() == b"original"
    assert _temporary_files(runtime_root) == []


def test_symlink_escape_is_rejected_where_supported(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    paths = RuntimePaths.from_root(runtime_root)
    paths.raw_uploads.mkdir(parents=True)
    inspection_id = str(uuid4())
    redirected_inspection = paths.raw_uploads / inspection_id
    try:
        redirected_inspection.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")

    with pytest.raises(ArtifactPathRedirectError):
        _storage(runtime_root).store(_input(inspection_id))

    assert not list(outside.iterdir())


def test_missing_inspection_foreign_key_rolls_back_new_file(tmp_path) -> None:
    async def scenario() -> None:
        runtime_root = tmp_path / "runtime"
        database, repositories, paths = await _initialized_database(runtime_root)
        registration = ArtifactRegistrationService(
            _storage(runtime_root), repositories.artifacts
        )
        try:
            with pytest.raises(ArtifactRegistrationError, match="registration failed"):
                await registration.store_and_register(_input(str(uuid4())))
            async with database.session() as session:
                count = await session.scalar(
                    select(func.count()).select_from(InspectionArtifact)
                )
            assert count == 0
            assert not list(paths.raw_uploads.rglob("rgb_raw.png"))
            assert _temporary_files(runtime_root) == []
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_database_registration_succeeds_for_real_inspection(tmp_path) -> None:
    async def scenario() -> None:
        runtime_root = tmp_path / "runtime"
        inspection_id = str(uuid4())
        database, repositories, _ = await _initialized_database(runtime_root)
        registration = ArtifactRegistrationService(
            _storage(runtime_root), repositories.artifacts
        )
        try:
            await _create_inspection(repositories, inspection_id)
            record = await registration.store_and_register(_input(inspection_id))
            retrieved = await repositories.artifacts.get(record.id)
            assert retrieved is not None
            assert retrieved.inspection_id == inspection_id
            assert retrieved.artifact_type is ArtifactType.RGB_RAW
            assert retrieved.relative_path == record.relative_path
            assert retrieved.sha256 == hashlib.sha256(b"artifact-bytes").hexdigest()
            assert retrieved.byte_size == len(b"artifact-bytes")
            assert retrieved.media_type == "image/png"
            assert retrieved.created_at is not None
            assert runtime_root.joinpath(*record.relative_path.split("/")).is_file()
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_database_failure_after_storage_triggers_owned_file_cleanup(
    tmp_path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        runtime_root = tmp_path / "runtime"
        inspection_id = str(uuid4())
        database, repositories, paths = await _initialized_database(runtime_root)
        await _create_inspection(repositories, inspection_id)
        registration = ArtifactRegistrationService(
            _storage(runtime_root), repositories.artifacts
        )

        async def fail_create(_data):
            raise RuntimeError("simulated registration failure")

        monkeypatch.setattr(repositories.artifacts, "create", fail_create)
        try:
            with pytest.raises(ArtifactRegistrationError, match="registration failed"):
                await registration.store_and_register(_input(inspection_id))
            assert not list(paths.raw_uploads.rglob("rgb_raw.png"))
            assert _temporary_files(runtime_root) == []
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_database_record_is_not_created_when_filesystem_storage_fails(
    tmp_path,
) -> None:
    async def scenario() -> None:
        runtime_root = tmp_path / "runtime"
        inspection_id = str(uuid4())
        database, repositories, _ = await _initialized_database(runtime_root)
        await _create_inspection(repositories, inspection_id)
        registration = ArtifactRegistrationService(
            _storage(runtime_root), repositories.artifacts
        )
        try:
            with pytest.raises(ArtifactHashMismatchError):
                await registration.store_and_register(
                    _input(inspection_id, expected_sha256="0" * 64)
                )
            async with database.session() as session:
                count = await session.scalar(
                    select(func.count()).select_from(InspectionArtifact)
                )
            assert count == 0
        finally:
            await database.dispose()

    asyncio.run(scenario())


def test_registered_artifact_and_file_remain_after_application_restart(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())
    settings = Settings(_env_file=None, runtime_root=runtime_root)

    first_application = create_app(settings)
    with TestClient(first_application):
        asyncio.run(
            _create_inspection(first_application.state.repositories, inspection_id)
        )
        record = asyncio.run(
            first_application.state.artifact_registration.store_and_register(
                _input(inspection_id)
            )
        )
        record_id = record.id
        relative_path = record.relative_path

    second_application = create_app(settings)
    with TestClient(second_application):
        retrieved = asyncio.run(
            second_application.state.repositories.artifacts.get(record_id)
        )
        assert retrieved is not None
        assert retrieved.relative_path == relative_path
        assert runtime_root.joinpath(*relative_path.split("/")).read_bytes() == (
            b"artifact-bytes"
        )


def test_two_concurrent_identical_stores_cannot_corrupt_final_file(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    inspection_id = str(uuid4())
    storage = _storage(runtime_root)
    content = b"concurrent-content" * 1000

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                storage.store,
                _input(inspection_id, source=content),
            )
            for _ in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]

    assert {result.idempotent_existing for result in results} == {False, True}
    assert _absolute(runtime_root, results[0]).read_bytes() == content
    assert _temporary_files(runtime_root) == []


def test_size_limit_settings_are_environment_driven(monkeypatch) -> None:
    monkeypatch.setenv("PCB_AOI_MAX_RGB_BYTES", "11")
    monkeypatch.setenv("PCB_AOI_MAX_HEIGHT_BYTES", "22")
    monkeypatch.setenv("PCB_AOI_MAX_MASK_BYTES", "33")
    monkeypatch.setenv("PCB_AOI_MAX_CALIBRATION_BYTES", "44")
    monkeypatch.setenv("PCB_AOI_MAX_GENERATED_ARTIFACT_BYTES", "55")

    settings = Settings(_env_file=None)

    assert settings.max_rgb_bytes == 11
    assert settings.max_height_bytes == 22
    assert settings.max_mask_bytes == 33
    assert settings.max_calibration_bytes == 44
    assert settings.max_generated_artifact_bytes == 55


@pytest.mark.parametrize(
    "field_name",
    [
        "max_rgb_bytes",
        "max_height_bytes",
        "max_mask_bytes",
        "max_calibration_bytes",
        "max_generated_artifact_bytes",
    ],
)
def test_size_limit_settings_must_be_positive(field_name) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field_name: 0})
