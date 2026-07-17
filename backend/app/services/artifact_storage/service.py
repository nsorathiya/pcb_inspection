from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from app.db.models import ArtifactType, InspectionArtifact
from app.db.repositories import (
    InspectionArtifactCreate,
    InspectionArtifactRepository,
)
from app.services.artifact_storage.exceptions import (
    ArtifactConflictError,
    ArtifactHashMismatchError,
    ArtifactPathError,
    ArtifactRegistrationError,
    ArtifactSizeLimitError,
    ArtifactSizeMismatchError,
)
from app.services.artifact_storage.hashing import hash_file, iter_binary_chunks
from app.services.artifact_storage.models import (
    ArtifactInput,
    ArtifactSizeLimits,
    ArtifactStorageResult,
)
from app.services.artifact_storage.paths import ArtifactPathPolicy, _ArtifactPathPlan


@dataclass(frozen=True)
class _StorageOperation:
    result: ArtifactStorageResult
    created_by_operation: bool


class ArtifactStorageService:
    """Store immutable artifact bytes independently of database concerns."""

    def __init__(
        self,
        path_policy: ArtifactPathPolicy,
        size_limits: ArtifactSizeLimits,
    ) -> None:
        self._paths = path_policy
        self._limits = size_limits

    def store(self, artifact: ArtifactInput) -> ArtifactStorageResult:
        return self._store_operation(artifact).result

    def _store_operation(self, artifact: ArtifactInput) -> _StorageOperation:
        plan = self._paths._plan(
            artifact.inspection_id,
            artifact.artifact_type,
            artifact.original_filename,
        )
        self._paths._prepare_parent(plan)
        limit = self._limits.for_type(plan.artifact_type)
        temporary_path: Path | None = None
        digest = sha256()
        byte_size = 0

        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=plan.destination.parent,
                prefix=f".{plan.destination.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as output:
                for chunk in iter_binary_chunks(artifact.source):
                    candidate_size = byte_size + len(chunk)
                    if candidate_size > limit:
                        raise ArtifactSizeLimitError(
                            f"{plan.artifact_type.value} exceeds its {limit}-byte limit"
                        )
                    output.write(chunk)
                    digest.update(chunk)
                    byte_size = candidate_size
                output.flush()
                os.fsync(output.fileno())

            calculated_sha256 = digest.hexdigest()
            self._verify_expectations(
                artifact,
                calculated_sha256=calculated_sha256,
                byte_size=byte_size,
            )
            self._paths._validate_before_finalization(plan)
            created = self._finalize_exclusively(
                temporary_path,
                plan,
                calculated_sha256,
                byte_size,
            )
            temporary_path = None
            result = ArtifactStorageResult(
                artifact_type=plan.artifact_type,
                relative_path=plan.relative_path,
                sha256=calculated_sha256,
                byte_size=byte_size,
                media_type=artifact.media_type,
                original_filename=artifact.original_filename,
                idempotent_existing=not created,
            )
            return _StorageOperation(result=result, created_by_operation=created)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _verify_expectations(
        artifact: ArtifactInput,
        *,
        calculated_sha256: str,
        byte_size: int,
    ) -> None:
        if (
            artifact.expected_sha256 is not None
            and artifact.expected_sha256 != calculated_sha256
        ):
            raise ArtifactHashMismatchError(
                "calculated SHA-256 does not match expected_sha256"
            )
        if (
            artifact.expected_byte_size is not None
            and artifact.expected_byte_size != byte_size
        ):
            raise ArtifactSizeMismatchError(
                "calculated byte size does not match expected_byte_size"
            )

    def _finalize_exclusively(
        self,
        temporary_path: Path,
        plan: _ArtifactPathPlan,
        calculated_sha256: str,
        byte_size: int,
    ) -> bool:
        try:
            os.link(temporary_path, plan.destination, follow_symlinks=False)
        except FileExistsError:
            self._verify_existing(
                plan,
                calculated_sha256=calculated_sha256,
                byte_size=byte_size,
            )
            temporary_path.unlink(missing_ok=True)
            return False
        except OSError as exc:
            raise ArtifactPathError(
                "filesystem does not support safe exclusive artifact finalization"
            ) from exc

        temporary_path.unlink(missing_ok=True)
        try:
            self._fsync_directory(plan.destination.parent)
        except OSError as exc:
            plan.destination.unlink(missing_ok=True)
            raise ArtifactPathError(
                "artifact destination directory could not be synchronized"
            ) from exc
        return True

    def _verify_existing(
        self,
        plan: _ArtifactPathPlan,
        *,
        calculated_sha256: str,
        byte_size: int,
    ) -> None:
        self._paths._validate_before_finalization(plan)
        if not plan.destination.is_file():
            raise ArtifactConflictError(
                "immutable artifact destination exists but is not a regular file"
            )
        existing_sha256, existing_size = hash_file(plan.destination)
        if existing_size != byte_size or existing_sha256 != calculated_sha256:
            raise ArtifactConflictError(
                "immutable artifact destination already contains different content"
            )

    def _rollback_created(self, operation: _StorageOperation) -> None:
        if not operation.created_by_operation:
            return
        try:
            target = self._paths._absolute_from_relative(
                operation.result.relative_path
            )
        except ArtifactPathError:
            return
        if not target.is_file() or target.is_symlink():
            return
        current_sha256, current_size = hash_file(target)
        if (
            current_sha256 == operation.result.sha256
            and current_size == operation.result.byte_size
        ):
            target.unlink()
            self._fsync_directory(target.parent)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
            return
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class ArtifactRegistrationService:
    """Coordinate immutable storage with one artifact metadata registration."""

    def __init__(
        self,
        storage: ArtifactStorageService,
        artifacts: InspectionArtifactRepository,
    ) -> None:
        self._storage = storage
        self._artifacts = artifacts

    async def store_and_register(self, artifact: ArtifactInput) -> InspectionArtifact:
        operation = self._storage._store_operation(artifact)
        result = operation.result
        try:
            existing = await self._artifacts.get_by_location(
                artifact.inspection_id,
                result.artifact_type,
                result.relative_path,
            )
            if existing is not None:
                if (
                    existing.sha256 != result.sha256
                    or existing.byte_size != result.byte_size
                    or existing.media_type != result.media_type
                ):
                    raise ArtifactConflictError(
                        "registered artifact metadata conflicts with stored content"
                    )
                return existing

            return await self._artifacts.create(
                InspectionArtifactCreate(
                    inspection_id=artifact.inspection_id,
                    artifact_type=result.artifact_type,
                    relative_path=result.relative_path,
                    sha256=result.sha256,
                    byte_size=result.byte_size,
                    media_type=result.media_type,
                )
            )
        except Exception as exc:
            registered = await self._artifacts.get_by_location(
                artifact.inspection_id,
                result.artifact_type,
                result.relative_path,
            )
            registered_matches = registered is not None and (
                registered.sha256 == result.sha256
                and registered.byte_size == result.byte_size
                and registered.media_type == result.media_type
            )
            if not registered_matches:
                self._storage._rollback_created(operation)
            if isinstance(exc, ArtifactConflictError):
                raise
            raise ArtifactRegistrationError(
                "artifact database registration failed; new storage was rolled back"
            ) from exc
