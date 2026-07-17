from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath

from app.core.runtime_paths import RuntimePaths
from app.db.models import ArtifactType
from app.db.repositories import InspectionArtifactRepository, InspectionRepository
from app.services.inspection_validation.exceptions import ArtifactResolutionError
from app.services.inspection_validation.interfaces import (
    RetrievedInspectionArtifacts,
    StoredArtifactReference,
)

_RAW_SUBDIRECTORY = {
    ArtifactType.RGB_RAW: "rgb",
    ArtifactType.HEIGHT_RAW: "height",
    ArtifactType.VALIDITY_MASK: "masks",
    ArtifactType.CALIBRATION: "calibration",
}


def _is_redirect(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


class ManagedArtifactPathResolver:
    def __init__(self, runtime_paths: RuntimePaths) -> None:
        self._paths = runtime_paths

    def resolve(self, artifact: StoredArtifactReference) -> Path:
        value = artifact.relative_path
        pure = PurePosixPath(value)
        if (
            not value
            or pure.is_absolute()
            or ".." in pure.parts
            or "\\" in value
            or ":" in value
            or artifact.artifact_type not in _RAW_SUBDIRECTORY
        ):
            raise ArtifactResolutionError("ARTIFACT_PATH_UNSAFE")
        expected_prefix = (
            "raw_uploads",
            artifact.inspection_id,
            _RAW_SUBDIRECTORY[artifact.artifact_type],
        )
        if pure.parts[:3] != expected_prefix or len(pure.parts) < 4:
            raise ArtifactResolutionError("ARTIFACT_PATH_UNSAFE")
        target = self._paths.root.joinpath(*pure.parts)
        root = self._paths.root
        try:
            target.resolve(strict=False).relative_to(root)
        except (OSError, ValueError) as exc:
            raise ArtifactResolutionError("ARTIFACT_PATH_UNSAFE") from exc
        current = root
        for part in pure.parts:
            current /= part
            if os.path.lexists(current) and _is_redirect(current):
                raise ArtifactResolutionError("ARTIFACT_SYMLINK_REJECTED")
        if not target.exists():
            raise ArtifactResolutionError("ARTIFACT_FILE_MISSING")
        if not target.is_file():
            raise ArtifactResolutionError("ARTIFACT_NOT_REGULAR_FILE")
        try:
            target.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise ArtifactResolutionError("ARTIFACT_PATH_UNSAFE") from exc
        return target


class DatabaseValidationArtifactRetriever:
    """Read inspection and artifact metadata without opening a write transaction."""

    def __init__(self, inspections: InspectionRepository, artifacts: InspectionArtifactRepository) -> None:
        self._inspections = inspections
        self._artifacts = artifacts

    async def get_validation_artifacts(self, inspection_id: str) -> RetrievedInspectionArtifacts:
        inspection = await self._inspections.get(inspection_id)
        records = [] if inspection is None else await self._artifacts.list_for_inspection(inspection_id)
        references = tuple(
            StoredArtifactReference(
                inspection_id=record.inspection_id,
                artifact_type=record.artifact_type,
                relative_path=record.relative_path,
                registered_sha256=record.sha256,
                registered_byte_size=record.byte_size,
                declared_media_type=record.media_type,
            )
            for record in records
        )
        return RetrievedInspectionArtifacts(
            inspection_id=inspection_id,
            artifacts=references,
            registration_evidence_available=False,
            synthetic_example=False,
        )
