from __future__ import annotations

import asyncio

from app.services.artifact_storage.hashing import hash_file
from app.services.inspection_validation.artifact_reader import ManagedArtifactPathResolver
from app.services.inspection_validation.exceptions import ArtifactResolutionError
from app.services.inspection_validation.interfaces import (
    ArtifactIntegrityInspection,
    ReadabilityStatus,
    StoredArtifactReference,
)


class StreamingFilesystemIntegrityInspector:
    def __init__(self, resolver: ManagedArtifactPathResolver) -> None:
        self._resolver = resolver

    async def inspect_integrity(self, artifact: StoredArtifactReference) -> ArtifactIntegrityInspection:
        try:
            resolved = self._resolver.resolve(artifact)
        except ArtifactResolutionError as exc:
            return ArtifactIntegrityInspection(
                artifact_type=artifact.artifact_type,
                sha256=artifact.registered_sha256,
                byte_size=artifact.registered_byte_size,
                declared_media_type=artifact.declared_media_type,
                readability_status=(ReadabilityStatus.MISSING if exc.finding_code == "ARTIFACT_FILE_MISSING" else ReadabilityStatus.INTEGRITY_FAILED),
                failure_code=exc.finding_code,
            )
        actual_sha256, actual_size = await asyncio.to_thread(hash_file, resolved)
        failure_code = None
        if actual_size != artifact.registered_byte_size:
            failure_code = "ARTIFACT_SIZE_MISMATCH"
        elif actual_sha256 != artifact.registered_sha256:
            failure_code = "ARTIFACT_SHA256_MISMATCH"
        return ArtifactIntegrityInspection(
            artifact_type=artifact.artifact_type,
            sha256=artifact.registered_sha256,
            byte_size=artifact.registered_byte_size,
            declared_media_type=artifact.declared_media_type,
            readability_status=ReadabilityStatus.READABLE if failure_code is None else ReadabilityStatus.INTEGRITY_FAILED,
            resolved_path=resolved,
            failure_code=failure_code,
        )
