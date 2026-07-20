from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from app.db.models import ArtifactType, InspectionStatus
from app.db.repositories import Repositories
from app.db.validation_types import ValidationOutcome
from app.services.artifact_storage.hashing import hash_file
from app.services.inspection_preprocessing.models import (
    ArtifactInputIdentity,
    ValidatedArtifactSource,
    ValidatedInspectionInput,
)
from app.services.inspection_processing.exceptions import (
    ProcessingArtifactPreflightError,
    ProcessingExecutionArtifactPairError,
    ProcessingExecutionConsistencyError,
    ProcessingExecutionInspectionNotFoundError,
    ProcessingExecutionValidationMissingError,
    ProcessingExecutionValidationNotPassedError,
)
from app.services.inspection_processing.models import canonical_uuid
from app.services.inspection_validation.artifact_reader import ManagedArtifactPathResolver
from app.services.inspection_validation.exceptions import ArtifactResolutionError
from app.services.inspection_validation.interfaces import ReadabilityStatus, StoredArtifactReference
from app.services.inspection_validation.models import (
    canonical_result_sha256,
    result_from_dict,
    result_to_dict,
)
from app.services.inspection_validation.persistence import PersistedInspectionValidation

SUPPORTED_VALIDATION_CONTRACT = "pcb-aoi-inspection-validation/1.0"
_EVIDENCE_TYPES = (ArtifactType.VALIDITY_MASK, ArtifactType.CALIBRATION)


@dataclass(frozen=True)
class RegisteredProcessingArtifact:
    artifact_id: str
    inspection_id: str
    artifact_type: ArtifactType
    relative_path: str
    sha256: str
    byte_size: int
    media_type: str | None

    def stored_reference(self) -> StoredArtifactReference:
        return StoredArtifactReference(
            inspection_id=self.inspection_id,
            artifact_type=self.artifact_type,
            relative_path=self.relative_path,
            registered_sha256=self.sha256,
            registered_byte_size=self.byte_size,
            declared_media_type=self.media_type,
        )


@dataclass(frozen=True)
class ProcessingInputSnapshot:
    inspection_id: str
    inspection_status: InspectionStatus
    validation: PersistedInspectionValidation
    rgb: RegisteredProcessingArtifact
    height: RegisteredProcessingArtifact
    evidence: tuple[RegisteredProcessingArtifact, ...]
    rgb_identity: ArtifactInputIdentity
    height_identity: ArtifactInputIdentity

    @property
    def artifacts(self) -> tuple[RegisteredProcessingArtifact, ...]:
        return (self.rgb, self.height, *self.evidence)


@dataclass(frozen=True)
class ResolvedProcessingArtifacts:
    rgb_path: Path
    height_path: Path


class InspectionProcessingInputReader:
    """Read and cross-check immutable processing identities from persistence only."""

    def __init__(self, repositories: Repositories) -> None:
        self._repositories = repositories

    async def read(self, inspection_id: str) -> ProcessingInputSnapshot:
        try:
            canonical_uuid(inspection_id, "inspection_id")
            inspection = await self._repositories.inspections.get(inspection_id)
        except ValueError as exc:
            raise ProcessingExecutionInspectionNotFoundError(
                "inspection ID is invalid"
            ) from exc
        except Exception as exc:
            raise ProcessingExecutionConsistencyError(
                "inspection processing metadata could not be read"
            ) from exc
        if inspection is None:
            raise ProcessingExecutionInspectionNotFoundError("inspection does not exist")

        try:
            validation = await self._repositories.validations.get_latest_for_inspection(
                inspection_id
            )
            records = await self._repositories.artifacts.list_for_inspection(inspection_id)
        except Exception as exc:
            raise ProcessingExecutionConsistencyError(
                "inspection processing metadata could not be read"
            ) from exc
        if validation is None:
            raise ProcessingExecutionValidationMissingError(
                "inspection has no persisted validation"
            )
        if validation.inspection_id != inspection_id:
            raise ProcessingExecutionConsistencyError(
                "persisted validation does not belong to inspection"
            )
        if validation.contract_version != SUPPORTED_VALIDATION_CONTRACT:
            raise ProcessingExecutionConsistencyError(
                "persisted validation contract is unsupported"
            )
        if validation.outcome is not ValidationOutcome.VALIDATION_PASSED:
            raise ProcessingExecutionValidationNotPassedError(
                "inspection validation did not pass"
            )

        by_type = {kind: [item for item in records if item.artifact_type is kind]
                   for kind in (ArtifactType.RGB_RAW, ArtifactType.HEIGHT_RAW, *_EVIDENCE_TYPES)}
        if len(by_type[ArtifactType.RGB_RAW]) != 1 or len(by_type[ArtifactType.HEIGHT_RAW]) != 1:
            raise ProcessingExecutionArtifactPairError(
                "registered raw artifact pair is incomplete or ambiguous"
            )
        if any(len(by_type[kind]) > 1 for kind in _EVIDENCE_TYPES):
            raise ProcessingExecutionArtifactPairError(
                "registered optional evidence is ambiguous"
            )

        rgb = self._artifact(by_type[ArtifactType.RGB_RAW][0])
        height = self._artifact(by_type[ArtifactType.HEIGHT_RAW][0])
        evidence = tuple(
            self._artifact(by_type[kind][0])
            for kind in _EVIDENCE_TYPES
            if by_type[kind]
        )
        rgb_identity, height_identity = self._validated_identities(
            validation, rgb, height
        )
        return ProcessingInputSnapshot(
            inspection_id=inspection_id,
            inspection_status=inspection.status,
            validation=validation,
            rgb=rgb,
            height=height,
            evidence=evidence,
            rgb_identity=rgb_identity,
            height_identity=height_identity,
        )

    @staticmethod
    def _artifact(record) -> RegisteredProcessingArtifact:
        return RegisteredProcessingArtifact(
            artifact_id=record.id,
            inspection_id=record.inspection_id,
            artifact_type=record.artifact_type,
            relative_path=record.relative_path,
            sha256=record.sha256,
            byte_size=record.byte_size,
            media_type=record.media_type,
        )

    @staticmethod
    def _validated_identities(
        validation: PersistedInspectionValidation,
        rgb: RegisteredProcessingArtifact,
        height: RegisteredProcessingArtifact,
    ) -> tuple[ArtifactInputIdentity, ArtifactInputIdentity]:
        try:
            result = result_from_dict(validation.result)
            document = result_to_dict(result)
            if (
                result.validation_id != validation.validation_id
                or result.inspection_id != validation.inspection_id
                or result.outcome is not ValidationOutcome.VALIDATION_PASSED
                or canonical_result_sha256(result) != validation.result_sha256
                or document["rgb_artifact"] != dict(validation.rgb_summary)
                or document["height_artifact"] != dict(validation.height_summary)
            ):
                raise ValueError("validation identity mismatch")

            def identity(summary, artifact, expected_type):
                if (
                    summary.artifact_type is not expected_type
                    or summary.readability_status is not ReadabilityStatus.READABLE
                    or summary.sha256 != artifact.sha256
                    or summary.byte_size != artifact.byte_size
                    or any(
                        value is None
                        for value in (
                            summary.detected_format,
                            summary.width,
                            summary.height,
                            summary.channels,
                            summary.bit_depth,
                        )
                    )
                ):
                    raise ValueError("validation artifact identity mismatch")
                return ArtifactInputIdentity(
                    artifact_type=expected_type.value,
                    sha256=artifact.sha256,
                    byte_size=artifact.byte_size,
                    detected_format=summary.detected_format,
                    width=summary.width,
                    height=summary.height,
                    channels=summary.channels,
                    bit_depth=summary.bit_depth,
                    storage_data_type=summary.storage_data_type,
                )

            return (
                identity(result.rgb_artifact, rgb, ArtifactType.RGB_RAW),
                identity(result.height_artifact, height, ArtifactType.HEIGHT_RAW),
            )
        except Exception as exc:
            raise ProcessingExecutionConsistencyError(
                "persisted validation evidence is internally inconsistent"
            ) from exc


class ProcessingArtifactPreflight:
    """Resolve managed raw files and verify current bytes after lifecycle begin."""

    def __init__(self, resolver: ManagedArtifactPathResolver) -> None:
        self._resolver = resolver

    async def resolve_and_verify(
        self, snapshot: ProcessingInputSnapshot
    ) -> ResolvedProcessingArtifacts:
        resolved: dict[ArtifactType, Path] = {}
        for artifact in snapshot.artifacts:
            try:
                path = self._resolver.resolve(artifact.stored_reference())
                digest, size = await asyncio.to_thread(hash_file, path)
            except (ArtifactResolutionError, OSError):
                raise ProcessingArtifactPreflightError(
                    "registered artifact failed execution integrity preflight"
                ) from None
            if digest != artifact.sha256 or size != artifact.byte_size:
                raise ProcessingArtifactPreflightError(
                    "registered artifact failed execution integrity preflight"
                )
            resolved[artifact.artifact_type] = path
        return ResolvedProcessingArtifacts(
            rgb_path=resolved[ArtifactType.RGB_RAW],
            height_path=resolved[ArtifactType.HEIGHT_RAW],
        )


def build_validated_preprocessing_input(
    snapshot: ProcessingInputSnapshot,
    resolved: ResolvedProcessingArtifacts,
) -> ValidatedInspectionInput:
    if snapshot.evidence:
        # The current result contract requires technical evidence summaries that
        # validation persistence does not store. Refuse rather than fabricate them.
        raise ProcessingExecutionArtifactPairError(
            "optional evidence is not supported by the selected synthetic executor"
        )
    return ValidatedInspectionInput(
        inspection_id=snapshot.inspection_id,
        validation_id=snapshot.validation.validation_id,
        inspection_status="READY",
        validation_outcome=ValidationOutcome.VALIDATION_PASSED.value,
        synthetic_input=True,
        rgb=ValidatedArtifactSource(snapshot.rgb_identity, resolved.rgb_path),
        height=ValidatedArtifactSource(snapshot.height_identity, resolved.height_path),
    )
