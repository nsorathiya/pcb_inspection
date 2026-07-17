"""Typed boundaries for a future paired-inspection validator.

This module deliberately contains no orchestration, filesystem access, database
write, status transition, or validation execution implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.db.models import ArtifactType


class ValidationOutcome(str, Enum):
    VALIDATION_PASSED = "VALIDATION_PASSED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class FindingSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class FindingCategory(str, Enum):
    PAIR = "PAIR"
    FILE_INTEGRITY = "FILE_INTEGRITY"
    FORMAT = "FORMAT"
    IMAGE_PROPERTIES = "IMAGE_PROPERTIES"
    HEIGHT_PROPERTIES = "HEIGHT_PROPERTIES"
    REGISTRATION_EVIDENCE = "REGISTRATION_EVIDENCE"
    CALIBRATION_EVIDENCE = "CALIBRATION_EVIDENCE"
    POLICY = "POLICY"
    INTERNAL = "INTERNAL"


class ReadabilityStatus(str, Enum):
    READABLE = "READABLE"
    MISSING = "MISSING"
    UNREADABLE = "UNREADABLE"
    INTEGRITY_FAILED = "INTEGRITY_FAILED"
    UNINSPECTED = "UNINSPECTED"


class DimensionRelationship(str, Enum):
    SAME_DIMENSIONS_REQUIRED = "SAME_DIMENSIONS_REQUIRED"
    DIFFERENT_DIMENSIONS_ALLOWED = "DIFFERENT_DIMENSIONS_ALLOWED"
    REGISTERED_TRANSFORM_REQUIRED = "REGISTERED_TRANSFORM_REQUIRED"


@dataclass(frozen=True)
class StoredArtifactReference:
    """Internal repository-to-filesystem reference; never part of public results."""

    inspection_id: str
    artifact_type: ArtifactType
    relative_path: str
    registered_sha256: str
    registered_byte_size: int
    declared_media_type: str | None


@dataclass(frozen=True)
class ArtifactIntegrityInspection:
    artifact_type: ArtifactType
    sha256: str | None
    byte_size: int | None
    declared_media_type: str | None
    readability_status: ReadabilityStatus
    resolved_path: Path | None = None
    failure_code: str | None = None


@dataclass(frozen=True)
class ArtifactTechnicalSummary:
    artifact_type: ArtifactType
    sha256: str | None
    byte_size: int | None
    declared_media_type: str | None
    detected_format: str | None
    width: int | None
    height: int | None
    channels: int | None
    bit_depth: int | None
    storage_data_type: str | None
    readability_status: ReadabilityStatus
    color_mode: str | None = None
    source_extension: str | None = None
    observed_storage_data_type: str | None = None


@dataclass(frozen=True)
class InspectionValidationPolicy:
    contract_version: str
    policy_id: str
    policy_version: str
    display_name: str
    description: str
    development_only: bool
    allowed_rgb_formats: tuple[str, ...]
    allowed_height_formats: tuple[str, ...]
    allowed_rgb_channels: tuple[int, ...]
    allowed_rgb_bit_depths: tuple[int, ...]
    allowed_height_storage_types: tuple[str, ...]
    allowed_height_invalid_value_policies: tuple[str, ...]
    minimum_height_bit_depth: int
    require_single_channel_height: bool
    require_explicit_height_invalid_value_policy: bool
    minimum_width: int
    minimum_height: int
    maximum_width: int
    maximum_height: int
    dimension_relationship: DimensionRelationship
    require_calibration_artifact: bool
    require_validity_mask: bool
    require_registration_evidence: bool
    warning_as_blocking: bool = False


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    severity: FindingSeverity
    category: FindingCategory
    message: str
    blocking: bool
    artifact_type: ArtifactType | None = None
    field: str | None = None
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ValidationSummary:
    finding_count: int
    info_count: int
    warning_count: int
    error_count: int
    blocking_count: int
    technically_ready: bool
    synthetic_example: bool = False


@dataclass(frozen=True)
class InspectionValidationResult:
    contract_version: str
    validation_id: str
    inspection_id: str
    validation_policy_id: str
    validation_policy_version: str
    outcome: ValidationOutcome
    started_at: datetime
    completed_at: datetime
    validator_version: str
    rgb_artifact: ArtifactTechnicalSummary
    height_artifact: ArtifactTechnicalSummary
    findings: tuple[ValidationFinding, ...]
    summary: ValidationSummary

    def to_dict(self) -> dict[str, Any]:
        from app.services.inspection_validation.models import result_to_dict

        return result_to_dict(self)


@dataclass(frozen=True)
class RetrievedInspectionArtifacts:
    inspection_id: str
    artifacts: tuple[StoredArtifactReference, ...]
    registration_evidence_available: bool = False
    synthetic_example: bool = False


@dataclass(frozen=True)
class NativeFormatInspection:
    summary: ArtifactTechnicalSummary
    findings: tuple[ValidationFinding, ...] = ()


class ValidationArtifactRetriever(Protocol):
    async def get_validation_artifacts(
        self,
        inspection_id: str,
    ) -> RetrievedInspectionArtifacts: ...


class FilesystemIntegrityInspector(Protocol):
    async def inspect_integrity(
        self,
        artifact: StoredArtifactReference,
    ) -> ArtifactIntegrityInspection: ...


class NativeFormatInspector(Protocol):
    async def inspect_native_format(
        self,
        artifact: StoredArtifactReference,
        integrity: ArtifactIntegrityInspection,
    ) -> NativeFormatInspection: ...


class ValidationPolicyEvaluator(Protocol):
    def evaluate_policy(
        self,
        policy: InspectionValidationPolicy,
        artifacts: Sequence[ArtifactTechnicalSummary],
        *,
        registered_artifacts: Sequence[StoredArtifactReference] = (),
        registration_evidence_available: bool = False,
    ) -> Sequence[ValidationFinding]: ...


class ValidationResultPersistence(Protocol):
    async def save_validation_result(
        self,
        result: InspectionValidationResult,
    ) -> None: ...


class InspectionValidationStatusTransition(Protocol):
    async def apply_validation_outcome(
        self,
        inspection_id: str,
        outcome: ValidationOutcome,
    ) -> None: ...


class InspectionPairValidator(Protocol):
    async def validate_inspection_pair(
        self,
        inspection_id: str,
        policy: InspectionValidationPolicy,
    ) -> InspectionValidationResult: ...
